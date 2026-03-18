"""
Output Guard — Data Leakage Prevention

The Output Guard is the final line of defense before any agent response
reaches the user. It scans every response for data leakage across four
threat categories:

  Layer 1 — Sensitive Data Scanner
    Regex-based detection of secrets, credentials, and infrastructure data:
    API keys, DB connection strings, internal URLs, password pairs, JWT tokens.
    Uses TextPreprocessor variants (hex/base64 decoded) for obfuscation hardening.

  Layer 2 — PII Detector (Microsoft Presidio)
    Detects Personally Identifiable Information: names, emails, phone numbers,
    SSNs, credit cards, medical records, IP addresses, passport numbers.
    Falls back to regex scanner if Presidio/spacy are not installed.

  Layer 3 — Entropy Analyzer
    Shannon entropy sliding-window analysis to catch encoded/steganographic
    data (base64 blobs, hex dumps) hidden inside responses.

  Layer 4 — Prompt Leak Detector
    Detects when an agent response reproduces or closely paraphrases its own
    system prompt (instruction extraction attacks).
    Uses n-gram Jaccard overlap + ChromaDB cosine similarity (threshold > 0.85).

Verdicts:
  CLEAN    — Response passes all checks unchanged.
  REDACTED — Sensitive data or PII found and redacted inline.
  BLOCKED  — System prompt leakage detected; full response suppressed.

Standard guard interface:
  guard.scan(response, system_prompt=None, session_id=None) → dict
  {
      "guard":             "output_guard",
      "verdict":           "CLEAN" | "REDACTED" | "BLOCKED",
      "confidence":        float,
      "threat_type":       str | None,
      "findings":          list,          # all detected issues
      "redacted_response": str,           # safe response to deliver
      "pii_result":        dict,
      "secrets_result":    dict,
      "entropy_result":    dict,
      "leak_result":       dict | None,
  }

Graph topology (Day 5):
  protected_agent → output_guard → END
"""

import logging
from typing import Optional

from guards.output_guard.pii_detector       import PIIDetector
from guards.output_guard.sensitive_data_scanner import SensitiveDataScanner
from guards.output_guard.entropy_analyzer   import EntropyAnalyzer
from guards.output_guard.prompt_leak_detector   import PromptLeakDetector

logger = logging.getLogger(__name__)


class OutputGuard:
    """
    Orchestrates all four output scanning layers.

    Each instance creates its own component objects. In production this
    would be a singleton; for LangGraph node use a new instance per call
    is fine (components are stateless except PromptLeakDetector's registry).

    Usage:
        guard = OutputGuard()
        guard.register_system_prompt(system_prompt, session_id)
        result = guard.scan(agent_response, system_prompt, session_id)
    """

    def __init__(self):
        self._secrets  = SensitiveDataScanner()
        self._pii      = PIIDetector()
        self._entropy  = EntropyAnalyzer()
        self._leak     = PromptLeakDetector()

    # ── Public API ─────────────────────────────────────────────────────────────

    def register_system_prompt(self, prompt: str, session_id: str) -> None:
        """
        Pre-register a system prompt for later leakage detection.
        Call this when a session starts (before any agent response is scanned).
        """
        self._leak.register_system_prompt(prompt, session_id)

    def scan(
        self,
        response: str,
        system_prompt: Optional[str] = None,
        session_id: Optional[str]   = None,
    ) -> dict:
        """
        Scan an agent response for data leakage.

        If system_prompt is provided and session_id is given, the prompt is
        registered automatically before running the leak check.

        Args:
            response:      Raw agent response text.
            system_prompt: Optional system prompt text for leakage detection.
            session_id:    Session ID (required for leak detection to work).

        Returns:
            Standard guard result dict (see module docstring).
        """
        if not response:
            return self._build_result(
                verdict="CLEAN",
                confidence=0.0,
                threat_type=None,
                findings=[],
                redacted_response="",
                pii_result={"found": False, "entity_count": 0},
                secrets_result={"clean": True, "findings": []},
                entropy_result={"suspicious": False},
                leak_result=None,
            )

        # ── Auto-register system prompt if provided ────────────────────────────
        if system_prompt and session_id:
            self._leak.register_system_prompt(system_prompt, session_id)

        current_response = response
        all_findings     = []
        threat_types     = []

        # ── Layer 1: Sensitive Data (secrets, credentials) ─────────────────────
        secrets_result = self._secrets.scan(current_response)
        if not secrets_result.clean:
            current_response = secrets_result.redacted_response
            all_findings.extend([
                {
                    "layer":    "sensitive_data",
                    "category": f.category,
                    "text":     f.matched_text[:50],
                }
                for f in secrets_result.findings
            ])
            threat_types.append("credential_leakage")
            logger.info(
                f"[OutputGuard] Secrets found: {secrets_result.categories_found}"
            )

        # ── Layer 2: PII Detection ─────────────────────────────────────────────
        pii_result = self._pii.analyze(current_response)
        if pii_result.found:
            current_response = pii_result.redacted_text
            all_findings.extend([
                {
                    "layer":       "pii",
                    "entity_type": e.entity_type,
                    "score":       e.score,
                }
                for e in pii_result.entities
            ])
            threat_types.append("pii_leakage")
            logger.info(
                f"[OutputGuard] PII found: "
                f"{[e.entity_type for e in pii_result.entities]} "
                f"engine={pii_result.engine_used}"
            )

        # ── Layer 3: Entropy Analysis ──────────────────────────────────────────
        entropy_result = self._entropy.analyze(current_response)
        if entropy_result.suspicious:
            all_findings.append({
                "layer":        "entropy",
                "max_entropy":  entropy_result.max_entropy,
                "base64_found": entropy_result.base64_like_found,
                "hex_found":    entropy_result.hex_dump_found,
                "segments":     entropy_result.segment_count,
            })
            threat_types.append("encoded_data")
            # Remove suspicious high-entropy segments from response
            current_response = self._strip_entropy_segments(
                current_response, entropy_result
            )
            logger.info(
                f"[OutputGuard] Entropy suspicious: "
                f"max={entropy_result.max_entropy:.2f} "
                f"base64={entropy_result.base64_like_found}"
            )

        # ── Layer 4: Prompt Leak Detection (highest priority) ──────────────────
        leak_result = None
        if session_id:
            leak_obj    = self._leak.check(response, session_id)  # scan ORIGINAL
            leak_result = {
                "is_leak":          leak_obj.is_leak,
                "method":           leak_obj.method,
                "similarity_score": leak_obj.similarity_score,
                "matched_excerpt":  leak_obj.matched_excerpt,
                "ngram_score":      leak_obj.ngram_score,
                "cosine_score":     leak_obj.cosine_score,
            }
            if leak_obj.is_leak:
                all_findings.append({
                    "layer":            "prompt_leak",
                    "method":           leak_obj.method,
                    "similarity_score": leak_obj.similarity_score,
                    "excerpt":          leak_obj.matched_excerpt,
                })
                threat_types.append("system_prompt_leakage")
                logger.warning(
                    f"[OutputGuard] System prompt leakage detected | "
                    f"method={leak_obj.method} score={leak_obj.similarity_score:.3f}"
                )

        # ── Determine final verdict ────────────────────────────────────────────
        has_leak    = leak_result is not None and leak_result.get("is_leak", False)
        has_pii     = pii_result.found
        has_secrets = not secrets_result.clean
        has_entropy = entropy_result.suspicious

        if has_leak:
            verdict    = "BLOCKED"
            confidence = max(leak_result["similarity_score"], 0.95)
            threat_type = "system_prompt_leakage"
            # Full block: use safe replacement, not the sanitized response
            final_response = (
                "[Output Guard] Response blocked: system prompt leakage detected."
            )

        elif has_pii or has_secrets or has_entropy:
            verdict     = "REDACTED"
            confidence  = self._compute_redact_confidence(
                has_pii, has_secrets, has_entropy, pii_result, secrets_result
            )
            threat_type = " | ".join(threat_types) if threat_types else None
            final_response = current_response

        else:
            verdict        = "CLEAN"
            confidence     = 0.0
            threat_type    = None
            final_response = response

        logger.info(
            f"[OutputGuard] verdict={verdict} | confidence={confidence:.2f} | "
            f"findings={len(all_findings)} | threats={threat_types}"
        )

        return self._build_result(
            verdict=verdict,
            confidence=confidence,
            threat_type=threat_type,
            findings=all_findings,
            redacted_response=final_response,
            pii_result={
                "found":        pii_result.found,
                "entity_count": pii_result.entity_count,
                "entities":     [
                    {"type": e.entity_type, "score": e.score}
                    for e in pii_result.entities
                ],
                "engine":       pii_result.engine_used,
            },
            secrets_result={
                "clean":            secrets_result.clean,
                "findings_count":   len(secrets_result.findings),
                "categories_found": secrets_result.categories_found,
            },
            entropy_result={
                "suspicious":    entropy_result.suspicious,
                "max_entropy":   entropy_result.max_entropy,
                "base64_found":  entropy_result.base64_like_found,
                "hex_found":     entropy_result.hex_dump_found,
                "segment_count": entropy_result.segment_count,
            },
            leak_result=leak_result,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_result(
        verdict: str,
        confidence: float,
        threat_type: Optional[str],
        findings: list,
        redacted_response: str,
        pii_result: dict,
        secrets_result: dict,
        entropy_result: dict,
        leak_result: Optional[dict],
    ) -> dict:
        return {
            "guard":             "output_guard",
            "verdict":           verdict,
            "confidence":        confidence,
            "threat_type":       threat_type,
            "findings":          findings,
            "redacted_response": redacted_response,
            "pii_result":        pii_result,
            "secrets_result":    secrets_result,
            "entropy_result":    entropy_result,
            "leak_result":       leak_result,
        }

    @staticmethod
    def _compute_redact_confidence(
        has_pii:     bool,
        has_secrets: bool,
        has_entropy: bool,
        pii_result,
        secrets_result,
    ) -> float:
        """
        Compute confidence score for REDACTED verdicts.
        Higher confidence = more certain there's a real leakage issue.
        """
        score = 0.0
        if has_secrets:
            score += 0.60
        if has_pii:
            # Average Presidio entity score, or 0.75 for regex fallback
            if pii_result.entities:
                avg = sum(e.score for e in pii_result.entities) / len(pii_result.entities)
                score += 0.50 * avg
            else:
                score += 0.40
        if has_entropy:
            score += 0.20
        return min(1.0, score)

    @staticmethod
    def _strip_entropy_segments(response: str, entropy_result) -> str:
        """
        Remove high-entropy segments from the response text.
        Works end→start to preserve offsets.
        """
        if not entropy_result.high_entropy_segments:
            return response

        result = response
        for seg in sorted(
            entropy_result.high_entropy_segments,
            key=lambda s: s.start,
            reverse=True
        ):
            if seg.start >= 0 and seg.end > seg.start:
                result = (
                    result[:seg.start]
                    + "[REDACTED:ENCODED_DATA]"
                    + result[seg.end:]
                )
        return result
