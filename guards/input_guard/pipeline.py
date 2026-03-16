"""
Input Guard Pipeline — Orchestrates all 3 detection layers.

Layer routing logic:

  Layer 1 (always):   Pattern matcher + encoding hardening
    ├─ MALICIOUS → short-circuit, return immediately (skip L2 + L3)
    ├─ SUSPICIOUS → run Layer 2 (LLM resolves the grey area)
    └─ SAFE → skip Layer 2, run Layer 3 only

  Layer 2 (SUSPICIOUS only):  LLM classifier with ChromaDB cache
    ├─ MALICIOUS → return blocked
    └─ SAFE or SUSPICIOUS → continue to Layer 3

  Layer 3 (SAFE after L1 + L2):  Scope validator
    ├─ out_of_scope → upgrade to SUSPICIOUS
    └─ in_scope → return SAFE

This ordering minimises Claude API calls:
  - Known MALICIOUS attacks are caught at Layer 1 for free
  - SUSPICIOUS inputs (uncertain) get LLM review
  - SAFE inputs only pass the lightweight scope check
"""
import logging
from pathlib import Path
from typing import Optional

from guards.input_guard.pattern_matcher import PatternMatcher, PatternMatchResult
from guards.input_guard.llm_classifier import LLMClassifier
from guards.input_guard.scope_validator import ScopeValidator

logger = logging.getLogger(__name__)

_SCOPE_POLICY_PATH = Path(__file__).parent / "patterns" / "scope_policy.yaml"


class InputGuard:
    """
    The 3-layer Input Firewall.

    Usage
    -----
        guard = InputGuard()
        result = guard.scan("user message here")
        # result["verdict"] → "SAFE" / "SUSPICIOUS" / "MALICIOUS"
    """

    def __init__(self):
        # Layer 1: Always active (regex, <1ms, free)
        self.pattern_matcher = PatternMatcher()

        # Layer 2: LLM classifier with ChromaDB semantic cache
        self.llm_classifier  = LLMClassifier()

        # Layer 3: YAML-driven scope policy validator
        self.scope_validator = ScopeValidator(policy_path=_SCOPE_POLICY_PATH)

        logger.info(
            "[InputGuard] All 3 layers active | "
            f"L1=PatternMatcher | L2=LLMClassifier | "
            f"L3=ScopeValidator('{self.scope_validator.policy_name()}')"
        )

    def scan(self, input_text: str, context: Optional[dict] = None) -> dict:
        """
        Run input through all 3 detection layers.

        Parameters
        ----------
        input_text : str
            Raw user input to scan.
        context : dict, optional
            Additional context forwarded to Layer 3 scope validation.

        Returns
        -------
        dict:
          verdict          — "SAFE" / "SUSPICIOUS" / "MALICIOUS"
          confidence       — float 0.0–1.0
          threat_type      — str or None (dominant attack category)
          matched_patterns — list of Layer 1 pattern names that fired
          details          — human-readable explanation
          sanitized_input  — cleaned text if SUSPICIOUS, else None
          layers_run       — list of layer names that actually executed
          l2_result        — Layer 2 result dict (or None if not run)
          l3_result        — Layer 3 result dict (or None if not run)
        """
        layers_run       = []
        current_verdict  = "SAFE"
        current_conf     = 0.0
        threat_type      = None
        matched_patterns = []
        sanitized_input  = None
        details          = "No threats detected."
        l2_result        = None
        l3_result        = None

        # ── LAYER 1: Pattern Matching ─────────────────────────────────────
        l1: PatternMatchResult = self.pattern_matcher.scan(input_text)
        layers_run.append("layer1_pattern_matcher")

        current_verdict  = l1.verdict
        current_conf     = l1.confidence
        threat_type      = l1.threat_type
        matched_patterns = [m.pattern_name for m in l1.matches]
        sanitized_input  = l1.sanitized_input

        if l1.verdict == "MALICIOUS":
            details = (
                f"Layer 1 blocked: {threat_type} attack | "
                f"patterns={matched_patterns[:5]} | "
                f"obfuscation={l1.obfuscation_methods} | "
                f"conf={current_conf:.0%}"
            )
            return self._build_result(
                verdict=current_verdict, confidence=current_conf,
                threat_type=threat_type, matched_patterns=matched_patterns,
                details=details, sanitized_input=sanitized_input,
                layers_run=layers_run, l2_result=l2_result, l3_result=l3_result,
            )

        # ── LAYER 2: LLM Classifier (SUSPICIOUS inputs only) ─────────────
        if l1.verdict == "SUSPICIOUS":
            l2_result = self.llm_classifier.classify(input_text)
            layers_run.append("layer2_llm_classifier")

            l2_verdict = l2_result["verdict"]
            l2_conf    = l2_result["confidence"]
            cached_tag = " [cached]" if l2_result.get("cached") else ""

            if l2_verdict == "MALICIOUS":
                current_verdict = "MALICIOUS"
                current_conf    = max(current_conf, l2_conf)
                details = (
                    f"Layer 2 blocked{cached_tag}: LLM classified as injection | "
                    f"reasoning='{l2_result.get('reasoning', '')}' | "
                    f"conf={current_conf:.0%}"
                )
                return self._build_result(
                    verdict=current_verdict, confidence=current_conf,
                    threat_type=threat_type or "llm_classified",
                    matched_patterns=matched_patterns,
                    details=details, sanitized_input=sanitized_input,
                    layers_run=layers_run, l2_result=l2_result, l3_result=l3_result,
                )

            if l2_verdict == "SAFE":
                # LLM overrules Layer 1's suspicion
                current_verdict = "SAFE"
                current_conf    = l2_conf
                details = (
                    f"Layer 1 suspicious, Layer 2 cleared{cached_tag} | "
                    f"reasoning='{l2_result.get('reasoning', '')}'"
                )
            else:
                # L2 also SUSPICIOUS — keep SUSPICIOUS but update confidence
                current_conf = max(current_conf, l2_conf)
                details = (
                    f"Layer 1 + Layer 2 both suspicious{cached_tag} | "
                    f"conf={current_conf:.0%}"
                )

        # ── LAYER 3: Scope Validator (SAFE inputs only) ───────────────────
        if current_verdict == "SAFE":
            l3_result = self.scope_validator.validate(input_text, context)
            layers_run.append("layer3_scope_validator")

            if l3_result["out_of_scope"]:
                current_verdict = "SUSPICIOUS"
                details = (
                    f"Layer 3 flagged: out-of-scope | "
                    f"rule={l3_result['rule_triggered']} | "
                    f"reason='{l3_result['reason']}'"
                )
            else:
                details = "All layers passed. Input is clean and in-scope."

        elif current_verdict == "SUSPICIOUS" and l2_result is None:
            # SUSPICIOUS from Layer 1 only, Layer 2 was skipped (shouldn't happen)
            details = (
                f"Layer 1 flagged: possible {threat_type} | "
                f"input sanitized | conf={current_conf:.0%}"
            )

        return self._build_result(
            verdict=current_verdict, confidence=current_conf,
            threat_type=threat_type, matched_patterns=matched_patterns,
            details=details, sanitized_input=sanitized_input,
            layers_run=layers_run, l2_result=l2_result, l3_result=l3_result,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_result(**kwargs) -> dict:
        return {"guard": "input_guard", **kwargs}
