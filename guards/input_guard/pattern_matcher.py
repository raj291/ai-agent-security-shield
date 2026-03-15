"""
Input Guard — Layer 1: Pattern Matcher

First and fastest line of defense. Three responsibilities:

  1. TEXT PREPROCESSING — defeat obfuscation before patterns run.
     (Delegated to TextPreprocessor in preprocessor.py)

  2. PATTERN MATCHING — 65+ regex patterns across 8 attack categories.
     Patterns are loaded from patterns/attack_patterns.yaml at startup.
     Runs against ALL text variants simultaneously.

  3. TYPOGLYCEMIA DETECTION — catches scrambled-word attacks.
     (Delegated to TypoglycemiaDetector in preprocessor.py)

Why regex first?
  - Zero API cost
  - Sub-millisecond latency
  - Deterministic — same input, same result, always
  - Catches 80–90% of known attacks after encoding hardening

Verdict thresholds:
  - SAFE:        confidence < 0.30
  - SUSPICIOUS:  0.30 ≤ confidence < 0.70  (sanitize + monitor)
  - MALICIOUS:   confidence ≥ 0.70         (block entirely)

Confidence formula:
  (0.7 × top_severity) + (0.3 × average_severity) + (0.10 × obfuscation_layers)
"""
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# Re-exported for backwards-compatible imports in tests and pipeline
from .preprocessor import TextPreprocessor, TypoglycemiaDetector  # noqa: F401

logger = logging.getLogger(__name__)

_PATTERNS_FILE = Path(__file__).parent / "patterns" / "attack_patterns.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# YAML LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _load_attack_patterns() -> dict[str, list[tuple[str, str, float]]]:
    """
    Load attack patterns from attack_patterns.yaml.

    Returns
    -------
    {
      "category_name": [(name, regex_string, severity), ...],
      ...
    }

    Fails gracefully — logs error and returns empty dict so the guard
    still runs (just with zero patterns, which will pass everything as SAFE).
    This avoids hard crashes on misconfigured deployments.
    """
    try:
        with open(_PATTERNS_FILE) as f:
            data = yaml.safe_load(f)

        patterns: dict[str, list[tuple[str, str, float]]] = {}
        for cat_name, cat_data in data.get("categories", {}).items():
            patterns[cat_name] = [
                (p["name"], p["regex"], float(p["severity"]))
                for p in cat_data.get("patterns", [])
            ]

        total = sum(len(v) for v in patterns.values())
        logger.info(
            f"[PatternMatcher] Loaded {total} patterns across "
            f"{len(patterns)} categories from {_PATTERNS_FILE.name}"
        )
        return patterns

    except FileNotFoundError:
        logger.error(
            f"[PatternMatcher] Pattern file not found: {_PATTERNS_FILE}. "
            "Guard will pass all inputs."
        )
        return {}
    except Exception as exc:
        logger.error(f"[PatternMatcher] Failed to load patterns: {exc}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PatternMatch:
    """A single pattern that fired on the input."""
    pattern_name: str
    category: str
    severity: float        # 0.0 (low) → 1.0 (critical)
    matched_text: str      # Substring that triggered the pattern
    detected_via: str      # Which text variant caught it: "original", "hex_decoded", etc.


@dataclass
class PatternMatchResult:
    """Full result from scanning one input through Layer 1."""
    input_text: str
    matches: list[PatternMatch]         = field(default_factory=list)
    confidence: float                   = 0.0
    verdict: str                        = "SAFE"     # SAFE / SUSPICIOUS / MALICIOUS
    threat_type: Optional[str]          = None
    sanitized_input: Optional[str]      = None

    # Obfuscation tracking
    obfuscation_methods: list[str]      = field(default_factory=list)
    contains_encoded: bool              = False
    typoglycemia_hits: list             = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN MATCHER
# ─────────────────────────────────────────────────────────────────────────────

class PatternMatcher:
    """
    Layer 1 of the Input Guard.

    Pipeline per request:
      1. TextPreprocessor generates all text variants (hex, b64, leet, unicode)
      2. All patterns (from YAML) run against ALL variants
      3. TypoglycemiaDetector scans original for scrambled keywords
      4. Confidence scoring + verdict
    """

    MALICIOUS_THRESHOLD  = 0.70
    SUSPICIOUS_THRESHOLD = 0.30

    def __init__(self):
        self._attack_patterns = _load_attack_patterns()
        self._compiled        = self._compile_patterns()
        self._preprocessor    = TextPreprocessor()
        self._typo_detector   = TypoglycemiaDetector()

        logger.info(
            f"[PatternMatcher] Ready | "
            f"Preprocessors: hex, base64, leet, unicode | "
            f"Typoglycemia keywords: {len(self._typo_detector.CRITICAL_KEYWORDS)}"
        )

    def _compile_patterns(self) -> dict:
        """Pre-compile all regex patterns at startup for performance."""
        compiled: dict[str, list] = {}
        for category, patterns in self._attack_patterns.items():
            compiled[category] = []
            for name, pattern, severity in patterns:
                try:
                    compiled[category].append(
                        (name, re.compile(pattern, re.IGNORECASE | re.DOTALL), severity)
                    )
                except re.error as exc:
                    logger.warning(
                        f"[PatternMatcher] Bad regex in '{name}': {exc}. Skipping."
                    )
        return compiled

    def scan(self, input_text: str) -> PatternMatchResult:
        """
        Full Layer 1 scan with encoding hardening.

        Steps
        -----
        1. Generate text variants via TextPreprocessor
        2. Run all compiled patterns against every variant
        3. Typoglycemia check on the original text
        4. Score and determine verdict
        """
        result = PatternMatchResult(input_text=input_text)

        # ── Step 1: Generate all text variants ─────────────────────────────
        variants = self._preprocessor.get_all_variants(input_text)

        obfuscation_found = [
            name for name, (_, was_diff) in variants.items()
            if was_diff and name != 'original'
        ]
        result.obfuscation_methods = obfuscation_found
        result.contains_encoded    = bool(obfuscation_found)

        # ── Step 2: Pattern matching across all variants ────────────────────
        all_matches: list[PatternMatch] = []

        for variant_name, (variant_text, _) in variants.items():
            for category, patterns in self._compiled.items():
                for name, compiled_re, severity in patterns:
                    match = compiled_re.search(variant_text)
                    if match:
                        all_matches.append(PatternMatch(
                            pattern_name=name,
                            category=category,
                            severity=severity,
                            matched_text=match.group()[:120],
                            detected_via=variant_name,
                        ))

        # Deduplicate: keep highest-severity match per pattern name
        seen: dict[str, PatternMatch] = {}
        for m in all_matches:
            if m.pattern_name not in seen or m.severity > seen[m.pattern_name].severity:
                seen[m.pattern_name] = m
        unique_matches = list(seen.values())
        result.matches = unique_matches

        # ── Step 3: Typoglycemia detection ─────────────────────────────────
        typo_hits = self._typo_detector.check(input_text)
        result.typoglycemia_hits = typo_hits

        # ── Step 4: Confidence scoring ──────────────────────────────────────
        if not unique_matches and not typo_hits:
            result.confidence = 0.0
        else:
            severities = sorted(
                [m.severity for m in unique_matches], reverse=True
            )

            # Typoglycemia contributions fold into severity list
            for _, _, typo_conf in typo_hits:
                severities.append(typo_conf)
                severities.sort(reverse=True)

            top = severities[0] if severities else 0.0
            avg = sum(severities) / len(severities) if severities else 0.0
            result.confidence = (0.7 * top) + (0.3 * avg)

        # Obfuscation boost: encoding itself is an attack signal (+10% per layer)
        if obfuscation_found:
            boost = 0.10 * len(obfuscation_found)
            result.confidence = min(1.0, result.confidence + boost)

        # Typoglycemia-only boost (no pattern match, but scrambled keyword found)
        if typo_hits and not unique_matches:
            result.confidence = max(
                result.confidence,
                max(c for _, _, c in typo_hits) * 0.75
            )

        result.confidence = min(1.0, result.confidence)

        # ── Step 5: Verdict ─────────────────────────────────────────────────
        if result.confidence >= self.MALICIOUS_THRESHOLD:
            result.verdict     = "MALICIOUS"
            result.threat_type = self._dominant_category(unique_matches, typo_hits)
        elif result.confidence >= self.SUSPICIOUS_THRESHOLD:
            result.verdict        = "SUSPICIOUS"
            result.threat_type    = self._dominant_category(unique_matches, typo_hits)
            result.sanitized_input = self._sanitize(input_text, unique_matches)
        else:
            result.verdict = "SAFE"

        # ── Logging ─────────────────────────────────────────────────────────
        if result.verdict != "SAFE":
            logger.warning(
                f"[PatternMatcher] {result.verdict} | "
                f"conf={result.confidence:.2f} | "
                f"patterns={[m.pattern_name for m in unique_matches[:3]]} | "
                f"obfuscation={obfuscation_found} | "
                f"typo={[h[0] for h in typo_hits[:2]]} | "
                f"input={input_text[:60]}..."
            )

        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _dominant_category(
        self,
        matches: list[PatternMatch],
        typo_hits: list[tuple],
    ) -> Optional[str]:
        """Return attack category with the highest cumulative severity."""
        scores: dict[str, float] = {}
        for m in matches:
            scores[m.category] = scores.get(m.category, 0) + m.severity
        if typo_hits and not matches:
            scores['typoglycemia'] = max(c for _, _, c in typo_hits)
        return max(scores, key=scores.get) if scores else None

    def _sanitize(self, text: str, matches: list[PatternMatch]) -> str:
        """Replace high-severity matched segments with [REDACTED]."""
        sanitized = text
        for m in matches:
            if m.severity >= 0.8:
                sanitized = sanitized.replace(m.matched_text, "[REDACTED]")
        return sanitized.strip()
