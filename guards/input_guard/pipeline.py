"""
Input Guard Pipeline - Wires all 3 detection layers.

Day 1: Layer 1 (Pattern Matcher) only.
Day 2: Will add Layer 2 (LLM Classifier) and Layer 3 (Scope Validator).

The pipeline runs layers in sequence. If any layer returns MALICIOUS,
the pipeline short-circuits and returns immediately (no need to check further).
"""
import logging
from typing import Optional
from guards.input_guard.pattern_matcher import PatternMatcher, PatternMatchResult

logger = logging.getLogger(__name__)


class InputGuard:
    """
    The 3-layer Input Firewall. Day 1 has Layer 1 active.

    Usage:
        guard = InputGuard()
        result = guard.scan("user message here")
        # result["verdict"] = SAFE / SUSPICIOUS / MALICIOUS
    """

    def __init__(self):
        # Layer 1: Always active
        self.pattern_matcher = PatternMatcher()

        # Layer 2 & 3: Initialized as None, added Day 2
        self.llm_classifier = None    # Day 2
        self.scope_validator = None   # Day 2

        logger.info("[InputGuard] Layer 1 (PatternMatcher) active. Layers 2&3 pending.")

    def scan(self, input_text: str, context: Optional[dict] = None) -> dict:
        """
        Run input through all active detection layers.

        Returns:
            dict with keys:
              - verdict: SAFE / SUSPICIOUS / MALICIOUS
              - confidence: float 0.0-1.0
              - threat_type: str or None
              - matched_patterns: list of pattern names
              - details: human-readable explanation
              - sanitized_input: cleaned text (if SUSPICIOUS)
              - layers_run: list of layer names that were executed
        """
        layers_run = []
        current_verdict = "SAFE"
        current_confidence = 0.0
        threat_type = None
        matched_patterns = []
        sanitized_input = None
        details = "No threats detected."

        # ── LAYER 1: Pattern Matching ──
        l1_result: PatternMatchResult = self.pattern_matcher.scan(input_text)
        layers_run.append("layer1_pattern_matcher")

        current_verdict = l1_result.verdict
        current_confidence = l1_result.confidence
        threat_type = l1_result.threat_type
        matched_patterns = [m.pattern_name for m in l1_result.matches]
        sanitized_input = l1_result.sanitized_input

        if l1_result.verdict == "MALICIOUS":
            details = (
                f"Layer 1 blocked: Detected {threat_type} attack. "
                f"Patterns: {', '.join(matched_patterns[:5])}. "
                f"Confidence: {current_confidence:.0%}"
            )
            return self._build_result(
                verdict=current_verdict,
                confidence=current_confidence,
                threat_type=threat_type,
                matched_patterns=matched_patterns,
                details=details,
                sanitized_input=sanitized_input,
                layers_run=layers_run,
            )

        if l1_result.verdict == "SUSPICIOUS":
            details = (
                f"Layer 1 flagged: Possible {threat_type}. "
                f"Input sanitized and forwarded with monitoring. "
                f"Confidence: {current_confidence:.0%}"
            )

        # ── LAYER 2: LLM Classifier (Day 2) ──
        if self.llm_classifier is not None:
            l2_result = self.llm_classifier.classify(input_text)
            layers_run.append("layer2_llm_classifier")
            if l2_result["verdict"] == "MALICIOUS":
                current_verdict = "MALICIOUS"
                current_confidence = max(current_confidence, l2_result["confidence"])
                details = f"Layer 2 blocked: LLM classified as injection attempt."

        # ── LAYER 3: Scope Validator (Day 2) ──
        if self.scope_validator is not None and current_verdict != "MALICIOUS":
            l3_result = self.scope_validator.validate(input_text, context)
            layers_run.append("layer3_scope_validator")
            if l3_result["out_of_scope"]:
                current_verdict = "SUSPICIOUS"
                details = f"Layer 3 flagged: Request outside agent's allowed scope."

        if current_verdict == "SAFE":
            details = "All active layers passed. Input is clean."

        return self._build_result(
            verdict=current_verdict,
            confidence=current_confidence,
            threat_type=threat_type,
            matched_patterns=matched_patterns,
            details=details,
            sanitized_input=sanitized_input,
            layers_run=layers_run,
        )

    def _build_result(self, **kwargs) -> dict:
        return {
            "guard": "input_guard",
            **kwargs
        }
