"""
Memory Guard — Context Window Monitor

Prevents context stuffing attacks that displace the system prompt.

How a context stuffing attack works
------------------------------------
An attacker submits an enormous document (e.g. 150,000 tokens). This pushes
the system prompt far from the LLM's active generation position. Many
transformer-based LLMs exhibit reduced adherence to instructions that appear
early in an extremely long context. The attacker hides their real payload at
the END of the large document, after the system prompt has been "buried".

This monitor tracks token ratios and raises a displacement risk level:

  NONE     : document < 20% of model limit       (safe)
  LOW      : document 20–40% of model limit      (fine)
  MEDIUM   : document 40–55% of model limit      (caution)
  HIGH     : document 55–70% of model limit      (flag it)
  CRITICAL : document > 70% of model limit       (block recommended)

Token estimation: len(text) // 4  (rough 4-chars-per-token heuristic)
This is intentionally conservative — it's better to over-estimate than miss.
"""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Displacement risk thresholds (document_tokens / model_limit)
_THRESHOLDS = {
    "NONE":     0.20,   # < 20%  → safe
    "LOW":      0.40,   # 20–40% → low risk
    "MEDIUM":   0.55,   # 40–55% → medium risk
    "HIGH":     0.70,   # 55–70% → high risk (flag)
    "CRITICAL": 1.00,   # > 70%  → critical (block recommended)
}

# At HIGH or above, safe_to_process is set to False
_UNSAFE_THRESHOLD = "HIGH"

_RISK_ORDER = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContextAssessment:
    """
    Assessment of context window usage and system prompt displacement risk.

    Fields
    ------
    system_prompt_tokens : Estimated token count of the system prompt
    document_tokens      : Estimated token count of the document
    total_tokens         : system_prompt_tokens + document_tokens
    model_limit          : Token limit for this model (default 200_000 for Claude)
    retrieved_ratio      : document_tokens / model_limit (0.0 – 1.0+)
    displacement_risk    : NONE / LOW / MEDIUM / HIGH / CRITICAL
    safe_to_process      : False at HIGH or CRITICAL risk
    recommendation       : Human-readable suggestion for the caller
    """
    system_prompt_tokens: int
    document_tokens:      int
    total_tokens:         int
    model_limit:          int
    retrieved_ratio:      float
    displacement_risk:    str    # "NONE" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    safe_to_process:      bool
    recommendation:       str


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT WINDOW MONITOR
# ─────────────────────────────────────────────────────────────────────────────

class ContextWindowMonitor:
    """
    Monitors context window token usage to detect context stuffing attacks.

    A context stuffing attack floods the context with a massive document so
    the system prompt gets displaced far from the LLM's active position,
    reducing its influence. The attacker's real payload is at the document end.

    Usage
    -----
        monitor = ContextWindowMonitor()

        assessment = monitor.assess(
            system_prompt="You are a helpful security assistant...",
            document="[150,000 tokens of text here]",
            model_limit=200_000,
        )

        if not assessment.safe_to_process:
            # Truncate document or reject
            print(assessment.recommendation)

        print(assessment.displacement_risk)   # e.g. "CRITICAL"
        print(assessment.retrieved_ratio)     # e.g. 0.82
    """

    DEFAULT_MODEL_LIMIT = 200_000   # Claude's context window

    def assess(
        self,
        system_prompt: str,
        document: str,
        model_limit: int = DEFAULT_MODEL_LIMIT,
    ) -> ContextAssessment:
        """
        Assess displacement risk of adding this document to the context.

        Parameters
        ----------
        system_prompt : str  — The system prompt text (or "" if unknown)
        document      : str  — The document/text to be added to context
        model_limit   : int  — Model's token context limit (default 200_000)

        Returns
        -------
        ContextAssessment with risk level, ratio, and recommendation
        """
        sp_tokens  = self.estimate_tokens(system_prompt)
        doc_tokens = self.estimate_tokens(document)
        total      = sp_tokens + doc_tokens

        ratio = doc_tokens / max(model_limit, 1)

        risk_level = self._classify_risk(ratio)
        safe       = _RISK_ORDER.index(risk_level) < _RISK_ORDER.index(_UNSAFE_THRESHOLD)

        recommendation = self._make_recommendation(
            risk_level, doc_tokens, model_limit, ratio
        )

        logger.info(
            f"[ContextMonitor] assess | "
            f"sp={sp_tokens}tok | doc={doc_tokens}tok | total={total}tok | "
            f"ratio={ratio:.2%} | risk={risk_level} | safe={safe}"
        )

        return ContextAssessment(
            system_prompt_tokens=sp_tokens,
            document_tokens=doc_tokens,
            total_tokens=total,
            model_limit=model_limit,
            retrieved_ratio=round(ratio, 4),
            displacement_risk=risk_level,
            safe_to_process=safe,
            recommendation=recommendation,
        )

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Rough token count estimate: len(text) // 4.

        The "4 chars per token" heuristic is standard for English text and
        intentionally errs on the side of over-estimation.

        Parameters
        ----------
        text : str — Any text string

        Returns
        -------
        int — Estimated token count (0 for empty / None input)
        """
        if not text:
            return 0
        return len(text) // 4

    @staticmethod
    def _classify_risk(ratio: float) -> str:
        """
        Map document_tokens / model_limit ratio to a risk level.

        Thresholds (document fraction of model limit):
          < 0.20 → NONE
          < 0.40 → LOW
          < 0.55 → MEDIUM
          < 0.70 → HIGH
          >= 0.70 → CRITICAL
        """
        if ratio < _THRESHOLDS["NONE"]:
            return "NONE"
        elif ratio < _THRESHOLDS["LOW"]:
            return "LOW"
        elif ratio < _THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        elif ratio < _THRESHOLDS["HIGH"]:
            return "HIGH"
        else:
            return "CRITICAL"

    @staticmethod
    def _make_recommendation(
        risk: str,
        doc_tokens: int,
        model_limit: int,
        ratio: float,
    ) -> str:
        """Return a human-readable recommendation string."""
        pct = ratio * 100

        if risk == "NONE":
            return (
                f"Document uses {pct:.1f}% of context window. "
                f"No displacement risk — safe to process."
            )
        elif risk == "LOW":
            return (
                f"Document uses {pct:.1f}% of context window. "
                f"Low displacement risk — safe to process."
            )
        elif risk == "MEDIUM":
            return (
                f"Document uses {pct:.1f}% of context window. "
                f"Moderate displacement risk. Consider chunking large documents."
            )
        elif risk == "HIGH":
            return (
                f"Document uses {pct:.1f}% of context window (~{doc_tokens:,} tokens). "
                f"HIGH displacement risk — system prompt may be partially displaced. "
                f"Truncate document to below {int(model_limit * 0.55):,} tokens before processing."
            )
        else:  # CRITICAL
            return (
                f"Document uses {pct:.1f}% of context window (~{doc_tokens:,} tokens). "
                f"CRITICAL displacement risk — context stuffing attack likely. "
                f"BLOCK: do not process this document without truncation to "
                f"< {int(model_limit * 0.40):,} tokens."
            )
