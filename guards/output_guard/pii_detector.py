"""
PII Detector — Output Guard Layer 1

Detects and redacts Personally Identifiable Information (PII) from agent responses
using Microsoft Presidio (presidio-analyzer + presidio-anonymizer).

Entities covered:
  PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, US_SSN, US_PASSPORT,
  MEDICAL_LICENSE, IP_ADDRESS, IBAN_CODE, LOCATION, NRP

Fail-safe design:
  If Presidio or the spacy model is unavailable, falls back to a regex-based
  pattern scanner that catches the most common PII types without spacy.
  This ensures the guard degrades gracefully rather than crashing.

Redaction format: <ENTITY_TYPE>  e.g. <EMAIL_ADDRESS>, <CREDIT_CARD>
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── PII Entities to detect ────────────────────────────────────────────────────
PRESIDIO_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "US_PASSPORT",
    "MEDICAL_LICENSE",
    "IP_ADDRESS",
    "IBAN_CODE",
    "LOCATION",
    "NRP",
]

# ── Fallback regex patterns (used when Presidio/spacy not available) ──────────
_FALLBACK_PATTERNS = {
    "EMAIL_ADDRESS": re.compile(
        r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"
    ),
    "PHONE_NUMBER": re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "US_SSN": re.compile(
        r"\b(?!000|666|9\d{2})\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}\b"
    ),
    "CREDIT_CARD": re.compile(
        r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13}|3(?:0[0-5]|[68]\d)\d{11}"
        r"|6(?:011|5\d{2})\d{12}|(?:2131|1800|35\d{3})\d{11})"
        r"(?:[-\s]?\d{4}){0,3}\b"
    ),
    "IP_ADDRESS": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
}


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class PIIEntity:
    """A single detected PII entity."""
    entity_type: str
    start: int
    end: int
    score: float
    text: str


@dataclass
class PIIResult:
    """Result from PIIDetector.analyze()"""
    found: bool
    entities: list = field(default_factory=list)   # list[PIIEntity]
    redacted_text: str = ""
    engine_used: str = "presidio"                   # "presidio" | "fallback_regex"
    entity_count: int = 0


# ─────────────────────────────────────────────
# PIIDetector
# ─────────────────────────────────────────────

class PIIDetector:
    """
    Detects and redacts PII in text using Microsoft Presidio.

    Falls back to regex-based detection if Presidio or the spacy model
    is not available in the environment.

    Usage:
        detector = PIIDetector()
        result = detector.analyze("Call me at 555-123-4567 or jane@email.com")
        print(result.redacted_text)
        # → "Call me at <PHONE_NUMBER> or <EMAIL_ADDRESS>"
    """

    def __init__(self):
        self._analyzer = None
        self._anonymizer = None
        self._mode = "fallback_regex"
        self._try_init_presidio()

    def _try_init_presidio(self) -> None:
        """
        Attempt to initialise Presidio engines.
        Silently falls back to regex mode if unavailable.
        """
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self._analyzer  = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            self._mode = "presidio"
            logger.info("[PIIDetector] Presidio initialised (full NLP mode)")
        except Exception as exc:
            logger.warning(
                f"[PIIDetector] Presidio unavailable ({exc}). "
                "Using regex fallback for PII detection."
            )
            self._mode = "fallback_regex"

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(self, text: str) -> PIIResult:
        """
        Scan text for PII. Returns a PIIResult with found entities and
        a version of the text with all PII replaced by <ENTITY_TYPE>.

        Args:
            text: Agent response text to scan.

        Returns:
            PIIResult with .found, .entities, .redacted_text
        """
        if not text or not text.strip():
            return PIIResult(found=False, redacted_text=text or "", engine_used=self._mode)

        if self._mode == "presidio":
            return self._presidio_analyze(text)
        return self._fallback_analyze(text)

    # ── Presidio path ──────────────────────────────────────────────────────────

    def _presidio_analyze(self, text: str) -> PIIResult:
        try:
            from presidio_anonymizer.entities import OperatorConfig

            # Step 1: Detect
            analyzer_results = self._analyzer.analyze(
                text=text,
                entities=PRESIDIO_ENTITIES,
                language="en",
            )

            if not analyzer_results:
                return PIIResult(
                    found=False,
                    redacted_text=text,
                    engine_used="presidio",
                )

            # Step 2: Anonymise — replace each entity with <ENTITY_TYPE>
            operators = {
                entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
                for entity in PRESIDIO_ENTITIES
            }
            anonymized = self._anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=operators,
            )

            # Step 3: Build entity list for the result
            entities = []
            for r in analyzer_results:
                entities.append(PIIEntity(
                    entity_type=r.entity_type,
                    start=r.start,
                    end=r.end,
                    score=r.score,
                    text=text[r.start:r.end],
                ))

            return PIIResult(
                found=True,
                entities=entities,
                redacted_text=anonymized.text,
                engine_used="presidio",
                entity_count=len(entities),
            )

        except Exception as exc:
            logger.warning(
                f"[PIIDetector] Presidio analysis failed ({exc}). "
                "Falling back to regex."
            )
            return self._fallback_analyze(text)

    # ── Regex fallback path ────────────────────────────────────────────────────

    def _fallback_analyze(self, text: str) -> PIIResult:
        """
        Regex-based PII detection for environments without Presidio/spacy.
        Covers: EMAIL, PHONE, SSN, CREDIT_CARD, IP_ADDRESS.
        """
        entities = []
        redacted = text

        # Process from end to start so offsets stay valid
        matches = []
        for entity_type, pattern in _FALLBACK_PATTERNS.items():
            for m in pattern.finditer(text):
                matches.append((m.start(), m.end(), entity_type, m.group()))

        # Sort by start position descending so we can safely replace from end
        matches.sort(key=lambda x: x[0], reverse=True)

        for start, end, entity_type, matched_text in matches:
            entities.append(PIIEntity(
                entity_type=entity_type,
                start=start,
                end=end,
                score=0.85,     # fixed confidence for regex matches
                text=matched_text,
            ))
            redacted = redacted[:start] + f"<{entity_type}>" + redacted[end:]

        found = len(entities) > 0
        return PIIResult(
            found=found,
            entities=entities,
            redacted_text=redacted,
            engine_used="fallback_regex",
            entity_count=len(entities),
        )
