"""
Memory Guard — Document Scanner

Scans documents BEFORE they are stored in the RAG knowledge base,
and AFTER retrieval before they are injected into the LLM's context.

Two attack surfaces:
  1. Storage-time poisoning:
     Attacker uploads a document containing hidden instructions.
     → Scanned here before ChromaDB stores it.
     → Poisoned docs are blocked or sanitized.

  2. Retrieval-time poisoning (defense in depth):
     Even if a document slipped through storage-time checks,
     we re-scan every retrieved chunk before it enters the prompt.
     → Second chance to catch anything that was added later.

Architecture
------------
                 Raw document text
                        │
                        ▼
            TextPreprocessor (from Input Guard)
            ← same normalization: unicode, leet, hex, b64
                        │
                        ▼
               Pattern Matcher
               ← memory_patterns.yaml  (60+ document-specific patterns)
               ← attack_patterns.yaml  (65+ input-level patterns, reused)
                        │
                        ▼
             Metadata Inspection
             ← suspicious_metadata_keys from memory_patterns.yaml
                        │
                        ▼
           Document Density Analysis
           ← invisible char density >1% = suspicious
           ← whitespace:content ratio anomaly
                        │
                        ▼
         Confidence scoring + ScanResult

Verdict thresholds (same as Layer 1 for consistency):
  CLEAN:       confidence < 0.30
  SUSPICIOUS:  0.30 ≤ confidence < 0.70  (sanitize + store with warning tag)
  POISONED:    confidence ≥ 0.70         (block from storage entirely)
"""
import re
import logging
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from guards.input_guard.preprocessor import TextPreprocessor, TypoglycemiaDetector

logger = logging.getLogger(__name__)

_MEMORY_PATTERNS_FILE = Path(__file__).parent / "patterns" / "memory_patterns.yaml"
_INPUT_PATTERNS_FILE  = Path(__file__).parent.parent / "input_guard" / "patterns" / "attack_patterns.yaml"

# Thresholds
POISONED_THRESHOLD   = 0.70
SUSPICIOUS_THRESHOLD = 0.30

# Flag docs where >1% of characters are invisible Unicode
INVISIBLE_CHAR_DENSITY_THRESHOLD = 0.01

# Chunk size for analysing extremely long documents
_MAX_SCAN_CHARS = 50_000


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DocumentThreat:
    """A single threat found within a document."""
    pattern_name:  str
    category:      str
    severity:      float
    matched_text:  str           # Matched snippet (truncated to 120 chars)
    detected_via:  str           # 'original', 'unicode_clean', 'metadata', etc.


@dataclass
class ScanResult:
    """Full result from scanning a document through the Memory Scanner."""
    document_id:        str
    verdict:            str            = "CLEAN"      # CLEAN / SUSPICIOUS / POISONED
    confidence:         float          = 0.0
    threats:            list[DocumentThreat] = field(default_factory=list)
    threat_categories:  list[str]      = field(default_factory=list)
    sanitized_content:  Optional[str]  = None         # Cleaned text (SUSPICIOUS only)
    metadata_threats:   list[str]      = field(default_factory=list)
    invisible_char_ratio: float        = 0.0
    obfuscation_methods: list[str]     = field(default_factory=list)
    typoglycemia_hits:   list          = field(default_factory=list)
    scan_note:          str            = ""


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _load_patterns(path: Path) -> dict[str, list[tuple[str, str, float]]]:
    """
    Load patterns from a YAML file.

    Returns {category: [(name, regex, severity), ...]}
    Fails gracefully — logs error, returns empty dict.
    """
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        patterns: dict[str, list[tuple[str, str, float]]] = {}
        for cat_name, cat_data in data.get("categories", {}).items():
            patterns[cat_name] = [
                (p["name"], p["regex"], float(p["severity"]))
                for p in cat_data.get("patterns", [])
            ]
        total = sum(len(v) for v in patterns.values())
        logger.debug(f"[MemoryScanner] Loaded {total} patterns from {path.name}")
        return patterns
    except FileNotFoundError:
        logger.warning(f"[MemoryScanner] Pattern file not found: {path}")
        return {}
    except Exception as exc:
        logger.error(f"[MemoryScanner] Failed to load {path.name}: {exc}")
        return {}


def _load_suspicious_metadata_keys(path: Path) -> list[str]:
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        return [k.lower() for k in data.get("suspicious_metadata_keys", [])]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY SCANNER
# ─────────────────────────────────────────────────────────────────────────────

class MemoryScanner:
    """
    Scans documents for RAG poisoning attempts before storage or after retrieval.

    Usage
    -----
        scanner = MemoryScanner()

        # Before storage
        result = scanner.scan("document text here", doc_id="doc_001")
        if result.verdict == "POISONED":
            reject the document

        # After retrieval (defense in depth)
        result = scanner.scan(retrieved_chunk, doc_id="chunk_123")
        if result.verdict != "CLEAN":
            exclude from prompt context
    """

    def __init__(self):
        # Load memory-specific patterns
        mem_pats   = _load_patterns(_MEMORY_PATTERNS_FILE)
        # Also load input-guard patterns — injection is injection regardless of vector
        input_pats = _load_patterns(_INPUT_PATTERNS_FILE)

        # Merge: memory patterns take precedence (same name = memory version wins)
        merged = {**input_pats, **mem_pats}
        self._patterns = merged
        self._compiled = self._compile(merged)

        # Suspicious metadata keys
        self._bad_meta_keys = _load_suspicious_metadata_keys(_MEMORY_PATTERNS_FILE)

        # Shared preprocessor + typoglycemia detector (reuse from Input Guard)
        self._preprocessor  = TextPreprocessor()
        self._typo_detector = TypoglycemiaDetector()

        total_patterns = sum(len(v) for v in self._compiled.values())
        logger.info(
            f"[MemoryScanner] Ready | "
            f"patterns={total_patterns} | "
            f"bad_meta_keys={len(self._bad_meta_keys)} | "
            f"typo_keywords={len(self._typo_detector.CRITICAL_KEYWORDS)}"
        )

    # ── Init helpers ──────────────────────────────────────────────────────────

    def _compile(self, patterns: dict) -> dict:
        """Pre-compile all regex patterns at startup."""
        compiled: dict[str, list] = {}
        for category, pat_list in patterns.items():
            compiled[category] = []
            for name, regex, severity in pat_list:
                try:
                    compiled[category].append(
                        (name, re.compile(regex, re.IGNORECASE | re.DOTALL | re.MULTILINE), severity)
                    )
                except re.error as exc:
                    logger.warning(f"[MemoryScanner] Bad regex '{name}': {exc}. Skipping.")
        return compiled

    # ── Public API ────────────────────────────────────────────────────────────

    def scan(
        self,
        content: str,
        doc_id: str = "unknown",
        metadata: Optional[dict] = None,
    ) -> ScanResult:
        """
        Scan document content (and optional metadata) for RAG poisoning.

        Parameters
        ----------
        content  : str   — Full document text
        doc_id   : str   — Identifier for logging/tracking
        metadata : dict  — Document metadata (title, source, tags, etc.)

        Returns
        -------
        ScanResult with verdict: "CLEAN" / "SUSPICIOUS" / "POISONED"
        """
        result = ScanResult(document_id=doc_id)

        # Truncate very long documents for scanning (keep both ends)
        scan_content = self._truncate_for_scan(content)

        # ── 1. Density analysis (invisible chars) ──────────────────────────
        result.invisible_char_ratio = self._invisible_char_ratio(content)

        # ── 2. Preprocessing: generate all text variants ───────────────────
        variants = self._preprocessor.get_all_variants(scan_content)
        obfuscation = [
            name for name, (_, diff) in variants.items()
            if diff and name != 'original'
        ]
        result.obfuscation_methods = obfuscation

        # ── 3. Pattern matching across all variants ────────────────────────
        all_threats: list[DocumentThreat] = []

        for variant_name, (variant_text, _) in variants.items():
            for category, patterns in self._compiled.items():
                for name, compiled_re, severity in patterns:
                    match = compiled_re.search(variant_text)
                    if match:
                        all_threats.append(DocumentThreat(
                            pattern_name=name,
                            category=category,
                            severity=severity,
                            matched_text=match.group()[:120],
                            detected_via=variant_name,
                        ))

        # Deduplicate: keep highest-severity per pattern name
        seen: dict[str, DocumentThreat] = {}
        for t in all_threats:
            if t.pattern_name not in seen or t.severity > seen[t.pattern_name].severity:
                seen[t.pattern_name] = t
        result.threats = list(seen.values())

        # ── 4. Typoglycemia check ──────────────────────────────────────────
        result.typoglycemia_hits = self._typo_detector.check(scan_content)

        # ── 5. Metadata inspection ─────────────────────────────────────────
        if metadata:
            result.metadata_threats = self._scan_metadata(metadata)

        # ── 6. Confidence scoring ──────────────────────────────────────────
        result.confidence = self._score(result)

        # ── 7. Verdict ────────────────────────────────────────────────────
        if result.confidence >= POISONED_THRESHOLD:
            result.verdict = "POISONED"
        elif result.confidence >= SUSPICIOUS_THRESHOLD:
            result.verdict = "SUSPICIOUS"
            result.sanitized_content = self._sanitize(scan_content, result.threats)
        else:
            result.verdict = "CLEAN"

        # Dominant threat categories
        cat_scores: dict[str, float] = {}
        for t in result.threats:
            cat_scores[t.category] = cat_scores.get(t.category, 0) + t.severity
        if result.typoglycemia_hits and not result.threats:
            cat_scores['typoglycemia'] = max(c for _, _, c in result.typoglycemia_hits)
        result.threat_categories = sorted(cat_scores, key=cat_scores.get, reverse=True)[:3]

        # Scan note
        if result.verdict != "CLEAN":
            result.scan_note = (
                f"{result.verdict} | conf={result.confidence:.2f} | "
                f"categories={result.threat_categories[:2]} | "
                f"patterns={[t.pattern_name for t in result.threats[:3]]} | "
                f"invisible_ratio={result.invisible_char_ratio:.3f}"
            )
            logger.warning(f"[MemoryScanner] doc={doc_id} | {result.scan_note}")
        else:
            logger.debug(f"[MemoryScanner] CLEAN | doc={doc_id}")

        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _invisible_char_ratio(self, text: str) -> float:
        """Ratio of invisible/zero-width Unicode chars to total chars."""
        if not text:
            return 0.0
        invisible_count = sum(
            1 for c in text
            if c in self._preprocessor.INVISIBLE_CHARS
            or unicodedata.category(c) in ('Cf', 'Cc')  # Format / Control
        )
        return invisible_count / len(text)

    def _scan_metadata(self, metadata: dict) -> list[str]:
        """
        Check metadata keys and values for suspicious patterns.

        Returns list of triggered rule names.
        """
        triggered = []
        for key, value in metadata.items():
            key_lower = key.lower()
            # Suspicious key name
            if any(bad in key_lower for bad in self._bad_meta_keys):
                triggered.append(f"suspicious_metadata_key:{key}")
            # Suspicious value content (treat as text, run a quick check)
            if isinstance(value, str) and len(value) > 5:
                for category, patterns in self._compiled.items():
                    for name, compiled_re, severity in patterns:
                        if severity >= 0.8 and compiled_re.search(value):
                            triggered.append(f"metadata_value_injection:{name}")
                            break
        return triggered

    def _score(self, result: ScanResult) -> float:
        """Compute overall confidence score."""
        severities = sorted(
            [t.severity for t in result.threats], reverse=True
        )
        for _, _, typo_conf in result.typoglycemia_hits:
            severities.append(typo_conf)
        severities.sort(reverse=True)

        if not severities:
            base = 0.0
        else:
            top = severities[0]
            avg = sum(severities) / len(severities)
            base = (0.7 * top) + (0.3 * avg)

        # Obfuscation boost: each encoding layer adds 10%
        boost = 0.10 * len(result.obfuscation_methods)

        # Invisible character density boost (>1% is very suspicious in docs)
        if result.invisible_char_ratio > INVISIBLE_CHAR_DENSITY_THRESHOLD:
            boost += 0.15
        elif result.invisible_char_ratio > 0.001:
            boost += 0.05

        # Metadata threat boost
        boost += 0.10 * len(result.metadata_threats)

        # Typoglycemia-only (no pattern match but scrambled keyword found)
        if result.typoglycemia_hits and not result.threats:
            base = max(base, max(c for _, _, c in result.typoglycemia_hits) * 0.75)

        return min(1.0, base + boost)

    def _sanitize(self, text: str, threats: list[DocumentThreat]) -> str:
        """Redact high-severity matched segments."""
        sanitized = text
        for t in threats:
            if t.severity >= 0.8:
                sanitized = sanitized.replace(t.matched_text, "[REDACTED]")
        return sanitized.strip()

    @staticmethod
    def _truncate_for_scan(text: str) -> str:
        """
        For very long documents, scan the first and last segments.
        Attackers often hide instructions at the END of documents
        (after legitimate content that establishes trust).
        """
        if len(text) <= _MAX_SCAN_CHARS:
            return text
        half = _MAX_SCAN_CHARS // 2
        return text[:half] + "\n[...middle truncated for scan...]\n" + text[-half:]
