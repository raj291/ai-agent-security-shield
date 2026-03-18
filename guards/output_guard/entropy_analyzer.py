"""
Entropy Analyzer — Output Guard Layer 3

Detects potentially encoded or steganographic content hidden inside agent
responses using Shannon entropy analysis and structural heuristics.

Threat model:
  An attacker could trick an agent into encoding sensitive data (API keys,
  PII, internal documents) as base64 or hex inside a "normal" looking
  response, bypassing simple keyword-based scanners.

Detection strategy:
  1. Shannon entropy scan — slide a window across the text and compute
     bits-per-character entropy. Random/encoded data scores > 4.5 bits/char.
  2. Base64 structural heuristic — long uninterrupted [A-Za-z0-9+/] runs
     of ≥ 60 characters (padded with 0-2 '=' chars) indicate base64 blobs.
  3. Hex dump heuristic — long runs of hex pairs, optionally space-separated.

Thresholds (tuned to minimise FP on normal English text):
  - entropy_threshold  : 4.5 bits/char  (English prose ≈ 3.5–4.2)
  - base64_min_length  : 60 chars       (real secrets are long)
  - hex_blob_min_length: 32 chars       (MD5 minimum)
  - window_size        : 50 chars

Verdict: returns EntropyResult.suspicious = True when any heuristic fires.
"""

import re
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Tunable thresholds ────────────────────────────────────────────────────────
_ENTROPY_THRESHOLD   = 4.5   # bits per character
_WINDOW_SIZE         = 50    # chars for sliding window
_BASE64_MIN_LEN      = 60    # minimum base64 blob length to flag
_HEX_BLOB_MIN_LEN    = 32    # minimum hex string length to flag (MD5 = 32)

# ── Structural patterns ───────────────────────────────────────────────────────
_BASE64_PATTERN = re.compile(
    r'(?:[A-Za-z0-9+/]{4}){10,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?'
)
_HEX_BLOB_PATTERN = re.compile(
    r'\b(?:[0-9a-fA-F]{2}[ \-]?){16,}\b'   # 16+ hex pairs = 32+ chars
)
_LONG_RANDOM_PATTERN = re.compile(
    r'[A-Za-z0-9+/=_\-]{' + str(_BASE64_MIN_LEN) + r',}'
)


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class EntropySegment:
    """A high-entropy or structurally suspicious text segment."""
    reason: str         # "high_entropy" | "base64_blob" | "hex_dump"
    text: str           # the suspicious text segment
    entropy: float      # Shannon entropy (0 if structural match only)
    start: int
    end: int


@dataclass
class EntropyResult:
    """Result from EntropyAnalyzer.analyze()"""
    suspicious: bool
    max_entropy: float = 0.0
    high_entropy_segments: list = field(default_factory=list)   # list[EntropySegment]
    base64_like_found: bool = False
    hex_dump_found: bool = False
    segment_count: int = 0


# ─────────────────────────────────────────────
# EntropyAnalyzer
# ─────────────────────────────────────────────

class EntropyAnalyzer:
    """
    Detects high-entropy / encoded content in agent responses.

    Uses Shannon entropy on sliding windows + structural patterns
    (base64 blobs, hex dumps) to flag potentially exfiltrated data.

    Usage:
        analyzer = EntropyAnalyzer()
        result = analyzer.analyze(response_text)
        if result.suspicious:
            print(result.high_entropy_segments)
    """

    def analyze(self, text: str) -> EntropyResult:
        """
        Scan text for high-entropy or structurally encoded segments.

        Returns EntropyResult with .suspicious flag and segment details.
        """
        if not text or len(text) < _WINDOW_SIZE:
            # Too short to meaningfully analyse
            return EntropyResult(suspicious=False)

        segments: list[EntropySegment] = []

        # ── Pass 1: Structural patterns ────────────────────────────────────────
        b64_segments = self._check_base64(text)
        hex_segments = self._check_hex_dump(text)
        segments.extend(b64_segments)
        segments.extend(hex_segments)

        # ── Pass 2: Sliding-window entropy scan ────────────────────────────────
        entropy_segments = self._sliding_window_entropy(text)
        # Merge: only add entropy findings not already covered by structural match
        for seg in entropy_segments:
            if not self._overlaps_existing(seg, segments):
                segments.append(seg)

        # ── Summary ────────────────────────────────────────────────────────────
        max_entropy = max((s.entropy for s in segments), default=0.0)
        base64_found = any(s.reason == "base64_blob" for s in segments)
        hex_found    = any(s.reason == "hex_dump"    for s in segments)
        suspicious   = len(segments) > 0

        if suspicious:
            logger.info(
                f"[EntropyAnalyzer] Suspicious: {len(segments)} segment(s), "
                f"max_entropy={max_entropy:.2f} | "
                f"base64={base64_found} hex={hex_found}"
            )

        return EntropyResult(
            suspicious=suspicious,
            max_entropy=max_entropy,
            high_entropy_segments=segments,
            base64_like_found=base64_found,
            hex_dump_found=hex_found,
            segment_count=len(segments),
        )

    # ── Shannon Entropy ────────────────────────────────────────────────────────

    def _shannon_entropy(self, text: str) -> float:
        """
        Compute Shannon entropy (bits per character) for a string.

        H = -Σ p(c) × log2(p(c))

        English prose: ~3.5–4.2 bits/char
        Random/encoded: > 4.5 bits/char
        """
        if not text:
            return 0.0
        from collections import Counter
        freq = Counter(text)
        n = len(text)
        return -sum((count / n) * math.log2(count / n) for count in freq.values())

    # ── Sliding-window entropy scan ────────────────────────────────────────────

    def _sliding_window_entropy(self, text: str) -> list[EntropySegment]:
        """
        Slide a window of _WINDOW_SIZE chars across text.
        Flag any window whose entropy exceeds _ENTROPY_THRESHOLD.
        Merge adjacent flagged windows into single segments.
        """
        flagged: list[tuple[int, int, float]] = []  # (start, end, entropy)

        step = max(1, _WINDOW_SIZE // 2)
        for i in range(0, len(text) - _WINDOW_SIZE + 1, step):
            window = text[i:i + _WINDOW_SIZE]
            e = self._shannon_entropy(window)
            if e >= _ENTROPY_THRESHOLD:
                flagged.append((i, i + _WINDOW_SIZE, e))

        if not flagged:
            return []

        # Merge overlapping/adjacent flagged windows
        merged: list[tuple[int, int, float]] = []
        cur_start, cur_end, cur_e = flagged[0]
        for start, end, e in flagged[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_start, cur_end, cur_e))
                cur_start, cur_end, cur_e = start, end, e
        merged.append((cur_start, cur_end, cur_e))

        return [
            EntropySegment(
                reason="high_entropy",
                text=text[s:e],
                entropy=en,
                start=s,
                end=e,
            )
            for s, e, en in merged
        ]

    # ── Structural Heuristics ──────────────────────────────────────────────────

    def _check_base64(self, text: str) -> list[EntropySegment]:
        """Flag long base64-like substrings."""
        segments = []
        for m in _BASE64_PATTERN.finditer(text):
            if len(m.group()) >= _BASE64_MIN_LEN:
                e = self._shannon_entropy(m.group())
                segments.append(EntropySegment(
                    reason="base64_blob",
                    text=m.group()[:80] + ("..." if len(m.group()) > 80 else ""),
                    entropy=e,
                    start=m.start(),
                    end=m.end(),
                ))
        return segments

    def _check_hex_dump(self, text: str) -> list[EntropySegment]:
        """Flag long hex-like substrings."""
        segments = []
        for m in _HEX_BLOB_PATTERN.finditer(text):
            clean = m.group().replace(" ", "").replace("-", "")
            if len(clean) >= _HEX_BLOB_MIN_LEN:
                e = self._shannon_entropy(m.group())
                segments.append(EntropySegment(
                    reason="hex_dump",
                    text=m.group()[:80] + ("..." if len(m.group()) > 80 else ""),
                    entropy=e,
                    start=m.start(),
                    end=m.end(),
                ))
        return segments

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _overlaps_existing(seg: EntropySegment, existing: list[EntropySegment]) -> bool:
        """Check if seg overlaps any segment already in list."""
        for ex in existing:
            if seg.start < ex.end and seg.end > ex.start:
                return True
        return False
