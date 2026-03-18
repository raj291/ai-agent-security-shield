"""
Sensitive Data Scanner — Output Guard Layer 2

Detects and redacts secrets, credentials, and sensitive infrastructure data
from agent responses using regex pattern matching.

Categories:
  - API Keys (OpenAI sk-*, AWS AKIA*, Slack xox*, Anthropic sk-ant-*)
  - Database connection strings (postgresql://, mysql://, mongodb://, redis://)
  - Internal / private network URLs (localhost, 127.x, 192.168.x, 10.x, 172.16-31.x)
  - Password / credential pairs (password=xyz, api_key=xyz)
  - Private key headers (-----BEGIN PRIVATE KEY-----)
  - AWS credentials (aws_secret_access_key=...)
  - JWT tokens (eyJxxx.yyy.zzz)
  - Generic high-entropy hex strings (32+ hex chars — looks like a hash/token)

Obfuscation hardening:
  Reuses TextPreprocessor from input_guard to generate decoded variants
  (hex-decoded, base64-decoded, leet-normalised) before scanning — catches
  secrets that have been trivially encoded before being placed in a response.

Redaction: matched text → [REDACTED:<CATEGORY>]
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Pattern library ───────────────────────────────────────────────────────────

# Each entry: (category_name, compiled_regex)
_PATTERNS: list[tuple[str, re.Pattern]] = [

    # ── API Keys ────────────────────────────────────────────────────────────
    ("OPENAI_API_KEY", re.compile(
        r'\bsk-[a-zA-Z0-9]{32,}\b'
    )),
    ("ANTHROPIC_API_KEY", re.compile(
        r'\bsk-ant-[a-zA-Z0-9\-_]{20,}\b'
    )),
    ("AWS_ACCESS_KEY", re.compile(
        r'\bAKIA[A-Z0-9]{16}\b'
    )),
    ("AWS_SECRET_KEY", re.compile(
        r'(?i)aws[_\-\s]?secret[_\-\s]?access[_\-\s]?key\s*[=:]\s*\S+'
    )),
    ("SLACK_TOKEN", re.compile(
        r'\bxox[bpao]-[0-9a-zA-Z\-]+'
    )),
    ("GITHUB_TOKEN", re.compile(
        r'\bghp_[a-zA-Z0-9]{36}\b'
        r'|\bgh[osr]_[a-zA-Z0-9]{36}\b'
    )),
    ("GOOGLE_API_KEY", re.compile(
        r'\bAIza[a-zA-Z0-9\-_]{35}\b'
    )),

    # ── Database Connection Strings ─────────────────────────────────────────
    ("DB_CONNECTION_STRING", re.compile(
        r'\b(?:postgresql|postgres|mysql|mongodb|redis|sqlite|mssql|oracle)'
        r'(?:\+\w+)?://[^\s\'"<>]+'
    )),

    # ── Private / Internal Network URLs ────────────────────────────────────
    ("INTERNAL_URL", re.compile(
        r'https?://(?:'
        r'localhost'
        r'|127(?:\.\d{1,3}){3}'
        r'|10(?:\.\d{1,3}){3}'
        r'|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}'
        r'|192\.168(?:\.\d{1,3}){2}'
        r'|::1'
        r')[^\s\'"<>]*'
    )),

    # ── Private Keys ────────────────────────────────────────────────────────
    ("PRIVATE_KEY", re.compile(
        r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----'
    )),
    ("CERTIFICATE", re.compile(
        r'-----BEGIN CERTIFICATE-----'
    )),

    # ── Password / Credential Pairs ─────────────────────────────────────────
    ("PASSWORD_PAIR", re.compile(
        r'(?i)(?:password|passwd|pass|pwd|secret|api[_\-]?key|auth[_\-]?token|access[_\-]?token)'
        r'\s*[=:]\s*[^\s\'"<>,;]{4,}'
    )),

    # ── JWT Tokens ──────────────────────────────────────────────────────────
    ("JWT_TOKEN", re.compile(
        r'\beyJ[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+\b'
    )),

    # ── Generic hex/base64 secrets (32+ char tokens) ───────────────────────
    ("HEX_SECRET", re.compile(
        r'\b[0-9a-f]{32,64}\b'
    )),
]


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class SecretFinding:
    """A single secret/sensitive-data finding in a response."""
    category: str
    matched_text: str
    redacted_text: str      # what replaces it in the output
    start: int
    end: int


@dataclass
class SensitiveDataResult:
    """Result from SensitiveDataScanner.scan()"""
    clean: bool
    findings: list = field(default_factory=list)    # list[SecretFinding]
    redacted_response: str = ""
    categories_found: list = field(default_factory=list)


# ─────────────────────────────────────────────
# SensitiveDataScanner
# ─────────────────────────────────────────────

class SensitiveDataScanner:
    """
    Scans agent responses for secrets, credentials, and sensitive
    infrastructure data. Redacts findings with [REDACTED:<CATEGORY>].

    Also checks decoded variants of the response (hex, base64, leet)
    to catch trivially obfuscated secrets.

    Usage:
        scanner = SensitiveDataScanner()
        result = scanner.scan("Here is your key: sk-abc123xyz789abc123xyz789abc123xy")
        print(result.clean)           # False
        print(result.redacted_response)  # "Here is your key: [REDACTED:OPENAI_API_KEY]"
    """

    def scan(self, text: str) -> SensitiveDataResult:
        """
        Scan for sensitive data in text AND its decoded variants.

        Returns SensitiveDataResult with all findings and a redacted
        version of the original text.
        """
        if not text or not text.strip():
            return SensitiveDataResult(clean=True, redacted_response=text or "")

        # ── Step 1: Check decoded variants (obfuscation awareness) ────────────
        obfuscated_categories = self._check_decoded_variants(text)

        # ── Step 2: Scan original text ─────────────────────────────────────────
        findings = self._scan_text(text)

        # Add obfuscation-only findings (no direct position to redact, but worth flagging)
        for cat in obfuscated_categories:
            if not any(f.category == cat for f in findings):
                findings.append(SecretFinding(
                    category=f"OBFUSCATED_{cat}",
                    matched_text="[encoded]",
                    redacted_text="[REDACTED:ENCODED_SECRET]",
                    start=-1,
                    end=-1,
                ))

        # ── Step 3: Build redacted text ────────────────────────────────────────
        redacted = self._redact(text, [f for f in findings if f.start >= 0])

        categories = list({f.category for f in findings})

        if findings:
            logger.info(
                f"[SensitiveDataScanner] Found {len(findings)} secret(s): {categories}"
            )

        return SensitiveDataResult(
            clean=len(findings) == 0,
            findings=findings,
            redacted_response=redacted,
            categories_found=categories,
        )

    # ── Internal ───────────────────────────────────────────────────────────────

    def _scan_text(self, text: str) -> list[SecretFinding]:
        """Run all regex patterns against text. Returns list of findings."""
        findings: list[SecretFinding] = []
        seen_spans: list[tuple[int, int]] = []

        for category, pattern in _PATTERNS:
            for m in pattern.finditer(text):
                # Skip if this span already covered by a previous match
                if any(s <= m.start() < e or s < m.end() <= e for s, e in seen_spans):
                    continue
                seen_spans.append((m.start(), m.end()))
                findings.append(SecretFinding(
                    category=category,
                    matched_text=m.group(),
                    redacted_text=f"[REDACTED:{category}]",
                    start=m.start(),
                    end=m.end(),
                ))

        return findings

    def _redact(self, text: str, findings: list[SecretFinding]) -> str:
        """Replace all finding spans with their redacted_text, working end→start."""
        if not findings:
            return text

        # Sort descending by start so replacement doesn't shift offsets
        sorted_findings = sorted(findings, key=lambda f: f.start, reverse=True)
        result = text
        for f in sorted_findings:
            result = result[:f.start] + f.redacted_text + result[f.end:]
        return result

    def _check_decoded_variants(self, text: str) -> list[str]:
        """
        Generate decoded variants (hex, base64, leet) and scan them.
        Returns list of categories found only in encoded form (not raw text).
        """
        try:
            import sys
            import os
            # Add project root to path for preprocessor import
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from guards.input_guard.preprocessor import TextPreprocessor
        except ImportError:
            return []

        preprocessor = TextPreprocessor()
        # get_all_variants returns {variant_name: (text, was_modified)}
        raw_variants = preprocessor.get_all_variants(text)

        obfuscated_categories: list[str] = []
        raw_categories = {f.category for f in self._scan_text(text)}

        for variant_name, (variant_text, was_modified) in raw_variants.items():
            if variant_name == "original":
                continue
            if not was_modified or variant_text == text:
                continue

            variant_findings = self._scan_text(variant_text)
            for f in variant_findings:
                if f.category not in raw_categories:
                    logger.warning(
                        f"[SensitiveDataScanner] Secret found only in {variant_name} variant: "
                        f"category={f.category}"
                    )
                    obfuscated_categories.append(f.category)

        return obfuscated_categories
