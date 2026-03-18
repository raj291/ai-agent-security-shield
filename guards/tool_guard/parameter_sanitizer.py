"""
Tool Guard — Parameter Sanitizer

Scans every string value in a tool call's parameter dict for injection attacks.

Attack types detected
---------------------
1. SQL Injection
   UNION SELECT, comment sequences (-- / # / /*), tautologies (1=1),
   stacked statements ('; DROP), blind injection (SLEEP / BENCHMARK),
   DROP TABLE

2. Path Traversal
   ../../etc/passwd style traversal, Windows variant (..\system32),
   URL-encoded sequences (%2e%2e/), null-byte injection, direct /etc/ access

3. Command Injection
   Semicolon chaining (; rm), pipe to shell (| bash), backtick execution,
   $() subshell substitution, && / || chaining to known system binaries

4. SSRF (Server-Side Request Forgery)
   AWS EC2 metadata IP (169.254.169.254), localhost / 127.0.0.1 / ::1,
   RFC-1918 internal ranges (10.x, 172.16-31.x, 192.168.x)

Why not just regex on user input?
----------------------------------
Parameters are STRUCTURED data — the attacker controls specific values like
a SQL query string, a file path, or a URL. These need purpose-built patterns
that match the specific attack surfaces of each parameter type, not the broad
"is this a prompt injection?" question that the Input Guard handles.

Obfuscation awareness
---------------------
We reuse TextPreprocessor from the Input Guard to generate decoded variants
(hex-decoded, base64-decoded, leet-normalized) before matching. This catches
encoded traversal sequences like %2e%2e%2f → ../../ or hex-encoded SQL.

Usage
-----
    sanitizer = ParameterSanitizer()
    result = sanitizer.scan({"query": "SELECT * FROM users WHERE id = 1"})
    # result.clean == True

    result = sanitizer.scan({"query": "1 UNION SELECT * FROM api_keys--"})
    # result.clean == False
    # result.threat_type == "sql_injection"
    # result.matched_pattern == "sql_union_select"
    # result.affected_param == "query"
"""
import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SanitizeResult:
    """
    Result of scanning tool call parameters.

    Fields
    ------
    clean           : True if no threats found
    threat_type     : "sql_injection" | "path_traversal" | "command_injection" | "ssrf" | ""
    matched_pattern : Which pattern name fired (e.g. "sql_union_select")
    affected_param  : Which parameter key contained the threat (e.g. "query")
    matched_text    : The matched substring, truncated to 80 chars
    """
    clean:           bool
    threat_type:     str
    matched_pattern: str
    affected_param:  str
    matched_text:    str


# Singleton clean result — returned when no threats found
_CLEAN = SanitizeResult(
    clean=True,
    threat_type="",
    matched_pattern="",
    affected_param="",
    matched_text="",
)


# ─────────────────────────────────────────────────────────────────────────────
# PATTERNS
# Each entry: (pattern_name, threat_type, compiled_regex)
# Ordered: most specific / highest severity first within each category
# ─────────────────────────────────────────────────────────────────────────────

_PATTERNS: list[tuple[str, str, re.Pattern]] = [

    # ── SQL Injection ─────────────────────────────────────────────────────────
    (
        "sql_union_select", "sql_injection",
        re.compile(r'\bUNION\s+(?:ALL\s+)?SELECT\b', re.IGNORECASE),
    ),
    (
        "sql_stacked_statement", "sql_injection",
        re.compile(r"';\s*(?:DROP|DELETE|INSERT|UPDATE|CREATE|ALTER|EXEC)\b", re.IGNORECASE),
    ),
    (
        "sql_drop_table", "sql_injection",
        re.compile(r'\bDROP\s+TABLE\b', re.IGNORECASE),
    ),
    (
        "sql_blind_sleep", "sql_injection",
        re.compile(r'\b(?:SLEEP|BENCHMARK|WAITFOR\s+DELAY|PG_SLEEP)\s*\(', re.IGNORECASE),
    ),
    (
        "sql_tautology", "sql_injection",
        re.compile(r"'\s*(?:OR|AND)\s+['\d].*?[=<>]", re.IGNORECASE | re.DOTALL),
    ),
    (
        "sql_comment", "sql_injection",
        re.compile(r'(?:--\s|#\s|/\*)', re.IGNORECASE),
    ),

    # ── Path Traversal ────────────────────────────────────────────────────────
    (
        "path_etc_passwd", "path_traversal",
        re.compile(r'/etc/(?:passwd|shadow|hosts|sudoers|group)\b', re.IGNORECASE),
    ),
    (
        "path_windows_system", "path_traversal",
        re.compile(r'(?:\.\.[/\\])+(?:windows|system32|winnt)', re.IGNORECASE),
    ),
    (
        "path_dotdot_unix", "path_traversal",
        re.compile(r'(?:\.\.[\\/]){2,}'),
    ),
    (
        "path_url_encoded", "path_traversal",
        re.compile(r'%2e%2e[%2f5c]', re.IGNORECASE),
    ),
    (
        "path_null_byte", "path_traversal",
        re.compile(r'\.\.[\\/].*\x00'),
    ),

    # ── Command Injection ─────────────────────────────────────────────────────
    (
        "cmd_subshell", "command_injection",
        re.compile(r'\$\([^)]+\)'),
    ),
    (
        "cmd_backtick", "command_injection",
        re.compile(r'`[^`]+`'),
    ),
    (
        "cmd_semicolon_exec", "command_injection",
        re.compile(
            r';\s*(?:rm|ls|cat|wget|curl|bash|sh|python|perl|ruby|nc|ncat|nmap)\b',
            re.IGNORECASE,
        ),
    ),
    (
        "cmd_pipe_to_shell", "command_injection",
        re.compile(
            r'\|\s*(?:bash|sh|cmd\.exe|powershell|zsh|ksh|csh)\b',
            re.IGNORECASE,
        ),
    ),
    (
        "cmd_chain", "command_injection",
        re.compile(
            r'(?:&&|\|\|)\s*(?:rm|ls|cat|wget|curl|bash|sh|python|nc)\b',
            re.IGNORECASE,
        ),
    ),

    # ── SSRF ──────────────────────────────────────────────────────────────────
    (
        "ssrf_aws_metadata", "ssrf",
        re.compile(r'169\.254\.169\.254'),
    ),
    (
        "ssrf_localhost", "ssrf",
        re.compile(r'(?:localhost|127\.0\.0\.1|::1)\b', re.IGNORECASE),
    ),
    (
        "ssrf_internal_10", "ssrf",
        re.compile(r'(?<!\d)10\.\d{1,3}\.\d{1,3}\.\d{1,3}'),
    ),
    (
        "ssrf_internal_172", "ssrf",
        re.compile(r'172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}'),
    ),
    (
        "ssrf_internal_192", "ssrf",
        re.compile(r'192\.168\.\d{1,3}\.\d{1,3}'),
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER SANITIZER
# ─────────────────────────────────────────────────────────────────────────────

class ParameterSanitizer:
    """
    Scans all string values in a tool call's parameter dict.

    Reuses TextPreprocessor from the Input Guard for obfuscation-aware
    scanning (hex decode, base64 decode, leet normalization, unicode clean).
    Falls back to plain regex if the preprocessor is unavailable.
    """

    def __init__(self):
        try:
            from guards.input_guard.preprocessor import TextPreprocessor
            self._preprocessor = TextPreprocessor()
        except ImportError:
            self._preprocessor = None
            logger.debug("[ParameterSanitizer] TextPreprocessor unavailable — using plain matching")

    def scan(self, parameters: dict, tool_name: str = "") -> SanitizeResult:
        """
        Scan all string values in a parameters dict.

        Returns the first threat found (stops on first match), or _CLEAN.

        Parameters
        ----------
        parameters : dict — Tool call parameters (e.g. {"query": "...", "limit": 10})
        tool_name  : str  — For logging only

        Returns
        -------
        SanitizeResult
        """
        for key, value in parameters.items():
            if not isinstance(value, str) or not value.strip():
                continue

            result = self._check_value(key, value)
            if not result.clean:
                logger.warning(
                    f"[ParameterSanitizer] THREAT | tool={tool_name} | param={key} | "
                    f"type={result.threat_type} | pattern={result.matched_pattern} | "
                    f"text='{result.matched_text}'"
                )
                return result

        return _CLEAN

    def _check_value(self, key: str, value: str) -> SanitizeResult:
        """
        Check a single parameter value against all patterns.
        Generates obfuscation-decoded variants if TextPreprocessor is available.
        """
        # Build text variants (original + obfuscation-decoded)
        variants: dict[str, str] = {"original": value}
        if self._preprocessor:
            try:
                all_variants = self._preprocessor.get_all_variants(value)
                variants = {name: text for name, (text, _) in all_variants.items()}
            except Exception:
                pass  # Fall back to original-only on any error

        # Check all variants against all patterns
        for variant_name, text in variants.items():
            for pattern_name, threat_type, compiled_re in _PATTERNS:
                match = compiled_re.search(text)
                if match:
                    return SanitizeResult(
                        clean=False,
                        threat_type=threat_type,
                        matched_pattern=pattern_name,
                        affected_param=key,
                        matched_text=match.group()[:80],
                    )

        return _CLEAN
