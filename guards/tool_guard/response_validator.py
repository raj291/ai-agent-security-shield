"""
Tool Guard — Response Validator

Validates tool responses BEFORE they are injected into the LLM context.

Two checks
----------
1. Size check — reject responses larger than 1MB (default).
   Oversized responses are a vector for context stuffing: a tool that returns
   a 2MB database dump could displace the system prompt from the LLM's active
   generation window.

2. Credential leakage scan — reject responses containing API keys, passwords,
   private keys, or recognisable token formats (OpenAI sk- tokens, Slack xox).
   A compromised or misconfigured tool might accidentally include secrets in
   its response. Blocking these before they enter the prompt prevents the LLM
   from echoing them back to the user.

Usage
-----
    validator = ResponseValidator()

    result = validator.validate({"rows": [{"id": 1, "name": "Alice"}]})
    # result.valid == True

    result = validator.validate({"data": "x" * 2_000_000})
    # result.valid == False
    # result.reason == "Response too large: 2,000,000 bytes (limit 1,000,000 bytes)"

    result = validator.validate({"config": "api_key=sk-abc123abc123abc123abc123abc123ab"})
    # result.valid == False
    # result.reason == "Response contains potential credential data"
"""
import json
import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_MAX_BYTES: int = 1_000_000  # 1MB

# Patterns that indicate credential data in serialized response JSON
_CREDENTIAL_PATTERNS: list[re.Pattern] = [
    # Key=value credential pairs
    re.compile(
        r'(?:api[_\-]?key|secret[_\-]?key|access[_\-]?key|password|passwd|passphrase|private[_\-]?key|token|auth[_\-]?token)\s*[:=]\s*\S{8,}',
        re.IGNORECASE,
    ),
    # PEM private key headers
    re.compile(r'-----BEGIN\s+(?:RSA\s+|EC\s+|DSA\s+|OPENSSH\s+)?PRIVATE\s+KEY-----', re.IGNORECASE),
    # OpenAI-style secret keys (sk-xxx)
    re.compile(r'\bsk-[a-zA-Z0-9]{32,}\b'),
    # Slack tokens (xoxb, xoxp, xoxa, xoxo)
    re.compile(r'\bxox[bpoa]-[0-9A-Za-z\-]{10,}\b'),
    # AWS access key format
    re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
    # Generic high-entropy quoted string that looks like a token (32+ hex chars)
    re.compile(r'["\']([0-9a-f]{32,})["\']', re.IGNORECASE),
]


@dataclass
class ValidationResult:
    """
    Result of validating a tool response.

    Fields
    ------
    valid      : True if the response passes all checks
    reason     : Human-readable reason for rejection (empty string if valid)
    size_bytes : Serialized size of the response in bytes
    """
    valid:      bool
    reason:     str
    size_bytes: int


class ResponseValidator:
    """
    Validates tool responses before they enter the LLM context.

    Usage
    -----
        validator = ResponseValidator()
        result = validator.validate(response_dict, tool_name="database_query")
        if not result.valid:
            # Block response from reaching LLM
    """

    def validate(
        self,
        response: dict,
        tool_name: str = "",
        max_size: int = DEFAULT_MAX_BYTES,
    ) -> ValidationResult:
        """
        Validate a tool response dict.

        Parameters
        ----------
        response  : dict — The tool's return value
        tool_name : str  — For logging only
        max_size  : int  — Maximum allowed byte size (default 1MB)

        Returns
        -------
        ValidationResult with valid=True/False
        """
        # Serialize to JSON for size and content checks
        try:
            serialized = json.dumps(response, default=str)
        except Exception as exc:
            logger.warning(f"[ResponseValidator] Serialization failed for tool={tool_name}: {exc}")
            return ValidationResult(
                valid=False,
                reason=f"Response not JSON-serializable: {exc}",
                size_bytes=0,
            )

        size_bytes = len(serialized.encode("utf-8"))

        # ── 1. Size check ─────────────────────────────────────────────────────
        if size_bytes > max_size:
            logger.warning(
                f"[ResponseValidator] OVERSIZED | tool={tool_name} | "
                f"size={size_bytes:,} | limit={max_size:,}"
            )
            return ValidationResult(
                valid=False,
                reason=(
                    f"Response too large: {size_bytes:,} bytes "
                    f"(limit {max_size:,} bytes)"
                ),
                size_bytes=size_bytes,
            )

        # ── 2. Credential leakage check ────────────────────────────────────────
        for pat in _CREDENTIAL_PATTERNS:
            if pat.search(serialized):
                logger.warning(
                    f"[ResponseValidator] CREDENTIAL_LEAK | tool={tool_name} | "
                    f"pattern={pat.pattern[:60]}"
                )
                return ValidationResult(
                    valid=False,
                    reason="Response contains potential credential data",
                    size_bytes=size_bytes,
                )

        logger.debug(
            f"[ResponseValidator] VALID | tool={tool_name} | size={size_bytes:,}"
        )
        return ValidationResult(valid=True, reason="", size_bytes=size_bytes)
