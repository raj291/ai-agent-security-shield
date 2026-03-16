"""
Input Guard — Layer 3: Scope Validator

The last checkpoint. Only runs when Layers 1 and 2 both returned SAFE,
meaning the input is not a security threat — but it might still be
outside what this particular agent is supposed to handle.

Example:
  Input: "DROP TABLE users"
  - Layer 1: No injection patterns → SAFE
  - Layer 2: Not a jailbreak → SAFE
  - Layer 3: Matches sql_destructive rule in scope_policy.yaml → SUSPICIOUS

Scope is defined in patterns/scope_policy.yaml — editable without code
changes. Uses the same single-quoted YAML scalar convention as
attack_patterns.yaml (backslashes are literal in single-quoted YAML).

Design choice — Python policy engine, not OPA:
  OPA requires a binary dependency and Rego policy language.
  A YAML + Python evaluator gives identical expressiveness for Day 2
  and is fully testable offline. OPA migration can happen in the
  hardening phase (Day 10-14) if policy complexity warrants it.
"""
import logging
import re
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_POLICY_PATH = Path(__file__).parent / "patterns" / "scope_policy.yaml"


class ScopeValidator:
    """
    Layer 3: Rule-based scope policy enforcer.

    Loads rules from scope_policy.yaml. Checks every SAFE input against:
      1. max_input_length — prevents context-stuffing
      2. forbidden_patterns — regex rules for out-of-scope requests

    Returns a standardized result dict matching the expected
    pipeline interface: {"out_of_scope": bool, "reason": str, ...}
    """

    def __init__(self, policy_path: Optional[Path] = None):
        self._policy_path = policy_path or _DEFAULT_POLICY_PATH
        self._policy      = self._load_policy()
        self._compiled    = self._compile_patterns()

        scope = self._policy.get("agent_scope", {})
        logger.info(
            f"[ScopeValidator] Ready | "
            f"agent='{scope.get('name', 'unknown')}' | "
            f"max_length={scope.get('max_input_length', 'unlimited')} | "
            f"rules={len(self._compiled)}"
        )

    # ── Init helpers ──────────────────────────────────────────────────────────

    def _load_policy(self) -> dict:
        """Load scope_policy.yaml. Returns empty dict on failure (fail-open)."""
        try:
            with open(self._policy_path) as f:
                data = yaml.safe_load(f)
            logger.debug(f"[ScopeValidator] Loaded policy from {self._policy_path.name}")
            return data or {}
        except FileNotFoundError:
            logger.warning(
                f"[ScopeValidator] Policy file not found: {self._policy_path}. "
                "All inputs will pass scope validation."
            )
            return {}
        except Exception as exc:
            logger.error(f"[ScopeValidator] Failed to load policy: {exc}")
            return {}

    def _compile_patterns(self) -> list[tuple[str, re.Pattern, str]]:
        """
        Pre-compile all forbidden_patterns from policy.

        Returns list of (name, compiled_regex, reason) tuples.
        Logs a warning and skips any pattern with invalid regex.
        """
        scope = self._policy.get("agent_scope", {})
        patterns = scope.get("forbidden_patterns", [])
        compiled = []

        for p in patterns:
            name   = p.get("name", "unnamed")
            regex  = p.get("regex", "")
            reason = p.get("reason", f"Rule '{name}' triggered")
            try:
                compiled.append(
                    (name, re.compile(regex, re.IGNORECASE | re.DOTALL), reason)
                )
            except re.error as exc:
                logger.warning(
                    f"[ScopeValidator] Bad regex in rule '{name}': {exc}. Skipping."
                )

        return compiled

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(self, input_text: str, context: Optional[dict] = None) -> dict:
        """
        Check whether input is within the agent's allowed scope.

        Parameters
        ----------
        input_text : str
            The user's (possibly sanitized) input text.
        context : dict, optional
            Additional request context (session_id, user_role, etc.)
            Reserved for future policy enrichment.

        Returns
        -------
        {
          "out_of_scope":   bool,
          "reason":         str,
          "rule_triggered": str | None,   # name of the rule that fired
          "layer":          "layer3_scope_validator",
        }
        """
        scope = self._policy.get("agent_scope", {})

        # ── Check 1: Max input length ─────────────────────────────────────
        max_len = scope.get("max_input_length")
        if max_len and len(input_text) > max_len:
            reason = (
                f"Input length {len(input_text)} exceeds policy "
                f"maximum of {max_len} characters"
            )
            logger.warning(
                f"[ScopeValidator] OUT_OF_SCOPE | rule=max_input_length | "
                f"length={len(input_text)} | limit={max_len}"
            )
            return self._result(out_of_scope=True, reason=reason, rule="max_input_length")

        # ── Check 2: Forbidden patterns ───────────────────────────────────
        for name, compiled_re, reason in self._compiled:
            match = compiled_re.search(input_text)
            if match:
                logger.warning(
                    f"[ScopeValidator] OUT_OF_SCOPE | rule={name} | "
                    f"matched='{match.group()[:60]}' | input={input_text[:60]}..."
                )
                return self._result(out_of_scope=True, reason=reason, rule=name)

        # ── All checks passed ─────────────────────────────────────────────
        return self._result(out_of_scope=False, reason="Input is within agent scope")

    def policy_name(self) -> str:
        """Return the configured agent name from policy."""
        return self._policy.get("agent_scope", {}).get("name", "unknown")

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _result(out_of_scope: bool, reason: str, rule: Optional[str] = None) -> dict:
        return {
            "out_of_scope":   out_of_scope,
            "reason":         reason,
            "rule_triggered": rule,
            "layer":          "layer3_scope_validator",
        }
