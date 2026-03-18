"""
Tool Guard — YAML-Driven Permission Matrix

Enforces least-privilege: every tool call must be explicitly permitted for
the calling agent's role. Unknown tools and unknown roles are denied.

Design
------
- Fail-closed: if tool_permissions.yaml is missing, ALL tools are denied.
  (Same principle as Layer 2 LLM classifier — unknown = dangerous.)
- YAML-driven: security analysts can add/remove roles or tools without
  touching Python code. Same principle as attack_patterns.yaml.
- Layered checks per tool:
    1. Is this tool known?
    2. Is this role allowed?
    3. Are any forbidden keywords in the parameters? (SQL keywords)
    4. Are any forbidden tables referenced? (sensitive DB tables)
    5. Is the file path in the approved list? (file_read)
    6. Is the URL domain in the approved list? (http_request)

Usage
-----
    matrix = PermissionMatrix()
    result = matrix.check("database_query", "analyst", {"query": "SELECT * FROM orders"})
    # result.allowed == True

    result = matrix.check("send_email", "analyst", {})
    # result.allowed == False
    # result.denial_reason == "Role 'analyst' is not authorized to call 'send_email'. ..."
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_POLICY_FILE = Path(__file__).parent / "policies" / "tool_permissions.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PermissionResult:
    """
    Result of a permission check for a tool call.

    Fields
    ------
    allowed       : True if the call is permitted
    denial_reason : Human-readable reason for denial (empty string if allowed)
    tool_policy   : The YAML config dict for this tool (empty dict if tool unknown)
    rate_limit    : Calls-per-60s limit for this tool (0 if denied early)
    """
    allowed:       bool
    denial_reason: str
    tool_policy:   dict
    rate_limit:    int


# ─────────────────────────────────────────────────────────────────────────────
# PERMISSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

class PermissionMatrix:
    """
    YAML-driven permission matrix.

    Loads tool_permissions.yaml once at init. All check() calls are in-memory.

    FAIL-CLOSED: if the YAML file is missing or corrupt, every check()
    returns denied. A missing policy file is a configuration problem that
    must be fixed before tools are accessible.
    """

    def __init__(self):
        self._policy = self._load_policy()
        tool_count = len(self._policy.get("tools", {}))
        logger.info(f"[PermissionMatrix] Loaded {tool_count} tools from policy")

    def check(
        self,
        tool_name: str,
        agent_role: str,
        parameters: dict,
    ) -> PermissionResult:
        """
        Check whether agent_role is allowed to call tool_name with parameters.

        Parameters
        ----------
        tool_name  : str  — Tool being called (must match key in policy YAML)
        agent_role : str  — Calling agent's role (e.g. "customer_support")
        parameters : dict — Tool parameters (checked for forbidden tables, paths, domains)

        Returns
        -------
        PermissionResult with allowed=True/False
        """
        tools       = self._policy.get("tools", {})
        rate_limits = self._policy.get("rate_limits", {})

        # ── 1. Unknown tool → DENIED (fail-closed) ────────────────────────────
        if tool_name not in tools:
            return PermissionResult(
                allowed=False,
                denial_reason=f"Tool '{tool_name}' is not registered in the permission matrix",
                tool_policy={},
                rate_limit=0,
            )

        tool_policy   = tools[tool_name]
        allowed_roles = tool_policy.get("allowed_roles", [])

        # ── 2. Role check ─────────────────────────────────────────────────────
        # "any" in allowed_roles grants access to all roles
        if "any" not in allowed_roles and agent_role not in allowed_roles:
            return PermissionResult(
                allowed=False,
                denial_reason=(
                    f"Role '{agent_role}' is not authorized to call '{tool_name}'. "
                    f"Allowed roles: {allowed_roles}"
                ),
                tool_policy=tool_policy,
                rate_limit=0,
            )

        # ── 3. Forbidden keywords (e.g. destructive SQL verbs) ────────────────
        forbidden_kw = [kw.upper() for kw in tool_policy.get("forbidden_keywords", [])]
        if forbidden_kw:
            # Concatenate all string parameter values for a single keyword scan
            param_text = " ".join(
                str(v) for v in parameters.values() if isinstance(v, str)
            ).upper()
            for kw in forbidden_kw:
                if kw in param_text:
                    return PermissionResult(
                        allowed=False,
                        denial_reason=(
                            f"Forbidden keyword '{kw}' detected in parameters "
                            f"for tool '{tool_name}'"
                        ),
                        tool_policy=tool_policy,
                        rate_limit=0,
                    )

        # ── 4. Forbidden tables (sensitive DB tables) ─────────────────────────
        forbidden_tables = [t.lower() for t in tool_policy.get("forbidden_tables", [])]
        if forbidden_tables:
            param_lower = " ".join(
                str(v) for v in parameters.values() if isinstance(v, str)
            ).lower()
            for tbl in forbidden_tables:
                if tbl in param_lower:
                    return PermissionResult(
                        allowed=False,
                        denial_reason=(
                            f"Access to sensitive table '{tbl}' is forbidden "
                            f"for tool '{tool_name}'"
                        ),
                        tool_policy=tool_policy,
                        rate_limit=0,
                    )

        # ── 5. Allowed paths (file_read) ──────────────────────────────────────
        allowed_paths = tool_policy.get("allowed_paths", [])
        if allowed_paths:
            path_val = str(parameters.get("path", "")).strip()
            if path_val and not any(path_val.startswith(p) for p in allowed_paths):
                return PermissionResult(
                    allowed=False,
                    denial_reason=(
                        f"Path '{path_val}' is not in the approved directories "
                        f"for '{tool_name}'. Allowed: {allowed_paths}"
                    ),
                    tool_policy=tool_policy,
                    rate_limit=0,
                )

        # ── 6. Allowed domains (http_request) ─────────────────────────────────
        allowed_domains = tool_policy.get("allowed_domains", [])
        if allowed_domains:
            url_val = str(parameters.get("url", "")).strip()
            if url_val and not any(domain in url_val for domain in allowed_domains):
                return PermissionResult(
                    allowed=False,
                    denial_reason=(
                        f"URL '{url_val[:80]}' is not in the approved domains "
                        f"for '{tool_name}'. Allowed: {allowed_domains}"
                    ),
                    tool_policy=tool_policy,
                    rate_limit=0,
                )

        # ── All checks passed ─────────────────────────────────────────────────
        limit = rate_limits.get(tool_name, rate_limits.get("default", 50))
        return PermissionResult(
            allowed=True,
            denial_reason="",
            tool_policy=tool_policy,
            rate_limit=int(limit),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_policy(self) -> dict:
        """
        Load and parse tool_permissions.yaml.
        Returns an empty dict on error — all subsequent check() calls will deny.
        """
        try:
            with open(_POLICY_FILE) as f:
                data = yaml.safe_load(f) or {}
            logger.debug(f"[PermissionMatrix] Policy loaded from {_POLICY_FILE}")
            return data
        except FileNotFoundError:
            logger.error(
                f"[PermissionMatrix] Policy file not found: {_POLICY_FILE}. "
                f"All tool calls will be DENIED until the file is restored."
            )
            return {}
        except Exception as exc:
            logger.error(f"[PermissionMatrix] Failed to load policy: {exc}")
            return {}
