"""
Tool Guard — Facade Class

Orchestrates all four tool call validation components:

  1. PermissionMatrix  → Is this agent role allowed to use this tool?
  2. RateLimiter       → Has this session exceeded the per-tool rate limit?
  3. ParameterSanitizer → Do any parameters contain SQL injection / path traversal / SSRF?
  4. ResponseValidator → Is the tool's response oversized or does it contain credentials?

Every tool call MUST pass all three pre-execution checks (1–3) before executing.
Every response MUST pass the post-execution check (4) before entering the LLM context.

Verdict values
--------------
  "ALLOWED"      — All checks passed; tool may execute
  "DENIED"       — Permission or parameter check failed; tool must not execute
  "RATE_LIMITED" — Session has exceeded the call rate; retry after reset_in_seconds

LangGraph integration
---------------------
    guard = ToolGuard()
    result = guard.check_for_graph(state["pending_tool_call"])
    # Returns plain dict compatible with GuardianState

Usage
-----
    guard = ToolGuard()

    # Pre-execution
    result = guard.check_tool_call({
        "tool_name": "database_query",
        "agent_role": "customer_support",
        "session_id": "sess_abc123",
        "parameters": {"query": "SELECT * FROM orders WHERE id = 42"},
    })
    if result.verdict == "ALLOWED":
        response = execute_tool(result.sanitized_params)
        # Post-execution
        val = guard.validate_response("database_query", response)
        if val.valid:
            inject_into_llm_context(response)
"""
import logging
from dataclasses import dataclass

from .permission_matrix import PermissionMatrix, PermissionResult
from .rate_limiter import RateLimiter, RateLimitResult
from .parameter_sanitizer import ParameterSanitizer, SanitizeResult
from .response_validator import ResponseValidator, ValidationResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolCallResult:
    """
    Full result from checking a tool call.

    Fields
    ------
    tool_name       : Tool being called
    agent_role      : Role of the calling agent
    session_id      : Session identifier (used for rate limiting)
    verdict         : "ALLOWED" | "DENIED" | "RATE_LIMITED"
    denial_reason   : Human-readable reason (empty string if ALLOWED)
    sanitized_params: Original parameters (unchanged — sanitizer detects, not modifies)
    permission      : Result from PermissionMatrix
    rate_limit      : Result from RateLimiter
    param_scan      : Result from ParameterSanitizer
    confidence      : 0.0 = ALLOWED; 0.9 = RATE_LIMITED; 0.95 = injection; 1.0 = permission denied
    """
    tool_name:        str
    agent_role:       str
    session_id:       str
    verdict:          str
    denial_reason:    str
    sanitized_params: dict
    permission:       PermissionResult
    rate_limit:       RateLimitResult
    param_scan:       SanitizeResult
    confidence:       float


# ─────────────────────────────────────────────────────────────────────────────
# SHARED RATE LIMITER
# Module-level — persists its _store across ToolGuard() instantiations within
# a single process. Prevents bypassing limits by creating new ToolGuard() per request.
# ─────────────────────────────────────────────────────────────────────────────

_rate_limiter = RateLimiter()

# Dummy rate limit result used when check is skipped (permission denied early)
_NO_RL = RateLimitResult(allowed=False, calls_in_window=0, limit=0, reset_in_seconds=0.0)

# Dummy sanitize result used when check is skipped (permission/rate denied early)
_NO_SCAN = SanitizeResult(
    clean=True, threat_type="", matched_pattern="", affected_param="", matched_text=""
)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL GUARD
# ─────────────────────────────────────────────────────────────────────────────

class ToolGuard:
    """
    Secure tool call proxy.

    Instantiate once per request (lightweight — all heavy state is in the
    module-level _rate_limiter and the YAML policy loaded by PermissionMatrix).
    """

    def __init__(self):
        self._permissions = PermissionMatrix()
        self._sanitizer   = ParameterSanitizer()
        self._validator   = ResponseValidator()
        logger.info("[ToolGuard] Ready")

    # ── Pre-execution check ───────────────────────────────────────────────────

    def check_tool_call(self, tool_call: dict) -> ToolCallResult:
        """
        Validate a tool call dict before the tool executes.

        Expected keys in tool_call
        --------------------------
        tool_name  : str  — Which tool is being called
        agent_role : str  — The calling agent's role
        session_id : str  — For rate limiting (use GuardianState.session_id)
        parameters : dict — Tool parameters to sanitize

        Returns
        -------
        ToolCallResult with verdict ALLOWED | DENIED | RATE_LIMITED
        """
        tool_name  = tool_call.get("tool_name", "")
        agent_role = tool_call.get("agent_role", "unknown")
        session_id = tool_call.get("session_id", "default")
        parameters = tool_call.get("parameters", {})

        # ── Step 1: Permission check ─────────────────────────────────────────
        perm = self._permissions.check(tool_name, agent_role, parameters)
        if not perm.allowed:
            logger.warning(
                f"[ToolGuard] DENIED | tool={tool_name} | role={agent_role} | "
                f"reason={perm.denial_reason}"
            )
            return ToolCallResult(
                tool_name=tool_name,
                agent_role=agent_role,
                session_id=session_id,
                verdict="DENIED",
                denial_reason=perm.denial_reason,
                sanitized_params=parameters,
                permission=perm,
                rate_limit=_NO_RL,
                param_scan=_NO_SCAN,
                confidence=1.0,
            )

        # ── Step 2: Rate limit check ─────────────────────────────────────────
        rl = _rate_limiter.check(session_id, tool_name, perm.rate_limit)
        if not rl.allowed:
            logger.warning(
                f"[ToolGuard] RATE_LIMITED | tool={tool_name} | session={session_id} | "
                f"calls={rl.calls_in_window}/{rl.limit} | reset_in={rl.reset_in_seconds:.1f}s"
            )
            return ToolCallResult(
                tool_name=tool_name,
                agent_role=agent_role,
                session_id=session_id,
                verdict="RATE_LIMITED",
                denial_reason=(
                    f"Rate limit exceeded: {rl.calls_in_window}/{rl.limit} calls "
                    f"in 60s. Retry in {rl.reset_in_seconds:.0f}s."
                ),
                sanitized_params=parameters,
                permission=perm,
                rate_limit=rl,
                param_scan=_NO_SCAN,
                confidence=0.9,
            )

        # ── Step 3: Parameter injection scan ────────────────────────────────
        scan = self._sanitizer.scan(parameters, tool_name)
        if not scan.clean:
            logger.warning(
                f"[ToolGuard] DENIED (injection) | tool={tool_name} | "
                f"threat={scan.threat_type} | param={scan.affected_param} | "
                f"pattern={scan.matched_pattern}"
            )
            return ToolCallResult(
                tool_name=tool_name,
                agent_role=agent_role,
                session_id=session_id,
                verdict="DENIED",
                denial_reason=(
                    f"{scan.threat_type.replace('_', ' ').title()} detected in "
                    f"parameter '{scan.affected_param}' "
                    f"(pattern: {scan.matched_pattern})"
                ),
                sanitized_params=parameters,
                permission=perm,
                rate_limit=rl,
                param_scan=scan,
                confidence=0.95,
            )

        # ── All checks passed — record rate limit call ───────────────────────
        _rate_limiter.record(session_id, tool_name)
        logger.info(
            f"[ToolGuard] ALLOWED | tool={tool_name} | role={agent_role} | "
            f"session={session_id} | calls={rl.calls_in_window + 1}/{rl.limit}"
        )
        return ToolCallResult(
            tool_name=tool_name,
            agent_role=agent_role,
            session_id=session_id,
            verdict="ALLOWED",
            denial_reason="",
            sanitized_params=parameters,
            permission=perm,
            rate_limit=rl,
            param_scan=scan,
            confidence=0.0,
        )

    # ── Post-execution check ──────────────────────────────────────────────────

    def validate_response(
        self,
        tool_name: str,
        response: dict,
        max_size: int = 1_000_000,
    ) -> ValidationResult:
        """
        Validate a tool's response before injecting it into the LLM context.

        Parameters
        ----------
        tool_name : str  — For logging
        response  : dict — The tool's return value
        max_size  : int  — Maximum byte size (default 1MB)

        Returns
        -------
        ValidationResult with valid=True/False
        """
        return self._validator.validate(response, tool_name=tool_name, max_size=max_size)

    # ── LangGraph interface ────────────────────────────────────────────────────

    def check_for_graph(self, tool_call: dict) -> dict:
        """
        LangGraph-compatible interface. Returns a plain dict suitable for
        writing to GuardianState.tool_guard_result.
        """
        result = self.check_tool_call(tool_call)
        return {
            "guard":         "tool_guard",
            "verdict":       result.verdict,
            "denial_reason": result.denial_reason,
            "tool_name":     result.tool_name,
            "agent_role":    result.agent_role,
            "session_id":    result.session_id,
            "threat_type":   result.param_scan.threat_type,
            "confidence":    result.confidence,
            "rate_info": {
                "calls_in_window": result.rate_limit.calls_in_window,
                "limit":           result.rate_limit.limit,
                "reset_in":        result.rate_limit.reset_in_seconds,
            },
        }
