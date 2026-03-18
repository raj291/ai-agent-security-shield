"""
Tool Guard — Least-Privilege Tool Call Proxy

Intercepts all agent tool calls before execution and validates:
  1. Permission: Is this agent role allowed to use this tool?
  2. Rate limit: Has this session exceeded the per-tool call limit?
  3. Parameters: Do any values contain SQL injection, path traversal, or SSRF?

Post-execution:
  4. Response: Is the tool response within size limits and free of credentials?

Public API
----------
    from guards.tool_guard import ToolGuard

    guard = ToolGuard()

    # Pre-execution check
    result = guard.check_tool_call({
        "tool_name":  "database_query",
        "agent_role": "customer_support",
        "session_id": "sess_abc123",
        "parameters": {"query": "SELECT * FROM orders WHERE id = 42"},
    })
    # result.verdict == "ALLOWED" | "DENIED" | "RATE_LIMITED"

    # Post-execution response validation
    val = guard.validate_response("database_query", tool_response_dict)
    # val.valid == True | False

    # LangGraph node interface
    result_dict = guard.check_for_graph(tool_call_dict)
"""
from .tool_guard import ToolGuard, ToolCallResult
from .permission_matrix import PermissionMatrix, PermissionResult
from .rate_limiter import RateLimiter, RateLimitResult
from .parameter_sanitizer import ParameterSanitizer, SanitizeResult
from .response_validator import ResponseValidator, ValidationResult

__all__ = [
    # Main facade
    "ToolGuard",
    "ToolCallResult",
    # Components (exposed for direct testing)
    "PermissionMatrix",
    "PermissionResult",
    "RateLimiter",
    "RateLimitResult",
    "ParameterSanitizer",
    "SanitizeResult",
    "ResponseValidator",
    "ValidationResult",
]
