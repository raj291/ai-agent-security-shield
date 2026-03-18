"""
Tool Guard Tests — Day 4

Covers all four components:
  - PermissionMatrix: role checks, forbidden keywords/tables, path/domain restrictions
  - RateLimiter: sliding-window enforcement, per-tool limits, session isolation
  - ParameterSanitizer: SQL injection, path traversal, command injection, SSRF
  - ResponseValidator: size limits, credential leakage detection
  - ToolGuard facade: end-to-end scenarios, LangGraph interface

All tests are fully offline — no Claude API calls, no ChromaDB writes.
"""
import time
import uuid
import pytest

from guards.tool_guard import (
    ToolGuard, ToolCallResult,
    PermissionMatrix, PermissionResult,
    RateLimiter, RateLimitResult,
    ParameterSanitizer, SanitizeResult,
    ResponseValidator, ValidationResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _session() -> str:
    """Generate a unique session ID for each test to avoid cross-test pollution."""
    return f"test_sess_{uuid.uuid4().hex[:8]}"


def _tool_call(
    tool_name="database_query",
    agent_role="customer_support",
    session_id=None,
    parameters=None,
) -> dict:
    return {
        "tool_name":  tool_name,
        "agent_role": agent_role,
        "session_id": session_id or _session(),
        "parameters": parameters or {"query": "SELECT * FROM orders WHERE id = 1"},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. PERMISSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

class TestPermissionMatrix:
    """YAML-driven permission matrix checks."""

    @pytest.fixture(scope="class")
    def matrix(self):
        return PermissionMatrix()

    def test_authorized_role_allowed(self, matrix):
        """customer_support is allowed to call database_query."""
        result = matrix.check("database_query", "customer_support", {"query": "SELECT id FROM orders"})
        assert result.allowed is True
        assert result.denial_reason == ""

    def test_authorized_analyst_allowed(self, matrix):
        """analyst is allowed to call database_query."""
        result = matrix.check("database_query", "analyst", {"query": "SELECT * FROM reports"})
        assert result.allowed is True

    def test_unauthorized_role_denied(self, matrix):
        """analyst is NOT allowed to call send_email."""
        result = matrix.check("send_email", "analyst", {})
        assert result.allowed is False
        assert "analyst" in result.denial_reason
        assert "send_email" in result.denial_reason

    def test_unknown_tool_denied(self, matrix):
        """Unregistered tool names must always be denied (fail-closed)."""
        result = matrix.check("hack_database", "analyst", {})
        assert result.allowed is False
        assert "not registered" in result.denial_reason

    def test_forbidden_table_denied(self, matrix):
        """Querying a sensitive table (users_credentials) must be denied."""
        result = matrix.check(
            "database_query",
            "customer_support",
            {"query": "SELECT * FROM users_credentials"},
        )
        assert result.allowed is False
        assert "users_credentials" in result.denial_reason

    def test_forbidden_keyword_denied(self, matrix):
        """SQL DROP keyword in a database_query must be denied."""
        result = matrix.check(
            "database_query",
            "analyst",
            {"query": "DROP TABLE orders"},
        )
        assert result.allowed is False
        assert "DROP" in result.denial_reason

    def test_any_role_can_send_agent_message(self, matrix):
        """agent_message has allowed_roles: [any] — all roles permitted."""
        for role in ["customer_support", "analyst", "admin", "unknown_role"]:
            result = matrix.check("agent_message", role, {"target_agent": "trust_agent", "message": "hi"})
            assert result.allowed is True, f"Role '{role}' should be allowed for agent_message"

    def test_file_read_path_outside_allowed_denied(self, matrix):
        """file_read with path outside approved directories must be denied."""
        result = matrix.check(
            "file_read",
            "analyst",
            {"path": "/etc/passwd"},
        )
        assert result.allowed is False
        assert "not in the approved directories" in result.denial_reason

    def test_file_read_approved_path_allowed(self, matrix):
        """file_read with approved path must be allowed."""
        result = matrix.check(
            "file_read",
            "analyst",
            {"path": "/data/reports/q3_report.pdf"},
        )
        assert result.allowed is True

    def test_http_request_unapproved_domain_denied(self, matrix):
        """http_request to an external attacker domain must be denied."""
        result = matrix.check(
            "http_request",
            "analyst",
            {"url": "https://attacker.com/exfiltrate"},
        )
        assert result.allowed is False
        assert "not in the approved domains" in result.denial_reason

    def test_rate_limit_returned_on_allowed(self, matrix):
        """Allowed result should include a non-zero rate_limit."""
        result = matrix.check("database_query", "customer_support", {"query": "SELECT 1"})
        assert result.allowed is True
        assert result.rate_limit > 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. RATE LIMITER
# ─────────────────────────────────────────────────────────────────────────────

class TestRateLimiter:
    """Sliding-window rate limiter tests."""

    def test_within_limit_allowed(self):
        """10 calls against a limit of 30 should all be allowed."""
        limiter = RateLimiter()
        sess = _session()
        for i in range(10):
            result = limiter.check(sess, "database_query", limit=30)
            assert result.allowed is True, f"Call {i+1} should be allowed"
            limiter.record(sess, "database_query")

    def test_exceeds_limit_blocked(self):
        """The (limit+1)th call within the window must be blocked."""
        limiter = RateLimiter()
        sess = _session()
        limit = 5
        # Fill the window
        for _ in range(limit):
            limiter.record(sess, "test_tool")
        # Next check should be blocked
        result = limiter.check(sess, "test_tool", limit=limit)
        assert result.allowed is False
        assert result.calls_in_window == limit

    def test_per_tool_email_limit(self):
        """send_email has a limit of 10. The 11th call must be blocked."""
        limiter = RateLimiter()
        sess = _session()
        for _ in range(10):
            limiter.record(sess, "send_email")
        result = limiter.check(sess, "send_email", limit=10)
        assert result.allowed is False
        assert result.calls_in_window == 10

    def test_different_sessions_independent(self):
        """Session A being at the limit must not block session B."""
        limiter = RateLimiter()
        sess_a = _session()
        sess_b = _session()
        limit = 3
        # Fill session A
        for _ in range(limit):
            limiter.record(sess_a, "database_query")
        # Session A at limit
        assert limiter.check(sess_a, "database_query", limit=limit).allowed is False
        # Session B unaffected
        assert limiter.check(sess_b, "database_query", limit=limit).allowed is True

    def test_result_fields_present(self):
        """RateLimitResult must have all required fields."""
        limiter = RateLimiter()
        result = limiter.check(_session(), "database_query", limit=50)
        assert hasattr(result, "allowed")
        assert hasattr(result, "calls_in_window")
        assert hasattr(result, "limit")
        assert hasattr(result, "reset_in_seconds")
        assert result.limit == 50
        assert isinstance(result.reset_in_seconds, float)

    def test_calls_in_window_accurate(self):
        """calls_in_window should reflect the number of recorded calls."""
        limiter = RateLimiter()
        sess = _session()
        for i in range(7):
            limiter.record(sess, "http_request")
        result = limiter.check(sess, "http_request", limit=20)
        assert result.calls_in_window == 7


# ─────────────────────────────────────────────────────────────────────────────
# 3. PARAMETER SANITIZER
# ─────────────────────────────────────────────────────────────────────────────

class TestParameterSanitizer:
    """Parameter injection detection tests."""

    @pytest.fixture(scope="class")
    def sanitizer(self):
        return ParameterSanitizer()

    # SQL Injection
    def test_sql_union_select_blocked(self, sanitizer):
        result = sanitizer.scan({"query": "1 UNION SELECT * FROM api_keys-- "})
        assert result.clean is False
        assert result.threat_type == "sql_injection"
        assert result.affected_param == "query"

    def test_sql_drop_table_blocked(self, sanitizer):
        result = sanitizer.scan({"query": "DROP TABLE users"})
        assert result.clean is False
        assert result.threat_type == "sql_injection"

    def test_sql_comment_injection_blocked(self, sanitizer):
        result = sanitizer.scan({"query": "SELECT * FROM orders WHERE id=1 -- AND active=1"})
        assert result.clean is False
        assert result.threat_type == "sql_injection"

    def test_sql_blind_sleep_blocked(self, sanitizer):
        result = sanitizer.scan({"query": "SELECT * FROM orders WHERE id=1 AND SLEEP(5)"})
        assert result.clean is False
        assert result.threat_type == "sql_injection"

    # Path Traversal
    def test_path_traversal_unix_blocked(self, sanitizer):
        result = sanitizer.scan({"path": "/data/reports/../../etc/passwd"})
        assert result.clean is False
        assert result.threat_type == "path_traversal"

    def test_path_etc_passwd_blocked(self, sanitizer):
        result = sanitizer.scan({"path": "/etc/passwd"})
        assert result.clean is False
        assert result.threat_type == "path_traversal"

    def test_path_url_encoded_blocked(self, sanitizer):
        result = sanitizer.scan({"path": "%2e%2e%2fetc%2fpasswd"})
        assert result.clean is False
        assert result.threat_type == "path_traversal"

    # Command Injection
    def test_command_injection_semicolon_blocked(self, sanitizer):
        result = sanitizer.scan({"filename": "report.pdf; rm -rf /"})
        assert result.clean is False
        assert result.threat_type == "command_injection"

    def test_command_injection_backtick_blocked(self, sanitizer):
        result = sanitizer.scan({"input": "value`whoami`"})
        assert result.clean is False
        assert result.threat_type == "command_injection"

    def test_command_injection_subshell_blocked(self, sanitizer):
        result = sanitizer.scan({"name": "$(id)"})
        assert result.clean is False
        assert result.threat_type == "command_injection"

    # SSRF
    def test_ssrf_localhost_blocked(self, sanitizer):
        result = sanitizer.scan({"url": "http://localhost:8080/admin"})
        assert result.clean is False
        assert result.threat_type == "ssrf"

    def test_ssrf_aws_metadata_blocked(self, sanitizer):
        result = sanitizer.scan({"url": "http://169.254.169.254/latest/meta-data/iam/credentials"})
        assert result.clean is False
        assert result.threat_type == "ssrf"

    def test_ssrf_internal_ip_blocked(self, sanitizer):
        result = sanitizer.scan({"url": "http://192.168.1.1/router-admin"})
        assert result.clean is False
        assert result.threat_type == "ssrf"

    # Clean parameters
    def test_clean_select_query_allowed(self, sanitizer):
        result = sanitizer.scan({"query": "SELECT id, name FROM orders WHERE status = 'shipped' LIMIT 50"})
        assert result.clean is True, f"Clean query flagged: {result.matched_pattern}"

    def test_clean_file_path_allowed(self, sanitizer):
        result = sanitizer.scan({"path": "/data/reports/q3_2024_revenue.pdf"})
        assert result.clean is True, f"Clean path flagged: {result.matched_pattern}"

    def test_clean_http_url_allowed(self, sanitizer):
        result = sanitizer.scan({"url": "https://api.company.com/v1/orders?page=1"})
        assert result.clean is True, f"Clean URL flagged: {result.matched_pattern}"

    def test_non_string_params_ignored(self, sanitizer):
        """Non-string parameter values (ints, lists) should not cause errors."""
        result = sanitizer.scan({"limit": 10, "page": 2, "active": True})
        assert result.clean is True

    def test_result_fields_present(self, sanitizer):
        """SanitizeResult must have all required fields."""
        result = sanitizer.scan({"query": "SELECT 1"})
        assert hasattr(result, "clean")
        assert hasattr(result, "threat_type")
        assert hasattr(result, "matched_pattern")
        assert hasattr(result, "affected_param")
        assert hasattr(result, "matched_text")


# ─────────────────────────────────────────────────────────────────────────────
# 4. RESPONSE VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

class TestResponseValidator:
    """Tool response validation tests."""

    @pytest.fixture(scope="class")
    def validator(self):
        return ResponseValidator()

    def test_valid_response_passes(self, validator):
        response = {"rows": [{"id": 1, "name": "Alice", "status": "active"}], "count": 1}
        result = validator.validate(response)
        assert result.valid is True
        assert result.reason == ""
        assert result.size_bytes > 0

    def test_empty_response_passes(self, validator):
        result = validator.validate({})
        assert result.valid is True

    def test_oversized_response_blocked(self, validator):
        """Response > 1MB must be rejected."""
        big_response = {"data": "x" * 1_100_000}
        result = validator.validate(big_response, max_size=1_000_000)
        assert result.valid is False
        assert "too large" in result.reason
        assert result.size_bytes > 1_000_000

    def test_custom_size_limit(self, validator):
        """Custom max_size should be respected."""
        response = {"data": "hello world"}
        # Very small limit
        result = validator.validate(response, max_size=10)
        assert result.valid is False
        assert "too large" in result.reason

    def test_credential_in_response_blocked(self, validator):
        """Response containing API key pattern must be rejected."""
        response = {"config": "api_key=sk-abc123abc123abc123abc123abc123ab"}
        result = validator.validate(response)
        assert result.valid is False
        assert "credential" in result.reason

    def test_private_key_in_response_blocked(self, validator):
        """Response with PEM private key header must be rejected."""
        response = {"key": "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAK...\n-----END RSA PRIVATE KEY-----"}
        result = validator.validate(response)
        assert result.valid is False
        assert "credential" in result.reason

    def test_size_bytes_field_accurate(self, validator):
        """size_bytes must be a non-negative integer."""
        response = {"message": "ok"}
        result = validator.validate(response)
        assert isinstance(result.size_bytes, int)
        assert result.size_bytes > 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. TOOL GUARD — End-to-End
# ─────────────────────────────────────────────────────────────────────────────

class TestToolGuard:
    """End-to-end ToolGuard scenarios."""

    @pytest.fixture(scope="class")
    def guard(self):
        return ToolGuard()

    def test_clean_authorized_call_allowed(self, guard):
        """Clean call from authorized role with safe parameters → ALLOWED."""
        result = guard.check_tool_call(_tool_call(
            tool_name="database_query",
            agent_role="customer_support",
            parameters={"query": "SELECT id, status FROM orders WHERE customer_id = 42"},
        ))
        assert result.verdict == "ALLOWED"
        assert result.denial_reason == ""
        assert result.confidence == 0.0

    def test_unauthorized_role_denied(self, guard):
        """Unauthorized role calling a tool → DENIED."""
        result = guard.check_tool_call(_tool_call(
            tool_name="send_email",
            agent_role="analyst",
        ))
        assert result.verdict == "DENIED"
        assert "analyst" in result.denial_reason or "not authorized" in result.denial_reason

    def test_unknown_tool_denied(self, guard):
        """Calling an unregistered tool → DENIED (fail-closed)."""
        result = guard.check_tool_call(_tool_call(tool_name="exfiltrate_database"))
        assert result.verdict == "DENIED"
        assert "not registered" in result.denial_reason

    def test_sql_injection_denied(self, guard):
        """SQL injection in parameters → DENIED."""
        result = guard.check_tool_call(_tool_call(
            tool_name="database_query",
            agent_role="analyst",
            parameters={"query": "1 UNION SELECT password FROM users--"},
        ))
        assert result.verdict == "DENIED"
        assert result.param_scan.threat_type == "sql_injection"

    def test_path_traversal_denied(self, guard):
        """Path traversal in file_read → DENIED."""
        result = guard.check_tool_call(_tool_call(
            tool_name="file_read",
            agent_role="analyst",
            parameters={"path": "/data/reports/../../etc/shadow"},
        ))
        assert result.verdict == "DENIED"

    def test_ssrf_denied(self, guard):
        """SSRF URL in agent_message (any-role tool, no domain restriction) → DENIED by sanitizer."""
        result = guard.check_tool_call(_tool_call(
            tool_name="agent_message",
            agent_role="analyst",
            parameters={"target_url": "http://169.254.169.254/latest/meta-data/"},
        ))
        assert result.verdict == "DENIED"
        assert result.param_scan.threat_type == "ssrf"

    def test_rate_limited_after_exceeding_limit(self):
        """Exceeding per-tool rate limit → RATE_LIMITED."""
        guard = ToolGuard()
        sess = _session()
        # The database_query limit is 30 per the YAML. Use the shared limiter.
        # Manually fill the rate limiter for this session:
        from guards.tool_guard.tool_guard import _rate_limiter
        _rate_limiter.reset_session(sess)  # Start clean
        # Fill to limit
        from guards.tool_guard.rate_limiter import WINDOW_SECONDS
        limit = 30  # database_query limit from YAML
        for _ in range(limit):
            _rate_limiter.record(sess, "database_query")
        # Next call must be rate limited
        result = guard.check_tool_call(_tool_call(
            tool_name="database_query",
            agent_role="customer_support",
            session_id=sess,
        ))
        assert result.verdict == "RATE_LIMITED"
        assert result.rate_limit.allowed is False

    def test_check_for_graph_returns_dict(self, guard):
        """check_for_graph() must return a plain dict with required keys."""
        result = guard.check_for_graph(_tool_call())
        required_keys = {"guard", "verdict", "denial_reason", "tool_name", "agent_role",
                         "session_id", "threat_type", "confidence", "rate_info"}
        assert required_keys.issubset(result.keys())
        assert result["guard"] == "tool_guard"
        assert isinstance(result["rate_info"], dict)

    def test_verdict_values_in_valid_set(self, guard):
        """All possible tool call scenarios must return one of the three valid verdicts."""
        test_cases = [
            _tool_call(),  # Should be ALLOWED
            _tool_call(tool_name="unknown_tool"),  # Should be DENIED
        ]
        valid_verdicts = {"ALLOWED", "DENIED", "RATE_LIMITED"}
        for tc in test_cases:
            result = guard.check_tool_call(tc)
            assert result.verdict in valid_verdicts, f"Invalid verdict: {result.verdict}"

    def test_forbidden_table_denied_end_to_end(self, guard):
        """Accessing a sensitive table goes through PermissionMatrix and is DENIED."""
        result = guard.check_tool_call(_tool_call(
            tool_name="database_query",
            agent_role="customer_support",
            parameters={"query": "SELECT * FROM api_keys WHERE active = 1"},
        ))
        assert result.verdict == "DENIED"
        assert "api_keys" in result.denial_reason

    def test_validate_response_passes_for_clean_data(self, guard):
        """validate_response should pass for a small clean dict."""
        val = guard.validate_response("database_query", {"rows": [{"id": 1}]})
        assert val.valid is True

    def test_validate_response_blocked_for_large_data(self, guard):
        """validate_response should fail for > 1MB response."""
        val = guard.validate_response("database_query", {"data": "z" * 1_100_000})
        assert val.valid is False
        assert "too large" in val.reason
