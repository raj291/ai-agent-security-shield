"""
Tool Guard — Sliding-Window In-Memory Rate Limiter

Tracks how many tool calls a session has made in the last 60 seconds.
Blocks calls that exceed the per-tool limit.

Design decisions
----------------
- In-memory (no Redis): sufficient for Day 4. Production upgrade path:
  swap _store for a Redis sorted set using ZADD + ZREMRANGEBYSCORE.
- Sliding window (not fixed bucket): 50 calls in any 60-second window,
  not "50 calls in the current minute bucket". This is more accurate —
  an attacker can't burst 50 calls at 11:59 and 50 more at 12:00.
- Class-level _store: shared across ToolGuard instances within one process
  so the count isn't reset when a new ToolGuard() is constructed per request.

Usage
-----
    limiter = RateLimiter()

    result = limiter.check("sess_abc", "database_query", limit=30)
    if result.allowed:
        limiter.record("sess_abc", "database_query")
        # ... execute the tool
    else:
        # Block: result.calls_in_window / result.limit exceeded
        # result.reset_in_seconds tells caller when the window clears
"""
import time
from collections import deque
from dataclasses import dataclass

WINDOW_SECONDS: int = 60


@dataclass
class RateLimitResult:
    """
    Result of a rate-limit check.

    Fields
    ------
    allowed          : True if the call is within the limit
    calls_in_window  : Number of calls already made in the current window
    limit            : Maximum calls allowed per window
    reset_in_seconds : Seconds until the oldest call expires (0.0 if window is empty)
    """
    allowed:          bool
    calls_in_window:  int
    limit:            int
    reset_in_seconds: float


class RateLimiter:
    """
    Sliding-window in-memory rate limiter.

    _store is a class-level dict so all instances share the same counters.
    Structure: {session_id: {tool_name: deque[float timestamps]}}

    Two-step usage:
      1. call check()  → read-only, returns RateLimitResult
      2. call record() → write, stamps the current timestamp (only on ALLOWED calls)
    """

    # Shared across all RateLimiter() instances — intentional class variable
    _store: dict = {}

    def check(self, session_id: str, tool_name: str, limit: int) -> RateLimitResult:
        """
        Check whether this call is within the rate limit.
        Does NOT record the call — call record() separately after approval.

        Parameters
        ----------
        session_id : str — Per-session identifier (from GuardianState.session_id)
        tool_name  : str — Tool being called (e.g. "database_query")
        limit      : int — Max calls per WINDOW_SECONDS (from policy YAML)

        Returns
        -------
        RateLimitResult with allowed=True/False
        """
        timestamps = self._get_timestamps(session_id, tool_name)
        self._clean_old(timestamps)
        count = len(timestamps)

        # How long until the oldest call expires (0 if no calls yet)
        reset_in = (timestamps[0] + WINDOW_SECONDS - time.time()) if timestamps else 0.0

        return RateLimitResult(
            allowed=count < limit,
            calls_in_window=count,
            limit=limit,
            reset_in_seconds=max(0.0, reset_in),
        )

    def record(self, session_id: str, tool_name: str) -> None:
        """
        Record a call timestamp. Call this AFTER check() returns allowed=True.

        Parameters
        ----------
        session_id : str — Session that made the call
        tool_name  : str — Tool that was called
        """
        timestamps = self._get_timestamps(session_id, tool_name)
        self._clean_old(timestamps)
        timestamps.append(time.time())

    def reset_session(self, session_id: str) -> None:
        """Clear all rate limit counters for a session (useful in tests)."""
        if session_id in self._store:
            del self._store[session_id]

    # ── Private helpers ──────────────────────────────────────────────────────

    def _get_timestamps(self, session_id: str, tool_name: str) -> deque:
        """Return (creating if necessary) the deque for session+tool."""
        self._store.setdefault(session_id, {})
        self._store[session_id].setdefault(tool_name, deque())
        return self._store[session_id][tool_name]

    def _clean_old(self, timestamps: deque) -> None:
        """Remove timestamps older than WINDOW_SECONDS from the left of the deque."""
        cutoff = time.time() - WINDOW_SECONDS
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()
