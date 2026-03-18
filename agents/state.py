"""
Shared State Schema for the Guardian System.

LangGraph passes this TypedDict between all nodes in the state graph.
Every node reads from and writes to this shared state.
"""
from typing import TypedDict, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime


# Verdict types - what the Input Guard returns
ThreatVerdict = Literal["SAFE", "SUSPICIOUS", "MALICIOUS"]

# Severity levels - what the Guardian Commander assigns
ThreatSeverity = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL", "NONE"]


@dataclass
class GuardResult:
    """Result from any guardian agent."""
    guard_name: str
    verdict: ThreatVerdict
    confidence: float          # 0.0 - 1.0
    threat_type: Optional[str] # e.g. "prompt_injection", "pii_leak"
    details: str               # Human-readable explanation
    matched_patterns: list     # Pattern names that fired
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class GuardianState(TypedDict):
    """
    Central state object that flows through the entire LangGraph pipeline.

    Architecture note: LangGraph uses immutable state updates — each node
    returns a dict of fields to update. Nodes never mutate state in place.
    """
    # --- Input ---
    user_input: str                         # Raw user input
    sanitized_input: Optional[str]          # Cleaned input (if SUSPICIOUS)

    # --- Guard Results ---
    input_guard_result: Optional[dict]      # Result from Input Guard
    memory_guard_result: Optional[dict]     # Day 3
    tool_guard_result: Optional[dict]       # Day 4
    output_guard_result: Optional[dict]     # Day 5
    trust_agent_result: Optional[dict]      # Day 7

    # --- Tool Call (Day 4) ---
    # Populated when a protected agent wants to call an external tool.
    # tool_guard_node reads this, validates it, and writes result to tool_guard_result.
    # Format: {tool_name, agent_role, session_id, parameters}
    pending_tool_call: Optional[dict]       # Tool call awaiting validation

    # --- Commander Decision ---
    threat_severity: ThreatSeverity         # Commander's severity assessment
    action_taken: str                       # What action was executed
    response_to_user: str                   # Final message back to user

    # --- Metadata ---
    session_id: str
    request_id: str
    is_blocked: bool                        # True = input never reached agent
    audit_log: list                         # Chronological event trail
