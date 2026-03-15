"""
Guardian Commander - Central Orchestrator

The Guardian Commander is the brain of the AI Agent Security Shield.
It sits at the root of the LangGraph StateGraph and:
  1. Receives every user input
  2. Routes it through appropriate guards
  3. Decides threat severity based on guard reports
  4. Triggers containment actions
  5. Coordinates between guards during active incidents

Architecture: LangGraph StateGraph
  - Each guard is a NODE in the graph
  - Edges define the routing logic
  - Conditional edges allow dynamic routing based on state

Day 1: Commander does routing only. Logic added each day as guards are built.
"""
import uuid
import logging
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from agents.state import GuardianState, ThreatSeverity

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# NODE: Guardian Commander
# ─────────────────────────────────────────────

def guardian_commander_node(state: GuardianState) -> dict:
    """
    Central decision node. Reads all guard results and decides:
      - Threat severity (LOW/MEDIUM/HIGH/CRITICAL)
      - What action to take
      - What message to return to user

    Called AFTER all relevant guards have processed the input.
    """
    input_result = state.get("input_guard_result", {})
    verdict = input_result.get("verdict", "SAFE")
    confidence = input_result.get("confidence", 0.0)

    # Severity mapping based on Input Guard verdict
    severity_map = {
        "SAFE": "NONE",
        "SUSPICIOUS": "MEDIUM",
        "MALICIOUS": "HIGH",
    }
    severity: ThreatSeverity = severity_map.get(verdict, "NONE")

    # Determine action and user response
    if verdict == "MALICIOUS":
        action = "BLOCK_INPUT"
        user_response = (
            "Your request was blocked by the security system. "
            "It appears to contain content that violates security policies. "
            "Please rephrase your request."
        )
        is_blocked = True

    elif verdict == "SUSPICIOUS":
        action = "SANITIZE_AND_MONITOR"
        user_response = (
            "Your request has been processed. "
            "[Security: Input was sanitized before processing]"
        )
        is_blocked = False

    else:  # SAFE
        action = "PASS_TO_AGENT"
        user_response = "Request processed normally."
        is_blocked = False

    # Audit log entry
    audit_entry = {
        "event": "COMMANDER_DECISION",
        "verdict": verdict,
        "severity": severity,
        "action": action,
        "confidence": confidence,
        "threat_type": input_result.get("threat_type"),
    }

    current_log = state.get("audit_log", [])

    logger.info(f"[Commander] verdict={verdict} severity={severity} action={action}")

    return {
        "threat_severity": severity,
        "action_taken": action,
        "response_to_user": user_response,
        "is_blocked": is_blocked,
        "audit_log": current_log + [audit_entry],
    }


# ─────────────────────────────────────────────
# NODE: Input Guard Proxy
# Wraps the actual Input Guard so LangGraph can call it as a node
# ─────────────────────────────────────────────

def input_guard_node(state: GuardianState) -> dict:
    """
    LangGraph node that delegates to the Input Guard pipeline.
    The Input Guard runs Layer 1 (pattern matching) today.
    Layer 2 (LLM classifier) and Layer 3 (scope validator) added Day 2.
    """
    from guards.input_guard import InputGuard

    guard = InputGuard()
    result = guard.scan(state["user_input"])

    audit_entry = {
        "event": "INPUT_GUARD_SCAN",
        "verdict": result["verdict"],
        "patterns_matched": result.get("matched_patterns", []),
        "threat_type": result.get("threat_type"),
    }

    current_log = state.get("audit_log", [])

    return {
        "input_guard_result": result,
        "sanitized_input": result.get("sanitized_input"),
        "audit_log": current_log + [audit_entry],
    }


# ─────────────────────────────────────────────
# NODE: Protected Agent (stub)
# The agent being protected — simulated for now
# ─────────────────────────────────────────────

def protected_agent_node(state: GuardianState) -> dict:
    """
    Represents the AI agent being protected.
    In a real deployment, this calls your actual agent logic.
    For Day 1, it's a stub that echoes the (potentially sanitized) input.
    """
    input_to_use = state.get("sanitized_input") or state["user_input"]
    logger.info(f"[ProtectedAgent] Processing: {input_to_use[:80]}...")

    audit_entry = {
        "event": "AGENT_PROCESSED",
        "input_length": len(input_to_use),
    }
    current_log = state.get("audit_log", [])

    return {
        "response_to_user": f"[Agent Response] Processed: {input_to_use[:100]}",
        "audit_log": current_log + [audit_entry],
    }


# ─────────────────────────────────────────────
# ROUTING: Conditional Edge Logic
# ─────────────────────────────────────────────

def route_after_input_guard(state: GuardianState) -> Literal["guardian_commander", "guardian_commander"]:
    """
    After Input Guard runs, always go to Commander.
    Commander decides what happens next based on severity.

    In Week 2, this will also route to Memory Guard for document uploads.
    """
    return "guardian_commander"


def route_after_commander(state: GuardianState) -> Literal["protected_agent", "__end__"]:
    """
    After Commander decides:
      - MALICIOUS → END (blocked, no agent access)
      - SUSPICIOUS → protected_agent (with sanitized input + monitoring)
      - SAFE → protected_agent (normal flow)
    """
    if state.get("is_blocked"):
        return "__end__"
    return "protected_agent"


# ─────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────

def build_guardian_graph() -> CompiledStateGraph:
    """
    Constructs and compiles the Guardian LangGraph StateGraph.

    Graph topology (Day 1):
      START → input_guard → guardian_commander → [END | protected_agent] → END

    Each arrow is an edge. Conditional edges use routing functions above.
    """
    graph = StateGraph(GuardianState)

    # Register nodes
    graph.add_node("input_guard", input_guard_node)
    graph.add_node("guardian_commander", guardian_commander_node)
    graph.add_node("protected_agent", protected_agent_node)

    # Entry point: all requests start at input_guard
    graph.set_entry_point("input_guard")

    # input_guard → guardian_commander (always)
    graph.add_edge("input_guard", "guardian_commander")

    # guardian_commander → conditional routing
    graph.add_conditional_edges(
        "guardian_commander",
        route_after_commander,
        {
            "protected_agent": "protected_agent",
            "__end__": END,
        }
    )

    # protected_agent → END
    graph.add_edge("protected_agent", END)

    return graph.compile()


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

class GuardianCommander:
    """
    Public interface for the Guardian System.
    Wraps the compiled LangGraph and provides a clean process() method.
    """

    def __init__(self):
        self.graph = build_guardian_graph()
        logger.info("[GuardianCommander] System initialized. All guards online.")

    def process(self, user_input: str, session_id: str = None) -> dict:
        """
        Process a user input through the full Guardian pipeline.

        Args:
            user_input: Raw text from the user
            session_id: Optional session identifier for audit trail

        Returns:
            Final state dict with response_to_user, threat info, audit_log
        """
        initial_state: GuardianState = {
            "user_input": user_input,
            "sanitized_input": None,
            "input_guard_result": None,
            "memory_guard_result": None,
            "tool_guard_result": None,
            "output_guard_result": None,
            "trust_agent_result": None,
            "threat_severity": "NONE",
            "action_taken": "",
            "response_to_user": "",
            "session_id": session_id or str(uuid.uuid4()),
            "request_id": str(uuid.uuid4()),
            "is_blocked": False,
            "audit_log": [],
        }

        final_state = self.graph.invoke(initial_state)
        return final_state
