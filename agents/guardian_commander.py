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

Graph topology (Day 5):
  START → input_guard → memory_guard → tool_guard → guardian_commander
        → [END if blocked | protected_agent → output_guard] → END

Day 1: Input Guard Layer 1 (pattern matching)
Day 2: Input Guard Layer 2 (LLM classifier) + Layer 3 (scope validator)
Day 3: Memory Guard (RAG poisoning prevention)
Day 4: Tool Guard (least-privilege tool proxy)
Day 5: Output Guard (PII + secrets + entropy + prompt leak detection)
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
        "event":      "COMMANDER_DECISION",
        "verdict":    verdict,
        "severity":   severity,
        "action":     action,
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

def memory_guard_node(state: GuardianState) -> dict:
    """
    LangGraph node for document scanning via Memory Guard (Day 3+).

    Handles two document types:
      - 'text': plain text or extracted document content
      - 'pdf':  file path → PyMuPDF extracts ALL text including hidden content

    Flow:
      1. Extract content (PDF path → fitz, or plain text as-is)
      2. Scan with MemoryScanner (patterns + semantic attack similarity)
      3. Context window assessment (displacement risk)
      4. If POISONED → block; if SUSPICIOUS → sanitize; if CLEAN → pass
      5. Log to audit trail

    State keys read:
      document_to_ingest  : str  — Document text OR PDF file path
      document_id         : str  — Identifier for logging
      document_type       : str  — 'text' (default) or 'pdf'
      user_input          : str  — Used for context window assessment

    State keys written:
      memory_guard_result : dict — Full scan result
      is_blocked          : bool — True if POISONED document
      sanitized_input     : str  — Sanitized content (SUSPICIOUS documents)
      audit_log           : list — Updated audit trail
    """
    from guards.memory_guard import MemoryGuard

    doc_content = state.get("document_to_ingest", "")
    doc_id      = state.get("document_id", "unknown")
    doc_type    = state.get("document_type", "text")

    if not doc_content:
        logger.debug("[MemoryGuardNode] No document to scan — skipping")
        return {}

    guard = MemoryGuard()
    updates: dict = {}

    # ── Step 1: PDF path or plain text ────────────────────────────────────────
    if doc_type == "pdf":
        # doc_content is a file path for PDFs
        pdf_result, scan_result = guard.scan_pdf(doc_content)
        result = {
            "guard":             "memory_guard",
            "verdict":           scan_result.verdict,
            "confidence":        scan_result.confidence,
            "threat_categories": scan_result.threat_categories,
            "threats_found":     len(scan_result.threats),
            "obfuscation":       scan_result.obfuscation_methods,
            "metadata_threats":  scan_result.metadata_threats,
            "scan_note":         scan_result.scan_note,
            "doc_id":            doc_id,
            "doc_type":          "pdf",
            "hidden_spans":      pdf_result.hidden_span_count,
            "hidden_text":       pdf_result.hidden_text[:300] if pdf_result.hidden_text else "",
            "pdf_scan_notes":    pdf_result.scan_notes,
        }
        if scan_result.verdict in ("SUSPICIOUS", "POISONED") and scan_result.sanitized_content:
            updates["sanitized_input"] = scan_result.sanitized_content
    else:
        # Plain text document
        scan_result = guard.scan_document(doc_content, doc_id=doc_id, doc_type="text")
        result = {
            "guard":             "memory_guard",
            "verdict":           scan_result.verdict,
            "confidence":        scan_result.confidence,
            "threat_categories": scan_result.threat_categories,
            "threats_found":     len(scan_result.threats),
            "obfuscation":       scan_result.obfuscation_methods,
            "metadata_threats":  scan_result.metadata_threats,
            "scan_note":         scan_result.scan_note,
            "doc_id":            doc_id,
            "doc_type":          "text",
        }
        if scan_result.verdict in ("SUSPICIOUS", "POISONED") and scan_result.sanitized_content:
            updates["sanitized_input"] = scan_result.sanitized_content

    # ── Step 2: Context window displacement check ──────────────────────────────
    system_prompt_hint = "You are a helpful, secure AI assistant."  # conservative default
    ctx_assessment = guard.assess_context(
        system_prompt=system_prompt_hint,
        document=doc_content if doc_type == "text" else "",
        model_limit=200_000,
    )
    result["context_risk"]        = ctx_assessment.displacement_risk
    result["context_safe"]        = ctx_assessment.safe_to_process
    result["context_doc_tokens"]  = ctx_assessment.document_tokens
    result["context_recommendation"] = ctx_assessment.recommendation

    # Context stuffing attack: CRITICAL displacement → escalate verdict
    if ctx_assessment.displacement_risk == "CRITICAL" and result["verdict"] == "CLEAN":
        result["verdict"]    = "SUSPICIOUS"
        result["scan_note"] += f" | Context stuffing: {ctx_assessment.displacement_risk}"
        logger.warning(
            f"[MemoryGuardNode] Context stuffing risk CRITICAL — escalating to SUSPICIOUS | "
            f"doc_tokens={ctx_assessment.document_tokens:,} | doc={doc_id}"
        )

    # ── Step 3: Block if POISONED ──────────────────────────────────────────────
    is_poisoned = result["verdict"] == "POISONED"
    if is_poisoned:
        updates["is_blocked"] = True
        logger.warning(
            f"[MemoryGuardNode] BLOCKED POISONED DOCUMENT | "
            f"doc={doc_id} | conf={result['confidence']:.2f} | "
            f"categories={result['threat_categories']}"
        )

    # ── Audit ─────────────────────────────────────────────────────────────────
    audit_entry = {
        "event":         "MEMORY_GUARD_SCAN",
        "verdict":       result["verdict"],
        "confidence":    result["confidence"],
        "doc_id":        doc_id,
        "doc_type":      doc_type,
        "threats_found": result["threats_found"],
        "context_risk":  ctx_assessment.displacement_risk,
        "blocked":       is_poisoned,
    }
    current_log = state.get("audit_log", [])

    logger.info(
        f"[MemoryGuardNode] verdict={result['verdict']} | "
        f"doc={doc_id} | conf={result['confidence']:.2f} | "
        f"context_risk={ctx_assessment.displacement_risk}"
    )

    return {
        "memory_guard_result": result,
        "audit_log": current_log + [audit_entry],
        **updates,
    }


def tool_guard_node(state: GuardianState) -> dict:
    """
    LangGraph node that validates pending tool calls (Day 4+).

    Called when state contains a 'pending_tool_call' key.
    Silently skips (returns {}) for regular text requests with no tool call.

    Flow:
      1. PermissionMatrix: is this role allowed to use this tool?
      2. RateLimiter: has this session exceeded the per-tool limit?
      3. ParameterSanitizer: do params contain SQL injection / path traversal / SSRF?
      4. If DENIED or RATE_LIMITED → set is_blocked=True

    State keys read:
      pending_tool_call : dict — {tool_name, agent_role, session_id, parameters}

    State keys written:
      tool_guard_result : dict — Full validation result
      is_blocked        : bool — True if tool call was denied
      audit_log         : list — Updated audit trail
    """
    from guards.tool_guard import ToolGuard

    tool_call = state.get("pending_tool_call")
    if not tool_call:
        logger.debug("[ToolGuardNode] No pending tool call — skipping")
        return {}

    guard  = ToolGuard()
    result = guard.check_for_graph(tool_call)

    audit_entry = {
        "event":         "TOOL_GUARD_CHECK",
        "verdict":       result["verdict"],
        "tool_name":     result["tool_name"],
        "agent_role":    result["agent_role"],
        "denial_reason": result["denial_reason"],
        "threat_type":   result["threat_type"],
        "confidence":    result["confidence"],
    }
    current_log = state.get("audit_log", [])
    updates: dict = {
        "tool_guard_result": result,
        "audit_log": current_log + [audit_entry],
    }

    if result["verdict"] != "ALLOWED":
        updates["is_blocked"] = True
        logger.warning(
            f"[ToolGuardNode] BLOCKED | tool={result['tool_name']} | "
            f"verdict={result['verdict']} | reason={result['denial_reason']}"
        )
    else:
        logger.info(
            f"[ToolGuardNode] ALLOWED | tool={result['tool_name']} | "
            f"role={result['agent_role']}"
        )

    return updates


def protected_agent_node(state: GuardianState) -> dict:
    """
    Represents the AI agent being protected.
    In a real deployment, this calls your actual agent logic.

    Day 5 change: sets 'agent_response' (the raw output) instead of
    directly writing to 'response_to_user'. The output_guard_node reads
    'agent_response', scans it, and sets the final 'response_to_user'.
    """
    input_to_use = state.get("sanitized_input") or state["user_input"]
    logger.info(f"[ProtectedAgent] Processing: {input_to_use[:80]}...")

    # Simulate the agent's raw response — in production this would call
    # your actual LLM/agent logic. We echo the input for the stub.
    raw_response = f"[Agent Response] Processed: {input_to_use[:100]}"

    audit_entry = {
        "event":        "AGENT_PROCESSED",
        "input_length": len(input_to_use),
    }
    current_log = state.get("audit_log", [])

    return {
        "agent_response":  raw_response,   # raw output, read by output_guard_node
        "response_to_user": raw_response,  # default; overwritten by output_guard if needed
        "audit_log": current_log + [audit_entry],
    }


def output_guard_node(state: GuardianState) -> dict:
    """
    LangGraph node that scans agent responses for data leakage (Day 5+).

    Runs AFTER protected_agent_node. Skips silently if the request was
    already blocked upstream (is_blocked=True) — the agent never ran,
    so there is no response to scan.

    Flow:
      1. SensitiveDataScanner: redact API keys, DB strings, internal URLs
      2. PIIDetector (Presidio): redact names, emails, SSNs, credit cards
      3. EntropyAnalyzer: flag/remove encoded/steganographic data
      4. PromptLeakDetector: block if response reproduces system prompt

    State keys read:
      agent_response : str  — Raw agent output to scan
      system_prompt  : str  — Registered system prompt (for leak detection)
      session_id     : str  — Used to namespace the prompt leak store
      is_blocked     : bool — If True, skip entirely

    State keys written:
      output_guard_result : dict — Full scan result
      response_to_user    : str  — Sanitized / blocked response
      is_blocked          : bool — Set True if BLOCKED verdict
      audit_log           : list — Updated audit trail
    """
    from guards.output_guard import OutputGuard

    # Skip: request was blocked before reaching the agent
    if state.get("is_blocked"):
        logger.debug("[OutputGuardNode] Request was already blocked — skipping")
        return {}

    response = state.get("agent_response") or state.get("response_to_user", "")
    if not response:
        logger.debug("[OutputGuardNode] No agent response to scan — skipping")
        return {}

    guard = OutputGuard()
    result = guard.scan(
        response=response,
        system_prompt=state.get("system_prompt"),
        session_id=state.get("session_id"),
    )

    updates: dict = {"output_guard_result": result}

    if result["verdict"] == "BLOCKED":
        updates["response_to_user"] = (
            "I'm sorry, but I cannot provide that response. "
            "The Output Guard detected potential data leakage. "
            "Please contact your administrator if you believe this is an error."
        )
        updates["is_blocked"] = True
        logger.warning(
            f"[OutputGuardNode] BLOCKED — system prompt leakage | "
            f"confidence={result['confidence']:.2f}"
        )

    elif result["verdict"] == "REDACTED":
        updates["response_to_user"] = result["redacted_response"]
        logger.info(
            f"[OutputGuardNode] REDACTED — {len(result['findings'])} finding(s) | "
            f"threats={result['threat_type']}"
        )

    else:
        logger.info("[OutputGuardNode] CLEAN — response passes")

    # Audit entry
    audit_entry = {
        "event":          "OUTPUT_GUARD_SCAN",
        "verdict":        result["verdict"],
        "threat_type":    result.get("threat_type"),
        "findings_count": len(result.get("findings", [])),
        "confidence":     result["confidence"],
        "pii_found":      result["pii_result"].get("found", False),
        "secrets_found":  not result["secrets_result"].get("clean", True),
        "entropy_flag":   result["entropy_result"].get("suspicious", False),
    }
    updates["audit_log"] = state.get("audit_log", []) + [audit_entry]

    return updates


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

    Graph topology (Day 5):
      START → input_guard → memory_guard → tool_guard → guardian_commander
            → [END if blocked | protected_agent → output_guard] → END

    Each arrow is an edge. Conditional edges use routing functions above.
    """
    graph = StateGraph(GuardianState)

    # Register nodes
    graph.add_node("input_guard",        input_guard_node)
    graph.add_node("memory_guard",       memory_guard_node)    # Day 3
    graph.add_node("tool_guard",         tool_guard_node)      # Day 4
    graph.add_node("guardian_commander", guardian_commander_node)
    graph.add_node("protected_agent",    protected_agent_node)
    graph.add_node("output_guard",       output_guard_node)    # Day 5

    # Entry point
    graph.set_entry_point("input_guard")

    # Pre-agent pipeline:
    # input_guard → memory_guard → tool_guard → guardian_commander
    # (memory_guard and tool_guard silently pass through when not applicable)
    graph.add_edge("input_guard",  "memory_guard")
    graph.add_edge("memory_guard", "tool_guard")
    graph.add_edge("tool_guard",   "guardian_commander")

    # commander → conditional: blocked → END, else → protected_agent
    graph.add_conditional_edges(
        "guardian_commander",
        route_after_commander,
        {
            "protected_agent": "protected_agent",
            "__end__": END,
        }
    )

    # Post-agent pipeline:
    # protected_agent → output_guard → END
    graph.add_edge("protected_agent", "output_guard")
    graph.add_edge("output_guard",    END)

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

    def process(
        self,
        user_input: str,
        session_id: str = None,
        system_prompt: str = None,
    ) -> dict:
        """
        Process a user input through the full Guardian pipeline.

        Args:
            user_input:     Raw text from the user
            session_id:     Optional session identifier for audit trail
            system_prompt:  Optional system prompt for Output Guard leakage detection

        Returns:
            Final state dict with response_to_user, threat info, audit_log
        """
        initial_state: GuardianState = {
            "user_input":           user_input,
            "sanitized_input":      None,
            "input_guard_result":   None,
            "memory_guard_result":  None,
            "tool_guard_result":    None,
            "output_guard_result":  None,
            "trust_agent_result":   None,
            "pending_tool_call":    None,
            "system_prompt":        system_prompt,   # Day 5
            "agent_response":       None,            # Day 5
            "threat_severity":      "NONE",
            "action_taken":         "",
            "response_to_user":     "",
            "session_id":           session_id or str(uuid.uuid4()),
            "request_id":           str(uuid.uuid4()),
            "is_blocked":           False,
            "audit_log":            [],
        }

        final_state = self.graph.invoke(initial_state)
        return final_state
