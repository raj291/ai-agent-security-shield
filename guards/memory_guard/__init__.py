"""
Memory Guard — Public API

Intercepts documents (PDFs, text) BEFORE they reach the LLM.
Extracts all content including hidden text, detects attacks, and either:
  - MALICIOUS → block, sanitize, store attack variant in knowledge base
  - CLEAN → pass full text to the LLM/protected agent node

ChromaDB stores only ATTACK examples (never clean documents).
When a new attack is found, it's stored for future semantic detection.

Usage
-----
    from guards.memory_guard import MemoryGuard

    mg = MemoryGuard()

    # Scan plain text
    result = mg.scan_document("document text here", doc_id="doc_001")
    if result.verdict == "CLEAN":
        pass_to_llm(result.sanitized_input or text)
    else:
        block_and_alert(result)

    # Scan a PDF file (extracts hidden text automatically)
    pdf_result, scan_result = mg.scan_pdf("/path/to/report.pdf")
    # pdf_result.hidden_text shows what was hidden
    # scan_result.verdict is the security decision

    # Check context window (prevent displacement attacks)
    ctx = mg.assess_context(system_prompt="...", document="large doc text")
    if not ctx.safe_to_process:
        truncate(document)
"""
import logging

from .memory_scanner import MemoryScanner, ScanResult, DocumentThreat
from .attack_store import AttackStore, AttackSimilarityResult
from .pdf_scanner import PdfScanner, PdfScanResult
from .context_monitor import ContextWindowMonitor, ContextAssessment

logger = logging.getLogger(__name__)


class MemoryGuard:
    """
    The Memory Guard — Day 3 of the AI Agent Security Shield.

    Scans documents for:
      1. Regex attack patterns (via MemoryScanner, 91+ patterns)
      2. Semantic similarity to known attacks (via AttackStore, ChromaDB)
      3. Hidden text in PDFs (via PdfScanner, PyMuPDF)
      4. Context window displacement risk (via ContextWindowMonitor)

    If a new attack is detected, it is stored in the attack knowledge base
    so future semantically similar attacks are caught faster.
    """

    def __init__(self):
        self._scanner = MemoryScanner()
        self._attack_store = AttackStore()
        self._pdf_scanner = PdfScanner()
        self._ctx_monitor = ContextWindowMonitor()
        logger.info("[MemoryGuard] Online | scanner + attack_store + pdf_scanner + ctx_monitor")

    # ── Document scanning ─────────────────────────────────────────────────────

    def scan_document(
        self,
        content: str,
        doc_id: str = "doc",
        doc_type: str = "text",
        metadata: dict | None = None,
    ) -> ScanResult:
        """
        Scan a document for attack content.

        Parameters
        ----------
        content  : str  — Document text to scan
        doc_id   : str  — Identifier for audit log
        doc_type : str  — "text" (default) or "pdf" (auto-detected if content is bytes)
        metadata : dict — Optional metadata to inspect for injection

        Returns
        -------
        ScanResult with verdict: "CLEAN" / "SUSPICIOUS" / "POISONED"
        """
        result = self._scanner.scan(content, doc_id=doc_id, metadata=metadata)

        # If MALICIOUS/SUSPICIOUS: grow the attack knowledge base
        if result.verdict in ("SUSPICIOUS", "POISONED") and result.threats:
            top_category = result.threat_categories[0] if result.threat_categories else "unknown"
            self._attack_store.store_new_attack(
                text=content[:500],  # store first 500 chars as the attack signature
                category=top_category,
                metadata={
                    "doc_id": doc_id,
                    "verdict": result.verdict,
                    "confidence": str(result.confidence),
                    "patterns": ",".join(t.pattern_name for t in result.threats[:3]),
                },
            )

        return result

    def scan_pdf(self, file_path: str) -> tuple[PdfScanResult, ScanResult]:
        """
        Scan a PDF file — extracts ALL text including hidden content, then scans.

        Returns
        -------
        (PdfScanResult, ScanResult)
          PdfScanResult: what PyMuPDF found (visible text, hidden text, OCG layers)
          ScanResult:    security verdict on the combined text
        """
        pdf_result = self._pdf_scanner.scan_file(file_path)

        # Scan the FULL text (visible + hidden) — the hidden part is the attack surface
        scan_result = self.scan_document(
            content=pdf_result.all_text,
            doc_id=file_path,
            doc_type="pdf",
        )

        # If PDF had hidden content but no pattern matches, boost confidence
        if pdf_result.has_hidden_content and scan_result.confidence < 0.30:
            # Hidden text in a PDF is suspicious even without explicit injection patterns
            logger.warning(
                f"[MemoryGuard] PDF has hidden content but no injection patterns | "
                f"doc={file_path} | hidden_spans={pdf_result.hidden_span_count}"
            )

        return pdf_result, scan_result

    # ── Context window monitoring ─────────────────────────────────────────────

    def assess_context(
        self,
        system_prompt: str,
        document: str,
        model_limit: int = 200_000,
    ) -> ContextAssessment:
        """
        Check if processing this document would displace the system prompt.

        A context stuffing attack submits a huge document to push the system
        prompt far from the active context window position.
        """
        return self._ctx_monitor.assess(system_prompt, document, model_limit)

    # ── LangGraph node interface ──────────────────────────────────────────────

    def scan_for_graph(
        self,
        content: str,
        doc_id: str = "graph_doc",
        doc_type: str = "text",
    ) -> dict:
        """
        LangGraph-compatible interface.

        Returns a flat dict matching GuardianState's memory_guard_result field.
        """
        scan = self._scanner.scan(content, doc_id=doc_id)
        return {
            "guard":              "memory_guard",
            "verdict":            scan.verdict,
            "confidence":         scan.confidence,
            "threat_categories":  scan.threat_categories,
            "threats_found":      len(scan.threats),
            "obfuscation":        scan.obfuscation_methods,
            "metadata_threats":   scan.metadata_threats,
            "scan_note":          scan.scan_note,
            "doc_id":             doc_id,
        }


__all__ = [
    "MemoryGuard",
    "MemoryScanner",
    "AttackStore",
    "PdfScanner",
    "ContextWindowMonitor",
    "ScanResult",
    "DocumentThreat",
    "AttackSimilarityResult",
    "PdfScanResult",
    "ContextAssessment",
]
