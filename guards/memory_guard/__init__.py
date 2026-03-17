"""
Memory Guard — Public API

Prevents RAG poisoning and memory injection attacks.

Two threats stopped here:
  1. Document-level injection: Attacker uploads a document containing
     hidden LLM instructions that execute when retrieved.
  2. Retrieval-time injection: A previously stored doc has since been
     modified (or a new poisoned doc was sneaked in) and gets retrieved.

Usage
-----
    from guards.memory_guard import MemoryGuard

    mg = MemoryGuard()

    # Ingest a new document
    add_result = mg.add_document(
        content="Our refund policy is 30 days...",
        doc_id="policy_001",
        metadata={"source": "internal_wiki", "author": "legal_team"},
    )
    # add_result.stored = True/False
    # add_result.verdict = "CLEAN" / "SUSPICIOUS" / "POISONED"

    # RAG retrieval (safe)
    retrieve_result = mg.retrieve("What is our refund policy?", k=5)
    # retrieve_result.chunks = [RetrievedChunk, ...]
    # Each chunk is pre-scanned; poisoned chunks are excluded

    # Scan a standalone document (no storage)
    scan_result = mg.scan_document("document text here", doc_id="scan_only")
    # scan_result.verdict = "CLEAN" / "SUSPICIOUS" / "POISONED"
"""
from .memory_scanner import MemoryScanner, ScanResult, DocumentThreat
from .rag_store import RagStore, AddResult, RetrieveResult, RetrievedChunk

import logging

logger = logging.getLogger(__name__)


class MemoryGuard:
    """
    The Memory Guard — Day 3 of the AI Agent Security Shield.

    Wraps MemoryScanner + RagStore into a single clean interface
    for the Guardian Commander to call.
    """

    def __init__(self):
        self._scanner  = MemoryScanner()
        self._rag      = RagStore()
        logger.info("[MemoryGuard] Online | scanner=ready | rag_store=ready")

    # ── Document lifecycle ────────────────────────────────────────────────────

    def add_document(
        self,
        content:  str,
        doc_id:   str | None  = None,
        metadata: dict | None = None,
    ) -> AddResult:
        """
        Ingest a document into the knowledge base.

        Internally: MemoryScanner → block/sanitize → RagStore.add_document()
        """
        return self._rag.add_document(content, doc_id=doc_id, metadata=metadata)

    def retrieve(self, query: str, k: int = 5) -> RetrieveResult:
        """
        Retrieve documents relevant to a query.

        Internally: ChromaDB.query() → MemoryScanner on each chunk → filter
        Returns only CLEAN or SUSPICIOUS (sanitized) chunks.
        """
        return self._rag.retrieve(query, k=k)

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document from the knowledge base."""
        return self._rag.delete_document(doc_id)

    # ── Standalone scan (no storage) ─────────────────────────────────────────

    def scan_document(
        self,
        content:  str,
        doc_id:   str = "scan_only",
        metadata: dict | None = None,
    ) -> ScanResult:
        """
        Scan a document without storing it.

        Use when you want to inspect a document but not ingest it
        (e.g. incoming email attachments, API responses, external feeds).
        """
        return self._scanner.scan(content, doc_id=doc_id, metadata=metadata)

    # ── LangGraph node interface ──────────────────────────────────────────────

    def scan_for_graph(self, content: str, doc_id: str = "graph_doc") -> dict:
        """
        LangGraph-compatible interface.

        Returns a flat dict matching GuardianState's memory_guard_result field.
        """
        scan = self._scanner.scan(content, doc_id=doc_id)
        return {
            "guard":            "memory_guard",
            "verdict":          scan.verdict,
            "confidence":       scan.confidence,
            "threat_categories": scan.threat_categories,
            "threats_found":    len(scan.threats),
            "obfuscation":      scan.obfuscation_methods,
            "metadata_threats": scan.metadata_threats,
            "scan_note":        scan.scan_note,
            "doc_id":           doc_id,
        }

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return knowledge base statistics."""
        return self._rag.stats()


__all__ = [
    "MemoryGuard",
    "MemoryScanner",
    "RagStore",
    "ScanResult",
    "DocumentThreat",
    "AddResult",
    "RetrieveResult",
    "RetrievedChunk",
]
