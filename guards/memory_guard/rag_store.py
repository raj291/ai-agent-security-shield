"""
Memory Guard — Safe RAG Store

A security-hardened ChromaDB wrapper for the agent's knowledge base.

Every operation goes through the Memory Scanner:
  add_document()   → scan BEFORE storage  (block/quarantine poisoned docs)
  retrieve()       → validate AFTER fetch (defense-in-depth)

Two ChromaDB collections are used across the whole project:
  ┌─────────────────────────────┬─────────────────────────────────────────┐
  │  Collection name            │  Purpose                                │
  ├─────────────────────────────┼─────────────────────────────────────────┤
  │  input_guard_verdicts       │  Layer 2 LLM classifier cache           │
  │  (Layer 2, LLMClassifier)   │  "Is this user message an injection?"  │
  ├─────────────────────────────┼─────────────────────────────────────────┤
  │  agent_knowledge_base       │  RAG document store (THIS FILE)         │
  │  (Memory Guard, RagStore)   │  "What documents should the agent see?" │
  └─────────────────────────────┴─────────────────────────────────────────┘

Same ChromaDB instance (PersistentClient at ./data/chroma_cache),
different collections — like two tables in the same database.

Public API
----------
    store = RagStore()

    # Add a document (runs MemoryScanner → stores only if CLEAN/SUSPICIOUS)
    result = store.add_document("document text", doc_id="doc_1", metadata={...})

    # Retrieve top-k similar docs for a query (validates each chunk)
    chunks = store.retrieve("what is our refund policy?", k=5)

    # Inspect what's in the store
    stats = store.stats()
"""
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from .memory_scanner import MemoryScanner, ScanResult

logger = logging.getLogger(__name__)

_CHROMA_PATH      = Path(__file__).parent.parent.parent / "data" / "chroma_cache"
_COLLECTION_NAME  = "agent_knowledge_base"


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AddResult:
    """Result from adding a document to the store."""
    doc_id:      str
    stored:      bool          # True if written to ChromaDB
    verdict:     str           # CLEAN / SUSPICIOUS / POISONED
    confidence:  float
    scan_result: ScanResult
    reason:      str           # Human-readable explanation


@dataclass
class RetrievedChunk:
    """A single document chunk returned from retrieval."""
    doc_id:     str
    content:    str
    distance:   float          # 0.0 = identical, 2.0 = opposite (cosine)
    metadata:   dict
    is_clean:   bool           # False if retrieval-time scan flagged it
    scan_result: Optional[ScanResult] = None


@dataclass
class RetrieveResult:
    """Result from a retrieve() call."""
    query:          str
    chunks:         list[RetrievedChunk]   = field(default_factory=list)
    blocked_chunks: int                    = 0   # Chunks excluded by scan
    total_fetched:  int                    = 0


# ─────────────────────────────────────────────────────────────────────────────
# RAG STORE
# ─────────────────────────────────────────────────────────────────────────────

class RagStore:
    """
    Security-hardened ChromaDB wrapper for the agent's knowledge base.

    All documents are scanned by MemoryScanner before storage.
    All retrieved chunks are re-validated before being returned.
    """

    def __init__(self, collection_name: str = _COLLECTION_NAME):
        self._scanner    = MemoryScanner()
        self._collection = self._init_chromadb(collection_name)
        self._collection_name = collection_name

        count = self._collection.count() if self._collection else 0
        logger.info(
            f"[RagStore] Ready | "
            f"collection={collection_name} | "
            f"documents={count} | "
            f"path={_CHROMA_PATH.name}"
        )

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init_chromadb(self, collection_name: str):
        """Open or create the ChromaDB collection."""
        try:
            _CHROMA_PATH.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
            collection = client.get_or_create_collection(
                name=collection_name,
                embedding_function=DefaultEmbeddingFunction(),
                metadata={"hnsw:space": "cosine"},
            )
            return collection
        except Exception as exc:
            logger.error(f"[RagStore] ChromaDB init failed: {exc}")
            return None

    # ── Public API ────────────────────────────────────────────────────────────

    def add_document(
        self,
        content:  str,
        doc_id:   Optional[str]  = None,
        metadata: Optional[dict] = None,
    ) -> AddResult:
        """
        Scan and store a document in the knowledge base.

        POISONED documents → rejected, not stored
        SUSPICIOUS documents → stored with a warning tag in metadata
        CLEAN documents → stored normally

        Parameters
        ----------
        content  : str   — Document text to store
        doc_id   : str   — Optional ID; auto-generated if not provided
        metadata : dict  — Document metadata (source, title, date, etc.)

        Returns
        -------
        AddResult with stored=True/False and full scan details
        """
        metadata = metadata or {}

        # Auto-generate doc_id from content hash if not provided
        if not doc_id:
            doc_id = "doc_" + hashlib.sha256(content.encode()).hexdigest()[:12]

        # ── Scan ──────────────────────────────────────────────────────────
        scan = self._scanner.scan(content, doc_id=doc_id, metadata=metadata)

        # ── POISONED: reject ───────────────────────────────────────────────
        if scan.verdict == "POISONED":
            logger.warning(
                f"[RagStore] BLOCKED doc={doc_id} | "
                f"conf={scan.confidence:.2f} | "
                f"categories={scan.threat_categories}"
            )
            return AddResult(
                doc_id=doc_id,
                stored=False,
                verdict="POISONED",
                confidence=scan.confidence,
                scan_result=scan,
                reason=(
                    f"Document blocked: RAG poisoning detected | "
                    f"categories={scan.threat_categories} | "
                    f"conf={scan.confidence:.0%}"
                ),
            )

        # ── SUSPICIOUS: store with warning tag ────────────────────────────
        store_content = content
        if scan.verdict == "SUSPICIOUS":
            # Tag the metadata so downstream components can see the warning
            metadata = {
                **metadata,
                "_security_verdict": "SUSPICIOUS",
                "_security_confidence": str(scan.confidence),
                "_security_threats": ",".join(
                    t.pattern_name for t in scan.threats[:5]
                ),
            }
            # Use sanitized content if available
            if scan.sanitized_content:
                store_content = scan.sanitized_content
            logger.warning(
                f"[RagStore] SUSPICIOUS doc stored with tag | doc={doc_id} | "
                f"conf={scan.confidence:.2f}"
            )

        # ── Store in ChromaDB ──────────────────────────────────────────────
        try:
            if self._collection is None:
                raise RuntimeError("ChromaDB collection unavailable")

            self._collection.upsert(
                ids=[doc_id],
                documents=[store_content],
                metadatas=[{**metadata, "_security_verdict": scan.verdict}],
            )

            logger.info(
                f"[RagStore] Stored doc={doc_id} | "
                f"verdict={scan.verdict} | "
                f"chars={len(store_content)}"
            )

            return AddResult(
                doc_id=doc_id,
                stored=True,
                verdict=scan.verdict,
                confidence=scan.confidence,
                scan_result=scan,
                reason=(
                    f"Document stored | verdict={scan.verdict} | "
                    f"conf={scan.confidence:.0%}"
                ),
            )

        except Exception as exc:
            logger.error(f"[RagStore] Store failed for doc={doc_id}: {exc}")
            return AddResult(
                doc_id=doc_id,
                stored=False,
                verdict=scan.verdict,
                confidence=scan.confidence,
                scan_result=scan,
                reason=f"Storage error: {exc}",
            )

    def retrieve(self, query: str, k: int = 5) -> RetrieveResult:
        """
        Retrieve the top-k most semantically similar documents.

        Each chunk is re-scanned after retrieval (defense in depth).
        Chunks with verdict POISONED are excluded from results.

        Parameters
        ----------
        query : str  — Natural language search query
        k     : int  — Max number of chunks to return

        Returns
        -------
        RetrieveResult with list of RetrievedChunk
        """
        result = RetrieveResult(query=query)

        if self._collection is None:
            logger.error("[RagStore] ChromaDB unavailable — returning empty results")
            return result

        doc_count = self._collection.count()
        if doc_count == 0:
            logger.debug("[RagStore] Knowledge base is empty — nothing to retrieve")
            return result

        # Clamp k to actual document count
        k_actual = min(k, doc_count)

        try:
            raw = self._collection.query(
                query_texts=[query],
                n_results=k_actual,
                include=["documents", "distances", "metadatas"],
            )
        except Exception as exc:
            logger.error(f"[RagStore] Query failed: {exc}")
            return result

        documents = raw.get("documents", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        ids       = raw.get("ids", [[]])[0]

        result.total_fetched = len(documents)

        for doc_id, content, distance, meta in zip(ids, documents, distances, metadatas):
            # Retrieval-time scan (defense in depth)
            scan = self._scanner.scan(content, doc_id=f"{doc_id}:retrieved")

            if scan.verdict == "POISONED":
                result.blocked_chunks += 1
                logger.warning(
                    f"[RagStore] Blocked retrieved chunk | "
                    f"doc={doc_id} | conf={scan.confidence:.2f}"
                )
                continue

            chunk = RetrievedChunk(
                doc_id=doc_id,
                content=scan.sanitized_content if scan.verdict == "SUSPICIOUS" else content,
                distance=distance,
                metadata=meta,
                is_clean=(scan.verdict == "CLEAN"),
                scan_result=scan if scan.verdict != "CLEAN" else None,
            )
            result.chunks.append(chunk)

        logger.info(
            f"[RagStore] Retrieved {len(result.chunks)}/{result.total_fetched} chunks | "
            f"blocked={result.blocked_chunks} | query='{query[:60]}'"
        )
        return result

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document from the knowledge base."""
        if self._collection is None:
            return False
        try:
            self._collection.delete(ids=[doc_id])
            logger.info(f"[RagStore] Deleted doc={doc_id}")
            return True
        except Exception as exc:
            logger.error(f"[RagStore] Delete failed for doc={doc_id}: {exc}")
            return False

    def stats(self) -> dict:
        """Return basic statistics about the knowledge base."""
        if self._collection is None:
            return {"collection": self._collection_name, "documents": 0, "available": False}
        return {
            "collection":  self._collection_name,
            "documents":   self._collection.count(),
            "chroma_path": str(_CHROMA_PATH),
            "available":   True,
        }
