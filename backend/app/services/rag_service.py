"""
ResumeIQ Vector Embeddings and In-Memory FAISS RAG Service.
Handles semantic chunking, dense embeddings, vector indexing, and grounded evidence retrieval.
"""

from typing import Dict, List, Optional
import numpy as np
import faiss
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.app.config import get_settings


# --- Data Models for RAG Pipeline ---
class ResumeChunk(BaseModel):
    """Represents an atomic, searchable textual unit of a resume with metadata."""
    chunk_id: int
    section: str = Field(description="Originating resume section, e.g. 'experience', 'skills'")
    text: str = Field(description="Textual content of the chunk")


class EvidenceResult(BaseModel):
    """Retrieved resume evidence for a specific job requirement."""
    chunk_id: int
    section: str
    text: str
    similarity: float = Field(description="Cosine similarity score between 0.0 and 1.0")


# --- Singleton Embedding Model Manager ---
class EmbeddingService:
    """
    Manages lazy-loading and inference for Sentence-Transformers embedding models.
    Cached as a singleton to avoid expensive re-initialization per request.
    """
    _instance: Optional["EmbeddingService"] = None
    _model: Optional[SentenceTransformer] = None

    def __new__(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model on first access."""
        if self._model is None:
            settings = get_settings()
            # BAAI/bge-small-en-v1.5 or all-MiniLM-L6-v2 are fast 384-dimensional models
            model_name = settings.EMBEDDING_MODEL_NAME or "BAAI/bge-small-en-v1.5"
            self._model = SentenceTransformer(model_name)
        return self._model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Compute dense vector embeddings for a list of strings.
        Returns a float32 2D NumPy array of shape (N, dimension).
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        # Generate embeddings (shape: [num_texts, embedding_dim])
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Ensures unit length for exact Cosine Similarity
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        Returns normalized 1D float32 array reshaped to (1, dimension).
        """
        embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)


# --- Semantic Document Chunking ---
def chunk_resume(sections: Dict[str, str], chunk_size: int = 350, chunk_overlap: int = 50) -> List[ResumeChunk]:
    """
    Break resume sections into semantic, searchable chunks using LangChain text splitters.
    Preserves list bullets and section contextual metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n- ", "\n• ", "\n", ". ", " "],
    )

    chunks: List[ResumeChunk] = []
    chunk_counter = 0

    for section_name, section_text in sections.items():
        if not section_text or not section_text.strip():
            continue

        raw_splits = splitter.split_text(section_text.strip())
        for split in raw_splits:
            cleaned_split = split.strip()
            # Discard trivial chunks (e.g. single character or empty bullets)
            if len(cleaned_split) >= 15:
                chunks.append(
                    ResumeChunk(
                        chunk_id=chunk_counter,
                        section=section_name,
                        text=cleaned_split,
                    )
                )
                chunk_counter += 1

    return chunks


# --- In-Memory FAISS Vector Index ---
class ResumeVectorStore:
    """
    Transient, in-memory FAISS vector index representing a single resume document.
    Enables sub-millisecond similarity search across resume bullet points.
    """

    def __init__(self, chunks: List[ResumeChunk]):
        self.chunks: List[ResumeChunk] = chunks
        self.chunk_map: Dict[int, ResumeChunk] = {c.chunk_id: c for c in chunks}
        self.embedding_service = EmbeddingService()
        self.index: Optional[faiss.IndexFlatIP] = None
        self.dimension: int = 384

        if chunks:
            self._build_index()

    def _build_index(self) -> None:
        """Vectorize all resume chunks and populate the FAISS Inner Product index."""
        texts = [c.text for c in self.chunks]
        vectors = self.embedding_service.embed_texts(texts)

        self.dimension = vectors.shape[1]
        # IndexFlatIP calculates Inner Product. Because vectors are L2-normalized,
        # Inner Product is mathematically identical to Cosine Similarity.
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)

    def search(self, query: str, top_k: int = 3, min_similarity: float = 0.20) -> List[EvidenceResult]:
        """
        Search for top-K resume chunks most semantically relevant to a query.

        Parameters:
        - query: Text string (e.g. a specific job requirement or keyword)
        - top_k: Maximum number of evidence chunks to retrieve
        - min_similarity: Minimum cosine similarity threshold (0.0 to 1.0)
        """
        if not self.index or self.index.ntotal == 0 or not query.strip():
            return []

        query_vec = self.embedding_service.embed_query(query.strip())
        k = min(top_k, self.index.ntotal)

        # faiss.search returns distances (cosine similarities) and indices
        similarities, indices = self.index.search(query_vec, k)

        results: List[EvidenceResult] = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1:
                continue
            sim_score = float(sim)
            # Clip between 0.0 and 1.0
            bounded_sim = max(0.0, min(1.0, sim_score))

            if bounded_sim >= min_similarity:
                chunk = self.chunk_map[idx]
                results.append(
                    EvidenceResult(
                        chunk_id=chunk.chunk_id,
                        section=chunk.section,
                        text=chunk.text,
                        similarity=round(bounded_sim, 4),
                    )
                )

        return results

    def batch_search(
        self, queries: List[str], top_k: int = 3, min_similarity: float = 0.20
    ) -> Dict[str, List[EvidenceResult]]:
        """
        Perform semantic search for multiple queries in a single pass.
        Returns a mapping of { query_text: [EvidenceResult, ...] }.
        """
        results_map = {}
        for query in queries:
            results_map[query] = self.search(query, top_k=top_k, min_similarity=min_similarity)
        return results_map


def build_rag_context(
    vector_store: ResumeVectorStore,
    requirements: List[str],
    top_k_per_req: int = 2,
) -> str:
    """
    Format retrieved resume evidence into a clean, structured context string
    specifically designed for the LLM prompt.
    """
    context_lines = ["### RETRIEVED RESUME EVIDENCE FOR JOB REQUIREMENTS:\n"]

    for idx, req in enumerate(requirements, start=1):
        evidence_list = vector_store.search(req, top_k=top_k_per_req)
        context_lines.append(f"Requirement {idx}: \"{req}\"")

        if evidence_list:
            context_lines.append("Supporting Evidence from Resume:")
            for ev in evidence_list:
                context_lines.append(
                    f"  - [{ev.section.upper()} | Similarity: {ev.similarity:.2f}]: {ev.text}"
                )
        else:
            context_lines.append("  - [NO MATCHING EVIDENCE FOUND IN RESUME]")

        context_lines.append("")  # Blank line separator

    return "\n".join(context_lines)
