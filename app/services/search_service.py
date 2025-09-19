from typing import List, Dict, Any, Optional
from app.config import DEFAULT_TOP_K, OVERFETCH_MULTIPLIER, MAX_OVERFETCH, MODEL_NAME, BATCH_SIZE, SEMANTIC_SEARCH_THRESHOLD
from app.utils.helper import normalize_str, normalize_vectors
from app.services.embedder import Embedder
from app.services.indexer import FaissIndexer
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SearchService:
    """
    Service for document ingestion and semantic search using embeddings and FAISS indexer.
    Handles filtering by source and author, and manages embedding model.
    """
    def __init__(self, model_name: str= MODEL_NAME, index_repo: FaissIndexer = None):
        """
        Initialize the SearchService with the specified embedding model and FAISS indexer.
        """
        self.embedder = Embedder(model_name)
        self.index_repo = index_repo if index_repo is not None else FaissIndexer()

    def search(self, query: str, top_k: int = DEFAULT_TOP_K, source: Optional[str] = None, author: Optional[str] = None):
        """
        Perform semantic search for the query string.
        Optionally filter by source and/or author.
        Returns a list of matching document dicts with scores.
        """
        try:
            # Encode the query to get its embedding and normalize it
            q_emb = np.array(self.embedder.encode([query])).astype("float32")
            q_emb = normalize_vectors(q_emb)
            fetch_k = min(MAX_OVERFETCH, top_k * OVERFETCH_MULTIPLIER)

            # Perform the search in the FAISS index
            scores, ids = self.index_repo.search(q_emb, fetch_k)
            allowed_set = None
            source_key = normalize_str(source) if source else None
            author_key = normalize_str(author) if author else None

            # Perform filtering based on source and author if provided
            if source_key and author_key:
                source_set = self.index_repo.source_map.get(source_key, set())
                author_set = self.index_repo.author_map.get(author_key, set())
                allowed_set = source_set & author_set
                logger.debug(f"Source set: {source_key}", source_set)
                logger.debug(f"Author set: {author_key}", author_set)
            elif source_key:
                allowed_set = self.index_repo.source_map.get(source_key, set())
                logger.debug(f"Source set: {source_key}", allowed_set)
            elif author_key:
                allowed_set = self.index_repo.author_map.get(author_key, set())
                logger.debug(f"Author set: {author_key}", allowed_set)

            if (source or author) and not allowed_set:
                logger.info("No documents match the given source/author filter.")
                return [] # filter was there but nothing matched
            
            # List to store final filtered responses
            hits = []

            # Debug logs to check filtering response
            logger.debug(f"Allowed set: {allowed_set}")
            logger.debug(f"Resultant Ids: {ids}")

            for idx, score in zip(ids, scores):
                if (allowed_set and idx not in allowed_set) or (score < SEMANTIC_SEARCH_THRESHOLD):
                    continue
                meta = self.index_repo.metadata_store[idx] if idx != -1 else None
                if meta:
                    logger.debug(f"Metadata for score:{score}", meta)
                    hits.append({**meta, "score": float(score)})
                if len(hits) >= top_k:
                    break
            return hits
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
