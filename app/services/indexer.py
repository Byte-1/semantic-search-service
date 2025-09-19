import faiss, threading
from app.config import EMBEDDER_DIM
from app.utils.helper import normalize_vectors, normalize_str
from collections import defaultdict
from typing import Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FaissIndexer:
    """
    Wrapper around FAISS for vector indexing and similarity search.
    Maintains metadata and inverted indices for source and author filtering.
    Thread-safe for concurrent access.
    """
    def __init__(self):
        """
        Initialize FAISS index, thread lock, and metadata/inverted indices.
        """
        self.index = faiss.IndexFlatIP(EMBEDDER_DIM)
        self.lock = threading.Lock()
        self.metadata_store = defaultdict(dict) # dictionary to map the doc with its vector_idx {vectoir_idx: doc}
        self.source_map = defaultdict(set) # inverted index map for source to vector_idx
        self.author_map = defaultdict(set) # inverted index map for author to vector_idx

    def add(self, embs, docs: list[dict[str, Any]]):
        """
        Add embeddings and associated documents to the index.
        Updates metadata and inverted indices for source/author.
        Returns number of failed additions.
        """
        failed_count = 0
        try:
            embs = np.array(embs).astype("float32")
            embs = normalize_vectors(embs)
            with self.lock:
                prev_ntotal = self.size()
                self.index.add(embs)

            for i, doc in enumerate(docs):
                # Assign a unique index to the document and transform inverted indices
                assigned_idx = prev_ntotal + i
                src_key = normalize_str(doc.get("source"))
                auth_key = normalize_str(doc.get("author"))

                # Store metadata and update inverted indices
                self.metadata_store[assigned_idx] = doc
                self.source_map[src_key].add(assigned_idx)
                self.author_map[auth_key].add(assigned_idx)
        except Exception as e:
            failed_count += 1
            logger.error(f"Error occurred while adding documents: {e}")
        return failed_count

    def search(self, q_emb, k: int):
        """
        Search for top-k most similar vectors to the query embedding.
        Returns similarity scores and vector indexes.
        """
        try:
            with self.lock:
                similarity_scores, vector_indexes = self.index.search(q_emb, k)
            return similarity_scores[0].tolist(), vector_indexes[0].tolist()
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return [], []

    def size(self):
        """
        Return the total number of vectors indexed.
        """
        return self.index.ntotal
