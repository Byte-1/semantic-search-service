from typing import List, Dict, Any, Optional
from app.config import MODEL_NAME, BATCH_SIZE
from app.utils.helper import normalize_str, normalize_vectors
from app.services.embedder import Embedder
from app.services.indexer import FaissIndexer
import numpy as np
import logging

logger = logging.getLogger(__name__)

class IngestionService:
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

    def ingest_in_batches(self, docs: List[Dict[str, Any]], batch_size: int = BATCH_SIZE):
        """
        Ingest multiple document batches to avoid memory or resource issues.
        Returns (total_docs, successful, failed)
        """
        total_docs = len(docs)
        successful = 0
        failed = 0
        batch_num = 0
        unique_docs = set() # To track unique document ids, so that in a same request if someone sends duplicate ids, we can ignore them.
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            _, batch_successful, batch_failed = self.ingest(batch, unique_docs)
            successful += batch_successful
            failed += batch_failed
            batch_num += 1
            logger.info(f"Ingestion Completed for batch {batch_num} , successful: {batch_successful}, failed: {batch_failed}")
        return (total_docs, successful, failed)

    def ingest(self, docs: List[Dict[str, Any]], unique_docs: set):
        """
        Ingest a subset of documents into the index.
        Returns a tuple: (total_docs, successful, failed)
        """
        try: 
            texts = []
            perfect_docs = []
            for d in docs:
                if d.get("id") in unique_docs:
                    logger.warning(f"Duplicate document id {d.get('id')} found in the same ingestion request. Ignoring this document.")
                else:
                    unique_docs.add(d.get("id"))
                    texts.append(d.get("text", ""))
                    perfect_docs.append(d)

            # Update docs to only include perfect unique documents
            docs = perfect_docs
            if texts:
                embs = self.embedder.encode(texts)
                failed_docs = self.index_repo.add(embs, docs)
                return (len(docs), len(docs)-failed_docs, failed_docs)
            return (len(docs), 0, len(docs))
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            return (len(docs), 0, len(docs))