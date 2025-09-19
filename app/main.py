import logging
from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
import uvicorn
from app.config import MODEL_NAME
from app.models.schemas import Document, IngestionResponse, SearchResponse, SearchResult
from app.services.search_service import SearchService
from app.services.ingestion_service import IngestionService
from app.services.indexer import FaissIndexer
from app.utils.helper import validate_ingestion_request, format_time
import time


# Centralized logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Initialize FastAPI app and SearchService
app = FastAPI(title="Semantic Search Service - Modular Prototype")
shared_index_repo = FaissIndexer()
search_service = SearchService(model_name=MODEL_NAME, index_repo=shared_index_repo)
ingestion_service = IngestionService(model_name=MODEL_NAME, index_repo=shared_index_repo)

@app.post("/ingest", response_model=IngestionResponse)
def ingest(docs: List[Document]):
    """
    Ingest a batch of documents into the semantic search index.
    Returns ingestion status and counts.
    """

    try:
        # Validate Ingestion Request
        validate_ingestion_request(docs)
        start_time = time.perf_counter()
        total_docs, successful, failed = ingestion_service.ingest_in_batches([d.dict() for d in docs])
        message = "Success" if failed == 0 else "Partial Success" if successful > 0 else "Failed"
        end_time = time.perf_counter()
        ingestion_time = end_time - start_time
        return {
            "message": message,
            "total_docs": total_docs,
            "ingestion_success": successful,
            "ingestion_failed": failed,
            "duplicate_ignored": total_docs - successful,
            "ingestion_time": format_time(ingestion_time)
        }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/search", response_model=SearchResponse)
def search(q: str, source: Optional[str] = None, author: Optional[str] = None, top_k: int = Query(default=5, ge=1, le=100, description="Number of results to return")):
    """
    Search indexed documents using semantic similarity.
    Supports optional filtering by source and author.
    Returns top-k results with scores.
    """
    if search_service.index_repo.size() == 0:
        raise HTTPException(status_code=400, detail="Index is empty")
    try:
        start_time = time.perf_counter()
        hits = search_service.search(q, top_k, source, author)
        end_time = time.perf_counter()
        search_time = end_time - start_time
        return {
            "query": q,
            "count": len(hits),
            "search_time": format_time(search_time),
            "results": [SearchResult(**h) for h in hits]
        }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/status")
def status():
    """
    Returns the number of indexed documents.
    """
    return {"indexed": search_service.index_repo.size()}

@app.get("/healthz")
def healthz():
    """
    Health check endpoint.
    Returns service status.
    """
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("semantic_search.main:app", host="0.0.0.0", port=8000, reload=True)
