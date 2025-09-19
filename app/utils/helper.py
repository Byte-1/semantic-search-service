import numpy as np
from datetime import datetime
from app.config import MAX_DOCS_PER_INGEST
from fastapi import HTTPException
import re

def normalize_vectors(v: np.ndarray) -> np.ndarray:
    """
    Normalize vectors to unit length for cosine similarity.
    """
    
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms

def validate_iso8601(date_str: str) -> bool:
    """
    Validate if date string is in ISO 8601 format.
    """

    try:
        _ = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return True
    except Exception:
        return False
    
def normalize_str(s: str) -> str:
    """
    Normalize string (here: author and source) for consistent indexing and lookup.
    """

    if not s:
        return s
    s = s.strip()                  
    s = re.sub(r"\s+", " ", s)     
    s = s.replace(" ", "_")    
    return s.lower()

def validate_ingestion_request(docs: list[dict]) -> bool:
    """
    Validate the ingestion request.
    """

    # Validate input documents
    if len(docs) > MAX_DOCS_PER_INGEST:
        raise HTTPException(status_code=413, detail=f"Too many documents. Limit is {MAX_DOCS_PER_INGEST} per request.")
    
    invalid_docs = [d for d in docs if not validate_iso8601(d.created_at)]
    if invalid_docs:
        raise HTTPException(status_code=400, detail=f"Invalid created_at for doc(s): {[d.id for d in invalid_docs]}")

def format_time(seconds: float) -> str:
    """
    Format time: if <1s, return ms; else return s (rounded to 3 decimals).
    """

    if seconds < 1.0:
        return f"{int(seconds * 1000)} ms"
    else:
        return f"{round(seconds, 3)} s"
