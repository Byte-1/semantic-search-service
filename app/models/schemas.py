from pydantic import BaseModel, Field

class Document(BaseModel):
    """
    Document Schema For Ingestion
    """
    id: str = Field(..., description="UUID string")
    source: str = Field(..., description="Source of the document (e.g., confluence, jira, git readme)")
    author: str = Field(..., description="Author of the document")
    text: str = Field(..., description="Document Data")
    created_at: str = Field(..., description="Document Creation Timestamp")

class IngestionResponse(BaseModel):
    """
    Response schema for ingestion endpoint.
    """
    message: str = Field(..., description="Status message for ingestion")
    total_docs: int = Field(..., description="Total number of documents received")
    ingestion_success: int = Field(..., description="Number of documents successfully ingested")
    ingestion_failed: int = Field(..., description="Number of documents failed to ingest")
    duplicate_ignored: int = Field(..., description="Number of duplicate documents ignored")
    ingestion_time: str = Field(..., description="Time taken for ingestion")


class SearchResult(BaseModel):
    """
    Individual search result schema.
    """
    id: str = Field(..., description="UUID string")
    source: str = Field(..., description="Source of the document")
    author: str = Field(..., description="Author of the document")
    text: str = Field(..., description="Text content of the document")
    created_at: str = Field(..., description="ISO8601 UTC timestamp when the document was created")
    score: float = Field(..., description="Semantic similarity score")


class SearchResponse(BaseModel):
    """
    Response schema for search endpoint.
    """
    query: str = Field(..., description="Search query string")
    count: int = Field(..., description="Number of results returned")
    search_time: str = Field(..., description="Search execution time")
    results: list[SearchResult] = Field(..., description="List of search results")