MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDER_DIM = 384
DEFAULT_TOP_K = 10
OVERFETCH_MULTIPLIER = 3
MAX_OVERFETCH = 200
MAX_DOCS_PER_INGEST = 10 # For Testing Purposes, set to 5 in production we can set it to 1000 or more based on the server capacity.
BATCH_SIZE = 5 # Same for testing set to more, based on the requirements and server capacity.
SEMANTIC_SEARCH_THRESHOLD = 0.4 # 1.0 to 0.7 is strict, 0.6 to 0.4 is average, below 0.4 is too low not to be considered.
