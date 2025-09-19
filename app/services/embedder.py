from sentence_transformers import SentenceTransformer
from app.config import MODEL_NAME, EMBEDDER_DIM

class Embedder:
    """
    Singleton wrapper for SentenceTransformer embedding model.
    Provides encoding functionality for text inputs.
    """
    _instance = None

    def __new__(cls, model_name:str = MODEL_NAME):
        """
        Create or return the singleton instance of Embedder with the specified model.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = SentenceTransformer(model_name,)
        return cls._instance

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Encode a list of text strings into vector embeddings.
        Raises TypeError or ValueError for invalid input.
        Returns a list of embedding vectors.
        """
        if not isinstance(texts, list):
            raise TypeError("Input must be a list of strings.")
        if not texts:
            raise ValueError("Input texts list is empty.")
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("All items in texts must be strings.")

        # Returns list of lists
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()
