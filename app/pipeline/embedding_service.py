"""
Shared embedding service for SentenceTransformer models.

This singleton service ensures that SentenceTransformer models are loaded only once
and shared across multiple components (SemanticReranker, ClaimMatcher).
"""
import logging
from typing import Optional, Any

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import]
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Singleton service for shared SentenceTransformer models.
    
    Prevents redundant model loading and reduces memory usage.
    """
    
    _instance: Optional["EmbeddingService"] = None
    _model: Optional[Any] = None
    _model_name: Optional[str] = None
    
    def __new__(cls):
        """Singleton pattern: return existing instance if available."""
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize embedding service (only called once due to singleton)."""
        # Avoid re-initialization
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._model = None
        self._model_name = None
    
    def get_model(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Get shared SentenceTransformer model instance.
        
        Args:
            model_name: HuggingFace model identifier (default: paraphrase-multilingual-MiniLM-L12-v2)
            
        Returns:
            SentenceTransformer model instance, or None if unavailable
        """
        if SentenceTransformer is None:
            logger.warning(
                "EmbeddingService: sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            return None
        
        # Return existing model if same name
        if self._model is not None and self._model_name == model_name:
            return self._model
        
        # Load new model
        try:
            logger.info(f"EmbeddingService: Loading model '{model_name}'")
            self._model = SentenceTransformer(model_name)
            self._model.eval()  # Set to evaluation mode
            self._model_name = model_name
            logger.info(f"EmbeddingService: Model '{model_name}' loaded successfully")
            return self._model
        except Exception as e:
            logger.warning(f"EmbeddingService: Failed to load model '{model_name}': {e}")
            self._model = None
            self._model_name = None
            return None
    
    def is_available(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model is not None


