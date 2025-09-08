from typing import Optional, Dict, Any, List
import logging
from .config import LIGHT_MODEL, HEAVY_MODEL

logger = logging.getLogger(__name__)

class ModelWrapper:
    """Wrapper for a machine learning model with pipeline interface"""
    
    def __init__(self, model_name: str, model_type: str):
        self.model_name = model_name
        self.model_type = model_type
        self.pipeline = None
        self.is_initialized = False
        
    def initialize(self):
        """Initialize the model pipeline"""
        try:
            # Try to import transformers for real models
            from transformers import pipeline
            
            # Initialize the pipeline based on model type
            if self.model_type == "light":
                self.pipeline = pipeline(
                    "text-classification",
                    model=self.model_name,
                    device=-1  # Use CPU
                )
            elif self.model_type == "heavy":
                self.pipeline = pipeline(
                    "text-classification", 
                    model=self.model_name,
                    device=-1  # Use CPU
                )
            
            self.is_initialized = True
            logger.info(f"✅ Initialized {self.model_type} model: {self.model_name}")
            
        except ImportError:
            logger.warning(f"⚠ Transformers not available for {self.model_name}")
            self.pipeline = MockPipeline(self.model_name, self.model_type)
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.model_name}: {e}")
            self.pipeline = MockPipeline(self.model_name, self.model_type)
            self.is_initialized = True

class MockPipeline:
    """Mock pipeline for when real models aren't available"""
    
    def __init__(self, model_name: str, model_type: str):
        self.model_name = model_name
        self.model_type = model_type
        
    def __call__(self, text: str, return_all_scores: bool = False) -> List[Dict[str, Any]]:
        """Mock classification that returns reasonable fake results"""
        
        # Simple keyword-based classification for mock
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['spam', 'scam', 'winner', 'congratulations']):
            primary = {"label": "spam", "score": 0.85}
            secondary = {"label": "personal", "score": 0.15}
        elif any(word in text_lower for word in ['work', 'meeting', 'project', 'deadline']):
            primary = {"label": "work", "score": 0.80}
            secondary = {"label": "personal", "score": 0.20}
        elif any(word in text_lower for word in ['sale', 'offer', 'discount', 'deal']):
            primary = {"label": "promotions", "score": 0.75}
            secondary = {"label": "personal", "score": 0.25}
        elif any(word in text_lower for word in ['support', 'help', 'issue', 'problem']):
            primary = {"label": "support", "score": 0.70}
            secondary = {"label": "personal", "score": 0.30}
        else:
            primary = {"label": "personal", "score": 0.65}
            secondary = {"label": "work", "score": 0.35}
        
        # Adjust confidence based on model type
        if self.model_type == "heavy":
            # Heavy model should be slightly more confident
            primary["score"] = min(0.95, primary["score"] + 0.05)
            secondary["score"] = 1.0 - primary["score"]
        
        if return_all_scores:
            return [primary, secondary]
        else:
            return [primary]

class ModelManager:
    """Manages multiple models for the email classification system"""
    
    def __init__(self):
        self.light_model: Optional[ModelWrapper] = None
        self.heavy_model: Optional[ModelWrapper] = None
        self.models_initialized = False
        
    def initialize_models(self):
        """Initialize all models"""
        logger.info("🔄 Initializing models...")
        
        # Initialize light model
        self.light_model = ModelWrapper(LIGHT_MODEL, "light")
        self.light_model.initialize()
        
        # Initialize heavy model  
        self.heavy_model = ModelWrapper(HEAVY_MODEL, "heavy")
        self.heavy_model.initialize()
        
        self.models_initialized = True
        logger.info("✅ All models initialized")
        
    def get_model(self, model_type: str) -> Optional[ModelWrapper]:
        """Get a specific model by type"""
        if not self.models_initialized:
            self.initialize_models()
            
        if model_type == "light":
            return self.light_model
        elif model_type == "heavy":
            return self.heavy_model
        else:
            logger.warning(f"⚠ Unknown model type: {model_type}")
            return None
            
    def is_ready(self) -> bool:
        """Check if all models are ready"""
        return (self.models_initialized and 
                self.light_model is not None and 
                self.heavy_model is not None and
                self.light_model.is_initialized and 
                self.heavy_model.is_initialized)