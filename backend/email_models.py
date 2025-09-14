# email_models.py - Fixed version with proper model handling
from typing import Optional, Dict, Any, List
import logging
from .config import LIGHT_MODEL, HEAVY_MODEL, EMAIL_CATEGORIES_NATURAL, EMAIL_CATEGORIES

logger = logging.getLogger(__name__)

class ModelWrapper:
    """Wrapper for a machine learning model with pipeline interface"""
    
    def __init__(self, model_name: str, model_type: str):
        self.model_name = model_name
        self.model_type = model_type
        self.pipeline = None
        self.is_initialized = False
        self.error_message = None
        
    def initialize(self):
        """Initialize the model pipeline with better error handling"""
        try:
            # Check if transformers is available
            try:
                from transformers import pipeline
                logger.info(f"🔄 Initializing {self.model_type} model: {self.model_name}")
            except ImportError as e:
                self.error_message = f"Transformers library not installed: {e}"
                logger.error(f"❌ {self.error_message}")
                self._fallback_to_mock()
                return
            
            # Initialize based on specific model
            if "bart-large-mnli" in self.model_name:
                # Zero-shot classification model
                self.pipeline = pipeline(
                    "zero-shot-classification",
                    model=self.model_name,
                    device=-1  # Use CPU
                )
                logger.info(f"✅ Initialized zero-shot model: {self.model_name}")
                
            elif "roberta-base-sentiment" in self.model_name:
                # Sentiment analysis model
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    device=-1
                )
                logger.info(f"✅ Initialized sentiment model: {self.model_name}")
                
            elif "distilbert" in self.model_name and "sst-2" in self.model_name:
                # Text classification model
                self.pipeline = pipeline(
                    "text-classification",
                    model=self.model_name,
                    device=-1
                )
                logger.info(f"✅ Initialized classification model: {self.model_name}")
                
            else:
                # Try text-classification as default
                try:
                    self.pipeline = pipeline(
                        "text-classification",
                        model=self.model_name,
                        device=-1
                    )
                    logger.info(f"✅ Initialized as text-classification: {self.model_name}")
                except Exception:
                    # Fallback to sentiment-analysis
                    self.pipeline = pipeline(
                        "sentiment-analysis", 
                        model=self.model_name,
                        device=-1
                    )
                    logger.info(f"✅ Initialized as sentiment-analysis: {self.model_name}")
            
            self.is_initialized = True
            
        except Exception as e:
            self.error_message = f"Failed to initialize model {self.model_name}: {str(e)}"
            logger.error(f"❌ {self.error_message}")
            logger.error(f"Full traceback: ", exc_info=True)
            self._fallback_to_mock()
    
    def _fallback_to_mock(self):
        """Fallback to mock pipeline"""
        logger.warning(f"⚠ Using mock pipeline for {self.model_name}")
        self.pipeline = MockPipeline(self.model_name, self.model_type)
        self.is_initialized = True

    def classify_email(self, text: str) -> List[Dict[str, Any]]:
        """Classify email text and return predictions in standard format"""
        if not self.is_initialized:
            return [{"label": "personal", "score": 0.5}]
        
        try:
            # Handle different pipeline types
            if "bart-large-mnli" in self.model_name:
                # Zero-shot classification
                candidate_labels = list(EMAIL_CATEGORIES_NATURAL.values())
                result = self.pipeline(text, candidate_labels=candidate_labels)
                
                # Convert zero-shot result to standard format
                predictions = []
                for label, score in zip(result['labels'], result['scores']):
                    # Map back to category keys
                    category_key = None
                    for key, natural_desc in EMAIL_CATEGORIES_NATURAL.items():
                        if natural_desc == label:
                            category_key = key
                            break
                    
                    predictions.append({
                        "label": category_key or label,
                        "score": float(score)
                    })
                return predictions
                
            elif "roberta-base-sentiment" in self.model_name:
                # Sentiment model - adapt for email classification
                result = self.pipeline(text)
                
                # Map sentiment to email categories (simple heuristic)
                sentiment_label = result[0]['label'].upper()
                sentiment_score = result[0]['score']
                
                if sentiment_label == 'NEGATIVE':
                    # Negative sentiment could be spam or support
                    if any(word in text.lower() for word in ['help', 'problem', 'issue', 'support']):
                        category = "support"
                    else:
                        category = "spam"
                elif sentiment_label == 'POSITIVE':
                    # Positive sentiment could be work or promotions
                    if any(word in text.lower() for word in ['sale', 'offer', 'discount', 'deal']):
                        category = "promotions"
                    else:
                        category = "work"
                else:
                    category = "personal"
                    
                return [{"label": category, "score": sentiment_score}]
                
            else:
                # Regular text classification
                result = self.pipeline(text, return_all_scores=True)
                return result if isinstance(result, list) else [result]
                
        except Exception as e:
            logger.error(f"❌ Classification failed for {self.model_name}: {e}")
            return [{"label": "personal", "score": 0.5}]


class MockPipeline:
    """Enhanced mock pipeline that returns better fake results"""
    
    def __init__(self, model_name: str, model_type: str):
        self.model_name = model_name
        self.model_type = model_type
        logger.info(f"🔄 Created mock pipeline for {model_name}")
        
    def __call__(self, text: str, candidate_labels: List[str] = None, return_all_scores: bool = False) -> Any:
        """Mock classification that returns reasonable fake results"""
        
        # Simple keyword-based classification for mock
        text_lower = text.lower()
        
        # Map keywords to categories with confidence
        if any(word in text_lower for word in ['spam', 'scam', 'winner', 'congratulations', 'click now', '$$$', 'urgent']):
            primary = {"label": "spam", "score": 0.85}
        elif any(word in text_lower for word in ['work', 'meeting', 'project', 'deadline', 'team', 'report', 'manager']):
            primary = {"label": "work", "score": 0.80}
        elif any(word in text_lower for word in ['sale', 'offer', 'discount', 'deal', '% off', 'limited time', 'shop now']):
            primary = {"label": "promotions", "score": 0.75}
        elif any(word in text_lower for word in ['support', 'help', 'issue', 'problem', 'account', 'reset', 'assistance']):
            primary = {"label": "support", "score": 0.70}
        elif any(word in text_lower for word in ['newsletter', 'weekly', 'news', 'update', 'subscribe', 'unsubscribe']):
            primary = {"label": "newsletter", "score": 0.65}
        else:
            primary = {"label": "personal", "score": 0.60}
        
        # Adjust confidence based on model type
        if self.model_type == "heavy":
            primary["score"] = min(0.95, primary["score"] + 0.05)
        
        # Handle zero-shot classification format
        if candidate_labels:
            scores = []
            labels = []
            for label in candidate_labels:
                # Find matching category from natural language description
                matching_category = None
                for cat, desc in EMAIL_CATEGORIES_NATURAL.items():
                    if desc == label:
                        matching_category = cat
                        break
                
                if matching_category == primary["label"]:
                    scores.append(primary["score"])
                else:
                    scores.append(max(0.05, (1.0 - primary["score"]) / len(candidate_labels)))
                labels.append(label)
            
            return {
                "labels": labels,
                "scores": scores
            }
        
        # Regular classification format
        if return_all_scores:
            results = [primary]
            remaining_categories = [cat for cat in EMAIL_CATEGORIES if cat != primary["label"]]
            remaining_prob = 1.0 - primary["score"]
            
            for i, cat in enumerate(remaining_categories[:4]):  # Top 4 alternatives
                score = remaining_prob * (0.5 ** (i + 1))
                results.append({"label": cat, "score": score})
            
            return results
        else:
            return [primary]


class ModelManager:
    """Manages multiple models for the email classification system"""
    
    def __init__(self):
        self.light_model: Optional[ModelWrapper] = None
        self.heavy_model: Optional[ModelWrapper] = None
        self.models_initialized = False
        
    def initialize_models(self):
        """Initialize all models with better error handling"""
        logger.info("🔄 Initializing models...")
        
        try:
            # Initialize light model (sentiment model)
            self.light_model = ModelWrapper(LIGHT_MODEL, "light")
            self.light_model.initialize()
            
            # Initialize heavy model (zero-shot model)
            self.heavy_model = ModelWrapper(HEAVY_MODEL, "heavy")
            self.heavy_model.initialize()
            
            self.models_initialized = True
            
            # Log initialization status
            light_status = "✅ Ready" if self.light_model.is_initialized else f"❌ Failed: {self.light_model.error_message}"
            heavy_status = "✅ Ready" if self.heavy_model.is_initialized else f"❌ Failed: {self.heavy_model.error_message}"
            
            logger.info(f"Light model ({LIGHT_MODEL}): {light_status}")
            logger.info(f"Heavy model ({HEAVY_MODEL}): {heavy_status}")
            logger.info("✅ Model initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Critical error during model initialization: {e}")
            self.models_initialized = False
        
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
        """Check if at least one model is ready"""
        return (self.models_initialized and 
                ((self.light_model and self.light_model.is_initialized) or
                 (self.heavy_model and self.heavy_model.is_initialized)))
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status of all models"""
        status = {
            "models_initialized": self.models_initialized,
            "light_model": {
                "ready": self.light_model.is_initialized if self.light_model else False,
                "error": self.light_model.error_message if self.light_model else "Not created",
                "name": LIGHT_MODEL
            },
            "heavy_model": {
                "ready": self.heavy_model.is_initialized if self.heavy_model else False,
                "error": self.heavy_model.error_message if self.heavy_model else "Not created", 
                "name": HEAVY_MODEL
            }
        }
        return status