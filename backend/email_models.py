import os
import logging
from typing import Dict, Any
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from .config import LIGHT_MODEL, HEAVY_MODEL, EMAIL_CATEGORIES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailModel:
    def _init_(self, model_name: str, model_type: str):
        self.model_name = model_name
        self.model_type = model_type
        self.pipeline = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the model pipeline"""
        try:
            logger.info(f"Loading {self.model_type} model: {self.model_name}")
            
            # For now, we'll use a general classification pipeline
            # In production, you'd want to use fine-tuned models for email classification
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                return_all_scores=True,
                device=-1  # Use CPU
            )
            
            # Note: You'll need to fine-tune these models on email data
            # For now, we'll use a mapping function to convert outputs
            self.is_loaded = True
            logger.info(f"✅ {self.model_type} model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {self.model_type} model: {e}")
            self.is_loaded = False
            raise
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Make prediction on email text"""
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.model_name} is not loaded")
            
        try:
            # Get predictions from the pipeline
            raw_predictions = self.pipeline(text)
            
            # Map predictions to email categories
            # This is a simplified mapping - you'd want to fine-tune the models
            mapped_predictions = self._map_to_email_categories(raw_predictions)
            
            return mapped_predictions
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def _map_to_email_categories(self, raw_predictions):
        """Map model outputs to email categories"""
        # This is a simplified mapping function
        # In production, you'd fine-tune the models on email-specific data
        
        # For demo purposes, we'll create a simple mapping
        email_predictions = []
        
        for i, category in enumerate(EMAIL_CATEGORIES):
            # Simple scoring based on text patterns
            # You would replace this with actual model predictions
            score = 1.0 / len(EMAIL_CATEGORIES)  # Equal probability initially
            
            if i < len(raw_predictions):
                # Use some of the original prediction confidence
                score = raw_predictions[i]['score'] * 0.7 + score * 0.3
            
            email_predictions.append({
                'label': category,
                'score': score
            })
        
        # Normalize scores
        total_score = sum(p['score'] for p in email_predictions)
        for pred in email_predictions:
            pred['score'] = pred['score'] / total_score
        
        return sorted(email_predictions, key=lambda x: x['score'], reverse=True)

class ModelManager:
    def _init_(self):
        self.light_model = EmailModel(LIGHT_MODEL, "light")
        self.heavy_model = EmailModel(HEAVY_MODEL, "heavy")
        self.models_initialized = False
        
    def initialize_models(self):
        """Initialize both light and heavy models"""
        try:
            logger.info("Initializing email classification models...")
            
            # Try to load light model first
            try:
                self.light_model.load_model()
            except Exception as e:
                logger.warning(f"Light model failed to load: {e}")
                self.light_model = self._create_fallback_model("light")
            
            # Try to load heavy model
            try:
                self.heavy_model.load_model()
            except Exception as e:
                logger.warning(f"Heavy model failed to load: {e}")
                self.heavy_model = self._create_fallback_model("heavy")
            
            self.models_initialized = True
            logger.info("✅ Model initialization completed")
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            # Create fallback models
            self.light_model = self._create_fallback_model("light")
            self.heavy_model = self._create_fallback_model("heavy")
            self.models_initialized = True
    
    def _create_fallback_model(self, model_type: str):
        """Create a simple fallback model"""
        
        class FallbackModel:
            def _init_(self, model_type):
                self.model_type = model_type
                self.is_loaded = True
                self.pipeline = self
            
            def _call_(self, text, return_all_scores=True):
                """Simple keyword-based classification"""
                text_lower = text.lower()
                predictions = []
                
                # Simple keyword matching
                scores = {
                    "spam": 0.1,
                    "work": 0.1,
                    "promotions": 0.1,
                    "personal": 0.1,
                    "support": 0.1,
                    "newsletter": 0.1
                }
                
                # Keyword patterns
                if any(word in text_lower for word in ['winner', 'congratulations', 'claim', '$$$', 'urgent']):
                    scores["spam"] = 0.8
                elif any(word in text_lower for word in ['meeting', 'report', 'project', 'deadline']):
                    scores["work"] = 0.7
                elif any(word in text_lower for word in ['sale', 'discount', 'offer', '% off']):
                    scores["promotions"] = 0.75
                elif any(word in text_lower for word in ['help', 'support', 'problem', 'issue']):
                    scores["support"] = 0.7
                elif any(word in text_lower for word in ['newsletter', 'weekly', 'update']):
                    scores["newsletter"] = 0.6
                else:
                    scores["personal"] = 0.6
                
                # Normalize scores
                total = sum(scores.values())
                for category in scores:
                    scores[category] = scores[category] / total
                
                # Convert to expected format
                for category in EMAIL_CATEGORIES:
                    predictions.append({
                        'label': category,
                        'score': scores.get(category, 0.1)
                    })
                
                return sorted(predictions, key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Created fallback model for {model_type}")
        return FallbackModel(model_type)
    
    def get_model_info(self):
        """Get information about loaded models"""
        return {
            "light_model": {
                "name": self.light_model.model_name if hasattr(self.light_model, 'model_name') else "fallback",
                "loaded": self.light_model.is_loaded,
                "type": "light"
            },
            "heavy_model": {
                "name": self.heavy_model.model_name if hasattr(self.heavy_model, 'model_name') else "fallback",
                "loaded": self.heavy_model.is_loaded,
                "type": "heavy"
            },
            "initialized": self.models_initialized
        }