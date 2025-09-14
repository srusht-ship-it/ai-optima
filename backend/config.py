# config.py - Fixed version
import os
from pathlib import Path

# Use proper models for email classification
# Light model: Fast sentiment model that can be adapted
LIGHT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Heavy model: Use zero-shot classification which works out of the box
HEAVY_MODEL = "facebook/bart-large-mnli"

# Alternative models that work better for classification
WORKING_MODELS = {
    "light_sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "zero_shot": "facebook/bart-large-mnli",
    "light_classification": "distilbert-base-uncased-finetuned-sst-2-english",
}

# Use working models
LIGHT_MODEL = WORKING_MODELS["light_sentiment"] 
HEAVY_MODEL = WORKING_MODELS["zero_shot"]

# Email categories with natural language descriptions for zero-shot
EMAIL_CATEGORIES_NATURAL = {
    "work": "work related business professional email",
    "personal": "personal private friendly email", 
    "spam": "spam junk unwanted promotional email",
    "promotions": "marketing sales promotional discount offer email",
    "support": "customer support help technical assistance email",
    "newsletter": "newsletter subscription informational update email"
}

EMAIL_CATEGORIES = ["work", "personal", "spam", "promotions", "support", "newsletter"]

# Configuration
DEFAULT_CONFIDENCE_THRESHOLD = 0.85
DB_PATH = "./data/email_classifier.db"

# Model list for testing multiple models
ALL_MODELS = [
    {"name": "light_sentiment", "type": "light"},
    {"name": "heavy_zero_shot", "type": "heavy"}, 
    {"name": "fallback", "type": "fallback"}
]

# Create data directory
Path("./data").mkdir(exist_ok=True)