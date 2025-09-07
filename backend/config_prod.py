import os
from pathlib import Path

# Production Models (you might want to use optimized versions)
LIGHT_MODEL = os.getenv("LIGHT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
HEAVY_MODEL = os.getenv("HEAVY_MODEL", "textattack/bert-base-uncased-SST-2")

# Performance settings for production
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.87"))
ENERGY_SAVING_THRESHOLD = float(os.getenv("ENERGY_SAVING_THRESHOLD", "0.25"))
MAX_EMAIL_LENGTH = int(os.getenv("MAX_EMAIL_LENGTH", "10000"))

# Database settings
DB_PATH = Path(os.getenv("DB_PATH", "/app/data/green_metrics.sqlite3"))
MODELS_CACHE_PATH = Path(os.getenv("MODELS_CACHE", "/app/models"))

# Redis for caching (production)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_EXPIRY = int(os.getenv("CACHE_EXPIRY", "3600"))  # 1 hour

# Monitoring
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"

# Categories
EMAIL_CATEGORIES = [
    "work", "spam", "promotions", 
    "personal", "support", "newsletter"
]