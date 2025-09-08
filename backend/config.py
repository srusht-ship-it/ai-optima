from pathlib import Path
# Configuration file for the email classification system

# Model configurations
LIGHT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
HEAVY_MODEL = "bert-base-uncased"

# Default thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.85

# Database configuration
DB_PATH = "./data/email_classifier.db"

# Email categories
EMAIL_CATEGORIES = [
    "work",
    "personal", 
    "spam",
    "promotions",
    "support",
    "newsletter"
]

# Model configurations for the orchestrator
ALL_MODELS = [
    {
        "name": "DistilBERT Light",
        "type": "light",
        "model_path": LIGHT_MODEL
    },
    {
        "name": "BERT Heavy", 
        "type": "heavy",
        "model_path": HEAVY_MODEL
    },
    {
        "name": "Keyword Fallback",
        "type": "fallback",
        "model_path": None
    }
]

# Energy tracking settings
ENERGY_TRACKING_ENABLED = True
CO2_EMISSION_FACTOR = 0.0004  # kg CO2 per kWh (varies by region)

# API settings
MAX_EMAIL_LENGTH = 10000  # Maximum characters
REQUEST_TIMEOUT = 30  # seconds
# Categories
EMAIL_CATEGORIES = ["work", "spam", "promotions", "personal", "support", "newsletter"]
NUM_CLASSES = len(EMAIL_CATEGORIES)

# Energy & Performance Thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.85
ENERGY_SAVING_THRESHOLD = 0.30  # 30% energy savings required to switch
ACCURACY_DROP_TOLERANCE = 0.02  # Max 2% accuracy drop allowed

# Paths
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "email_classification.db"
MODELS_PATH = ROOT / "models"
EMAIL_SAMPLES_PATH = ROOT / "data" / "email_samples.txt"