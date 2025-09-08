from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import time
import sqlite3
import hashlib
from datetime import datetime
import os
import traceback
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enhanced fallback classifier (moved before imports)
def simple_classification(text: str, subject: str = "", sender: str = "") -> Dict[str, Any]:
    """Enhanced keyword-based classification as fallback"""
    logger.info(f"Running simple_classification for text length: {len(text)}")

    text_lower = (text + " " + subject).lower()

    if any(word in text_lower for word in ['congratulations', 'winner', 'claim', '$$$', 'urgent', 'click now']):
        category, confidence = "spam", 0.85
    elif any(word in text_lower for word in ['meeting', 'report', 'project', 'deadline', 'team']):
        category, confidence = "work", 0.75
    elif any(word in text_lower for word in ['offer', 'sale', 'discount', '% off', 'limited time']):
        category, confidence = "promotions", 0.8
    elif any(word in text_lower for word in ['help', 'support', 'problem', 'issue', 'account']):
        category, confidence = "support", 0.7
    elif any(word in text_lower for word in ['newsletter', 'weekly', 'news', 'update']):
        category, confidence = "newsletter", 0.65
    else:
        category, confidence = "personal", 0.6

    all_predictions = [
        {"category": category, "confidence": confidence},
        {"category": "personal", "confidence": max(0.1, 1.0 - confidence)},
    ]

    result = {
        "email_text": text,
        "predicted_category": category,
        "confidence": confidence,
        "all_predictions": all_predictions,
        "model_used": "keyword_fallback_v2",
        "model_type": "fallback",
        "escalated": False,
        "energy_metrics": {
            "co2_emissions_g": 0.001,
            "co2_emissions_kg": 0.000001,
            "processing_time_seconds": 0.05,
            "memory_used_gb": 0.001,
            "cpu_utilization_start": 5.0,
            "cpu_utilization_end": 6.0,
            "gpu_metrics": {},
            "energy_efficiency_score": 9.8
        },
        "ai_insights": {
            "environmental_impact": {
                "co2_this_classification": 0.001,
                "yearly_projection_g": 18.25,
                "equivalent_km_driven": 0.045,
                "impact_level": "minimal"
            },
            "accuracy_assessment": {
                "confidence_level": "medium" if confidence > 0.7 else "low",
                "accuracy_assessment": "Enhanced keyword matching",
                "should_review": confidence < 0.7
            },
            "suggestions": [
                "⚡ Ultra-low energy classification used",
                "🔄 AI models loading - accuracy will improve soon" if not ORCHESTRATOR_READY else "✅ System running optimally"
            ]
        },
        "timestamp": time.time(),
    }
    return result

# Imports with fallbacks and detailed error reporting
try:
    from backend.agent_logic import AgentOrchestrator
    from backend.config import DB_PATH, EMAIL_CATEGORIES
    BACKEND_AVAILABLE = True
    logger.info("✅ Backend modules imported successfully")
except ImportError as e:
    logger.warning(f"⚠ Backend modules not found: {e}")
    logger.info("Using fallback configuration...")
    BACKEND_AVAILABLE = False

    # Fallback configuration
    DB_PATH = "./email_classifier.db"
    EMAIL_CATEGORIES = ["work", "personal", "spam", "promotions", "support", "newsletter"]

# Initialize the orchestrator
orchestrator = None
ORCHESTRATOR_READY = False
try:
    if BACKEND_AVAILABLE:
        orchestrator = AgentOrchestrator()
        ORCHESTRATOR_READY = True
        logger.info("✅ AgentOrchestrator initialized successfully")
except Exception as e:
    logger.error(f"❌ Could not initialize orchestrator: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")

# Initialize FastAPI
app = FastAPI(
    title="🌱 Green AI Email Classification API",
    version="2.0",
    description="Intelligent email classification with energy optimization",
    debug=True
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request/Response Models
# -------------------------
class EmailClassificationRequest(BaseModel):
    text: str
    subject: Optional[str] = ""
    sender: Optional[str] = ""
    preferences: Optional[Dict[str, Any]] = {}
    user_id: Optional[str] = "default"

class Prediction(BaseModel):
    category: str
    confidence: float

class EmailClassificationResponse(BaseModel):
    predicted_category: str
    confidence: float
    all_predictions: List[Prediction]
    model_used: str
    escalated: bool
    energy_metrics: Dict[str, Any]
    ai_insights: Dict[str, Any]
    processing_time: float
    timestamp: float

# -------------------------
# Database initialization
# -------------------------
def initialize_database():
    """Initialize SQLite database with required tables"""
    try:
        db_dir = os.path.dirname(DB_PATH)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS email_classifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    email_hash TEXT,
                    predicted_category TEXT,
                    confidence REAL,
                    model_used TEXT,
                    escalated BOOLEAN,
                    co2_emissions_g REAL,
                    processing_time REAL,
                    energy_efficiency_score REAL
                )
            """)
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")

# -------------------------
# API Routes
# -------------------------
@app.get("/")
def root():
    return {
        "message": "🌱 Green AI Email Classification API",
        "version": "2.0",
        "status": "running",
        "orchestrator_ready": ORCHESTRATOR_READY,
        "backend_available": BACKEND_AVAILABLE,
        "docs": "/docs"
    }

@app.post("/classify-email", response_model=EmailClassificationResponse)
async def classify_email(request: EmailClassificationRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Email text cannot be empty")

        start_time = time.time()

        result = None
        if ORCHESTRATOR_READY and orchestrator:
            try:
                result = orchestrator.process_email(request.dict())
                logger.info("✅ Used orchestrator for classification")
            except Exception as e:
                logger.error(f"❌ Orchestrator failed: {e}")
                result = None

        if result is None:
            result = simple_classification(request.text, request.subject or "", request.sender or "")
            logger.info("✅ Used fallback classification")

        processing_time = time.time() - start_time
        log_classification(result, processing_time)

        return EmailClassificationResponse(
            predicted_category=result["predicted_category"],
            confidence=result["confidence"],
            all_predictions=result["all_predictions"],
            model_used=result["model_used"],
            escalated=result["escalated"],
            energy_metrics=result["energy_metrics"],
            ai_insights=result["ai_insights"],
            processing_time=processing_time,
            timestamp=result["timestamp"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ classify_email failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent-stats")
async def get_agent_stats():
    """Get performance statistics"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Total classifications
            total = cursor.execute("SELECT COUNT(*) FROM email_classifications").fetchone()[0]
            
            # Escalation rate
            escalated = cursor.execute("SELECT COUNT(*) FROM email_classifications WHERE escalated = 1").fetchone()[0]
            escalation_rate = (escalated / total) if total > 0 else 0
            
            # Average CO2 per email
            avg_co2 = cursor.execute("SELECT AVG(co2_emissions_g) FROM email_classifications").fetchone()[0] or 0
            
            # Estimated energy savings (rough calculation)
            energy_savings = max(0, (total * 2.5) - (total * avg_co2))  # Assuming 2.5g baseline
            
            return {
                "total_classifications": total,
                "escalation_rate": escalation_rate,
                "avg_co2_per_email_g": avg_co2,
                "energy_savings_estimate": energy_savings
            }
    except Exception as e:
        logger.error(f"❌ Stats calculation failed: {e}")
        return {
            "total_classifications": 0,
            "escalation_rate": 0.0,
            "avg_co2_per_email_g": 0.0,
            "energy_savings_estimate": 0.0
        }

# -------------------------
# Database logging
# -------------------------
def log_classification(result: Dict, processing_time: float):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            email_hash = hashlib.sha256(str(result.get("predicted_category", "")).encode()).hexdigest()[:16]
            conn.execute("""
                INSERT INTO email_classifications 
                (timestamp, email_hash, predicted_category, confidence, model_used, 
                 escalated, co2_emissions_g, processing_time, energy_efficiency_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result["timestamp"],
                email_hash,
                result["predicted_category"],
                result["confidence"],
                result["model_used"],
                result["escalated"],
                result["energy_metrics"]["co2_emissions_g"],
                processing_time,
                result["energy_metrics"]["energy_efficiency_score"]
            ))
    except Exception as e:
        logger.warning(f"⚠ DB logging error: {e}")

# -------------------------
# Startup
# -------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("🌱 API starting...")
    initialize_database()
    if not ORCHESTRATOR_READY:
        logger.warning("⚠ Running in fallback mode - full AI models not available")
    logger.info("✅ Ready!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True, log_level="debug")