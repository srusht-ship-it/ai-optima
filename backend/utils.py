# backend/utils.py
from typing import Dict, Any

def simple_classification(text: str) -> Dict[str, Any]:
    """Simple keyword-based classification as fallback"""
    text_lower = text.lower()

    if any(word in text_lower for word in ['congratulations', 'winner', 'claim', '$$$', 'urgent', 'click now']):
        category, confidence = "spam", 0.8
    elif any(word in text_lower for word in ['meeting', 'report', 'project', 'deadline', 'team']):
        category, confidence = "work", 0.7
    else:
        category, confidence = "general", 0.5

    return {"predicted_category": category, "confidence": confidence}
