#!/usr/bin/env python3
"""
Simple test script to debug the FastAPI email classification endpoint
"""

import requests
import json
import time

API_BASE = "http://127.0.0.1:8000"

def test_endpoint(endpoint, method="GET", data=None):
    """Test a specific endpoint"""
    url = f"{API_BASE}{endpoint}"
    print(f"\n{'='*50}")
    print(f"Testing: {method} {url}")
    print(f"{'='*50}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2)}")
            except:
                print(f"Response Text: {response.text}")
        else:
            print(f"Error Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Is the server running?")
    except requests.exceptions.Timeout:
        print("❌ Timeout Error: Server took too long to respond")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

def main():
    print("🧪 FastAPI Email Classification API Test")
    print("=" * 60)
    
    # Test 1: Root endpoint
    test_endpoint("/")
    
    # Test 2: Health check
    test_endpoint("/health")
    
    # Test 3: Debug config
    test_endpoint("/debug/config")
    
    # Test 4: Debug test classification
    test_endpoint("/debug/test-classification")
    
    # Test 5: Agent stats
    test_endpoint("/agent-stats")
    
    # Test 6: Simple email classification
    simple_email = {
        "text": "Hello, this is a test email about our meeting tomorrow at 2 PM.",
        "subject": "Meeting Tomorrow",
        "sender": "test@example.com",
        "preferences": {"priority": "balanced", "confidence_threshold": 0.85},
        "user_id": "test_user"
    }
    test_endpoint("/classify-email", "POST", simple_email)
    
    # Test 7: Another email classification
    spam_email = {
        "text": "Congratulations! You've won $1000! Click here now to claim your prize!!!",
        "subject": "You've Won!",
        "sender": "winner@spam.com",
        "preferences": {"priority": "speed"},
    }
    test_endpoint("/classify-email", "POST", spam_email)
    
    # Test 8: Edge case - empty text (should fail)
    empty_email = {
        "text": "",
        "subject": "Empty Email Test"
    }
    test_endpoint("/classify-email", "POST", empty_email)
    
    print("\n" + "="*60)
    print("🏁 Testing Complete!")
    print("="*60)

if __name__ == "__main__":
    main()