# test_models.py - Test if models can load
import logging
logging.basicConfig(level=logging.INFO)

def test_model_loading():
    """Test if the models can actually load"""
    
    try:
        from transformers import pipeline
        print("✅ Transformers library imported successfully")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return
    
    # Test models one by one
    models_to_test = [
        ("facebook/bart-large-mnli", "zero-shot-classification"),
        ("cardiffnlp/twitter-roberta-base-sentiment-latest", "sentiment-analysis"),
        ("distilbert-base-uncased-finetuned-sst-2-english", "sentiment-analysis"),
    ]
    
    for model_name, task in models_to_test:
        print(f"\n🔄 Testing {model_name}...")
        try:
            pipe = pipeline(task, model=model_name, device=-1)
            print(f"✅ {model_name} loaded successfully")
            
            # Test with sample text
            if task == "zero-shot-classification":
                result = pipe("This is a work email about a meeting", 
                            candidate_labels=["work", "personal", "spam"])
                print(f"📝 Test result: {result['labels'][0]} ({result['scores'][0]:.3f})")
            else:
                result = pipe("This is a test email")
                print(f"📝 Test result: {result[0]['label']} ({result[0]['score']:.3f})")
                
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
    
    print("\n" + "="*50)
    print("If all models failed, your transformers installation may have issues.")
    print("Try: pip install --upgrade transformers torch")

if __name__ == "__main__":
    test_model_loading()