# Add this to your FastAPI app (api.py)

@app.get("/debug/models")
async def debug_models():
    """Debug endpoint to check model status"""
    try:
        if not ORCHESTRATOR_READY:
            return {
                "orchestrator_ready": False,
                "error": "Orchestrator not initialized",
                "backend_available": BACKEND_AVAILABLE
            }
        
        # Get model manager status
        model_status = orchestrator.email_agent.model_manager.get_status()
        
        # Test transformers import
        transformers_available = False
        transformers_error = None
        try:
            import transformers
            transformers_available = True
            transformers_version = transformers.__version__
        except ImportError as e:
            transformers_error = str(e)
        
        # Test individual model loading
        model_tests = {}
        if transformers_available:
            from transformers import pipeline
            
            test_models = [
                ("facebook/bart-large-mnli", "zero-shot-classification"),
                ("cardiffnlp/twitter-roberta-base-sentiment-latest", "sentiment-analysis")
            ]
            
            for model_name, task in test_models:
                try:
                    pipe = pipeline(task, model=model_name, device=-1)
                    model_tests[model_name] = {"status": "success", "task": task}
                except Exception as e:
                    model_tests[model_name] = {"status": "failed", "error": str(e)}
        
        return {
            "orchestrator_ready": ORCHESTRATOR_READY,
            "backend_available": BACKEND_AVAILABLE,
            "transformers_available": transformers_available,
            "transformers_version": transformers_version if transformers_available else None,
            "transformers_error": transformers_error,
            "model_manager_status": model_status,
            "individual_model_tests": model_tests,
            "light_model_config": LIGHT_MODEL,
            "heavy_model_config": HEAVY_MODEL
        }
        
    except Exception as e:
        return {
            "error": f"Debug failed: {str(e)}",
            "orchestrator_ready": ORCHESTRATOR_READY,
            "backend_available": BACKEND_AVAILABLE
        }