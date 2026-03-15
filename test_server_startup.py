import os
import sys

# Workaround for OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    print("Attempting to import dependencies...")
    import torch
    from transformers import pipeline
    from fastapi import FastAPI
    import uvicorn
    print("Imports successful.")
    
    print("Testing model loading...")
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="daekeun-ml/koelectra-small-v3-nsmc",
        device=-1 # CPU for testing
    )
    print("Model load successful.")
    
    # If we got here, the main things are fine.
    # The server might have other issues.
    
except Exception as e:
    print(f"Startup test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Startup test PASSED.")
