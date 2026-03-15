import os
# openmp fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from transformers import pipeline
    import torch
    print(f"Torch version: {torch.__version__}")
    
    print("Loading sentiment analyzer...")
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="daekeun-ml/koelectra-small-v3-nsmc",
        device=0 if torch.cuda.is_available() else -1
    )
    
    test_text = "이 영화 정말 재미있어요! 추천합니다."
    print(f"Testing text: {test_text}")
    result = sentiment_analyzer(test_text)[0]
    print(f"Result: {result}")
    
    label_map = {"LABEL_0": "Negative", "LABEL_1": "Positive"}
    label = label_map.get(result["label"], result["label"])
    print(f"Sentiment: {label} (Score: {result['score']:.4f})")
    
except Exception as e:
    print(f"Error during sentiment analysis test: {e}")
    import traceback
    traceback.print_exc()
