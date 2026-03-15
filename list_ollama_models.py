import requests

def list_ollama_models():
    url = "http://localhost:11434/api/tags"
    try:
        response = requests.get(url)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [m["name"] for m in models]
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    print("Available Ollama models:")
    models = list_ollama_models()
    if isinstance(models, list):
        for m in models:
            print(f"- {m}")
    else:
        print(models)
