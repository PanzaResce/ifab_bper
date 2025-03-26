import requests

OLLAMA_URL = "http://172.30.96.1:11434/api/generate"

def call_ollama():
    payload = {
        "model": "llama3.2",
        "prompt": "Hello, how are you?",
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    print(response.json())

if __name__ == "__main__":
    call_ollama()
