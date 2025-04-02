from google import genai
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"


def call_ollama():
    payload = {
        "model": "llama3.2",
        "prompt": "Hello, how are you?",
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    print(response.json())


def queryGemini(prompt, googleAPI):

    client = genai.Client(api_key= googleAPI["API_Key"] )
    response = client.models.generate_content(
        model = googleAPI["Model"],
        contents=prompt,
    )
    return response
    

