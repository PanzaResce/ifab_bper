from google import genai
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"


def call_ollama():
    payload = {
        "model": "llama3.2",
        "prompt": "Hello, how are you?",
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    print(response.json())


def query_google_ai(prompt, googleAPI):
    url = f"https://generativelanguage.googleapis.com/v1beta2/models/{googleAPI['Model']}:generateText"
    headers = {
        "Authorization": f"Bearer {googleAPI['API_Key']}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": {"text": prompt},
        "temperature": 0.7,
        "maxOutputTokens": 512
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["candidates"][0]["output"]
    else:
        return f"Error {response.status_code}: {response.json().get('error', {}).get('message', 'Unknown error')}"



def queryGemini(prompt, googleAPI):

    client = genai.Client(api_key= googleAPI["API_Key"] )
    response = client.models.generate_content(
        model = googleAPI["Model"],
        contents=prompt,
    )
    return response
    

