"""Check available models on OpenRouter."""
import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
headers = {
    "Authorization": f"Bearer {api_key}",
}

# Get models list
response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)

if response.status_code == 200:
    models = response.json()
    print("Available models (showing first 20 free models):")
    count = 0
    for model in models.get('data', []):
        if ':free' in model['id'] or model.get('pricing', {}).get('prompt', '0') == '0':
            print(f"  - {model['id']}")
            count += 1
            if count >= 20:
                break
else:
    print(f"Error: {response.status_code}")
    print(response.text)
