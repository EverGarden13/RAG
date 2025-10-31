"""Simple test for OpenRouter API."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
print(f"API Key loaded: {api_key[:20]}..." if api_key else "No API key found")

# Test with a simple free model
models_to_try = [
    "qwen/qwen3-coder:free",
    "deepseek/deepseek-chat-v3.1:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
]

for model in models_to_try:
    print(f"\n{'='*60}")
    print(f"Testing model: {model}")
    print('='*60)
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Say 'Hello World' in one word"}
        ],
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                answer = data['choices'][0]['message']['content']
                print(f"✓ SUCCESS! Response: {answer}")
                print(f"This model works: {model}")
                break
            else:
                print(f"✗ No response in data: {data}")
        else:
            print(f"✗ Error: {response.text}")
    except Exception as e:
        print(f"✗ Exception: {e}")

print("\n" + "="*60)
print("Test complete")
print("="*60)
