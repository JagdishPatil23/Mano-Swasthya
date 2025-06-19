

import requests

API_URL = "https://api-inference.huggingface.co/models/tanusrich/Mental_Health_Chatbot"
headers = {"Authorization": "Bearer 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"}

data = {
    "inputs": "User: How can I improve mental health?\nBot:",
    "parameters": {
        "max_new_tokens": 100,
        "return_full_text": False,
        "stop": ["\nUser:", "\nBot:"]
    }
}

response = requests.post(API_URL, headers=headers, json=data)

try:
    print(response.json())
except Exception as e:
    print("Decode error:", e)
    print("Raw response:", response.text)

