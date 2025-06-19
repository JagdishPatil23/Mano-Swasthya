from flask import Flask, request, jsonify, render_template

app = Flask(__name__, static_folder="../static", template_folder="../templates")  # Set template and static folder

import requests

API_URL = "https://api-inference.huggingface.co/models/google/gemma-2-2b-it"
HEADERS = {"Authorization": "Bearer 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"}

chat_history = []  # Store previous messages

def chat_with_bot(user_input):
    global chat_history
    chat_history.append(f"User: {user_input}")

    prompt = "\n".join(chat_history[-5:]) + "\nBot:"

    response = requests.post(API_URL, headers=HEADERS, json={
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 100,
            "return_full_text": False,
            "stop": ["\nUser:", "\nBot:"]
        }
    })

    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        return "Error: Unable to decode response."

    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"

    if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
        bot_response = data[0]["generated_text"].strip()
    else:
        bot_response = "Error: Unexpected response format."

    chat_history.append(f"Bot: {bot_response}")  
    return bot_response

@app.route("/")
def home():
    return render_template("folder11.html")  # Serve the HTML file

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    bot_response = chat_with_bot(user_input)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
