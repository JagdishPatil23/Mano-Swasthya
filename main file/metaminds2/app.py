from flask import Flask, request, jsonify, render_template, Response
import cv2
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import requests
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# === Chatbot Configuration ===
API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
HEADERS = {"Authorization": "Bearer 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"}
chat_history = []  # Store previous messages

# === Flask App Setup ===
app = Flask(__name__, static_folder="static", template_folder="templates")

# === Load Hugging Face Face Emotion Detection Model ===
extractor = AutoFeatureExtractor.from_pretrained("dpaul93/face_emotions_image_detection-v2")
model = AutoModelForImageClassification.from_pretrained("dpaul93/face_emotions_image_detection-v2")

# === Transform for Face Emotion Detection ===
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=extractor.image_mean, std=extractor.image_std)
])

# === Camera Setup ===
camera = cv2.VideoCapture(0)

# === Function to Generate Camera Frames ===
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        input_tensor = transform(pil_image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()
            label = model.config.id2label[predicted_class]

        cv2.putText(frame, f'Emotion: {label}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# === Chatbot Logic ===
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


# === Routes ===
@app.route("/")
def index():
    return render_template("folder11.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    bot_response = chat_with_bot(user_input)
    return jsonify({"response": bot_response})

@app.route("/run-script", methods=["GET"])
def run_script():
    try:
        subprocess.run(['python', 'emo.py'])  # Optional: run additional script
        return jsonify({'message': 'Script executed successfully!'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face')
def face():
    return render_template('face/face.html')

@app.route('/signup')
def signup():
    return render_template('signup/signup.html')

@app.route('/email')
def email():
    return render_template('email/email.html')

@app.route('/voice')
def voice():
    return render_template('voice/voice.html')

@app.route('/passive')
def passive():
    return render_template('passive/passive.html')

@app.route('/nextpage')
def nextpage():
    return render_template('nextpage/nextpage.html')


# === Run Server ===
if __name__ == "__main__":
    app.run(debug=True)