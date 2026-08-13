from flask import Flask, render_template, request, jsonify, send_file
import speech_recognition as sr
import pyttsx3
from groq import Groq
from pydub import AudioSegment
import os

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
recognizer = sr.Recognizer()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    audio_file = request.files["audio"]
    webm_path = os.path.join(UPLOAD_FOLDER, "input.webm")
    wav_path = os.path.join(UPLOAD_FOLDER, "input.wav")
    audio_file.save(webm_path)

    # convert webm to proper wav format
    sound = AudioSegment.from_file(webm_path)
    sound.export(wav_path, format="wav")

    # step 1 - speech to text
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
    try:
        user_text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio"}), 400

    # step 2 - send to groq llm
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful, concise voice assistant. Keep answers short."},
            {"role": "user", "content": user_text}
        ]
    )
    ai_text = response.choices[0].message.content

    # step 3 - text to speech, save as file
    output_path = os.path.join(UPLOAD_FOLDER, "response.mp3")
    engine = pyttsx3.init()
    engine.save_to_file(ai_text, output_path)
    engine.runAndWait()

    return jsonify({"user_text": user_text, "ai_text": ai_text, "audio_url": "/audio"})

@app.route("/audio")
def audio():
    return send_file(os.path.join(UPLOAD_FOLDER, "response.mp3"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)