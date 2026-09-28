import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "gemini-3.1-flash-lite"

app = Flask(__name__)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

config_path = BASE_DIR / "chatbot_config.txt"
SYSTEM_PROMPT = config_path.read_text(encoding="utf-8")


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/chat")
def chat():
    if not client:
        return jsonify({"error": "Gemini API key is not configured."}), 500

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=message,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0.3,
            },
        )

        reply = (response.text or "").strip()
        if not reply:
            reply = "I couldn't generate a response. Please try again."

        return jsonify({"reply": reply})

    except Exception:
        return jsonify({
            "error": "Unable to reach Gemini right now. Please try again later."
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
