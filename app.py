from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import json
import requests
import os

# -------------------------
# Initialize FastAPI app
# -------------------------
app = FastAPI(title="🌱 Plant Disease Prediction API")

# ✅ CORS setup
origins = [
    "*",  # allow all origins (dev/testing)
    "http://localhost:8081",  # Expo web
    "http://localhost:3000",  # React web
    "https://agri-disease-api.onrender.com",  # Deployed backend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load Model & Class Data
# -------------------------
MODEL_PATH = "plant_disease_model.h5"
CLASS_INDICES_PATH = "class_indices.json"

# Load ML model
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

# Load class indices
try:
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    print("✅ Class indices loaded")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load class indices: {e}")

# -------------------------
# Gemini API Setup
# -------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAc_g9Oek-wavaFWDeyncuD-PywXS-GI90")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

def fetch_gemini_suggestion(disease_name: str):
    """Call Gemini API to get suggestions dynamically"""
    if disease_name.lower() == "healthy":
        prompt = "The crop leaf is healthy. Give 3 simple precautionary steps farmers should take based on season and weather."
    else:
        prompt = f"The crop is affected by {disease_name}. Give 3 simple farmer-friendly treatment suggestions, including general care and medication names if possible."

    body = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    try:
        response = requests.post(GEMINI_URL, json=body, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        data = response.json()
        suggestion = data.get("candidates", [])[0]["content"]["parts"][0]["text"].strip()
        return suggestion
    except Exception as e:
        return f"❌ Failed to fetch suggestions: {e}"

# -------------------------
# Routes
# -------------------------
@app.get("/")
def home():
    return {"message": "🌱 Plant Disease Prediction API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File is not an image")

        # Load image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        predictions = model.predict(img_array)
        predicted_index = int(np.argmax(predictions, axis=1)[0])
        confidence = round(float(np.max(predictions)) * 100, 2)

        disease_name = idx_to_class.get(predicted_index, "Unknown")

        # ✅ Get dynamic suggestion from Gemini API
        suggestion = fetch_gemini_suggestion(disease_name)

        return {
            "status": "success",
            "prediction": disease_name,
            "confidence": confidence,
            "suggestion": suggestion
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# Run Locally
# -------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
