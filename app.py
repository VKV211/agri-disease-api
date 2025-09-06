from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import io
from PIL import Image
import google.generativeai as genai

# -------------------------
# Initialize FastAPI app
# -------------------------
app = FastAPI()

# ✅ CORS setup
origins = [
    "*",  # allow all origins (for dev/testing)
    "http://localhost:8081",  # Expo web (React Native dev)
    "http://localhost:3000",  # React web app
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
# Load Model and Class Indices
# -------------------------
MODEL_PATH = "plant_disease_model.h5"
CLASS_INDICES_PATH = "class_indices.json"

try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    raise RuntimeError(f"❌ Error loading model: {str(e)}")

try:
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    # Reverse mapping {index: class_name}
    idx_to_class = {v: k for k, v in class_indices.items()}
    print("✅ Class indices loaded successfully")
except Exception as e:
    raise RuntimeError(f"❌ Error loading class indices: {str(e)}")


# -------------------------
# Gemini Setup
# -------------------------
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"  # 🔑 Replace with your key
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def get_disease_suggestion(disease_name: str) -> str:
    """Fetch farmer-friendly suggestions from Gemini."""
    try:
        prompt = f"""
        The detected plant disease is: {disease_name}.
        Please give short, clear, farmer-friendly treatment steps,
        including natural remedies and preventive measures.
        Limit to 5 bullet points.
        """
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Could not fetch AI suggestion: {str(e)}"


# -------------------------
# Routes
# -------------------------
@app.get("/")
def home():
    return {"message": "🌱 Plant Disease Prediction API with Gemini is running!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Ensure it's an image
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File is not an image")

        # Read file as PIL image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess image
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediction
        predictions = model.predict(img_array)
        predicted_index = int(np.argmax(predictions, axis=1)[0])
        confidence = round(float(np.max(predictions)) * 100, 2)

        disease_name = idx_to_class.get(predicted_index, "Unknown")

        # ✅ Get AI suggestions from Gemini
        suggestion = get_disease_suggestion(disease_name)

        return {
            "status": "success",
            "prediction": disease_name,
            "confidence": confidence,
            "suggestion": suggestion
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# Run locally
# -------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
