from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import io
from PIL import Image

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
DISEASE_SUGGESTIONS_PATH = "disease_suggestions.json"

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

try:
    with open(DISEASE_SUGGESTIONS_PATH, "r") as f:
        disease_suggestions = json.load(f)
    print("✅ Disease suggestions loaded successfully")
except Exception as e:
    raise RuntimeError(f"❌ Error loading disease suggestions: {str(e)}")

# -------------------------
# Helper function
# -------------------------
def get_disease_suggestion(disease_name: str):
    """Fetch farmer-friendly suggestions from static JSON."""
    return disease_suggestions.get(disease_name, ["No suggestion available for this disease"])

# -------------------------
# Routes
# -------------------------
@app.get("/")
def home():
    return {"message": "🌱 Plant Disease Prediction API is running!"}

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

        # ✅ Get suggestions from static JSON
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
