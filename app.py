from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import json

# -------------------------
# Initialize FastAPI app
# -------------------------
app = FastAPI(title="🌱 Plant Disease Prediction API")

# -------------------------
# CORS configuration
# -------------------------
origins = [
    "*",  # allow all origins (for dev/testing)
    "http://localhost:8081",
    "http://localhost:3000",
    "https://agri-disease-api.onrender.com",
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

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

try:
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
except Exception as e:
    raise RuntimeError(f"Failed to load class indices: {e}")

# -------------------------
# Routes
# -------------------------
@app.get("/")
def home():
    return {"message": "🌱 Plant Disease Prediction API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict plant disease from uploaded image.
    - file: leaf image
    """
    try:
        # Validate image
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image")

        # Load and preprocess image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        predictions = model.predict(img_array)
        predicted_index = int(np.argmax(predictions, axis=1)[0])
        confidence = round(float(np.max(predictions)) * 100, 2)
        disease_name = idx_to_class.get(predicted_index, "Unknown")

        # Low confidence handling
        if confidence < 70:
            return {
                "status": "low_confidence",
                "prediction": disease_name,
                "confidence": confidence,
                "message": "AI is not confident about this prediction. Please retake the photo or consult an expert.",
            }

        return {
            "status": "success",
            "prediction": disease_name,
            "confidence": confidence,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# Run locally
# -------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
