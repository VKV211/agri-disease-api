import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from io import BytesIO
from PIL import Image
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and labels
try:
    MODEL_PATH = "plant_disease_model.h5"
    CLASS_INDICES_PATH = "class_indices.json"
    
    model = load_model(MODEL_PATH)
    logger.info("✅ Model loaded successfully")
    
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    
    class_labels = {v: k for k, v in class_indices.items()}
    logger.info(f"✅ Class labels loaded: {len(class_labels)} classes")
except Exception as e:
    logger.error(f"❌ Error loading model or class indices: {str(e)}")
    raise e

def predict_image(img: Image.Image):
    """Preprocess image and make prediction"""
    try:
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        predictions = model.predict(img_array)
        predicted_class_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        return class_labels[predicted_class_idx], confidence
    except Exception as e:
        logger.error(f"❌ Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
async def home():
    return {"message": "🌱 Plant Disease Detection API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        
        if isinstance(contents, str):
            logger.warning("Received string content, attempting to extract image data")
            if "base64" in contents:
                base64_data = re.search(r"base64,(.*)", contents)
                if base64_data:
                    import base64
                    contents = base64.b64decode(base64_data.group(1))
                else:
                    raise HTTPException(status_code=422, detail="Invalid image format")
        
        img = Image.open(BytesIO(contents)).convert("RGB")
        
        label, confidence = predict_image(img)
        
        return {
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ✅ Local testing only
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
