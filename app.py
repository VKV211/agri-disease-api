from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
# Initialize app
app = FastAPI()

# Allow CORS (React Native / Web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and class indices
MODEL_PATH = "plant_disease_model.h5"
CLASS_INDICES_PATH = "class_indices.json"

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"❌ Error loading model: {str(e)}")

try:
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    # Reverse mapping {index: class_name}
    idx_to_class = {v: k for k, v in class_indices.items()}
except Exception as e:
    raise RuntimeError(f"❌ Error loading class indices: {str(e)}")


@app.get("/")
def home():
    return {"message": "✅ Plant Disease Prediction API is running!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Ensure it's an image
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File is not an image")

        # Read image
        contents = await file.read()
        with open("temp.jpg", "wb") as f:
            f.write(contents)

        img = image.load_img("temp.jpg", target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = round(float(np.max(predictions)) * 100, 2)

        disease_name = idx_to_class.get(predicted_class, "Unknown")

        return {
            "status": "success",
            "prediction": disease_name,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
