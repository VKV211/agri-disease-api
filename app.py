from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

# ✅ Allow only your frontend origin for security
origins = [
    "http://localhost:8081",  # React Native Web (Expo web dev server)
    "http://localhost:3000",  # (optional, in case you run React web)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # You can set ["*"] during testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "🌱 API is running fine!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Replace this dummy response with your actual ML model prediction.
    """
    # Example: Just echoing file name
    return JSONResponse({
        "status": "success",
        "prediction": "Tomato___Late_blight",
        "confidence": 92.5,
        "filename": file.filename
    })

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
