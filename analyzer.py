from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import shutil
import os

app = FastAPI()

# CORS (Java + Frontend )
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # sınır
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  model 
model = load_model("ai_detector_model.h5")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # temp
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # read edit
    try:
        img = Image.open(temp_path).convert("RGB").resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # guess
        prediction = model.predict(img_array)
        score = float(prediction[0][0])
        is_fake = score > 0.5
    except Exception as e:
        os.remove(temp_path)
        return {"error": str(e)}


    os.remove(temp_path)

    
    return {
        "is_ai_generated": is_fake,
        "confidence": round(score, 4)  #: 0.8732
    }

# uvicorn analyzer:app --reload --port 8000
