import os
import uuid
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import cv2

# FastAPI uygulaması başlatılır
app = FastAPI(title="NeoVision AI Analyzer")

# Geliştirme için CORS (prod ortamında domainini ekle!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sonuçları başka bir endpoint'e göndermek için Pydantic şema
class AnalyzeReport(BaseModel):
    path: str
    score: float
    is_fake: bool

# Sonuçları almak için (veritabanı, log, vs.)
@app.post("/endpoint")
async def receive_report(report: AnalyzeReport):
    print("Received report:", report)
    # Burada ekstra işleme, log veya db kaydı yapabilirsin
    return {"ok": True}

# Model yükleme (ilk başta sadece 1 kez yapılır)
base_cnn = VGG16(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
feature_extractor = Model(
    inputs=base_cnn.input,
    outputs=base_cnn.get_layer("block5_pool").output
)
try:
    top_model = load_model("deepfake_model.h5")
except Exception as e:
    print(f"Error loading top model: {e}")
    raise

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # 1. Dosyayı uploads/ klasörüne kaydet
    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    print("File saved:", file_path)

    # 2. Görüntü ya da video AI analizi
    ext = os.path.splitext(unique_name)[1].lower()
    frames = []
    try:
        if ext in [".mp4", ".avi", ".mov", ".mkv"]:
            # Video için 10 frame çekilir
            cap = cv2.VideoCapture(file_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                raise Exception("Video okunamadı veya geçersiz.")
            if total_frames < 10:
                indices = list(range(total_frames)) + [max(0, total_frames - 1)] * (10 - total_frames)
            else:
                step = max(1, total_frames // 10)
                indices = [i * step for i in range(10)]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    cap.release()
                    raise Exception(f"Videodan frame alınamadı. Index: {idx}")
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb).convert("RGB").resize((224, 224))
                arr = img_to_array(pil_img)
                arr = np.expand_dims(arr, axis=0)
                arr = vgg_preprocess(arr)
                feat = feature_extractor.predict(arr)
                feat = np.squeeze(feat, axis=0)
                frames.append(feat)
            cap.release()
        else:
            # Resim için tek resmin 10 kopyası
            pil_img = Image.open(file_path).convert("RGB").resize((224, 224))
            arr = img_to_array(pil_img)
            arr = np.expand_dims(arr, axis=0)
            arr = vgg_preprocess(arr)
            feat = feature_extractor.predict(arr)
            feat = np.squeeze(feat, axis=0)
            frames = [feat for _ in range(10)]
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"AI analizi sırasında hata: {e}")

    feature_stack = np.stack(frames, axis=0)      # (10,7,7,512)
    feature_stack = np.expand_dims(feature_stack, axis=0)  # (1,10,7,7,512)

    try:
        prediction = top_model.predict(feature_stack)
        score = float(np.squeeze(prediction))
        is_fake = score > 0.5
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Model tahmini sırasında hata: {e}")

    # 3. Sonucu /endpoint'e POST et
    try:
        data = {
            "path": file_path,
            "score": score,
            "is_fake": is_fake
        }
        r = requests.post("http://localhost:8000/endpoint", json=data, timeout=3)
        print("/endpoint response:", r.text)
    except Exception as e:
        print("Error posting to /endpoint:", e)

    # 4. Frontend'e sade bir cevap dön
    return {
        "confidence": score,
        "is_ai_generated": is_fake
    }
