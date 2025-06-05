# src/yolo_infer.py
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import tempfile

app = FastAPI()
model = YOLO("models/yolo_classifier.pt")

@app.post("/classify/")
async def classify(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    results = model(tmp_path)
    pred = results[0].probs
    prediction = {
        "class": model.names[pred.top1],
        "confidence": float(pred.top1conf)
    }
    return prediction
