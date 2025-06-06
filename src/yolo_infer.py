# src/yolo_infer.py
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import tempfile
import os

app = FastAPI()

# Load YOLOv8 classification model (must be trained and saved at this path)
try:
    model = YOLO("models/yolo_classifier.pt")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

@app.post("/classify/", summary="Classify an image")
async def classify(file: UploadFile = File(...)):
    """
    Upload a JPG or PNG image for classification using a YOLOv8 model.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Perform inference
        results = model(tmp_path)

        if not results or results[0].probs is None:
            return {"error": "⚠️ Model did not return classification probabilities."}

        pred = results[0].probs
        prediction = {
            "class": model.names[pred.top1],
            "confidence": float(pred.top1conf)
        }
        return prediction

    except Exception as e:
        return {"error": str(e)}
    
    finally:
        os.remove(tmp_path)
