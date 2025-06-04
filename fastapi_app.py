# fastapi_app.py
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = FastAPI()
model = load_model("models/image_classifier.h5")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).resize((28, 28)).convert("L")
    img_array = np.array(image).reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(img_array)
    return {"prediction": int(np.argmax(prediction))}
