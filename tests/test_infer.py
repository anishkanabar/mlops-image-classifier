import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from yolo_infer import app

import pytest
from fastapi.testclient import TestClient
from yolo_infer import app
from PIL import Image
import io

client = TestClient(app)

def test_classify_endpoint():
    # Generate a dummy image
    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    response = client.post(
        "/classify/",
        files={"file": ("dummy.jpg", img_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    json_data = response.json()
    assert "class" in json_data and "confidence" in json_data
    assert isinstance(json_data["class"], str)
    assert isinstance(json_data["confidence"], float)
