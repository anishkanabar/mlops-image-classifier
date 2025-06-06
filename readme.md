# 🧠 YOLOv8 Image Classifier - MLOps-Ready

This project implements an **image classification pipeline using YOLOv8** as a classifier, wrapped in a modern MLOps framework. It integrates:

- ✅ PyTorch + [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- ✅ FastAPI inference service
- ✅ MLflow for experiment tracking
- ✅ DVC for dataset and model versioning
- ✅ Pytest for automated testing
- ✅ Torch Lightning for training structure (optional hooks)
- ✅ Reproducible structure for deployment or CI/CD

---

## 🚀 Project Structure

mlops-image-classifier/
├── data/ # Raw + processed data
├── models/ # Saved YOLOv8 classifier models
├── params.yaml # Training configuration
├── dvc.yaml # DVC pipeline stages
├── requirements.txt # All dependencies
├── mlruns/ # MLflow tracking data (auto-generated)
├── src/
│ ├── yolo_train.py # Training pipeline with MLflow
│ ├── yolo_infer.py # FastAPI inference API
│ ├── mlflow_utils.py # MLflow helpers
├── tests/
│ ├── test_train.py # Unit tests for training
│ ├── test_infer.py # Unit tests for API
├── .dvc/ # DVC config folder
└── README.md