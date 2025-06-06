import os
import yaml
from ultralytics import YOLO
import mlflow
from mlflow_utils import init_mlflow, log_params, log_model, log_metrics
from pathlib import Path

def train():
    from ultralytics.utils import SETTINGS
    SETTINGS['mlflow'] = False

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    model = YOLO(params["model_name"])

    init_mlflow("YOLOv8-Classifier")
    with mlflow.start_run():
        results = model.train(
            data="data/classification",
            epochs=params["epochs"],
            imgsz=params["imgsz"],
            batch=params["batch"],
            project="runs/classifier",
            name="yolo_classifier"
        )

        model_path = Path("models/yolo_classifier.pt")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))

        log_params(params)
        log_model(model_path)

        log_metrics({
            "train_loss": results.results_dict.get("train/cls_loss", 0.0)
        })

if __name__ == "__main__":
    train()
