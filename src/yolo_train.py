import os
import yaml
from ultralytics import YOLO
import mlflow
from mlflow_utils import init_mlflow, log_params, log_model, log_metrics

def train():
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

        model_path = "models/yolo_classifier.pt"
        model.save(model_path)

        log_params(params)
        log_model(model_path)
        log_metrics({
            "train_loss": results.results_dict["train/cls_loss"]
        })

if __name__ == "__main__":
    train()
