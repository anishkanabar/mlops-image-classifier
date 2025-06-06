import mlflow
from pathlib import Path

def init_mlflow(experiment_name: str = "YOLOv8-Classifier"):
    mlruns_path = Path("mlruns")
    mlruns_path.mkdir(parents=True, exist_ok=True)  # Create folder if missing
    tracking_uri = mlruns_path.resolve().as_uri()  # Now it's guaranteed absolute
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_params(params: dict):
    """
    Logs all hyperparameters to MLflow.
    """
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_metrics(metrics: dict):
    """
    Logs evaluation or training metrics to MLflow.
    """
    for key, value in metrics.items():
        mlflow.log_metric(key, value)


def log_model(model_path: str):
    """
    Logs the trained model file to MLflow as an artifact.
    """
    model_path_obj = Path(model_path)
    if model_path_obj.exists():
        mlflow.log_artifact(str(model_path_obj.resolve()))
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
