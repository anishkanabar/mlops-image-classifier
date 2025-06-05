import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from yolo_infer import app

import mlflow
from mlflow_utils import init_mlflow, log_params, log_metrics

def test_mlflow_logging():
    init_mlflow("test-mlflow-utils")
    with mlflow.start_run():
        log_params({"epochs": 1, "batch": 2})
        log_metrics({"accuracy": 0.99})
