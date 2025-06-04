import mlflow
import mlflow.pytorch

with mlflow.start_run():
    mlflow.pytorch.log_model(model, "model")
    mlflow.register_model("runs:/{}/model".format(mlflow.active_run().info.run_id), "ImageClassifier")
