# src/train.py
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from src.model import LitClassifier
from src.datamodule import ImageDataModule

def train():
    model = LitClassifier()
    data = ImageDataModule()

    mlf_logger = MLFlowLogger(experiment_name="image_classifier", tracking_uri="file:./mlruns")
    trainer = pl.Trainer(max_epochs=5, logger=mlf_logger)
    trainer.fit(model, datamodule=data)

if __name__ == "__main__":
    train()
