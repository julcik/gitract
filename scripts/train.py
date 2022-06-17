from pathlib import Path
from typing import  Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from gitract.data.data_module import LitDataModule
from gitract.model.runner import LitModule
import click
from pytorch_lightning.loggers import TensorBoardLogger



KAGGLE_DIR = Path("/") / "kaggle"
INPUT_DIR = KAGGLE_DIR / "input"
OUTPUT_DIR = KAGGLE_DIR / "working"

INPUT_DATA_DIR = INPUT_DIR / "uw-madison-gi-tract-image-segmentation"
INPUT_DATA_NPY_DIR = INPUT_DIR / "uw-madison-gi-tract-image-segmentation-masks"

N_SPLITS = 5
RANDOM_SEED = 42
VAL_FOLD = 0
BATCH_SIZE = 32
NUM_WORKERS = 6
OPTIMIZER = "AdamW"
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05
SCHEDULER = "CosineAnnealingLR"
MIN_LR = 1e-8
SPATIAL_SIZE = 320 #384

GPUS = -1
MAX_EPOCHS = 30
PRECISION = 32

DEVICE = "cuda"
THR = 0.45

DEBUG = False # Debug complete pipeline

@click.command()
@click.option('--out_dir')
@click.option('--data_path', default="../mmseg_train/splits/fold_0.csv")
@click.option('--holdout_path', default="../mmseg_train/splits/holdout_0.csv")
@click.option('--batch_size', default=BATCH_SIZE)
@click.option('--num_workers', default=NUM_WORKERS)
@click.option('--learning_rate', default=LEARNING_RATE)
@click.option('--max_epochs', default=MAX_EPOCHS)
@click.option('--device', default=DEVICE)
@click.option('--model', default="smpUnet")
@click.option('--spatial_size', default=SPATIAL_SIZE)
@click.option('--background', default=False)
def train(
        out_dir,
        data_path: str,
        holdout_path: str,
        batch_size: int,
        num_workers: int,
        learning_rate: float,
        weight_decay: float = WEIGHT_DECAY,
        scheduler: Optional[str] = SCHEDULER,
        min_lr: float = MIN_LR,
        device: str = DEVICE,
        gpus: int = GPUS,
        max_epochs: int = MAX_EPOCHS,
        precision: int = PRECISION,
        debug: bool = DEBUG,
        random_seed: int = RANDOM_SEED,
        model: str = "smpUnet",
        spatial_size: int = SPATIAL_SIZE,
        background: bool = True
):
    out_dir = Path(out_dir)
    pl.seed_everything(random_seed)

    data_module = LitDataModule(
        data_path=data_path,
        holdout_path=holdout_path,
        batch_size=batch_size,
        num_workers=num_workers,
        spatial_size=(spatial_size,spatial_size),
    )

    module = LitModule(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler=scheduler,
        T_max=int(30_000 / batch_size * max_epochs) + 50,
        T_0=25,
        min_lr=min_lr,
        model=model,
        background=background
    )

    if device == "cpu":
        gpus = 0

    trainer = pl.Trainer(
        accelerator=device,
        gpus=gpus,
        log_every_n_steps=10,
        logger=[pl.loggers.CSVLogger(save_dir=out_dir/'logs/'),
                TensorBoardLogger(out_dir, name="tb_logs")],
        max_epochs=max_epochs,
        precision=precision,
        limit_train_batches=20 if device == "cpu" else None,
        limit_val_batches=20 if device == "cpu" else None,
        callbacks=[LearningRateMonitor(logging_interval='step')]
    )

    trainer.fit(module, datamodule=data_module)

    return trainer


if __name__ == '__main__':
    train()