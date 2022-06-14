from pathlib import Path
from typing import  Optional
import pytorch_lightning as pl
from gitract.data.data_module import LitDataModule
from gitract.model.runner import LitModule

KAGGLE_DIR = Path("/") / "kaggle"
INPUT_DIR = KAGGLE_DIR / "input"
OUTPUT_DIR = KAGGLE_DIR / "working"

INPUT_DATA_DIR = INPUT_DIR / "uw-madison-gi-tract-image-segmentation"
INPUT_DATA_NPY_DIR = INPUT_DIR / "uw-madison-gi-tract-image-segmentation-masks"

SPATIAL_SIZE = (192, 192, 128)
N_SPLITS = 5
RANDOM_SEED = 42
VAL_FOLD = 0
BATCH_SIZE = 8
NUM_WORKERS = 0 #2
OPTIMIZER = "Adam"
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-6
SCHEDULER = None
MIN_LR = 1e-6

FAST_DEV_RUN = False # Debug training
GPUS = 1
MAX_EPOCHS = 30
PRECISION = 32

DEVICE = "cuda"
THR = 0.45

DEBUG = False # Debug complete pipeline


def train(
        random_seed: int = RANDOM_SEED,
        data_path: str = "../mmseg_train/splits/fold_0.csv",
        holdout_path: str = "../mmseg_train/splits/holdout_0.csv",
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        scheduler: Optional[str] = SCHEDULER,
        min_lr: float = MIN_LR,
        gpus: int = GPUS,
        fast_dev_run: bool = FAST_DEV_RUN,
        max_epochs: int = MAX_EPOCHS,
        precision: int = PRECISION,
        debug: bool = DEBUG,
):
    pl.seed_everything(random_seed)

    if debug:
        max_epochs = 2

    data_module = LitDataModule(
        data_path=data_path,
        holdout_path=holdout_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    module = LitModule(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler=scheduler,
        T_max=int(30_000 / batch_size * max_epochs) + 50,
        T_0=25,
        min_lr=min_lr,
    )

    trainer = pl.Trainer(
        # fast_dev_run=fast_dev_run,
        # gpus=gpus,
        log_every_n_steps=10,
        logger=pl.loggers.CSVLogger(save_dir='logs/'),
        max_epochs=max_epochs,
        precision=precision,
    )

    trainer.fit(module, datamodule=data_module)

    # if not fast_dev_run:
    #     trainer.test(module, datamodule=data_module)

    return trainer


if __name__ == '__main__':
    train()