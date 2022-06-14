from pathlib import Path
from typing import  Optional
import pytorch_lightning as pl
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
OPTIMIZER = "Adam"
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-6
SCHEDULER = None
MIN_LR = 1e-6

FAST_DEV_RUN = False # Debug training
GPUS = 0
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
        fast_dev_run: bool = FAST_DEV_RUN,
        max_epochs: int = MAX_EPOCHS,
        precision: int = PRECISION,
        debug: bool = DEBUG,
        random_seed: int = RANDOM_SEED,
):
    out_dir = Path(out_dir)
    pl.seed_everything(random_seed)

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
        accelerator=device,
        gpus=gpus,
        log_every_n_steps=10,
        logger=[pl.loggers.CSVLogger(save_dir=out_dir/'logs/'),
                TensorBoardLogger(out_dir, name="tb_logs")],
        max_epochs=max_epochs,
        precision=precision,
    )

    trainer.fit(module, datamodule=data_module)

    # if not fast_dev_run:
    #     trainer.test(module, datamodule=data_module)

    return trainer


if __name__ == '__main__':
    train()