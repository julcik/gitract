import pytorch_lightning as pl
from typing import Optional, Callable
import pandas as pd
import monai
from monai.data import CSVDataset
from monai.data import DataLoader
import torch
from torchmetrics import MetricCollection
from gitract.model.metrics import DiceMetric

SPATIAL_SIZE=(384,384)

class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        holdout_path: str,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.data_path = data_path

        train_transforms, val_transforms, test_transforms = self._init_transforms()

        self.train_dataset = CSVDataset(src=data_path, transform=train_transforms)
        self.val_dataset = CSVDataset(src=holdout_path, transform=val_transforms)


    def _init_transforms(self):
        spatial_size = SPATIAL_SIZE

        transforms = [
            monai.transforms.LoadImaged(keys=["image", "masks"]),
            # monai.transforms.AddChanneld(keys=["image", "masks"]),
            monai.transforms.AsChannelFirstd(keys=["image", "masks"], channel_dim=2),
            monai.transforms.ScaleIntensityd(keys=["image", "masks"]),
            # monai.transforms.ResizeWithPadOrCrop(keys=["image", "masks"], spatial_size=spatial_size),
            monai.transforms.Resized(keys=["image", "masks"], spatial_size=spatial_size, mode="nearest"),
        ]

        test_transforms = [
            monai.transforms.LoadImaged(keys=["image", "masks"]),
            # monai.transforms.AddChanneld(keys=["image", "masks"]),
            monai.transforms.AsChannelFirstd(keys=["image", "masks"], channel_dim=2),
            monai.transforms.ScaleIntensityd(keys=["image", "masks"]),
            # monai.transforms.ResizeWithPadOrCrop(keys=["image_3d"], spatial_size=spatial_size),
            monai.transforms.Resized(keys=["image", "masks"], spatial_size=spatial_size, mode="nearest"),
        ]

        train_transforms = monai.transforms.Compose(transforms)
        val_transforms = monai.transforms.Compose(test_transforms)
        test_transforms = monai.transforms.Compose(test_transforms)

        return train_transforms, val_transforms, test_transforms


    def _dataset(self, df: pd.DataFrame, transforms: Callable) -> CSVDataset:
        return CSVDataset(src=df, transform=transforms)

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def _dataloader(self, dataset: CSVDataset, train: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=train,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )