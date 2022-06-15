import numpy as np
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

        self.train_dataset = CSVDataset(src=data_path, col_names=["image", "masks"], transform=train_transforms)
        self.val_dataset = CSVDataset(src=holdout_path, col_names=["image", "masks"], transform=val_transforms)


    def _init_transforms(self):
        spatial_size = SPATIAL_SIZE
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])


        transforms = [
            monai.transforms.LoadImaged(keys=["image", "masks"]),
            # monai.transforms.AddChanneld(keys=["image", "masks"]),
            monai.transforms.AsChannelFirstd(keys=["image", "masks"], channel_dim=2),
            monai.transforms.ScaleIntensityd(keys="image"),
            monai.transforms.NormalizeIntensityd(keys="image", subtrahend=mean, divisor=std, channel_wise=True),
            # RandAdjustContrastd(keys=["image"], prob=0.3),
            # RandGaussianNoised(keys=["image"], prob=0.5),
            # RandRotate90d(keys=["image", "label"], prob=0.5),
            # RandFlipd(keys=["image", "label"], prob=0.5),
            # RandRotated(keys=["image", "label"], range_x=180, range_y=180, prob=0.5),
            # RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=1, max_zoom=2),
            # RandAffined(keys=["image", "label"], prob=0.5),
            # Rand2DElasticd(keys=["image", "label"], magnitude_range=(0, 1), spacing=(0.3, 0.3), prob=0.5),
            # monai.transforms.ResizeWithPadOrCrop(keys=["image", "masks"], spatial_size=spatial_size),
            monai.transforms.Resized(keys=["image", "masks"], spatial_size=spatial_size, mode="nearest"),
        ]

        test_transforms = [
            monai.transforms.LoadImaged(keys=["image", "masks"]),
            # monai.transforms.AddChanneld(keys=["image", "masks"]),
            monai.transforms.AsChannelFirstd(keys=["image", "masks"], channel_dim=2),
            monai.transforms.ScaleIntensityd(keys=["image", "masks"]),
            monai.transforms.NormalizeIntensityd(keys="image", subtrahend=mean, divisor=std, channel_wise=True),
            # monai.transforms.ResizeWithPadOrCrop(keys=["image_3d"], spatial_size=spatial_size),
            monai.transforms.Resized(keys=["image", "masks"], spatial_size=spatial_size, mode="nearest"),
        ]

        train_transforms = monai.transforms.Compose(transforms)
        val_transforms = monai.transforms.Compose(test_transforms)
        test_transforms = monai.transforms.Compose(test_transforms)

        return train_transforms, val_transforms, test_transforms

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