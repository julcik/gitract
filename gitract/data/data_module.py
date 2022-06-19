import numpy as np
import pytorch_lightning as pl
from typing import Optional, Callable, Tuple
import pandas as pd
import monai
from monai.data import CSVDataset, CacheNTransDataset
from monai.data import DataLoader
from monai.utils import set_determinism
import torch
from torchmetrics import MetricCollection
from gitract.model.metrics import DiceMetric

class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: Optional[str],
        holdout_path: Optional[str],
        spatial_size: Tuple[int,int],
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        set_determinism(seed=42)
        self.save_hyperparameters()
        self.data_path = data_path

        train_transforms, val_transforms, test_transforms = self._init_transforms()

        if data_path and holdout_path:
            self.train_dataset = CSVDataset(src=data_path, col_names=["image", "masks"], transform=train_transforms)
            self.val_dataset = CSVDataset(src=holdout_path, col_names=["image", "masks"], transform=val_transforms)


    def _init_transforms(self):
        spatial_size = self.hparams.spatial_size

        mean = np.array([0.456, 0.456, 0.456])
        std = np.array([0.224, 0.224, 0.224])


        transforms = [
            monai.transforms.LoadImaged(keys=["image", "masks"]),
            monai.transforms.AsChannelFirstd(keys=["image", "masks"], channel_dim=2),
            monai.transforms.ScaleIntensityd(keys="image", minv=None, maxv= None, factor=1/255.0 - 1),
            monai.transforms.NormalizeIntensityd(keys="image", subtrahend=mean, divisor=std, channel_wise=True),

            monai.transforms.RandAdjustContrastd(keys=["image"], prob=0.2),
            monai.transforms.RandGaussianNoised(keys=["image"], prob=0.1),
            # monai.transforms.RandFlipd(keys=["image", "masks"], prob=0.5),
            monai.transforms.RandRotated(keys=["image", "masks"], range_x=5, range_y=0, prob=0.1),
            monai.transforms.RandZoomd(keys=["image", "masks"], prob=0.1, min_zoom=0.8, max_zoom=1.3),
            monai.transforms.Rand2DElasticd(keys=["image", "masks"], magnitude_range=(0, 1), spacing=(0.3, 0.3), prob=0.1),

            monai.transforms.Resized(keys=["image", "masks"], size_mode="longest", spatial_size=spatial_size[0], mode="nearest"),
            monai.transforms.ResizeWithPadOrCropd(keys=["image", "masks"], spatial_size=spatial_size),
            monai.transforms.ToTensord(keys=["image", "masks"]),
        ]

        val_transforms = [
            monai.transforms.LoadImaged(keys=["image", "masks"]),
            # monai.transforms.AddChanneld(keys=["image", "masks"]),
            monai.transforms.AsChannelFirstd(keys=["image", "masks"], channel_dim=2),
            monai.transforms.ScaleIntensityd(keys="image", minv=None, maxv=None, factor=1 / 255.0 - 1),
            monai.transforms.NormalizeIntensityd(keys="image", subtrahend=mean, divisor=std, channel_wise=True),
            # monai.transforms.ResizeWithPadOrCrop(keys=["image_3d"], spatial_size=spatial_size),
            monai.transforms.Resized(keys=["image", "masks"], size_mode="longest", spatial_size=spatial_size[0],
                                     mode="nearest"),
            monai.transforms.ResizeWithPadOrCropd(keys=["image", "masks"], spatial_size=spatial_size),
            monai.transforms.ToTensord(keys=["image", "masks"]),
        ]

        test_transforms = [
            monai.transforms.LoadImaged(keys=["image", "masks"], allow_missing_keys=True),
            monai.transforms.ScaleIntensityd(keys="image", minv=None, maxv=None, factor=1 / 255.0 - 1, allow_missing_keys=True),
            monai.transforms.NormalizeIntensityd(keys="image", subtrahend=mean, divisor=std, channel_wise=True, allow_missing_keys=True),
            monai.transforms.Resized(keys=["image", "masks"], size_mode="longest", spatial_size=spatial_size[0],
                                     mode="nearest", allow_missing_keys=True),
            monai.transforms.ResizeWithPadOrCropd(keys=["image", "masks"], spatial_size=spatial_size, allow_missing_keys=True),
            monai.transforms.ToTensord(keys=["image", "masks"], allow_missing_keys=True),
        ]

        train_transforms = monai.transforms.Compose(transforms)
        val_transforms = monai.transforms.Compose(val_transforms)
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
            drop_last=train,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )