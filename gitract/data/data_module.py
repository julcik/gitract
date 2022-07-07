from os import PathLike
from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as pl
from typing import Optional, Callable, Tuple, List, Sequence, Union, Dict
import pandas as pd
import monai
from PIL import Image
from monai.data import CSVDataset, CacheNTransDataset, PILReader, CacheDataset
from monai.data import DataLoader
from monai.data.image_reader import PILImage, ImageReader, _copy_compatible_dict, _stack_images
from monai.utils import set_determinism, ensure_tuple
import torch
from torchmetrics import MetricCollection

from gitract.data.utils import parse_train, rle_decode
from gitract.model.metrics import DiceMetric


def image_reader(file_name, numpy=False):
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED).astype('float32')
    img = cv2.normalize(img, None, alpha=0, beta=255,
                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = img.astype(np.uint8)
    if numpy:
        return img
    return Image.fromarray(img)


def mask_reader(rle, shape, numpy=False):
    if len(rle)>0:
        mask = rle_decode(rle, shape)
    else:
        mask = np.zeros(shape, dtype=np.uint8)
    if numpy:
        return mask
    return Image.fromarray(mask)


class GitractReader(PILReader):

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        return True

    def read(self, data, **kwargs):
        img_: List[PILImage.Image] = []
        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            if "." in name:
                img = image_reader(name)
            else:
                rle = name.split("[")[0]
                shape = [int(i) for i in name.split("[")[1][:-1].split(",")]
                img = mask_reader(rle, shape)
            if callable(self.converter):
                img = self.converter(img)
            img_.append(img)
        if "." not in filenames[0] and len(img_) > 3: # if masks combine by class len
            img_ = [Image.fromarray(np.stack(img_[i:i+3], axis=-1).astype(np.uint8)) for i in range(0,len(img_),3)]
            # print(img_[0].shape)
        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img):

        img_array: List[np.ndarray] = []
        compatible_meta: Dict = {}

        for i in ensure_tuple(img):
            header = self._get_meta_dict(i)
            header["spatial_shape"] = self._get_spatial_shape(i)
            data = np.moveaxis(np.asarray(i), 0, 1)
            if data.ndim == 3:
                data = np.moveaxis(data,-1,0)
            data = data[...,None]
            img_array.append(data)
            header["original_channel_dim"] = "no_channel" if len(data.shape) == len(header["spatial_shape"]) else -1
            _copy_compatible_dict(header, compatible_meta)
        res = _stack_images(img_array, compatible_meta), compatible_meta
        return res


class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: Optional[PathLike],
        spatial_size: Tuple[int,int],
        batch_size: int,
        num_workers: int,
        fold: int = 0,
        slices: int = 5,
        stride: int = 1

    ):
        super().__init__()
        set_determinism(seed=42)
        self.save_hyperparameters()
        self.data_path = data_path
        self.fold = fold

        train_transforms, val_transforms, test_transforms = self._init_transforms()

        if data_path:
            df = parse_train(data_path,  slices=slices, stride = stride)
            df = df.sample(frac=1.0, replace=False, random_state=42).reset_index(drop=True)

            from sklearn.model_selection import GroupKFold
            split = list(GroupKFold(5).split(df["patient"], groups=df["patient"]))
            train_idx, valid_idx = split[self.fold]

            self.df_train = df.iloc[train_idx][["image", "masks"]].reset_index(drop=True)
            self.df_val = df.iloc[valid_idx][["image", "masks"]].reset_index(drop=True)

            self.train_dataset = CSVDataset(self.df_train[["image", "masks"]],
                              col_names=["image", "masks"],
                              transform=train_transforms
                              )
            #CSVDataset(src=data_path, col_names=["image", "masks"], transform=train_transforms)
            self.val_dataset = CSVDataset(self.df_val[["image", "masks"]],
                              col_names=["image", "masks"],
                              transform=val_transforms
                              )

            # for d in self.train_dataset:
            #     print(d)


    def _init_transforms(self):
        spatial_size = self.hparams.spatial_size

        mean = np.array([0.456]*self.hparams.slices)
        std = np.array([0.224]*self.hparams.slices)

        transforms = [
            # monai.transforms.LoadImaged(keys=["image", "masks"]),
            monai.transforms.LoadImaged(keys=["image", "masks"], reader=GitractReader(), image_only=True),
            # monai.transforms.AsChannelFirstd(keys=["masks"], channel_dim=2),
            monai.transforms.ScaleIntensityd(keys="image", minv=None, maxv= None, factor=1/255.0 - 1),
            monai.transforms.NormalizeIntensityd(keys="image", subtrahend=mean, divisor=std, channel_wise=True),

            monai.transforms.RandAdjustContrastd(keys=["image"], prob=0.2),
            monai.transforms.RandGaussianNoised(keys=["image"], prob=0.1),
            # monai.transforms.RandFlipd(keys=["image", "masks"], prob=0.5),
            # monai.transforms.RandRotate90d(keys=["image", "masks"], prob=0.5),
            monai.transforms.RandRotated(keys=["image", "masks"], range_x=5, range_y=0, prob=0.2),
            monai.transforms.RandZoomd(keys=["image", "masks"], prob=0.2, min_zoom=0.8, max_zoom=1.2),
            monai.transforms.Rand2DElasticd(keys=["image", "masks"], magnitude_range=(0, 1), spacing=(0.3, 0.3), prob=0.2),

            # monai.transforms.Resized(keys=["image", "masks"], size_mode="longest", spatial_size=spatial_size[0], mode="nearest"),
            monai.transforms.ResizeWithPadOrCropd(keys=["image", "masks"], spatial_size=spatial_size),
            monai.transforms.ToTensord(keys=["image", "masks"]),
        ]

        val_transforms = [
            monai.transforms.LoadImaged(keys=["image", "masks"], reader=GitractReader(), image_only=True),
            # monai.transforms.LoadImaged(keys=["image", "masks"]),
            # monai.transforms.AddChanneld(keys=["image", "masks"]),
            # monai.transforms.AsChannelFirstd(keys=["masks"], channel_dim=2),
            monai.transforms.ScaleIntensityd(keys="image", minv=None, maxv=None, factor=1 / 255.0 - 1),
            monai.transforms.NormalizeIntensityd(keys="image", subtrahend=mean, divisor=std, channel_wise=True),
            # monai.transforms.Resized(keys=["image", "masks"], size_mode="longest", spatial_size=spatial_size[0],
            #                          mode="nearest"),
            monai.transforms.ResizeWithPadOrCropd(keys=["image", "masks"], spatial_size=spatial_size),
            monai.transforms.ToTensord(keys=["image", "masks"]),
        ]

        test_transforms = [
            monai.transforms.LoadImaged(keys=["image", "masks"], reader=GitractReader(), allow_missing_keys=True),
            monai.transforms.ScaleIntensityd(keys="image", minv=None, maxv=None, factor=1 / 255.0 - 1,
                                             allow_missing_keys=True),
            monai.transforms.NormalizeIntensityd(keys="image", subtrahend=mean, divisor=std, channel_wise=True,
                                                 allow_missing_keys=True),
            # monai.transforms.Resized(keys=["image", "masks"], size_mode="longest", spatial_size=spatial_size,
            #                          mode="nearest", allow_missing_keys=True),
            monai.transforms.ResizeWithPadOrCropd(keys=["image", "masks"], spatial_size=(spatial_size, spatial_size),
                                                  allow_missing_keys=True),
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


class LitDataModule3d(LitDataModule):
    def __init__(
        self,
        cache_rate_train=0.7,
        cache_rate_val=0.5,
        **kwargs

    ):
        super().__init__(**kwargs)
        set_determinism(seed=42)
        self.save_hyperparameters()

        train_transforms, val_transforms, test_transforms = self._init_transforms()

        if self.data_path:
            df = parse_train(self.data_path, slices=-11, stride=1)
            df = df.sample(frac=1.0, replace=False, random_state=42).reset_index(drop=True)

            from sklearn.model_selection import GroupKFold
            split = list(GroupKFold(5).split(df["patient"], groups=df["patient"]))
            train_idx, valid_idx = split[self.fold]

            self.df_train = df.iloc[train_idx][["image", "masks"]].reset_index(drop=True)
            self.df_val = df.iloc[valid_idx][["image", "masks"]].reset_index(drop=True)



            train_csv_dataset = CSVDataset(self.df_train[["image", "masks"]],col_names=["image", "masks"],
                                           transform=monai.transforms.Compose([
                                               monai.transforms.LoadImaged(keys=["image", "masks"], reader=GitractReader(), image_only=True),
                                               monai.transforms.AddChanneld(keys=["image"])])

                                           )
            print("dataset len", self.df_train.shape[0], len(train_csv_dataset), self.df_val.shape[0])
            assert len(train_csv_dataset) == self.df_train.shape[0]
            self.train_dataset = CacheDataset(
                                train_csv_dataset,
                              transform=train_transforms,
                              cache_rate=cache_rate_train
                              )
            #CSVDataset(src=data_path, col_names=["image", "masks"], transform=train_transforms)
            self.val_dataset = CacheDataset(
                CSVDataset(self.df_val[["image", "masks"]], col_names=["image", "masks"],
                           transform=monai.transforms.Compose([
                               monai.transforms.LoadImaged(keys=["image", "masks"], reader=GitractReader(),
                                                           image_only=True),
                               monai.transforms.AddChanneld(keys=["image"])])
                           ),
                              transform=val_transforms,
                              cache_rate=cache_rate_val
                              )

            # for d in self.train_dataset:
            #     print(d)


    def _init_transforms(self):
        spatial_size = self.hparams.spatial_size

        mean = np.array([0.456])
        std = np.array([0.224])

        transforms = [
            # monai.transforms.LoadImaged(keys=["image", "masks"]),
            # monai.transforms.LoadImaged(keys=["image", "masks"], reader=GitractReader(), image_only=True),
            # monai.transforms.AddChanneld(keys=["image"]),
            # monai.transforms.EnsureChannelFirstd(keys=["image", "masks"]),
            # # monai.transforms.AsChannelFirstd(keys=["masks"], channel_dim=2),
            # # monai.transforms.Orientationd(keys=["image", "masks"], axcodes="RAS"),
            # # monai.transforms.Spacingd(keys=["image", "masks"], pixdim=(
            # #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            monai.transforms.ScaleIntensityd(keys="image", minv=None, maxv= None, factor=1/255.0 - 1),
            monai.transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            #
            # monai.transforms.RandAdjustContrastd(keys=["image"], prob=0.2),
            monai.transforms.RandGaussianNoised(keys=["image"], prob=0.1),
            #
            # # monai.transforms.RandFlipd(keys=["image", "masks"], prob=0.5),
            # monai.transforms.RandRotate90d(keys=["image", "masks"], prob=0.5),
            monai.transforms.RandRotated(keys=["image", "masks"], range_x=5, range_y=0, prob=0.2),
            # monai.transforms.RandZoomd(keys=["image", "masks"], prob=0.2, min_zoom=0.8, max_zoom=1.2),
            # monai.transforms.Rand2DElasticd(keys=["image", "masks"], magnitude_range=(0, 1), spacing=(0.3, 0.3),
            #                                 prob=0.3),
            #
            # monai.transforms.CropForegroundd(keys=["image", "masks"], source_key="image"),
            # monai.transforms.RandCropByPosNegLabeld(
            #     keys=["image", "masks"],
            #     label_key="masks",
            #     spatial_size=spatial_size,
            #     pos=1,
            #     neg=1,
            #     num_samples=4,
            #     image_key="image",
            #     image_threshold=0,
            # ),

            monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            monai.transforms.RandSpatialCropd(keys=["image", "masks"],
                                              roi_size=spatial_size,
                                              random_size=False),
            monai.transforms.ResizeWithPadOrCropd(keys=["image", "masks"], spatial_size=spatial_size),
            monai.transforms.ToTensord(keys=["image", "masks"]),
        ]

        val_transforms = [
            # monai.transforms.LoadImaged(keys=["image", "masks"], reader=GitractReader(), image_only=True),
            # monai.transforms.EnsureChannelFirstd(keys=["image", "masks"]),
            # monai.transforms.LoadImaged(keys=["image", "masks"]),
            # monai.transforms.AddChanneld(keys=["image"]),
            # monai.transforms.AsChannelFirstd(keys=["masks"], channel_dim=2),
            # monai.transforms.Orientationd(keys=["image", "masks"], axcodes="RAS"),
            # monai.transforms.Spacingd(keys=["image", "masks"], pixdim=(
            #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            monai.transforms.ScaleIntensityd(keys="image", minv=None, maxv=None, factor=1 / 255.0 - 1),
            monai.transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # monai.transforms.ResizeWithPadOrCropd(keys=["image", "masks"], spatial_size=[spatial_size[0],spatial_size[0]]),
            # monai.transforms.CropForegroundd(keys=["image", "masks"], source_key="image"),
            # monai.transforms.CenterSpatialCropd(keys=["image", "masks"], roi_size=[spatial_size[0],spatial_size[0],spatial_size[0]]),
            monai.transforms.ToTensord(keys=["image", "masks"]),
        ]

        test_transforms = [
            monai.transforms.LoadImaged(keys=["image", "masks"], reader=GitractReader(), allow_missing_keys=True),
            monai.transforms.ScaleIntensityd(keys="image", minv=None, maxv=None, factor=1 / 255.0 - 1,
                                             allow_missing_keys=True),
            monai.transforms.NormalizeIntensityd(keys="image", subtrahend=mean, divisor=std, channel_wise=True,
                                                 allow_missing_keys=True),
            monai.transforms.Resized(keys=["image", "masks"], size_mode="longest", spatial_size=spatial_size,
                                     mode="nearest", allow_missing_keys=True),
            monai.transforms.ResizeWithPadOrCropd(keys=["image", "masks"], spatial_size=(spatial_size, spatial_size),
                                                  allow_missing_keys=True),
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
            batch_size=self.hparams.batch_size if train else 1,
            shuffle=train,
            drop_last=train,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )