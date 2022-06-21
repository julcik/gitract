import os
import glob
import click

import numpy as np
import pandas as pd
from pathlib import Path

import cv2
from PIL import Image
from tqdm.auto import tqdm

from gitract.data.data_module import image_reader
from gitract.data.utils import rle_decode, parse_train


def make_2_5_d(df_train, out, stride = 2, add_background=False):
    for day, group in tqdm(df_train.groupby("days")):
        # patient = group.patient.iloc[0]
        imgs = []
        msks = []
        # file_names = []
        for file_name in group.image_files.unique():

            img = image_reader(file_name, numpy=True)

            segms = group.loc[group.image_files == file_name]
            masks = {}
            for segm, label in zip(segms.segmentation, segms["class"]):
                if not pd.isna(segm):
                    mask = rle_decode(segm, img.shape[:2])
                    masks[label] = mask
                else:
                    masks[label] = np.zeros(img.shape[:2], dtype = np.uint8)
            masks = np.stack([masks[k] for k in sorted(masks)], -1)
            if add_background:
                bg_mask = 1 - masks.max(-1, keepdims=True)
                masks = np.concatenate([bg_mask, masks], axis=-1)

            imgs.append(img)
            msks.append(masks)

        imgs = np.stack(imgs, 0)
        msks = np.stack(msks, 0)
        slices = group.slice.unique().tolist()
        for i in range(msks.shape[0]):
            img = imgs[[max(0, i - stride), i, min(imgs.shape[0] - 1, i + stride)]].transpose(1, 2, 0)  # 2.5d data
            msk = msks[i]

            new_file_name = f"{day}_{slices[i]}.png"

            # RGBA <-> BGRA problem, mixed classes
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if add_background:
                msk = cv2.cvtColor(msk, cv2.COLOR_RGBA2BGRA)
            else:
                msk = cv2.cvtColor(msk, cv2.COLOR_RGB2BGR)

            cv2.imwrite(str(out / "images" / new_file_name), img)
            cv2.imwrite(str(out / "labels" / new_file_name), msk)


def is_empty(mask_file):
    msk = cv2.imread(mask_file)
    return np.sum(msk[:, :, -3:]) == 0

def split(out, downsample_empty=1.0):
    # all_image_files = pd.DataFrame([f.relative_to(out) for f in out.glob("images/*.png")], columns=["image"])
    all_image_files = pd.DataFrame([f.absolute() for f in out.glob("images/*.png")], columns=["image"])
    all_image_files["masks"] = [str(f).replace("images", "labels") for f in all_image_files["image"]]
    all_image_files["key"] = [f.name for f in all_image_files["image"]]
    all_image_files["patients"] = [f.name.split("_")[0] for f in all_image_files["image"]]
    # patients = [f.name.split("_")[0] for f in all_image_files]
    all_image_files["is_empty"] = all_image_files["masks"].apply(is_empty)

    from sklearn.model_selection import GroupKFold

    split = list(GroupKFold(5).split(all_image_files["patients"], groups = all_image_files["patients"]))

    for fold, (train_idx, valid_idx) in enumerate(split):
        train = all_image_files.iloc[train_idx]
        if downsample_empty<1:
            train = pd.concat((train[~train.is_empty],train[train.is_empty].sample(frac=downsample_empty, random_state=42)), axis=0)
        train.to_csv(out / f"splits/fold_{fold}.csv", index=False)

        all_image_files.iloc[valid_idx].to_csv(out / f"splits/holdout_{fold}.csv", index=False)

        # with open(out / f"splits/fold_{fold}.txt", "w") as f:
        #     for idx in train_idx:
        #         f.write(all_image_files[idx].stem + "\n")
        # with open(out / f"splits/holdout_{fold}.txt", "w") as f:
        #     for idx in valid_idx:
        #         f.write(all_image_files[idx].stem + "\n")


@click.command()
@click.option('-d', '--data_path')
@click.option('-o', '--out_dir')
@click.option('--add_background', default=False)
@click.option('--downsample_empty', default=1.0)
def main(data_path:str,
         out_dir:str,
         add_background: bool,
         downsample_empty: float):
    data_path = Path(data_path)
    out_dir = Path(out_dir) if out_dir else data_path
    out_dir.mkdir(exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    (out_dir / "labels").mkdir(exist_ok=True)
    (out_dir / "splits").mkdir(exist_ok=True)

    df_train = parse_train(data_path)
    make_2_5_d(df_train, out_dir, add_background=add_background)
    split(out_dir, downsample_empty=downsample_empty)

if __name__ == '__main__':
    np.random.seed(42)
    main()