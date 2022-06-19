import os
import glob
import click
import pytorch_lightning as pl

import numpy as np
import pandas as pd
from pathlib import Path
import monai

import cv2
import torch
from PIL import Image
from monai.data import CSVDataset
from tqdm.auto import tqdm

from gitract.data.data_module import LitDataModule
from gitract.model.runner import LitModule


def parse_test(data_path):
    test_dir = data_path / "test"
    sub = pd.read_csv(data_path / "sample_submission.csv")

    test_images = list(test_dir.glob("**/*.png"))
    if len(test_images) == 0:
        print("No test, using train for debug")
        test_dir = data_path / "train"
        sub = pd.read_csv(data_path / "train.csv")[["id", "class"]].iloc[:50 * 3]
        sub["predicted"] = ""
        test_images = list(test_dir.glob("**/*.png"))

    id2img = {str(_).rsplit("/", 4)[2] + "_" + "_".join(str(_).rsplit("/", 4)[4].split("_")[:2]): _ for _ in test_images}
    sub["file_name"] = sub.id.map(id2img)
    # sub["size_x"] = sub.file_name.apply(lambda x: int(os.path.basename(x)[:-4].split("_")[-4]))
    # sub["size_x"] = sub.file_name.apply(lambda x: int(os.path.basename(x)[:-4].split("_")[-3]))

    for imageid in id2img:
        file_name = str(id2img[imageid])
        s = int(os.path.basename(file_name).split("_")[1])
        file_names = [file_name.replace(f"slice_{s:04d}", f"slice_{s + i:04d}") for i in range(-2, 3)]
        file_names = [_ for _ in file_names if os.path.exists(_)]
        file_names = [file_names[0], file_name, file_names[-1]]
        id2img[imageid] = file_names

    sub["image"] = sub.id.map(id2img)
    sub["days"] = sub.id.apply(lambda x: "_".join(x.split("_")[:2]))

    return sub


def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


@click.command()
@click.option('-d', '--data_path')
@click.option('-o', '--out_path', default="submission.csv")
@click.option('--checkpoint_path')
def main(data_path:str,
         out_path:str,
         checkpoint_path:str = '../models/unet_f0.ckpt'):
    data_path = Path(data_path)

    df_test = parse_test(data_path)
    idclass2index = {f + c: i for f, c, i in zip(df_test.id, df_test["class"], df_test.index)}

    spatial_size = 320
    model = "smpUnet"
    classes = ['large_bowel', 'small_bowel', 'stomach']

    pl.seed_everything(42)

    data_module = LitDataModule(
        data_path=None,
        holdout_path=None,
        batch_size=1,
        num_workers=2,
        spatial_size=(spatial_size, spatial_size),
    )
    _, _, test_transforms = data_module._init_transforms()
    test_dataset = CSVDataset(df_test[["image", "id"]].drop_duplicates(subset=["id"]).reset_index(drop=True),
                              col_names=["image", "id"],
                              transform=test_transforms
                              )
    assert len(test_dataset) == df_test[["image", "id"]].drop_duplicates(subset=["id"]).shape[0]

    litmodule = LitModule(
        model=model,
        background=False
    )
    litmodule.load_state_dict(torch.load(checkpoint_path, map_location=litmodule.device)['state_dict'])
    litmodule.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataset):
            id = batch["id"]
            pred = litmodule.predict_step(batch["image"], batch_idx)[0]
            # print(batch["image_transforms"])
            inverted_pred = test_transforms.inverse(
                {"masks": pred,
                 "masks_transforms": batch["image_transforms"]})["masks"]
            # print(pred.shape, inverted_pred.shape)

            for clz in range(3):
                rle = rle_encode(inverted_pred[clz,...])
                index = idclass2index[id + classes[clz]]
                df_test.loc[index, "predicted"] = rle

    df_test = df_test[["id", "class", "predicted"]]
    # print(df_test.head())
    df_test.to_csv(out_path, index=False)
    return df_test

if __name__ == '__main__':
    np.random.seed(42)
    main()