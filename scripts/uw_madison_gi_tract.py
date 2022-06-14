import argparse
import os
import glob

import mmcv
import numpy as np
import pandas as pd
from pathlib import Path

import cv2
from PIL import Image
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert to 2.5D')
    parser.add_argument('--data-path', help='data path')
    parser.add_argument('-o', '--out-dir', help='output path')
    args = parser.parse_args()
    return args

def parse_train(data_path):
    df_train = pd.read_csv(data_path / "train.csv")
    df_train = df_train.sort_values(["id", "class"]).reset_index(drop = True)
    df_train["patient"] = df_train.id.apply(lambda x: x.split("_")[0])
    df_train["days"] = df_train.id.apply(lambda x: "_".join(x.split("_")[:2]))

    all_image_files = sorted([str(f) for f in data_path.glob("**/scans/*.png")], key = lambda x: x.split("/")[-3] + "_" + x.split("/")[-1])

    size_x = [int(os.path.basename(_)[:-4].split("_")[-4]) for _ in all_image_files]
    size_y = [int(os.path.basename(_)[:-4].split("_")[-3]) for _ in all_image_files]
    spacing_x = [float(os.path.basename(_)[:-4].split("_")[-2]) for _ in all_image_files]
    spacing_y = [float(os.path.basename(_)[:-4].split("_")[-1]) for _ in all_image_files]
    df_train["image_files"] = np.repeat(all_image_files, 3)
    df_train["spacing_x"] = np.repeat(spacing_x, 3)
    df_train["spacing_y"] = np.repeat(spacing_y, 3)
    df_train["size_x"] = np.repeat(size_x, 3)
    df_train["size_y"] = np.repeat(size_y, 3)
    df_train["slice"] = np.repeat([int(os.path.basename(_)[:-4].split("_")[-5]) for _ in all_image_files], 3)
    return df_train

def parse_test(data_path):
    # test_dir = data_path / "test"
    # sub = pd.read_csv(data_path / "sample_submission.csv")
    #
    # test_images = list(test_dir.glob("**/*.png"))
    # assert len(test_images) > 0
    #
    # id2img = {_.rsplit("/", 4)[2] + "_" + "_".join(_.rsplit("/", 4)[4].split("_")[:2]): _ for _ in test_images}
    # sub["file_name"] = sub.id.map(id2img)
    # sub["days"] = sub.id.apply(lambda x: "_".join(x.split("_")[:2]))
    # fname2index = {f + c: i for f, c, i in zip(sub.file_name, sub["class"], sub.index)}
    # print(sub)
    # return sub
    pass

def rle_decode(mask_rle, shape):
    s = np.array(mask_rle.split(), dtype=int)
    starts, lengths = s[0::2] - 1, s[1::2]
    ends = starts + lengths
    h, w = shape
    img = np.zeros((h * w,), dtype = np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = 1
    return img.reshape(shape)

def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def make_2_5_d(df_train, out, stride = 2):
    for day, group in tqdm(df_train.groupby("days")):
        # patient = group.patient.iloc[0]
        imgs = []
        msks = []
        # file_names = []
        for file_name in group.image_files.unique():
            img = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
            segms = group.loc[group.image_files == file_name]
            masks = {}
            for segm, label in zip(segms.segmentation, segms["class"]):
                if not pd.isna(segm):
                    mask = rle_decode(segm, img.shape[:2])
                    masks[label] = mask
                else:
                    masks[label] = np.zeros(img.shape[:2], dtype = np.uint8)
            masks = np.stack([masks[k] for k in sorted(masks)], -1)
            imgs.append(img)
            msks.append(masks)

        imgs = np.stack(imgs, 0)
        msks = np.stack(msks, 0)
        for i in range(msks.shape[0]):
            img = imgs[[max(0, i - stride), i, min(imgs.shape[0] - 1, i + stride)]].transpose(1,2,0) # 2.5d data
            msk = msks[i]
            new_file_name = f"{day}_{i}.png"
            cv2.imwrite(str(out / "images" / new_file_name), img)
            cv2.imwrite(str(out / "labels" / new_file_name), msk)

def split(out):
    # all_image_files = pd.DataFrame([f.relative_to(out) for f in out.glob("images/*.png")], columns=["image"])
    all_image_files = pd.DataFrame([f.absolute() for f in out.glob("images/*.png")], columns=["image"])
    all_image_files["masks"] = [str(f).replace("images", "labels") for f in all_image_files["image"]]
    all_image_files["patients"] = [f.name.split("_")[0] for f in all_image_files["image"]]
    # patients = [f.name.split("_")[0] for f in all_image_files]

    from sklearn.model_selection import GroupKFold

    split = list(GroupKFold(5).split(all_image_files["patients"], groups = all_image_files["patients"]))

    for fold, (train_idx, valid_idx) in enumerate(split):
        all_image_files.iloc[train_idx].to_csv(out / f"splits/fold_{fold}.csv", index=False)
        all_image_files.iloc[train_idx].to_csv(out / f"splits/holdout_{fold}.csv", index=False)

        # with open(out / f"splits/fold_{fold}.txt", "w") as f:
        #     for idx in train_idx:
        #         f.write(all_image_files[idx].stem + "\n")
        # with open(out / f"splits/holdout_{fold}.txt", "w") as f:
        #     for idx in valid_idx:
        #         f.write(all_image_files[idx].stem + "\n")


def main():
    args = parse_args()
    data_path = Path(args.data_path)
    out_dir = Path(args.out_dir) if args.out_dir else data_path
    mmcv.mkdir_or_exist(out_dir)
    mmcv.mkdir_or_exist(out_dir / "images")
    mmcv.mkdir_or_exist(out_dir / "labels")
    mmcv.mkdir_or_exist(out_dir / "splits")

    df_train = parse_train(data_path)
    print(df_train)
    make_2_5_d(df_train, out_dir)
    split(out_dir)

if __name__ == '__main__':
    main()