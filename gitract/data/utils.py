import os

import numpy as np
import pandas as pd


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
    df_train['segmentation'] = df_train.segmentation.fillna('')
    df_train['rle_len'] = df_train.segmentation.map(len)
    df_train['empty'] = (df_train.rle_len == 0)
    return df_train


def parse_test(data_path):
    test_dir = data_path / "test"
    sub = pd.read_csv(data_path / "sample_submission.csv")

    test_images = list(test_dir.glob("**/*.png"))
    if len(test_images) == 0:
        print("No test, using train for debug")
        test_dir = data_path / "train"
        sub = pd.read_csv(data_path / "train.csv")[["id", "class"]].iloc[:200 * 3]
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