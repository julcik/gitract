import os

import numpy as np
import pandas as pd

CLASSES = ["large_bowel", "small_bowel", "stomach"]

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


def get_slice(path):
    return int(os.path.basename(path)[:-4].split("_")[-5])

def parse_train(data_path, slices=3, stride = 2):
    df_train = pd.read_csv(data_path / "train.csv")
    df_train['segmentation'] = df_train.segmentation.fillna('')
    df_train = df_train.set_index(["id", "class"]).unstack()
    df_train = df_train.droplevel(None, axis=1)
    df_train = df_train.reset_index()

    df_train["patient"] = df_train.id.apply(lambda x: x.split("_")[0])
    df_train["days"] = df_train.id.apply(lambda x: "_".join(x.split("_")[:2]))

    all_image_files = {str(f).split("/")[-3] + "_" + "_".join(str(f).split("/")[-1].split("_")[:2]) : str(f) for f in data_path.glob("**/scans/*.png")}

    # size_x = [int(os.path.basename(_)[:-4].split("_")[-4]) for _ in all_image_files]
    # size_y = [int(os.path.basename(_)[:-4].split("_")[-3]) for _ in all_image_files]
    # spacing_x = [float(os.path.basename(_)[:-4].split("_")[-2]) for _ in all_image_files]
    # spacing_y = [float(os.path.basename(_)[:-4].split("_")[-1]) for _ in all_image_files]

    df_train["image_files"] = df_train.id.map(all_image_files) #np.repeat(all_image_files, 3)
    # df_train["spacing_x"] = np.repeat(spacing_x, 3)
    # df_train["spacing_y"] = np.repeat(spacing_y, 3)
    df_train["shape"] = df_train["image_files"].map(lambda s: "[" + ",".join([os.path.basename(s)[:-4].split("_")[-3], os.path.basename(s)[:-4].split("_")[-4]])+"]")

    for cl in CLASSES:
        df_train[cl] = df_train[cl] + df_train["shape"]

    df_train["masks"] = df_train[CLASSES].values.tolist()

    # df_train["slice"] = np.repeat([get_slice(path) for path in all_image_files], 3)

    # df_train['rle_len'] = df_train.segmentation.map(len)
    # df_train['empty'] = (df_train.rle_len == 0)

    id2img = {id: path for id, path in df_train[["id","image_files"]].drop_duplicates().values}
    # print(id2img)
    id2imgs = {}

    half_dist = slices // 2
    for id, path in id2img.items():
        s = get_slice(path)

        file_names = [path.replace(f"slice_{s:04d}", f"slice_{max(0,s + i):04d}") for i in range(-stride * half_dist, 1 + stride * half_dist, stride)]

        for i in range(half_dist-1,-1,-1):
            if not os.path.exists(file_names[i]):
                file_names[i] = file_names[i+1]
        for i in range(half_dist+1,2 * half_dist+1):
            if not os.path.exists(file_names[i]):
                file_names[i] = file_names[i-1]
        id2imgs[id] = file_names

    df_train["image"] = df_train.id.map(id2imgs)

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