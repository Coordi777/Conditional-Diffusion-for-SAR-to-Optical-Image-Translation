import random
import pandas as pd
import blobfile as bf
import math
import numpy as np
from PIL import Image
from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_dir_sar,
    data_dir_opt,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    print("data_dir_sar: ", data_dir_sar)
    print("data_dir_opt: ", data_dir_opt)
    if not data_dir_sar or not data_dir_opt:
        raise ValueError("unspecified data directory")

    # all_files_sar = _list_image_files_recursively(data_dir_sar)
    # all_files_opt = _list_image_files_recursively(data_dir_opt)
    classes = None
    #
    # all_files_sar.sort(key=lambda x: int(x[len(data_dir_sar) + 1:].split('.')[0]))
    # all_files_opt.sort(key=lambda x: int(x[len(data_dir_opt) + 1:].split('.')[0]))
    sar_path = pd.read_csv('/remote-home/bxy/data/Sen1_2/sar_path_train.csv')
    opt_path = pd.read_csv('/remote-home/bxy/data/Sen1_2/opt_path_train.csv')
    sar_path = sar_path.values.tolist()
    opt_path = opt_path.values.tolist()
    # 去掉多余维度
    for i in range(len(sar_path)):
        sar_path[i] = sar_path[i][0]
        opt_path[i] = opt_path[i][0]

    all_files_sar = sar_path
    all_files_opt = opt_path

    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in data_dir_opt]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files_sar,
        all_files_opt,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths_sar,
        image_paths_opt,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images_sar = image_paths_sar[shard:][::num_shards]
        self.local_images_opt = image_paths_opt[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images_sar)

    def __getitem__(self, idx):
        path_sar = self.local_images_sar[idx]
        path_opt = self.local_images_opt[idx]
        with bf.BlobFile(path_sar, "rb") as f:
            pil_image_sar = Image.open(f)
            pil_image_sar.load()
        pil_image_sar = pil_image_sar.convert("RGB")
        with bf.BlobFile(path_opt, "rb") as f:
            pil_image_opt = Image.open(f)
            pil_image_opt.load()
        pil_image_opt = pil_image_opt.convert("RGB")

        if self.random_crop:
            arr_sar = random_crop_arr(pil_image_sar, self.resolution)
            arr_opt = random_crop_arr(pil_image_opt, self.resolution)
        else:
            arr_sar = center_crop_arr(pil_image_sar, self.resolution)
            arr_opt = center_crop_arr(pil_image_opt, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr_sar = arr_sar[:, ::-1]
            arr_opt = arr_opt[:, ::-1]

        arr_sar = arr_sar.astype(np.float32) / 127.5 - 1
        arr_opt = arr_opt.astype(np.float32) / 127.5 - 1
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        arr_sar = np.transpose(arr_sar, [2, 0, 1])
        arr_opt = np.transpose(arr_opt, [2, 0, 1])
        arr = np.concatenate((arr_sar, arr_opt), axis=0)
        return arr, out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
