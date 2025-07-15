import io
import json

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
import webdataset as wds
from huggingface_hub import get_token
from torch.utils.data import IterableDataset


class IndexedTarDataset(Dataset):
    def __init__(
        self,
        imagenet_tar,
        imagenet_index,
        size=None,
        random_crop=False,
        aug_mode="default",
    ):
        self.size = size
        self.random_crop = random_crop

        self.aug_mode = aug_mode

        if aug_mode == "default":
            assert self.size is not None and self.size > 0
            self.rescaler = T.Resize(self.size)
            if not self.random_crop:
                self.cropper = T.CenterCrop((self.size, self.size))
            else:
                self.cropper = T.RandomCrop((self.size, self.size))
            self.preprocessor = T.Compose([self.rescaler, self.cropper])
        else:
            raise NotImplementedError

        # Tar setup
        self.imagenet_tar = imagenet_tar
        self.imagenet_index = imagenet_index
        with open(self.imagenet_index, "r") as fp:
            self.index = json.load(fp)
        self.index = sorted(self.index, key=lambda d: d["name"].split("/")[-1])
        self.id_to_handle = {}

    def __len__(self):
        return len(self.index)

    def get_image(self, image_info):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        if worker_id not in self.id_to_handle:
            self.id_to_handle[worker_id] = open(self.imagenet_tar, "rb")
        handle = self.id_to_handle[worker_id]

        handle.seek(image_info["offset"])
        img_bytes = handle.read(image_info["size"])
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image.load()
        return image

    def preprocess_image(self, image_info):
        image = self.get_image(image_info)
        image = self.preprocessor(image)
        image = np.array(image)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.index[i])
        return example


class ImageNetWebDataset(IterableDataset):
    def __init__(
        self,
        imagenet_tar=None,
        imagenet_index=None,
        split="train",
        size=None,
        random_crop=False,
        aug_mode="default",
        buffer_size=1000,
    ):
        self.size = size
        self.random_crop = random_crop
        self.aug_mode = aug_mode

        if aug_mode == "default":
            assert self.size is not None and self.size > 0
            self.rescaler = T.Resize(self.size)
            if not self.random_crop:
                self.cropper = T.CenterCrop((self.size, self.size))
            else:
                self.cropper = T.RandomCrop((self.size, self.size))
            self.preprocessor = T.Compose([self.rescaler, self.cropper])
        else:
            raise NotImplementedError

        hf_token = get_token()
        if hf_token is None:
            raise ValueError(
                "Hugging Face token not found. Please login using `huggingface-cli login`."
            )

        if split == "train":
            self.url = "https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/imagenet1k-train-{{0000..1023}}.tar"
            self.length = 1281167
        elif split == "val" or split == "validation":
            self.url = "https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/imagenet1k-validation-{{0000..0063}}.tar"
            self.length = 50000
        else:
            raise ValueError(f"Unknown split: {split}")

        # Use curl to get around redirection issues
        self.url = f"pipe:curl -s -L '{self.url}' -H 'Authorization:Bearer {hf_token}'"

        self.dataset = (
            wds.WebDataset(self.url, shardshuffle=100, nodesplitter=wds.split_by_node)
            .shuffle(buffer_size)
            .decode("pil")
            .rename(image="jpg;jpeg", label="cls")
            .map(self.process_sample)
            .with_length(self.length)
        )

    def process_sample(self, sample):
        image = self.preprocessor(sample["image"])
        image = np.array(image)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return {"image": image}

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return self.length
