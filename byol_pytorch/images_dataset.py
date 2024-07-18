import os
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset

IMAGE_EXTS = [".jpg", ".png", ".jpeg"]


class TwoImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.paths = {
            "camera_1": [],
            "camera_2": [],
            "camera_3": [],
            "camera_4": [],
        }
        self.combinations = [
            ("camera_1", "camera_2"),
            ("camera_1", "camera_3"),
            ("camera_1", "camera_4"),
            ("camera_2", "camera_3"),
            ("camera_2", "camera_4"),
            ("camera_3", "camera_4"),
        ]
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        for camera in self.paths.keys():
            sorted_paths = sorted(Path(folder, camera).glob("*"))
            for path in sorted_paths:
                _, ext = os.path.splitext(path)
                if ext.lower() in IMAGE_EXTS:
                    self.paths[camera].append(path)
            print(f"{len(self.paths[camera])} images found for {camera}")
        self.min_length = min(len(paths) for paths in self.paths.values())
        print(f"Number of samples: {self.min_length * len(self.combinations)}")

    def __len__(self):
        return self.min_length * len(self.combinations)

    def __getitem__(self, index):
        combination_index = index // self.min_length
        image_index = index % self.min_length

        camera_a, camera_b = self.combinations[combination_index]

        path_a = self.paths[camera_a][image_index]
        path_b = self.paths[camera_b][image_index]

        image_a = self.transform(Image.open(path_a).convert("RGB"))
        image_b = self.transform(Image.open(path_b).convert("RGB"))

        return image_a, image_b


class TwoImagesStackedDataset(TwoImagesDataset):
    def __getitem__(self, index):
        image_a, image_b = super().__getitem__(index)
        return torch.stack([image_a, image_b], dim=0)


class TwoImagesLabelledDataset(TwoImagesDataset):
    def __init__(self, folder, image_size):
        super().__init__(folder, image_size)
        self.combinations = [
            ("camera_1", "camera_2"),
            ("camera_3", "camera_4"),
        ]

    def __getitem__(self, index):
        combination_index = index // self.min_length
        image_index = index % self.min_length

        camera_a, camera_b = self.combinations[combination_index]

        path_a = self.paths[camera_a][image_index]
        path_b = self.paths[camera_b][image_index]

        image_a = self.transform(Image.open(path_a).convert("RGB"))
        image_b = self.transform(Image.open(path_b).convert("RGB"))

        return image_a, image_b, camera_a, camera_b


class TripletDataset(Dataset):

    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.paths = {
            "camera_1": [],
            "camera_2": [],
            "camera_3": [],
            "camera_4": [],
        }
        self.combinations = [
            ("camera_1", "camera_2"),
            ("camera_1", "camera_3"),
            ("camera_1", "camera_4"),
            ("camera_2", "camera_1"),
            ("camera_2", "camera_3"),
            ("camera_2", "camera_4"),
            ("camera_3", "camera_1"),
            ("camera_3", "camera_2"),
            ("camera_3", "camera_4"),
            ("camera_4", "camera_1"),
            ("camera_4", "camera_2"),
            ("camera_4", "camera_3"),
        ]
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        for camera in self.paths.keys():
            sorted_paths = sorted(Path(folder, camera).glob("*"))
            for path in sorted_paths:
                _, ext = os.path.splitext(path)
                if ext.lower() in IMAGE_EXTS:
                    self.paths[camera].append(path)
            print(f"{len(self.paths[camera])} images found for {camera}")
        self.min_length = min(len(paths) for paths in self.paths.values())
        print(f"Number of samples: {self.min_length * len(self.combinations)}")

    def __len__(self):
        return self.min_length * len(self.combinations)

    def __getitem__(self, index):
        combination_index = index // self.min_length
        image_index = index % self.min_length
        # the negative example is from the same camera as the anchor and randomly
        # sampled excluding a margin around the anchor image
        margin = 10  # 1 second
        random_range = self.min_length - 2 * margin
        random_offset = torch.randint(1, random_range, (1,)).item()
        random_index = (image_index + margin + random_offset) % self.min_length

        camera_a, camera_p = self.combinations[combination_index]

        path_a = self.paths[camera_a][image_index]
        path_p = self.paths[camera_p][image_index]
        path_n = self.paths[camera_a][random_index]

        image_a = self.transform(Image.open(path_a).convert("RGB"))
        image_p = self.transform(Image.open(path_p).convert("RGB"))
        image_n = self.transform(Image.open(path_n).convert("RGB"))

        return torch.stack([image_a, image_p, image_n], dim=0)
