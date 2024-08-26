import os
from pathlib import Path
from PIL import Image
import random
import pickle
import numpy as np
import cv2

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

        image_a = self.transform(Image.open(path_a))
        image_b = self.transform(Image.open(path_b))

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

        image_a = self.transform(Image.open(path_a))
        image_b = self.transform(Image.open(path_b))

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

        image_a = self.transform(Image.open(path_a))
        image_p = self.transform(Image.open(path_p))
        image_n = self.transform(Image.open(path_n))

        return torch.stack([image_a, image_p, image_n], dim=0)


class TripletDatasetAugmentedPositives(TripletDataset):
    def __getitem__(self, index):
        combination_index = index // self.min_length
        image_index = index % self.min_length
        # the negative example is from the same camera as the anchor and randomly
        # sampled excluding a margin around the anchor image
        margin = 10
        random_range = self.min_length - 2 * margin
        random_offset = torch.randint(1, random_range, (1,)).item()
        random_index = (image_index + margin + random_offset) % self.min_length

        camera_a, camera_p = self.combinations[combination_index]

        path_a = self.paths[camera_a][image_index]
        path_n = self.paths[camera_a][random_index]

        # positive example is either same frame from another camera or a close frame
        # from the same camera
        if random.random() < 0.2:
            disturbance = random.randint(-margin / 2, margin / 2)
            distubed_index = (image_index + disturbance) % self.min_length
            path_p = self.paths[camera_a][distubed_index]
        else:
            path_p = self.paths[camera_p][image_index]

        image_a = self.transform(Image.open(path_a))
        image_p = self.transform(Image.open(path_p))
        image_n = self.transform(Image.open(path_n))

        return torch.stack([image_a, image_p, image_n], dim=0)


class ImageDataset(Dataset):
    def __init__(self, folder, image_size, camera):
        super().__init__()
        self.folder = folder
        self.camera = camera
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        self.paths = []
        sorted_paths = sorted(Path(folder, camera).glob("*"))
        for path in sorted_paths:
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)
        print(f"{len(self.paths)} images found for {camera}")

    def __len__(self):
        return len(self.paths) * 10

    def __getitem__(self, index):
        index = index % len(self.paths)
        image = self.transform(Image.open(self.paths[index]))
        return image


class ImageDatasetEncDec(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        camera_x,
        camera_y,
        data_multiplier=1,
        data_percentage=1.0,
    ):
        super().__init__()
        self.folder = folder
        self.data_multiplier = data_multiplier
        self.data_percentage = data_percentage
        self.camera_x = camera_x
        self.camera_y = camera_y
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        self.paths = {
            camera_x: [],
            camera_y: [],
        }
        for camera in self.paths.keys():
            sorted_paths = sorted(Path(folder, camera).glob("*"))
            for path in sorted_paths:
                _, ext = os.path.splitext(path)
                if ext.lower() in IMAGE_EXTS:
                    self.paths[camera].append(path)
            print(f"{len(self.paths[camera])} images found for {camera}")
        self.min_length = int(
            min(len(paths) for paths in self.paths.values()) * self.data_percentage
        )

    def __len__(self):
        return self.min_length * self.data_multiplier

    def __getitem__(self, index):
        index = index % self.min_length
        x = self.transform(Image.open(self.paths[self.camera_x][index]))
        y = self.transform(Image.open(self.paths[self.camera_y][index]))

        return torch.stack([x, y], dim=0)


class TwoImageDataset(ImageDataset):
    def __init__(self, folder, image_size, camera, camera2):
        super().__init__(folder, image_size, camera)
        self.camera2 = camera2
        self.paths2 = []
        sorted_paths2 = sorted(Path(folder, camera2).glob("*"))
        for path in sorted_paths2:
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths2.append(path)
        print(f"{len(self.paths2)} images found for {camera2}")

    def __getitem__(self, index):
        index = index % len(self.paths)
        image = self.transform(Image.open(self.paths[index]))
        image2 = self.transform(Image.open(self.paths2[index]))
        return image, image2


class ImagePoseDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        num_negatives=2,
        ratio_positives=1.0,
        threshold_positives=0.0,
        threshold_negatives=0.0,
        data_percentage=1.0,
        data_multiplier=1,
        paths=None,
        combinations=None,
        transform=None,
    ):
        super().__init__()
        self.folder = folder
        self.num_negatives = num_negatives
        self.ratio_positives = ratio_positives
        self.threshold_positives = threshold_positives
        self.threshold_negatives = threshold_negatives
        self.data_percentage = data_percentage
        self.data_multiplier = data_multiplier
        if paths is None:
            self.paths = {
                "camera_1": [],
                "camera_2": [],
                "camera_3": [],
                "camera_4": [],
            }
        else:
            self.paths = paths
        if combinations is None:
            self.combinations = [
                # ("camera_1", "camera_2"),
                # ("camera_1", "camera_3"),
                # ("camera_1", "camera_4"),
                # ("camera_2", "camera_1"),
                # ("camera_2", "camera_3"),
                ("camera_2", "camera_4"),
                # ("camera_3", "camera_1"),
                # ("camera_3", "camera_2"),
                # ("camera_3", "camera_4"),
                # ("camera_4", "camera_1"),
                ("camera_4", "camera_2"),
                # ("camera_4", "camera_3"),
            ]
        else:
            self.combinations = combinations
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transform

        with open(Path(folder, "poses.pkl"), "rb") as f:
            self.poses = pickle.load(f)
        for camera in self.paths.keys():
            sorted_paths = sorted(Path(folder, camera).glob("*"))
            for path in sorted_paths:
                _, ext = os.path.splitext(path)
                if ext.lower() in IMAGE_EXTS:
                    self.paths[camera].append(path)
            print(f"{len(self.paths[camera])} images found for {camera}")
        self.min_length = int(
            min(len(paths) for paths in self.paths.values()) * self.data_percentage
        )
        self.length = self.min_length * len(self.combinations)
        print(f"Number of samples: {self.length}")

        # compute distances between all poses
        poses_torch = torch.tensor(self.poses.values, dtype=torch.float32)
        diff = poses_torch.unsqueeze(1) - poses_torch.unsqueeze(0)
        diff *= torch.tensor(
            [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1],
            dtype=torch.float32,
        )
        self.distances = (diff**2).sum(-1).sqrt()

    def __len__(self):
        return self.length * self.data_multiplier

    def __getitem__(self, index):
        index = index % self.length
        combination_index = index // self.min_length
        image_index = index % self.min_length

        camera_a, camera_b = self.combinations[combination_index]

        # anchor
        path_a = self.paths[camera_a][image_index]
        image_a = self.transform(Image.open(path_a))

        # positive
        if random.random() < self.ratio_positives:
            # same frame different camera
            path_p = self.paths[camera_b][image_index]
        else:
            raise ValueError("not useful for object data!")
            # different frame same camera
            dist = 999 * self.threshold_positives
            while dist > self.threshold_positives:
                random_index = torch.randint(0, self.min_length, (1,)).item()
                dist = self.distances[image_index, random_index]
            path_p = self.paths[camera_a][random_index]
        image_p = self.transform(Image.open(path_p))

        # negative
        images_n = []
        for i in range(self.num_negatives):
            dist = -1
            while dist < self.threshold_negatives:
                random_index = torch.randint(0, self.min_length, (1,)).item()
                dist = self.distances[image_index, random_index]
            path_n = self.paths[camera_a][random_index]
            images_n.append(self.transform(Image.open(path_n)))

        # import matplotlib.pyplot as plt

        # fig, axes = plt.subplots(1, 2 + self.num_negatives, figsize=(20, 5))
        # axes[0].imshow(image_a.permute(1, 2, 0))
        # axes[1].imshow(image_p.permute(1, 2, 0))
        # for i, image_n in enumerate(images_n):
        #     axes[2 + i].imshow(image_n.permute(1, 2, 0))
        # plt.show()

        return torch.stack([image_a, image_p, *images_n], dim=0)


class PickleFileDataset(Dataset):
    def __init__(self, file_path, image_size):
        super().__init__()
        with open(file_path, "rb") as f:
            self.data = pickle.load(f)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        images = row["observations", "images"]
        state = row["observations", "state"]
        action = row["policy_action", "action"]

        camera_1 = self.transform(Image.fromarray(images[:240, :, ::-1]))
        camera_2 = self.transform(Image.fromarray(images[240:480, :, ::-1]))
        camera_3 = self.transform(Image.fromarray(images[480:720, :, ::-1]))

        return camera_1, camera_2, camera_3, state, action


class SpartanFileDataset(Dataset):
    def __init__(self, file_path, target_freq_ratio, width, height, image_size):
        super().__init__()
        self.target_freq_ratio = target_freq_ratio
        self.width = width  # needed for first resize to match legacy pipeline
        self.height = height  # needed for first resize to match legacy pipeline
        obs = np.load(os.path.join(file_path, "observations.npz"))
        actions = np.load(os.path.join(file_path, "actions.npz"))

        self.image_1 = obs["camera__image__1"]
        self.image_2 = obs["camera__image__3"]
        self.image_3 = obs["camera__image__2"]
        self.pos_left = obs["robot__actual__joint_position__left"]
        self.pos_right = obs["robot__actual__joint_position__right"]
        self.vel_left = obs["robot__actual__joint_velocity__left"]
        self.vel_right = obs["robot__actual__joint_velocity__right"]
        self.actions = actions["actions"]

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_1) // self.target_freq_ratio

    def __getitem__(self, index):
        index *= self.target_freq_ratio
        camera_1 = self.transform(
            Image.fromarray(
                cv2.resize(
                    self.image_1[index],
                    (self.width, self.height),
                    interpolation=cv2.INTER_LINEAR,
                )
            )
        )
        camera_2 = self.transform(
            Image.fromarray(
                cv2.resize(
                    self.image_2[index],
                    (self.width, self.height),
                    interpolation=cv2.INTER_LINEAR,
                )
            )
        )
        camera_3 = self.transform(
            Image.fromarray(
                cv2.resize(
                    self.image_3[index],
                    (self.width, self.height),
                    interpolation=cv2.INTER_LINEAR,
                )
            )
        )
        state = np.concatenate(
            [
                self.pos_left[index],
                self.pos_right[index],
                self.vel_left[index],
                self.vel_right[index],
            ]
        )
        action = self.actions[index]
        return camera_1, camera_2, camera_3, state, action
