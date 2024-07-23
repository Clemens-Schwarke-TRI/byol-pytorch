import numpy as np
import torch
import time
import argparse
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import multiprocessing

from MulticoreTSNE import MulticoreTSNE as TSNE
from scipy.spatial.distance import cdist
from torch import nn
from torchvision import transforms as T
from torchvision import models

from byol_pytorch import BYOL, TwoImagesLabelledDataset

# arguments
parser = argparse.ArgumentParser(description="plot_byol")
parser.add_argument(
    "--image_folder",
    type=str,
    required=True,
    help="path to your folder of images for self-supervised learning",
)
parser.add_argument(
    "--plot",
    type=str,
    choices=["cameras", "points", "distance", "live_plot", "live_plot_2"],
    default="cameras",
    help="plot cameras or points",
)
parser.add_argument(
    "--use_saved_projections",
    action="store_true",
    help="use saved projections",
)
parser.add_argument(
    "--use_saved_tsne",
    action="store_true",
    help="use saved t-SNE datapoints",
)
args = parser.parse_args()

# constants
IMAGE_SIZE = 256
BATCH_SIZE = 64
LR = 3e-4


# pytorch lightning module
class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        image_a = images[:, 0]
        image_b = images[:, 1]
        return self.learner(image_a, image_b)

    def training_step(self, images, _):
        loss = self.forward(images)
        self.log("loss", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()


# main
if __name__ == "__main__":
    # load model and compute projections
    if not args.use_saved_projections and not args.use_saved_tsne:
        # create dataset
        dataset = TwoImagesLabelledDataset(args.image_folder, IMAGE_SIZE)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=BATCH_SIZE,
            num_workers=multiprocessing.cpu_count(),
        )

        # create model
        net = models.resnet50()
        model = SelfSupervisedLearner.load_from_checkpoint(
            "/home/clemensschwarke/git/byol-pytorch/lightning_logs/version_64_only_cam_2_and_4/checkpoints/epoch=85-step=8686.ckpt",
            net=net,
            image_size=IMAGE_SIZE,
            hidden_layer="avgpool",
        )
        model.learner.augment1 = model.learner.augment2 = nn.Sequential(
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]),
            ),
        )
        model.eval()

        with torch.no_grad():
            # play model
            projections = {
                "camera_1": [],
                "camera_2": [],
                "camera_3": [],
                "camera_4": [],
            }
            start = time.time()
            for idx, (image_a, image_b, camera_a, camera_b) in enumerate(dataloader):
                image_a, image_b = image_a.to(model.device), image_b.to(model.device)
                image_a_out, image_b_out = model.learner(
                    image_a, image_b, return_embedding=True
                )
                projection_a = image_a_out[0]
                projection_b = image_b_out[0]

                # projection_a = model.learner.online_predictor(projection_a)
                # projection_b = model.learner.online_predictor(projection_b)

                for i in range(len(camera_a)):
                    projections[camera_a[i]].append(projection_a[i])
                    projections[camera_b[i]].append(projection_b[i])
                print(f"Step {idx+1} of {len(dataloader)}")
            print(f"Projections took {time.time() - start:.2f} seconds")

            # convert to numpy
            projections_camera_1 = torch.stack(projections["camera_1"]).cpu().numpy()
            projections_camera_2 = torch.stack(projections["camera_2"]).cpu().numpy()
            projections_camera_3 = torch.stack(projections["camera_3"]).cpu().numpy()
            projections_camera_4 = torch.stack(projections["camera_4"]).cpu().numpy()

            # save projections
            np.save("projections/projections_camera_1.npy", projections_camera_1)
            np.save("projections/projections_camera_2.npy", projections_camera_2)
            np.save("projections/projections_camera_3.npy", projections_camera_3)
            np.save("projections/projections_camera_4.npy", projections_camera_4)

    if args.use_saved_projections or args.use_saved_tsne:
        # load saved projections
        projections_camera_1 = np.load("projections/projections_camera_1.npy")
        projections_camera_2 = np.load("projections/projections_camera_2.npy")
        projections_camera_3 = np.load("projections/projections_camera_3.npy")
        projections_camera_4 = np.load("projections/projections_camera_4.npy")

    projections_all = np.vstack(
        [
            projections_camera_1,
            projections_camera_2,
            projections_camera_3,
            projections_camera_4,
        ]
    )

    # t-SNE visualization
    labels = np.array(
        [0] * len(projections_camera_1)
        + [1] * len(projections_camera_2)
        + [2] * len(projections_camera_3)
        + [3] * len(projections_camera_4)
    )

    if not args.use_saved_tsne:
        projections_norm = projections_all / np.linalg.norm(
            projections_all, axis=1, keepdims=True
        )
        tsne = TSNE(n_components=2, random_state=42, n_jobs=multiprocessing.cpu_count())
        print("Fitting t-SNE ...")
        start = time.time()
        projections_tsne = tsne.fit_transform(projections_norm)
        print(f"t-SNE took {time.time() - start:.2f} seconds")

        np.save("projections/projections_tsne.npy", projections_tsne)
    else:
        projections_tsne = np.load("projections/projections_tsne.npy")

    if args.plot == "cameras":
        plt.figure(figsize=(10, 8))
        plt.scatter(
            projections_tsne[labels == 0, 0],
            projections_tsne[labels == 0, 1],
            label="camera_1",
            alpha=0.1,
        )
        plt.scatter(
            projections_tsne[labels == 1, 0],
            projections_tsne[labels == 1, 1],
            label="camera_2",
            alpha=0.1,
        )
        plt.scatter(
            projections_tsne[labels == 2, 0],
            projections_tsne[labels == 2, 1],
            label="camera_3",
            alpha=0.1,
        )
        plt.scatter(
            projections_tsne[labels == 3, 0],
            projections_tsne[labels == 3, 1],
            label="camera_4",
            alpha=0.1,
        )
        plt.legend()
        plt.title("t-SNE Visualization of 256-Dimensional Projections")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.show()

    elif args.plot == "points":
        plt.figure(figsize=(10, 8))
        num_points = 100
        colormap = cm.get_cmap("viridis", num_points)
        for i in range(num_points):
            color = colormap(i)
            plt.scatter(
                projections_tsne[i :: len(projections_camera_1), 0],
                projections_tsne[i :: len(projections_camera_1), 1],
                color=color,
                alpha=0.5,
                s=100,
            )
        plt.title(
            "t-SNE Visualization of 256-Dimensional Projections with Corresponding Points"
        )
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.show()

    elif args.plot == "distance":
        # compute pairwise distances
        num_points = len(projections_camera_1)
        camera_base = "camera_1"
        distances = {
            "camera_1": np.zeros((num_points, num_points)),
            "camera_2": np.zeros((num_points, num_points)),
            "camera_3": np.zeros((num_points, num_points)),
            "camera_4": np.zeros((num_points, num_points)),
        }
        projections = {
            "camera_1": projections_camera_1,
            "camera_2": projections_camera_2,
            "camera_3": projections_camera_3,
            "camera_4": projections_camera_4,
        }
        # normalize the projections
        normalized_projections = {
            camera: projections[camera]
            / np.linalg.norm(projections[camera], axis=1, keepdims=True)
            for camera in projections
        }
        # compute cosine similarities
        for camera in distances:
            distances[camera] = 1 - cdist(
                normalized_projections[camera_base],
                normalized_projections[camera],
                "cosine",
            )
        # plot distances
        fig, axs = plt.subplots(2, 2, figsize=(20, 16))
        axs_flat = axs.flatten()
        for idx, (camera, dist_matrix) in enumerate(distances.items()):
            im = axs_flat[idx].imshow(dist_matrix, cmap="viridis")
            axs_flat[idx].set_title(f"Pairwise Cosine Similarity: {camera}")
            fig.colorbar(im, ax=axs_flat[idx])
        plt.suptitle("Pairwise Cosine Similarity of 256-Dimensional Projections")
        plt.tight_layout()
        plt.show()

    elif args.plot == "live_plot":
        disp_camera = 2
        dataset = TwoImagesLabelledDataset(args.image_folder, IMAGE_SIZE)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
            num_workers=multiprocessing.cpu_count(),
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        idx = 0
        for image_a, image_b, camera_a, camera_b in dataloader:
            if disp_camera == 1:
                image = image_a
                camera = camera_a
                if idx == 0:
                    projections = projections_tsne[0 : len(projections_camera_1)]
                if idx > len(projections_camera_1):
                    break
            elif disp_camera == 2:
                image = image_b
                camera = camera_b
                if idx == 0:
                    projections = projections_tsne[
                        len(projections_camera_1) : 2 * len(projections_camera_1)
                    ]
                if idx > len(projections_camera_1):
                    break
            elif disp_camera == 3:
                image = image_a
                camera = camera_a
                if idx == 0:
                    projections = projections_tsne[
                        2 * len(projections_camera_1) : 3 * len(projections_camera_1)
                    ]
                if idx < len(projections_camera_1):
                    idx += 1
                    continue
            elif disp_camera == 4:
                image = image_b
                camera = camera_b
                if idx == 0:
                    projections = projections_tsne[
                        3 * len(projections_camera_1) : 4 * len(projections_camera_1)
                    ]
                if idx < len(projections_camera_1):
                    idx += 1
                    continue

            ax1.clear()
            ax1.imshow(image[0].permute(1, 2, 0).cpu().numpy())
            ax1.axis("off")
            ax1.set_title(f"Camera {camera[0]}")

            ax2.clear()
            if disp_camera == 1 or disp_camera == 2:
                ax2.scatter(
                    projections[0:idx, 0],
                    projections[0:idx, 1],
                    alpha=0.5,
                )
            elif disp_camera == 3 or disp_camera == 4:
                ax2.scatter(
                    projections[0 : idx - len(projections_camera_1), 0],
                    projections[0 : idx - len(projections_camera_1), 1],
                    alpha=0.5,
                )
            ax2.set_xlim(-75, 75)
            ax2.set_ylim(-75, 75)

            plt.draw()
            plt.pause(0.01)

            idx += 1
            print(idx)

    elif args.plot == "live_plot_2":
        print(
            "For this plot, set the dataset to desired combination and adapt offsets!"
        )
        offset_a = 0  # len(projections_camera_1)
        offset_b = len(projections_camera_1)
        dataset = TwoImagesLabelledDataset(args.image_folder, IMAGE_SIZE)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
            num_workers=multiprocessing.cpu_count(),
        )
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        idx = 0
        for image_a, image_b, camera_a, camera_b in dataloader:
            if idx % 10 == 0:
                ax1.clear()
                ax1.imshow(image_a[0].permute(1, 2, 0).cpu().numpy())
                ax1.axis("off")
                ax1.set_title(camera_a[0])

                ax2.clear()
                ax2.scatter(
                    projections_tsne[offset_a : offset_a + idx, 0],
                    projections_tsne[offset_a : offset_a + idx, 1],
                    alpha=0.5,
                    label=camera_a[0],
                )
                ax2.scatter(
                    projections_tsne[offset_b : offset_b + idx, 0],
                    projections_tsne[offset_b : offset_b + idx, 1],
                    alpha=0.5,
                    label=camera_b[0],
                )
                ax2.set_xlim(-75, 75)
                ax2.set_ylim(-75, 75)
                ax2.legend()

                ax3.clear()
                ax3.imshow(image_b[0].permute(1, 2, 0).cpu().numpy())
                ax3.axis("off")
                ax3.set_title(camera_b[0])

                plt.draw()
                plt.pause(0.01)

            idx += 1
            print(idx)