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

from byol_pytorch import InfoNCE, TwoImagesLabelledDataset, CNN

# arguments
parser = argparse.ArgumentParser(description="plot_info_nce")
parser.add_argument(
    "--image_folder",
    type=str,
    required=True,
    help="path to your folder of images for self-supervised learning",
)
parser.add_argument(
    "--plot",
    type=str,
    choices=[
        "cameras",
        "split_logs",
        "embedding",
        "points",
        "distance",
        "live_plot",
        "live_plot_2",
    ],
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
BATCH_SIZE = 128
LR = 3e-4


# pytorch lightning module
class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = InfoNCE(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        self.log("loss", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)


def check_for_nan_hook(module, input, output):
    if torch.isnan(output).any():
        print(f"NaN detected in {module}")
        print(f"Input: {input}")


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
        net = models.resnet18()
        # net = CNN()
        model = SelfSupervisedLearner.load_from_checkpoint(
            "/home/clemensschwarke/git/byol-pytorch/lightning_logs/version_201_box_large/checkpoints/epoch=19-step=1560.ckpt",
            net=net,
            image_size=IMAGE_SIZE,
            hidden_layer="avgpool",
            projection_size=32,
            projection_hidden_size=256,
            # map_location={"cuda:1": "cuda:0"},
        )
        model.eval()

        hooks = []
        for name, layer in model.named_modules():
            hook = layer.register_forward_hook(check_for_nan_hook)
            hooks.append(hook)

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
                images = torch.stack([image_a, image_b, image_a], dim=1).to(
                    model.device
                )
                projections_out = model.learner(images, return_embedding=True)
                projection_a = projections_out[:, 0]
                projection_b = projections_out[:, 1]

                for i in range(len(camera_a)):
                    projections[camera_a[i]].append(projection_a[i])
                    projections[camera_b[i]].append(projection_b[i])
                print(f"Step {idx+1} of {len(dataloader)}")
            print(f"Projections took {time.time() - start:.2f} seconds")

            for hook in hooks:
                hook.remove()

            # convert to numpy
            # projections_camera_1 = torch.stack(projections["camera_1"]).cpu().numpy()
            projections_camera_2 = torch.stack(projections["camera_2"]).cpu().numpy()
            # projections_camera_3 = torch.stack(projections["camera_3"]).cpu().numpy()
            projections_camera_4 = torch.stack(projections["camera_4"]).cpu().numpy()

            # save projections
            # np.save("projections/projections_camera_1.npy", projections_camera_1)
            np.save("projections/projections_camera_2.npy", projections_camera_2)
            # np.save("projections/projections_camera_3.npy", projections_camera_3)
            np.save("projections/projections_camera_4.npy", projections_camera_4)

    if args.use_saved_projections or args.use_saved_tsne:
        # load saved projections
        # projections_camera_1 = np.load("projections/projections_camera_1.npy")
        projections_camera_2 = np.load("projections/projections_camera_2.npy")
        # projections_camera_3 = np.load("projections/projections_camera_3.npy")
        projections_camera_4 = np.load("projections/projections_camera_4.npy")

    projections_all = np.vstack(
        [
            # projections_camera_1,
            projections_camera_2,
            # projections_camera_3,
            projections_camera_4,
        ]
    )

    # compute l1 norm of projections
    # norm = np.linalg.norm(projections_all, ord=1, axis=1)
    # print(f"Mean L1 norm of projections: {np.mean(norm):.2f}")

    # t-SNE visualization
    labels = np.array(
        # [0] * len(projections_camera_1)
        [1] * len(projections_camera_2)
        # + [2] * len(projections_camera_3)
        + [3] * len(projections_camera_4)
    )

    if not args.use_saved_tsne:
        tsne = TSNE(n_components=2, random_state=42, n_jobs=multiprocessing.cpu_count())
        print("Fitting t-SNE ...")
        start = time.time()
        projections_tsne = tsne.fit_transform(projections_all)
        # projections_tsne = projections_all
        print(f"t-SNE took {time.time() - start:.2f} seconds")

        np.save("projections/projections_tsne.npy", projections_tsne)
    else:
        projections_tsne = np.load("projections/projections_tsne.npy")

    if args.plot == "cameras":
        plt.figure(figsize=(10, 8))
        # plt.scatter(
        #     projections_tsne[labels == 0, 0],
        #     projections_tsne[labels == 0, 1],
        #     label="camera_1",
        #     alpha=0.1,
        # )
        plt.scatter(
            projections_tsne[labels == 1, 0],
            projections_tsne[labels == 1, 1],
            label="camera_2",
            alpha=0.1,
        )
        # plt.scatter(
        #     projections_tsne[labels == 2, 0],
        #     projections_tsne[labels == 2, 1],
        #     label="camera_3",
        #     alpha=0.1,
        # )
        plt.scatter(
            projections_tsne[labels == 3, 0],
            projections_tsne[labels == 3, 1],
            label="camera_4",
            alpha=0.1,
        )
        plt.legend()
        plt.title("t-SNE Visualization of 32-Dimensional Projections")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.show()

    if args.plot == "split_logs":
        # List of image counts
        image_counts = [
            1162,
            1054,
            955,
            367,
            658,
            449,
            1659,
            684,
            428,
            300,
            649,
            465,
            577,
            625,
            498,
            504,
            491,
            482,
            441,
            1417,
            713,
            971,
            2214,
            619,
            483,
            633,
            301,
        ]

        # List of labels corresponding to the image counts
        labels = [
            "0804_1_log_5",
            "0804_1_log_6",
            "0804_1_log_7",
            "0804_1_log_8",
            "0804_2_log_1",
            "0804_2_log_2",
            "0804_2_log_3",
            "0804_2_log_4",
            "0804_2_log_5",
            "0804_2_log_6",
            "0804_2_log_7",
            "0804_2_log_8",
            "0804_3_log_1",
            "0804_3_log_2",
            "0804_3_log_3",
            "0823_1_log_1",
            "0823_1_log_2",
            "0823_1_log_3",
            "0823_1_log_4",
            "0823_1_log_5",
            "0823_1_log_6",
            "0823_1_log_7",
            "0823_1_log_8",
            "0823_1_log_9",
            "0823_1_log_10",
            "0823_1_log_11",
            "0823_1_log_12",
        ]

        # Group the image counts and labels by their prefixes
        grouped_data = {}
        for count, label in zip(image_counts, labels):
            # prefix = label.split("_")[0] + "_" + label.split("_")[1]
            prefix = label.split("_")[0]
            if prefix not in grouped_data:
                grouped_data[prefix] = count
            else:
                grouped_data[prefix] += count

        # Initialize start variable
        start = 0

        # Create the plot
        plt.figure(figsize=(10, 8))

        # Loop through the grouped data and plot the scatter plots
        for prefix, count in grouped_data.items():
            end = start + count
            plt.scatter(
                projections_tsne[start:end, 0],
                projections_tsne[start:end, 1],
                label=prefix,
                alpha=0.5,
            )
            start = end

        # Add legend and labels
        plt.legend()
        plt.title("t-SNE Visualization of 32-Dimensional Projections")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.show()

    if args.plot == "embedding":
        # plot all dimensions
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        for i in range(4):
            for j in range(4):
                ax = axes[i, j]
                dim1 = 2 * (i * 4 + j)
                dim2 = dim1 + 1
                ax.scatter(
                    projections_all[labels == 0, dim1],
                    projections_all[labels == 0, dim2],
                    label="camera_1",
                    alpha=0.1,
                )
                ax.scatter(
                    projections_all[labels == 1, dim1],
                    projections_all[labels == 1, dim2],
                    label="camera_2",
                    alpha=0.1,
                )
                ax.scatter(
                    projections_all[labels == 2, dim1],
                    projections_all[labels == 2, dim2],
                    label="camera_3",
                    alpha=0.1,
                )
                ax.scatter(
                    projections_all[labels == 3, dim1],
                    projections_all[labels == 3, dim2],
                    label="camera_4",
                    alpha=0.1,
                )
                ax.set_title(f"Dimensions {dim1} and {dim2}")
                ax.legend()
        plt.tight_layout()
        plt.show()

    elif args.plot == "points":
        plt.figure(figsize=(10, 8))
        num_points = 100
        colormap = cm.get_cmap("viridis", num_points)
        for i in range(num_points):
            color = colormap(i)
            plt.scatter(
                projections_tsne[i :: len(projections_camera_2), 0],
                projections_tsne[i :: len(projections_camera_2), 1],
                color=color,
                alpha=0.5,
                s=100,
            )
        plt.title(
            "t-SNE Visualization of 32-Dimensional Projections with Corresponding Points"
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
        plt.suptitle("Pairwise Cosine Similarity of 32-Dimensional Projections")
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
                    projections = projections_tsne[0 : len(projections_camera_2)]
                if idx > len(projections_camera_2):
                    break
            elif disp_camera == 2:
                image = image_b
                camera = camera_b
                if idx == 0:
                    projections = projections_tsne[
                        len(projections_camera_2) : 2 * len(projections_camera_2)
                    ]
                if idx > len(projections_camera_2):
                    break
            elif disp_camera == 3:
                image = image_a
                camera = camera_a
                if idx == 0:
                    projections = projections_tsne[
                        2 * len(projections_camera_2) : 3 * len(projections_camera_2)
                    ]
                if idx < len(projections_camera_2):
                    idx += 1
                    continue
            elif disp_camera == 4:
                image = image_b
                camera = camera_b
                if idx == 0:
                    projections = projections_tsne[
                        3 * len(projections_camera_2) : 4 * len(projections_camera_2)
                    ]
                if idx < len(projections_camera_2):
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
                    projections[0 : idx - len(projections_camera_2), 0],
                    projections[0 : idx - len(projections_camera_2), 1],
                    alpha=0.5,
                )
            ax2.set_xlim(-75, 75)
            ax2.set_ylim(-75, 75)

            plt.draw()
            plt.pause(0.01)

            idx += 1
            print(idx)

    elif args.plot == "live_plot_2":
        print("For this plot, set the dataset to desired camera combination!")
        offset_a = 0
        offset_b = 1 * len(projections_camera_2)
        offset_c = 2 * len(projections_camera_2)
        offset_d = 3 * len(projections_camera_2)
        dataset = TwoImagesLabelledDataset(args.image_folder, IMAGE_SIZE)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
            num_workers=multiprocessing.cpu_count(),
        )
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
        idx = 0
        for image_a, image_b, camera_a, camera_b in dataloader:
            ax1.clear()
            ax1.imshow(image_a[0].permute(1, 2, 0).cpu().numpy())
            ax1.axis("off")
            ax1.set_title(camera_a[0])

            ax2.clear()
            ax2.scatter(
                projections_tsne[offset_a : offset_a + idx, 0],
                projections_tsne[offset_a : offset_a + idx, 1],
                alpha=1,
                label="camera 1",
            )
            ax2.scatter(
                projections_tsne[offset_b : offset_b + idx, 0],
                projections_tsne[offset_b : offset_b + idx, 1],
                alpha=0.5,
                label="camera 2",
            )
            ax2.scatter(
                projections_tsne[offset_c : offset_c + idx, 0],
                projections_tsne[offset_c : offset_c + idx, 1],
                alpha=0.2,
                label="camera 3",
            )
            ax2.scatter(
                projections_tsne[offset_d : offset_d + idx, 0],
                projections_tsne[offset_d : offset_d + idx, 1],
                alpha=0.2,
                label="camera 4",
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
