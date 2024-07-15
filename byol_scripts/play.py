import numpy as np
import torch
import time
import argparse
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from torch import nn
from torchvision import transforms as T
from torchvision import models

from byol_pytorch import BYOL, TwoImagesLabelledDataset

# arguments
parser = argparse.ArgumentParser(description="byol-lightning-test")
parser.add_argument(
    "--image_folder",
    type=str,
    required=True,
    help="path to your folder of images for self-supervised learning",
)
parser.add_argument(
    "--use_saved_projections",
    action="store_true",
    help="use saved projections",
)
args = parser.parse_args()

# constants
IMAGE_SIZE = 256
BATCH_SIZE = 256
LR = 3e-4


# pytorch lightning module
class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        image_a = images[0]
        image_b = images[1]
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
    if not args.use_saved_projections:
        # create dataset
        dataset = TwoImagesLabelledDataset(args.image_folder, IMAGE_SIZE)
        dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=False, batch_size=BATCH_SIZE
        )

        # create model
        net = models.resnet50()
        model = SelfSupervisedLearner.load_from_checkpoint(
            "/home/clemensschwarke/git/byol-pytorch/lightning_logs/version_5/checkpoints/epoch=999-step=151000.ckpt",
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
                image_a_out, image_b_out = model.learner(
                    image_a, image_b, return_embedding=True
                )
                projection_a = image_a_out[0]
                projection_b = image_b_out[0]
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
    else:
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

    tsne = TSNE(n_components=2, random_state=42)
    print("Fitting t-SNE ...")
    start = time.time()
    projections_tsne = tsne.fit_transform(projections_all)
    print(f"t-SNE took {time.time() - start:.2f} seconds")

    plt.figure(figsize=(10, 8))
    plt.scatter(
        projections_tsne[labels == 0, 0],
        projections_tsne[labels == 0, 1],
        label="camera_1",
        alpha=0.5,
    )
    plt.scatter(
        projections_tsne[labels == 1, 0],
        projections_tsne[labels == 1, 1],
        label="camera_2",
        alpha=0.5,
    )
    plt.scatter(
        projections_tsne[labels == 2, 0],
        projections_tsne[labels == 2, 1],
        label="camera_3",
        alpha=0.5,
    )
    plt.scatter(
        projections_tsne[labels == 3, 0],
        projections_tsne[labels == 3, 1],
        label="camera_4",
        alpha=0.5,
    )
    plt.legend()
    plt.title("t-SNE Visualization of 256-Dimensional Projections")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()
