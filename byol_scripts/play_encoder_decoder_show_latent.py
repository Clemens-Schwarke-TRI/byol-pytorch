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

from byol_pytorch import InfoNCE, TwoImagesLabelledDataset, CNN, EncoderDecoder

# arguments
parser = argparse.ArgumentParser(description="plot_info_nce")
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
        model = InfoNCE(net, **kwargs)
        encoder = model.online_encoder
        self.learner = EncoderDecoder(
            encoder, IMAGE_SIZE, train_encoder=False, train_decoder=False
        )

    def forward(self, x):
        return self.learner.get_encoding(x)


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
            "/home/clemensschwarke/git/byol-pytorch/lightning_logs/version_188_encoder_for_186/checkpoints/epoch=49-step=550.ckpt",
            net=net,
            image_size=IMAGE_SIZE,
            hidden_layer="avgpool",
            projection_size=32,
            projection_hidden_size=256,
            # map_location={"cuda:1": "cuda:0"},
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
                images = torch.stack([image_a, image_b], dim=1).to(model.device)
                projections_out = model(images)
                projection_a = projections_out[:, 0]
                projection_b = projections_out[:, 1]

                for i in range(len(camera_a)):
                    projections[camera_a[i]].append(projection_a[i])
                    projections[camera_b[i]].append(projection_b[i])
                print(f"Step {idx+1} of {len(dataloader)}")
            print(f"Projections took {time.time() - start:.2f} seconds")

            # convert to numpy
            projections_camera_2 = torch.stack(projections["camera_2"]).cpu().numpy()
            projections_camera_4 = torch.stack(projections["camera_4"]).cpu().numpy()

            # save projections
            np.save("projections/projections_camera_2.npy", projections_camera_2)
            np.save("projections/projections_camera_4.npy", projections_camera_4)

    if args.use_saved_projections or args.use_saved_tsne:
        # load saved projections
        projections_camera_2 = np.load("projections/projections_camera_2.npy")
        projections_camera_4 = np.load("projections/projections_camera_4.npy")

    projections_all = np.vstack(
        [
            projections_camera_2,
            projections_camera_4,
        ]
    )

    # t-SNE visualization
    labels = np.array([0] * len(projections_camera_2) + [1] * len(projections_camera_4))

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

    plt.figure(figsize=(10, 8))
    plt.scatter(
        projections_tsne[labels == 0, 0],
        projections_tsne[labels == 0, 1],
        label="camera_2",
        alpha=0.25,
    )
    plt.scatter(
        projections_tsne[labels == 1, 0],
        projections_tsne[labels == 1, 1],
        label="camera_4",
        alpha=0.25,
    )
    plt.legend()
    plt.title("t-SNE Visualization of 32-Dimensional Projections")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()
