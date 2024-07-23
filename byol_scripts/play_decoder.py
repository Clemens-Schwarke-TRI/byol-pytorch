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
from torchvision import transforms
from torchvision import models

from byol_pytorch import Triplet, Decoder, ImageDataset, TwoImageDataset

# arguments
parser = argparse.ArgumentParser(description="plot_decoder")
parser.add_argument(
    "--image_folder",
    type=str,
    required=True,
    help="path to your folder of images for self-supervised learning",
)
args = parser.parse_args()

# constants
IMAGE_SIZE = 256
BATCH_SIZE = 1
LR = 3e-4


# pytorch lightning module
class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        triplet = Triplet(net, **kwargs)
        encoder = triplet.online_encoder
        self.learner = Decoder(encoder)

    def forward(self, images):
        return self.learner.compute_loss(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        self.log("loss", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)


# main
if __name__ == "__main__":
    dataset = TwoImageDataset(args.image_folder, IMAGE_SIZE, "camera_4", "camera_2")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=multiprocessing.cpu_count(),
    )

    # create model
    net = models.resnet50()
    model = SelfSupervisedLearner.load_from_checkpoint(
        "/home/clemensschwarke/git/byol-pytorch/lightning_logs/version_92/checkpoints/epoch=99-step=50300.ckpt",
        net=net,
        image_size=IMAGE_SIZE,
        hidden_layer="avgpool",
    )
    model.eval()

    denormalize = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
        ]
    )

    with torch.no_grad():
        fig, ax = plt.subplots(1, 3, figsize=(30, 10))
        for image, image2 in dataloader:
            output_image = model.learner(image.to(model.device))
            image = denormalize(image[0])
            output_image = denormalize(output_image[0])
            image2 = denormalize(image2[0])

            ax[0].clear()
            ax[0].imshow(image.cpu().numpy().transpose(1, 2, 0))
            ax[0].set_title("Input Image")
            ax[0].axis("off")
            ax[1].clear()
            ax[1].imshow(output_image.cpu().numpy().transpose(1, 2, 0))
            ax[1].set_title("Output Image")
            ax[1].axis("off")
            ax[2].clear()
            ax[2].imshow(image2.cpu().numpy().transpose(1, 2, 0))
            ax[2].set_title("Ground Truth Image")
            ax[2].axis("off")

            plt.pause(0.1)
