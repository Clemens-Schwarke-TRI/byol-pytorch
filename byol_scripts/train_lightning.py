import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image

import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

from byol_pytorch import BYOL, TwoImagesStackedDataset
import pytorch_lightning as pl

# test model, a resnet 50
resnet = models.resnet50(models.ResNet50_Weights.DEFAULT)

# arguments
parser = argparse.ArgumentParser(description="byol-lightning-test")

parser.add_argument(
    "--image_folder",
    type=str,
    required=True,
    help="path to your folder of images for self-supervised learning",
)

args = parser.parse_args()

# constants
BATCH_SIZE = 256
EPOCHS = 1000
LR = 3e-4
IMAGE_SIZE = 256
IMAGE_EXTS = [".jpg", ".png", ".jpeg"]
NUM_WORKERS = multiprocessing.cpu_count()


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
    ds = TwoImagesStackedDataset(args.image_folder, IMAGE_SIZE)
    train_loader = DataLoader(
        ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True
    )

    model = SelfSupervisedLearner(
        resnet,
        image_size=IMAGE_SIZE,
        hidden_layer="avgpool",
        projection_size=256,
        projection_hidden_size=4096,
        moving_average_decay=0.99,
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS, accumulate_grad_batches=1, sync_batchnorm=True
    )

    trainer.fit(model, train_loader)
