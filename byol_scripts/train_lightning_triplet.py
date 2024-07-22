import argparse
import multiprocessing

import torch
from torchvision import models
from torch.utils.data import DataLoader

from byol_pytorch import Triplet, TripletDataset
import pytorch_lightning as pl

# test model, a resnet 50
resnet = models.resnet50(models.ResNet50_Weights.DEFAULT)

# arguments
parser = argparse.ArgumentParser(description="triplet_lightning")

parser.add_argument(
    "--image_folder",
    type=str,
    required=True,
    help="path to your folder of images for self-supervised learning",
)

args = parser.parse_args()

# constants
BATCH_SIZE = 48
EPOCHS = 100
LR = 3e-4
IMAGE_SIZE = 256
IMAGE_EXTS = [".jpg", ".png", ".jpeg"]


# pytorch lightning module
class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = Triplet(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        self.log("loss", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)


# main
if __name__ == "__main__":
    ds = TripletDataset(args.image_folder, IMAGE_SIZE)
    train_loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=multiprocessing.cpu_count()
    )

    model = SelfSupervisedLearner(
        resnet,
        image_size=IMAGE_SIZE,
        hidden_layer="avgpool",
        projection_size=256,
        projection_hidden_size=4096,
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
    )

    trainer.fit(model, train_loader)
