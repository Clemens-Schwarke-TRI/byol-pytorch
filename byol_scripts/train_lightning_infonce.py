import argparse
import multiprocessing

import torch
from torchvision import models
from torch.utils.data import DataLoader

from byol_pytorch import InfoNCE, TwoDatasetsDataset, ImagePoseDataset
import pytorch_lightning as pl

# test model, a resnet 18
resnet = models.resnet18(models.ResNet18_Weights.DEFAULT)

# arguments
parser = argparse.ArgumentParser(description="infonce_lightning")

parser.add_argument(
    "--image_folder_cc",
    type=str,
    required=True,
    help="path to your folder of images for self-supervised learning",
)

parser.add_argument(
    "--image_folder_ac",
    type=str,
    required=True,
    help="path to your folder of images for self-supervised learning",
)

args = parser.parse_args()

# constants
BATCH_SIZE = 128
EPOCHS = 100
LR = 3e-4
IMAGE_SIZE = 256
IMAGE_EXTS = [".jpg", ".png", ".jpeg"]


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


# main
if __name__ == "__main__":
    dscc = ImagePoseDataset(args.image_folder_cc, IMAGE_SIZE)
    dsac = ImagePoseDataset(
        args.image_folder_ac,
        IMAGE_SIZE,
        paths={"camera_1": [], "camera_2": [], "camera_3": []},
        combinations=[
            ("camera_1", "camera_2"),
            ("camera_1", "camera_3"),
            ("camera_2", "camera_1"),
            ("camera_2", "camera_3"),
            ("camera_3", "camera_1"),
            ("camera_3", "camera_2"),
        ],
    )

    ds = TwoDatasetsDataset(dscc, dsac)
    train_loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=multiprocessing.cpu_count()
    )

    model = SelfSupervisedLearner(
        resnet,
        image_size=IMAGE_SIZE,
        hidden_layer="avgpool",
        projection_size=32,
        projection_hidden_size=256,
    )

    trainer = pl.Trainer(
        devices=[0, 1, 2, 3],
        max_epochs=EPOCHS,
        strategy="ddp_find_unused_parameters_true",
        sync_batchnorm=True,
    )

    trainer.fit(model, train_loader)
