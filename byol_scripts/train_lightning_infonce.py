import argparse
import multiprocessing

import torch
from torchvision import models
from torch.utils.data import DataLoader
import torch.nn as nn

from byol_pytorch import InfoNCE, ImagePoseDataset, CNN
import pytorch_lightning as pl


# test model
net = models.resnet18(models.ResNet18_Weights.DEFAULT)
# net = CNN()

# arguments
parser = argparse.ArgumentParser(description="infonce_lightning")

parser.add_argument(
    "--train_folder",
    type=str,
    required=True,
    help="path to your folder of images for self-supervised learning",
)
parser.add_argument(
    "--val_folder",
    type=str,
    required=True,
    help="path to your folder of images for validation",
)

args = parser.parse_args()

# constants
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3
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

    def validation_step(self, images, _):
        loss = self.forward(images)
        self.log("val_loss", loss, sync_dist=True)
        return {"val_loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_validation_model_train(self):
        super().on_validation_model_train()


# main
if __name__ == "__main__":
    ds = ImagePoseDataset(args.train_folder, IMAGE_SIZE, ratio_positives=1.0)
    train_loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=multiprocessing.cpu_count()
    )
    ds_val = ImagePoseDataset(args.val_folder, IMAGE_SIZE, ratio_positives=1.0)
    val_loader = DataLoader(
        ds_val, batch_size=BATCH_SIZE, num_workers=multiprocessing.cpu_count()
    )

    model = SelfSupervisedLearner(
        net,
        image_size=IMAGE_SIZE,
        hidden_layer="avgpool",
        projection_size=32,
        projection_hidden_size=256,
    )

    trainer = pl.Trainer(
        devices=[0, 1, 2, 3],
        callbacks=[pl.callbacks.ModelCheckpoint(every_n_epochs=1, save_top_k=1)],
        max_epochs=EPOCHS,
        strategy="ddp_find_unused_parameters_true",
        sync_batchnorm=True,
    )

    trainer.fit(model, train_loader, val_loader)
