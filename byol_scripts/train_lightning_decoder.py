import argparse
import multiprocessing

import torch
from torchvision import models
from torch.utils.data import DataLoader

from byol_pytorch import (
    InfoNCE,
    Decoder,
    ImageDataset,
)
import pytorch_lightning as pl

# test model, a resnet 50
resnet = models.resnet18()

# arguments
parser = argparse.ArgumentParser(description="decoder_lightning")

parser.add_argument(
    "--image_folder",
    type=str,
    required=True,
    help="path to your folder of images for self-supervised learning",
)

args = parser.parse_args()

# constants
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
IMAGE_SIZE = 256
IMAGE_EXTS = [".jpg", ".png", ".jpeg"]


# pytorch lightning module
class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        model = InfoNCE(net, **kwargs)
        encoder = model.online_encoder
        self.learner = Decoder(encoder, IMAGE_SIZE)

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
    ds = ImageDataset(args.image_folder, IMAGE_SIZE, "camera_2")
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

    # load encoder
    checkpoint = torch.load(
        "/home/clemensschwarke/git/byol-pytorch/lightning_logs/version_138_new_dataset_ccm_v1/checkpoints/epoch=91-step=32476.ckpt"
    )
    encoder_weights = {
        k.replace("learner.online_encoder.", ""): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("learner.online_encoder.")
    }
    model.learner.encoder.load_state_dict(encoder_weights)

    trainer = pl.Trainer(
        devices=1,
        max_epochs=EPOCHS,
    )

    trainer.fit(model, train_loader)
