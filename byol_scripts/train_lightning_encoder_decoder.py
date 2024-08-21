import argparse
import multiprocessing

import torch
from torchvision import models
from torch.utils.data import DataLoader
from PIL import Image

from byol_pytorch import (
    InfoNCE,
    EncoderDecoder,
    ImageDataset,
    ImageDatasetEncDec,
    CNN,
)
import pytorch_lightning as pl

# test model
net = CNN() #models.resnet18(models.ResNet18_Weights.DEFAULT)

# arguments
parser = argparse.ArgumentParser(description="encoder_decoder_lightning")

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
BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3
IMAGE_SIZE = 256
IMAGE_EXTS = [".jpg", ".png", ".jpeg"]


# pytorch lightning module
class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        model = InfoNCE(net, **kwargs)
        encoder = model.online_encoder
        self.learner = EncoderDecoder(encoder, IMAGE_SIZE)

    def forward(self, images):
        return self.learner.compute_loss(images)

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
    ds_train = ImageDatasetEncDec(args.train_folder, IMAGE_SIZE, data_percentage=1.0)
    ds_val = ImageDatasetEncDec(args.val_folder, IMAGE_SIZE, data_percentage=1.0)
    
    train_loader = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=BATCH_SIZE,
        num_workers=multiprocessing.cpu_count(),
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
        log_every_n_steps=1,
        callbacks=[pl.callbacks.ModelCheckpoint(every_n_epochs=5, save_top_k=1)],
        max_epochs=EPOCHS,
        strategy="ddp_find_unused_parameters_true",
        sync_batchnorm=True,
    )

    trainer.fit(model, train_loader, val_loader)
