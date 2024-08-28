import argparse
import multiprocessing

import torch
from torchvision import models
from torch.utils.data import DataLoader

from byol_pytorch import (
    InfoNCE,
    EncoderDecoder,
    ImageDatasetEncDec,
    CNN,
)
import pytorch_lightning as pl

# test model
net = models.resnet18()
# net = CNN()

# arguments
parser = argparse.ArgumentParser(description="decoder_lightning")

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
        self.learner = EncoderDecoder(
            encoder, IMAGE_SIZE, train_encoder=False, train_decoder=True
        )

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
    ds_train_1 = ImageDatasetEncDec(
        args.train_folder, IMAGE_SIZE, "camera_2", "camera_2", data_multiplier=1
    )
    ds_train_2 = ImageDatasetEncDec(
        args.train_folder, IMAGE_SIZE, "camera_4", "camera_2", data_multiplier=1
    )
    ds_train = torch.utils.data.ConcatDataset([ds_train_1, ds_train_2])

    ds_val = ImageDatasetEncDec(args.val_folder, IMAGE_SIZE, "camera_4", "camera_2")
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

    # load encoder
    checkpoint = torch.load(
        "/home/clemensschwarke/git/byol-pytorch/lightning_logs/version_215_like_205_sanity/checkpoints/epoch=19-step=1480.ckpt"
    )
    encoder_weights = {
        k.replace("learner.online_encoder.", ""): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("learner.online_encoder.")
    }
    model.learner.encoder.load_state_dict(encoder_weights)

    trainer = pl.Trainer(
        # devices=[0, 1, 2, 3],
        log_every_n_steps=1,
        callbacks=[pl.callbacks.ModelCheckpoint(every_n_epochs=5, save_top_k=1)],
        max_epochs=EPOCHS,
        strategy="ddp",
        sync_batchnorm=True,
    )

    trainer.fit(model, train_loader, val_loader)
