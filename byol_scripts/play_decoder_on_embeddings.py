
import numpy as np
import torch
import time
import argparse
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import multiprocessing
import pickle

from MulticoreTSNE import MulticoreTSNE as TSNE
from scipy.spatial.distance import cdist
from torch import nn
from torchvision import transforms as T
from torchvision import models

from byol_pytorch import InfoNCE, Decoder, TwoImageDataset, CNN

# constants
IMAGE_SIZE = 256
BATCH_SIZE = 1
LR = 3e-4


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
    # create model
    net = models.resnet18()
    # net = CNN()
    model = SelfSupervisedLearner.load_from_checkpoint(
        "/home/clemensschwarke/git/byol-pytorch/lightning_logs/version_161_decoder_for_160/checkpoints/epoch=99-step=40500.ckpt",
        net=net,
        image_size=IMAGE_SIZE,
        hidden_layer="avgpool",
        projection_size=32,
        projection_hidden_size=256,
    )
    model.eval()
    denormalize = T.Compose(
        [
            T.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
        ]
    )

    # import the replay buffer
    replay_buffer_file = (
        "/home/clemensschwarke/Desktop/rl_logs/2024-08-20_17-31-42/replay_buffer.pkl"
    )
    with open(replay_buffer_file, "rb") as f:
        replay_buffer = pickle.load(f)
    replay_latent = (
        replay_buffer.observations[:, 0, -32:].astype(np.float32)
    )

    # decode the latent vectors
    with torch.no_grad():
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        for i, latent in enumerate(replay_latent):
            output_image = model.learner.decoder(torch.tensor(latent,device=model.device).unsqueeze(0))
            output_image = denormalize(output_image)

            ax.clear()
            ax.imshow(output_image[0].cpu().numpy().transpose(1, 2, 0))
            ax.set_title("Output Image")
            ax.axis("off")

            plt.pause(0.1)
