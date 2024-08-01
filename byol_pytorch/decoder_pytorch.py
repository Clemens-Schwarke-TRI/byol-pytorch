import matplotlib.pyplot as plt
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from torchvision import transforms as T

# augmentation utils
class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

class DecoderNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize the FCN
        fc_layers = nn.Sequential(
            nn.Linear(32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 2048),
        )

        # Initialize the CNN network
        conv_layers = nn.Sequential(
            nn.Conv2d(32, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 3, 3, padding=1),
        )

        self.cnn = nn.Sequential(
            fc_layers,
            nn.Unflatten(-1, (32, 8, 8)),
            conv_layers,
        )

    def forward(self, x):
        return self.cnn(x)


class Decoder(nn.Module):
    def __init__(self, encoder, image_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = DecoderNet()

        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # default SimCLR augmentation
        self.augment = torch.nn.Sequential(
            RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.1),
            T.RandomGrayscale(p=0.1),
            RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.1),
            T.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]),
            ),
        )
       
        self.normalize = T.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.augment(x)
            x = self.encoder(x)
        return self.decoder(x)

    def compute_loss(self, x):
        return F.mse_loss(self(x), self.normalize(x))
