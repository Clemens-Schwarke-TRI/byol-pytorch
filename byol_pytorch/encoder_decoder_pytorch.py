import matplotlib.pyplot as plt
import random
from functools import wraps
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from torchvision import transforms as T

from .networks import MLP, DecoderNet


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


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, image_size, train_encoder, train_decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = DecoderNet()
        self.image_size = image_size
        self.train_encoder = train_encoder
        self.train_decoder = train_decoder
        self.augment = None

        if not self.train_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        if not self.train_decoder:
            self.decoder.eval()
            for param in self.decoder.parameters():
                param.requires_grad = False

        self.normalize = T.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        )
        self.train()

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.augment = torch.nn.Sequential(
                RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.1),
                T.RandomGrayscale(p=0.1),
                RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.1),
                T.RandomResizedCrop(
                    size=(self.image_size, self.image_size), scale=(0.8, 1.0)
                ),
                T.Normalize(
                    mean=torch.tensor([0.485, 0.456, 0.406]),
                    std=torch.tensor([0.229, 0.224, 0.225]),
                ),
            )
            if not self.train_encoder:
                self.encoder.eval()
            if not self.train_decoder:
                self.decoder.eval()
        else:
            self.augment = nn.Sequential(
                T.Normalize(
                    mean=torch.tensor([0.485, 0.456, 0.406]),
                    std=torch.tensor([0.229, 0.224, 0.225]),
                ),
            )

    def compute_loss(self, images):
        x = images[:, 0]
        y = images[:, 1]
        x = self.augment(x)
        x = self.encoder(x.contiguous())
        x = self.decoder(x)
        return F.mse_loss(x, self.normalize(y))

    def get_encoding(self, x):
        x_shape = x.shape
        x = x.view(-1, 3, self.image_size, self.image_size)
        x = self.augment(x)
        x = self.encoder(x.contiguous())
        return x.view(x_shape[0], x_shape[1], 32)

    def get_translation(self, x):
        x_shape = x.shape
        x = x.view(-1, 3, self.image_size, self.image_size)
        x = self.augment(x)
        x = self.encoder(x.contiguous())
        x = self.decoder(x)
        return x.view(x_shape)
