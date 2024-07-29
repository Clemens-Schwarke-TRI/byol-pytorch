import matplotlib.pyplot as plt
import random
from functools import wraps

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from torchvision import transforms as T

from info_nce import InfoNCE as InfoNCELoss


def flatten(t):
    return t.reshape(t.shape[0], -1)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


def get_module_device(module):
    return next(module.parameters()).device


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


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


# MLP class for projector and predictor
def MLP(dim, projection_size, hidden_size):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size),
    )


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets
class NetWrapper(nn.Module):
    def __init__(
        self,
        net,
        projection_size,
        projection_hidden_size,
        layer,
    ):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f"hidden layer ({self.layer}) not found"
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton("projector")
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(
            dim,
            self.projection_size,
            self.projection_hidden_size,
        )
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f"hidden layer {self.layer} never emitted an output"
        return hidden

    def forward(self, x, return_projection=True):
        representation = self.get_representation(x)

        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection


# main class
class InfoNCE(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer,
        projection_size,
        projection_hidden_size,
        reg_lambda=1e-3,
    ):
        super().__init__()
        self.net = net
        self.loss = InfoNCELoss(negative_mode="paired")
        self.reg_lambda = reg_lambda

        # default SimCLR augmentation
        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.1),
            T.RandomGrayscale(p=0.1),
            RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.1),
            T.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]),
            ),
        )

        self.augment1 = DEFAULT_AUG
        self.augment2 = DEFAULT_AUG
        self.augment3 = DEFAULT_AUG

        self.online_encoder = NetWrapper(
            net,
            projection_size,
            projection_hidden_size,
            layer=hidden_layer,
        )

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(
            torch.randn(2, 3, 3, image_size, image_size, device=device),
        )

    def forward(self, images, return_embedding=False, return_projection=True):
        assert not (
            self.training and images.shape[0] == 1
        ), "you must have greater than 1 sample when training, due to the batchnorm in the projection layer"

        batch_size = images.shape[0]  # B
        num_negatives = images.shape[1] - 2  # N
        sizes = [batch_size, batch_size, batch_size * num_negatives]

        # augment and reshape from [B, 1+1+N, C, H, W] to [B*1+1+N, C, H, W]
        image_a = images[:, 0]
        image_p = images[:, 1]
        image_n = images[:, 2:].flatten(0, 1)

        image_a, image_p, image_n = (
            self.augment1(image_a),
            self.augment2(image_p),
            self.augment3(image_n),
        )
        images = torch.cat((image_a, image_p, image_n), dim=0)

        if return_embedding:
            online_projections = self.online_encoder(
                images, return_projection=return_projection
            )
            return F.normalize(online_projections, dim=-1, p=2)

        online_projections = self.online_encoder(images)
        online_projections = F.normalize(online_projections, dim=-1, p=2)
        online_projections_a, online_projections_p, online_projections_n = (
            online_projections.split(sizes, dim=0)
        )
        online_projections_n = online_projections_n.unflatten(
            0, (batch_size, num_negatives)
        )

        nce_loss = self.loss(
            online_projections_a,
            online_projections_p,
            online_projections_n,
        )

        # regularize with L1
        reg_loss = self.reg_lambda * torch.sum(torch.abs(online_projections), dim=-1)

        return (nce_loss + reg_loss).mean()
