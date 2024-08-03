import numpy as np
import torch
import time
import argparse
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import multiprocessing
import pandas as pd

from MulticoreTSNE import MulticoreTSNE as TSNE
from scipy.spatial.distance import cdist
from torch import nn
from torchvision import transforms as T
from torchvision import models

from byol_pytorch import InfoNCE, PickleFileDataset

# arguments
parser = argparse.ArgumentParser(description="export_embeddings")
parser.add_argument(
    "--file_path",
    type=str,
    required=True,
    help="path to the pickle file",
)
args = parser.parse_args()

# constants
IMAGE_SIZE = 256
BATCH_SIZE = 128
LR = 3e-4


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
    # load model and compute projections
    # create dataset
    dataset = PickleFileDataset(args.file_path, IMAGE_SIZE)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=multiprocessing.cpu_count(),
    )

    # create model
    net = models.resnet18()
    model = SelfSupervisedLearner.load_from_checkpoint(
        "/home/clemensschwarke/git/byol-pytorch/lightning_logs/version_141_ccmt/checkpoints/epoch=99-step=66200.ckpt",
        net=net,
        image_size=IMAGE_SIZE,
        hidden_layer="avgpool",
        projection_size=32,
        projection_hidden_size=256,
        # map_location={"cuda:1": "cuda:0"},
    )
    model.learner.augment = nn.Sequential(
        T.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        ),
    )
    model.eval()

    with torch.no_grad():
        # play model
        projections = {
            "camera_1": [],
            "camera_2": [],
            "camera_3": [],
        }
        states = []
        actions = []

        for idx, (camera_1, camera_2, camera_3, state, action) in enumerate(dataloader):
            images = torch.stack([camera_1, camera_2, camera_3], dim=1).to(model.device)
            projections_out = model.learner(images, return_embedding=True)

            for i in range(len(camera_1)):
                projections["camera_1"].append(projections_out[i, 0])
                projections["camera_2"].append(projections_out[i, 1])
                projections["camera_3"].append(projections_out[i, 2])
                states.append(state[i])
                actions.append(action[i])

            print(f"Step {idx+1} of {len(dataloader)}")

    # create and save new dataframe
    projections_camera_1 = [proj.cpu().numpy() for proj in projections["camera_1"]]
    projections_camera_2 = [proj.cpu().numpy() for proj in projections["camera_2"]]
    projections_camera_3 = [proj.cpu().numpy() for proj in projections["camera_3"]]
    states = [state.cpu().numpy().astype(np.float32) for state in states]
    actions = [action.cpu().numpy().astype(np.float32) for action in actions]
    data = {
        ("observations", "latent_camera_1"): projections_camera_1,
        ("observations", "latent_camera_2"): projections_camera_2,
        ("observations", "latent_camera_3"): projections_camera_3,
        ("observations", "state"): states,
        ("policy_action", "action"): actions,
    }
    df = pd.DataFrame(data)

    df.to_pickle(args.file_path.replace(".pkl", "_with_latents.pkl"))
    print(f"Saved dataframe to {args.file_path.replace('.pkl', '_with_latents.pkl')}")
