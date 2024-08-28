import numpy as np
import torch
import time
import argparse
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import multiprocessing
import pandas as pd
import os

from MulticoreTSNE import MulticoreTSNE as TSNE
from scipy.spatial.distance import cdist
from torch import nn
from torchvision import transforms as T
from torchvision import models

from byol_pytorch import InfoNCE, SpartanFileDataset

# arguments
parser = argparse.ArgumentParser(description="export_embeddings")
parser.add_argument(
    "--save_path",
    type=str,
    required=True,
    help="path to the save folder",
)
parser.add_argument(
    "--target_freq_ratio",
    type=int,
    required=True,
    help="target frequency ratio",
)
parser.add_argument(
    "--width",
    type=int,
    required=True,
    help="width of the image for preprocessing",
)
parser.add_argument(
    "--height",
    type=int,
    required=True,
    help="height of the image for preprocessing",
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


file_paths = [
    # "/home/clemensschwarke/Desktop/data/0804_1/log_5/",
    # "/home/clemensschwarke/Desktop/data/0804_1/log_6/",
    # "/home/clemensschwarke/Desktop/data/0804_1/log_7/",
    # "/home/clemensschwarke/Desktop/data/0804_1/log_8/",
    # "/home/clemensschwarke/Desktop/data/0804_2/log_1/",
    # "/home/clemensschwarke/Desktop/data/0804_2/log_2/",
    # "/home/clemensschwarke/Desktop/data/0804_2/log_3/",
    # "/home/clemensschwarke/Desktop/data/0804_2/log_4/",
    # "/home/clemensschwarke/Desktop/data/0804_2/log_5/",
    # "/home/clemensschwarke/Desktop/data/0804_2/log_6/",
    # "/home/clemensschwarke/Desktop/data/0804_2/log_7/",
    # "/home/clemensschwarke/Desktop/data/0804_2/log_8/",
    # "/home/clemensschwarke/Desktop/data/0804_3/log_1/",
    # "/home/clemensschwarke/Desktop/data/0804_3/log_2/",
    # "/home/clemensschwarke/Desktop/data/0804_3/log_3/",
    # "/home/clemensschwarke/Desktop/data/0823_1/log_1/",
    # "/home/clemensschwarke/Desktop/data/0823_1/log_2/",
    # "/home/clemensschwarke/Desktop/data/0823_1/log_3/",
    # "/home/clemensschwarke/Desktop/data/0823_1/log_4/",
    # "/home/clemensschwarke/Desktop/data/0823_1/log_5/",
    # "/home/clemensschwarke/Desktop/data/0823_1/log_6/",
    # "/home/clemensschwarke/Desktop/data/0823_1/log_8/",
    # "/home/clemensschwarke/Desktop/data/0823_1/log_9/",
    # "/home/clemensschwarke/Desktop/data/0823_1/log_10/",
    # "/home/clemensschwarke/Desktop/data/0823_1/log_11/",
    # "/home/clemensschwarke/Desktop/data/0823_1/log_12/",
]

file_paths_val = [
    # "/home/clemensschwarke/Desktop/data/0804_3/log_4/",
    # "/home/clemensschwarke/Desktop/data/0823_1/log_7/",
]


def export_embeddings(file_path, is_val=False):
    dataset = SpartanFileDataset(
        file_path, args.target_freq_ratio, args.width, args.height, IMAGE_SIZE
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        # num_workers=multiprocessing.cpu_count(),
    )

    # create model
    net = models.resnet18()
    model = SelfSupervisedLearner.load_from_checkpoint(
        # Do not forget to change version for file name!
        "/home/clemensschwarke/git/byol-pytorch/lightning_logs/version_215_like_205_sanity/checkpoints/epoch=19-step=1480.ckpt",  # Do not forget to change version for file name!
        # Do not forget to change version for file name!
        net=net,
        image_size=IMAGE_SIZE,
        hidden_layer="avgpool",
        projection_size=32,
        projection_hidden_size=256,
        # map_location={"cuda:1": "cuda:0"},
    )
    model.eval()

    version = "215"  # Do not forget to change version for file name!

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
    states = [s.cpu().numpy().astype(np.float32) for s in states]
    actions = [a.cpu().numpy().astype(np.float32) for a in actions]
    data = {
        ("observations", "latent_camera_1"): projections_camera_1,
        ("observations", "latent_camera_2"): projections_camera_2,
        ("observations", "latent_camera_3"): projections_camera_3,
        ("observations", "state"): states,
        ("policy_action", "action"): actions,
    }
    df = pd.DataFrame(data)

    path_parts = file_path.strip(os.sep).split(os.sep)
    try:
        data_index = path_parts.index("data")
        filename_parts = path_parts[data_index + 1 :]
        filename = f"latent_{version}_{'_'.join(filename_parts)}.pkl"
    except ValueError:
        filename = f"latent_{version}.pkl"

    path = os.path.join(args.save_path, "val" if is_val else "train")
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, filename)
    df.to_pickle(path)
    print(f"Saved dataframe to {path}")


# main
if __name__ == "__main__":
    # load model and compute projections
    # create dataset
    for file_path in file_paths:
        export_embeddings(file_path)
    for file_path in file_paths_val:
        export_embeddings(file_path, is_val=True)
