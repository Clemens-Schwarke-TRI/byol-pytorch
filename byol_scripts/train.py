import argparse
from torchvision import models

from byol_pytorch import BYOLTrainer, TwoImagesDataset

# arguments
parser = argparse.ArgumentParser(description="byol-lightning-test")
parser.add_argument(
    "--image_folder",
    type=str,
    required=True,
    help="path to your folder of images for self-supervised learning",
)
args = parser.parse_args()


# constants
BATCH_SIZE = 128
NUM_TRAIN_STEPS = 10000
LR = 3e-4
IMAGE_SIZE = 256

# main
if __name__ == "__main__":
    # create dataset
    dataset = TwoImagesDataset(args.image_folder, IMAGE_SIZE)
    # create model, trainer, and optimizer
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # create trainer
    trainer = BYOLTrainer(
        resnet,
        dataset=dataset,
        image_size=IMAGE_SIZE,
        hidden_layer="avgpool",
        learning_rate=LR,
        num_train_steps=NUM_TRAIN_STEPS,
        batch_size=BATCH_SIZE,
    )

    # train
    trainer()
