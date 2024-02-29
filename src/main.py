import warnings

# Ignore DeprecationWarning warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tqdm import tqdm

from dataloader import _dataloaders
from login_wandb import wandb
from model import define_model
from training import engine
from wandb_config import config


def run():
    """
    Main training loop for the U-Net model.

    Iterates through epochs, performs training and validation,
    and logs metrics using Weights & Biases.
    """
    for epoch in range(config.epoch_num):
        print("####################")
        print(f"       Epoch: {epoch}   ")
        print("####################")

        # Training loop
        for bx, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            # Perform training for a batch and retrieve loss and accuracy
            train_loss, train_acc = engine.train_batch(model, data, optimizer, criterion)

        # Validation loop
        for bx, data in tqdm(enumerate(valloader), total=len(valloader)):
            # Perform validation for a batch and retrieve loss and accuracy
            val_loss, val_acc = engine.validate_batch(model, data, criterion)

        # Log metrics using Weights & Biases
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        print()


if __name__ == "__main__":
    # dataloaders
    trainloader, valloader = _dataloaders()
    # model
    model, criterion, optimizer = define_model()
    # lauch training
    run()
