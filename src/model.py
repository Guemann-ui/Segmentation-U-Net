import torch

from loss import UNetLoss
from unet_model import UNet
from wandb_config import config


def define_model():
    """
    Define and initialize the U-Net model, loss criterion, and optimizer.

    Returns:
    - Tuple[torch.nn.Module, Callable, torch.optim.Optimizer]:
      U-Net model, loss criterion, and optimizer.
    """
    # U-Net defined architecture
    model = UNet().to(config.device)
    # loss function
    criterion = UNetLoss
    # define model hyperparamters
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    return model, criterion, optimizer
