import torch
import torch.nn as nn

loss_func = nn.CrossEntropyLoss()


def UNetLoss(preds, labels):
    """
    Compute the U-Net loss using a specified loss function and accuracy.

    Args:
    - preds (torch.Tensor): Predictions from the model.
    - labels (torch.Tensor): Ground truth labels.
    - loss_func (torch.nn.Module): Loss function (e.g., nn.CrossEntropyLoss).

    Returns:
    - Tuple[torch.Tensor, torch.Tensor]: Loss and accuracy.
    """
    loss = loss_func(preds, labels)
    accuracy = (torch.max(preds, 1)[1] == labels).float().mean()
    return loss, accuracy
