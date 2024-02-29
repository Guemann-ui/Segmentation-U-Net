import torch
class engine():
    def train_batch(model, data, optimizer, criterion):
        """
        Train the model for one batch.

        Args:
        - model (torch.nn.Module): The neural network model.
        - data (Tuple[torch.Tensor, torch.Tensor]): Tuple containing input images (ims) and ground truth masks (ce_masks).
        - optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        - criterion: Loss and accuracy computation function.

        Returns:
        - Tuple[float, float]: Training loss and accuracy for the batch.
        """
        # Set the model to training mode
        model.train()
        # unpack/unzip input data
        ims, ce_masks = data
        ce_masks = ce_masks.long()
        # print(ce_masks.shape)
        # Forward pass
        _masks = model(ims)
        # print(_masks.shape)
        # zero the gradients
        optimizer.zero_grad()
        # compute loss and accuracy
        loss, acc = criterion(_masks, ce_masks)
        # Backward pass and optimization (update weights)
        loss.backward()
        optimizer.step()
        return loss.item(), acc.item()

    @torch.no_grad()
    def validate_batch(model, data, criterion):
        """
        Validate the model for one batch.

        Args:
        - model (torch.nn.Module): The neural network model.
        - data (Tuple[torch.Tensor, torch.Tensor]): Tuple containing input images (ims) and ground truth masks (masks).
        - criterion: Loss and accuracy computation function.

        Returns:
        - Tuple[float, float]: Validation loss and accuracy for the batch.
        """
        # set the model to evaluation mode
        model.eval()
        # unzip input data
        ims, ce_masks = data
        ce_masks = ce_masks.long()
        # forward pass
        masks = model(ims)
        # compute loss and accuracy
        loss, acc = criterion(masks, ce_masks)

        return loss.item(), acc.item()
