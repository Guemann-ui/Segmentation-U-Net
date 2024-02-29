import torch
import wandb


class config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.001  # learning rate
    epoch_num = 10

    # wandb config
    WANDB_CONFIG = {'_wandb_kernel': 'neuracort'}
    # initialize W&B
    run = wandb.init(
        project="semantic_seg_unet_workshop",
        config=WANDB_CONFIG)
