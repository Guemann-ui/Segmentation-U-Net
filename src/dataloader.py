from torch.utils.data import DataLoader

from dataset import segmentationDataset


def _dataloaders():
    # split dataset
    trainset = segmentationDataset('train')
    valset = segmentationDataset('test')
    # Print the size of the training and validation sets
    print(f"Training set size: {len(trainset)}")
    print(f"Validation set size: {len(valset)}")
    # dataloaders
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, collate_fn=trainset.collate_fn)
    valloader = DataLoader(valset, batch_size=1, shuffle=True, collate_fn=valset.collate_fn)
    return trainloader, valloader



