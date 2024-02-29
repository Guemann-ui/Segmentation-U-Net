from torch_snippets import *

from transformation import apply_transformation
from wandb_config import config


# download data from here: https://www.dropbox.com/s/0pigmmmynbf9xwq/dataset1.zip
# unzip it and rename it to dataset in the root dir

class segmentationDataset(Dataset):

    # __init__: specifies the image location
    def __init__(self, split):
        self.split = split
        self.items = stems(f'dataset/images_prepped_{split}')

    # __len__: define the length of the dataset.
    def __len__(self):
        return len(self.items)

    # __getitem__: loads an image and mask and resizes them to the same size
    def __getitem__(self, idx):
        # read data
        image = read(f'dataset/images_prepped_{self.split}/{self.items[idx]}.png', 1)
        mask = read(f'dataset/annotations_prepped_{self.split}/{self.items[idx]}.png')
        # resize
        image = cv2.resize(image, (224, 224))
        mask = cv2.resize(mask, (224, 224))
        return image, mask

    # Randomly choose an item from the class instance
    def choose(self): return self[randint(len(self))]

    # Collate function for PyTorch DataLoader
    def collate_fn(self, batch):
        ims, masks = list(zip(*batch))

        ims = torch.cat([apply_transformation()(im.copy() / 255.)[None] for im in ims]).float().to(config.device)

        # Convert masks to grayscale if they have 3 channels
        masks = [mask[:, :, 0] if mask.shape[-1] == 3 else mask for mask in masks]

        # Transform and concatenate masks along a new dimension, convert to long tensor
        ce_masks = torch.cat([torch.Tensor(mask[None]) for mask in masks]).long().to(config.device)

        return ims, ce_masks
