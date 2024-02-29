from torchvision import transforms
def apply_transformation():
    """
    Returns a composition of image transformations commonly used in neural network training.
    Transforms:
    1. ToTensor(): Converts the image to a PyTorch tensor.
    2. Normalize(mean, std): Normalizes the tensor by subtracting the mean and dividing by the standard deviation
       along each channel. The provided mean and std correspond to the ImageNet dataset values.

    Returns:
    transforms.Compose: Composition of the specified transformations.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
