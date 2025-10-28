from torch import Tensor
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms

from .util import EUROSAT_MEAN, EUROSAT_STD, unnormalize_to01
from torchvision.utils import save_image
import os

from typing import Tuple

def eurosat_dataloaders(root: str, 
                        image_size: int = 64, 
                        batch_size: int = 64, 
                        num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Creates training and validation dataloaders for the Eurosat dataset.
    Also exports the validation images to root/val_images/class_label,
    where class_label is an int, for FID calculation.

    Args:
        root (str): Root directory for the dataset.
        image_size (int): Resized image size.
        batch_size (int): The batch size for the dataloaders.
        num_workers (int): The number of worker threads for the dataloaders.
    
    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation dataloaders.
    """

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=EUROSAT_MEAN, std=EUROSAT_STD)
    ])

    dataset = datasets.EuroSAT(root=root, download=True, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    export_validation_images(dataset=val_dataset, 
                             save_dir=os.path.join(root, 'val_images'),
                             mean=EUROSAT_MEAN,
                             std=EUROSAT_STD)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def compute_mean_std(dataset: Dataset,
                     batch_size: int = 64,
                     num_workers: int = 4) -> Tuple[Tensor, Tensor]:
    """
    Computes the mean and std of a dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to compute mean and std for.
        batch_size (int): The batch size for the DataLoader.
        num_workers (int): The number of workers for the DataLoader.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The mean and std tensors.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    mean = 0.0
    std = 0.0
    total_imgs = 0

    for images, labels in loader:
        batch = images.size(0)
        images = images.view(batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_imgs += batch

    mean /= total_imgs
    std /= total_imgs
    return mean, std

def export_validation_images(dataset: Dataset, save_dir: str, mean: Tensor, std: Tensor):
    """
    Saves validation images to disk (save_dir/label) after unnormalizing them for FID calculation.

    Args:
        dataset (torch.utils.data.Dataset): The dataset containing validation images.
        save_dir (str): save directory for the validation images.
        mean (torch.Tensor): mean used to normalize.
        std (torch.Tensor): std used to normalize.
    """
    os.makedirs(save_dir, exist_ok=True)
    for i, (img, label) in enumerate(dataset):
        img = unnormalize_to01(img, mean, std)
        class_dir = os.path.join(save_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        save_image(img, os.path.join(class_dir, f"img_{i}.png"))

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.EuroSAT(root='data/eurosat', download=True, transform=transform)
    mean, std = compute_mean_std(dataset, batch_size=64, num_workers=4)
    print("Mean:", mean)
    print("Std:", std)