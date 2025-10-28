from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .util import EUROSAT_MEAN, EUROSAT_STD, unnormalize_to01

def eurosat_dataloaders(root, image_size=64, batch_size=64, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=EUROSAT_MEAN, std=EUROSAT_STD)
    ])

    dataset = datasets.EuroSAT(root=root, download=True, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def compute_mean_std(dataset, batch_size=64, num_workers=4):
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

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.EuroSAT(root='data/eurosat', download=True, transform=transform)
    mean, std = compute_mean_std(dataset, batch_size=64, num_workers=4)
    print("Mean:", mean)
    print("Std:", std)