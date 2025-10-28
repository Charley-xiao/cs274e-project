# Must provide: eurosat_dataloaders(root, image_size, batch_size, num_workers)
# You may also explore the dataset a little bit in a separate notebook
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def compute_mean_std(dataset, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
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