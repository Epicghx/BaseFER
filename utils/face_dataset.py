import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# VGGFACE_MEAN = [0.5, 0.5, 0.5]
# VGGFACE_STD  = [0.5, 0.5, 0.5]
VGGFACE_MEAN = [0.485, 0.456, 0.406]
VGGFACE_STD  = [0.229, 0.224, 0.225]
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def train_loader(path, train_batch_size, num_workers=8, pin_memory=False, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=VGGFACE_MEAN, std=VGGFACE_STD)
    rotation = 30


    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(),
        # transforms.RandomRotation(rotation),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = datasets.ImageFolder(path, train_transform)

    return torch.utils.data.DataLoader(train_dataset,
        batch_size=train_batch_size, shuffle=True, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)


def val_loader(path, val_batch_size, num_workers=8, pin_memory=False, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=VGGFACE_MEAN, std=VGGFACE_STD)

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])
    val_dataset = datasets.ImageFolder(path, val_transform)

    return torch.utils.data.DataLoader(val_dataset,
        batch_size=val_batch_size, shuffle=False, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)
