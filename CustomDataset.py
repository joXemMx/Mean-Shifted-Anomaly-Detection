import math
import torch
import torchvision.transforms
from torchvision import datasets, transforms


def create_dataset(path, transforms):
    # ToTensor() automatically normalizes pixel values to [0,1]
    dataset = datasets.ImageFolder(path, transform=transforms)
    return dataset


# def create_dataset(path):
#     # ToTensor() automatically normalizes pixel values to [0,1]
#     dataset = datasets.ImageFolder(path)
#     return dataset


def get_mean_std(dataset, batch_size):
    # create a data loader
    loader = torch.utils.data.DataLoader(dataset,
                             batch_size=batch_size,
                             num_workers=0,
                             shuffle=False)

    # calculate dataset mean and standard deviation
    mean = 0.0
    var = 0.0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    std = torch.sqrt(var / (len(loader.dataset) * 224 * 224))

    mean = [round(float(x), 3) for x in mean]
    std = [round(float(x), 3) for x in std]

    return mean, std


# ds = create_dataset('E:\\parralelcomputed_trackdata\\Heatmaps_test\\', transforms=torchvision.transforms.ToTensor())
# ds1 = create_dataset('E:\\parralelcomputed_trackdata\\Heatmaps_train_norm\\', transforms=torchvision.transforms.ToTensor())
# ds2 = create_dataset('E:\\parralelcomputed_trackdata\\Heatmaps_train_noc\\', transforms=torchvision.transforms.ToTensor())
# whole_ds = torch.utils.data.ConcatDataset([ds, ds1, ds2])
# print(len(whole_ds))
# mean, std = get_mean_std(whole_ds, 1)
# print(mean)
# print(std)
