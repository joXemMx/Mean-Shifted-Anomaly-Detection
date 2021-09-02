import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import faiss
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import ImageFilter
import random
import CustomDataset

# include mean and std from CustomDataset.ipynb here
mean = [0.249, 0.292, 0.734]
std = [0.075, 0.035, 0.079]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


# do transformations on the image data
transform_color = transforms.Compose([

    # resize the input image to the given size
    # if size is an int, smaller edge of the image will be matched to this number
    transforms.Resize(256),

    # crops the given image at the center
    # if image size is smaller than output size along any edge, image is padded with 0 and then center cropped
    # if size is an int, a square crop (size, size) is made
    transforms.CenterCrop(224),

    # Converts a PIL (Python Imaging Library) Image or numpy.ndarray (H x W x C) in the range [0, 255] to a
    # torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    transforms.ToTensor(),

    # Normalize a tensor image with mean and standard deviation. This transform does not support PIL Image.
    # Given mean: (mean[1],...,mean[n]) and std: (std[1],..,std[n]) for n channels, this transform will normalize
    # each channel of the input.
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transforms.Normalize(mean=mean, std=std)])

moco_transform = transforms.Compose([

    # Crop a random portion of image and resize it to a given size.
    # scale: Specifies the lower and upper bounds for the random area of the crop, before resizing.
    # The scale is defined with respect to the area of the original image.
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),

    # Apply randomly a list of transformations with a given probability.
    transforms.RandomApply([
        # Randomly change the brightness, contrast, saturation and hue of an image.
        # variables: brightness, contrast, saturation, hue
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),

    # Randomly convert image to grayscale with a probability of p (default 0.1).
    transforms.RandomGrayscale(p=0.2),

    transforms.RandomApply([
        # Blurs image with randomly chosen Gaussian blur.
        # kernel_size (int or sequence) – Size of the Gaussian kernel
        GaussianBlur([.1, 2.])
    ], p=0.5),

    # Horizontally flip the given image randomly with a given probability. Default value is 0.5.
    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])


class Transform:
    def __init__(self):
        self.moco_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        x_1 = self.moco_transform(x)
        x_2 = self.moco_transform(x)
        return x_1, x_2


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet152(pretrained=True)

        # A placeholder identity operator that is argument-insensitive.
        # basically a 'dummy' layer
        self.backbone.fc = torch.nn.Identity()
        freeze_parameters(self.backbone, train_fc=False)

    def forward(self, x):
        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n


def freeze_parameters(model, train_fc=False):
    for p in model.conv1.parameters():
        # not update (freeze) parts of the network
        p.requires_grad = False
    for p in model.bn1.parameters():
        p.requires_grad = False
    for p in model.layer1.parameters():
        p.requires_grad = False
    for p in model.layer2.parameters():
        p.requires_grad = False
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False


def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN (K-nearest neighbour) distance
    """
    # Faiss is a library for efficient similarity search and clustering of dense vectors.
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)


def get_loaders(dataset, label_class, batch_size):

    # if dataset == "cifar10":
    #     ds = torchvision.datasets.CIFAR10
    #     transform = transform_color
    #     coarse = {}
    #     trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
    #     testset = ds(root='data', train=False, download=True, transform=transform, **coarse)
    #     trainset_1 = ds(root='data', train=True, download=True, transform=Transform(), **coarse)
    #
    #     # True where target is part of wanted label_class, False else
    #     idx = np.array(trainset.targets) == label_class
    #
    #     # 0 where target class is label_class, 1 else
    #     testset.targets = [int(t != label_class) for t in testset.targets]
    #
    #     # take the train data from the indexes where the label is the one of label_class
    #     # thereby cuts len(trainset.data) from 50.000 to 5.000, as there are 10 sets equally distributed
    #     trainset.data = trainset.data[idx]
    #
    #     # targets are of the label class
    #     trainset.targets = [trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
    #
    #     # again: data where label == label_class, targets are label_class
    #     trainset_1.data = trainset_1.data[idx]
    #     trainset_1.targets = [trainset_1.targets[i] for i, flag in enumerate(idx, 0) if flag]
    #
    #     train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,
    #                                                drop_last=False)
    #     test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2,
    #                                               drop_last=False)
    #     return train_loader, test_loader, torch.utils.data.DataLoader(trainset_1, batch_size=batch_size,
    #                                                                   shuffle=True, num_workers=2, drop_last=False)

    if dataset == 'custom':
        test_path = '...'
        train_path = '...'
        transform = transform_color
        testset = CustomDataset.create_dataset(test_path, transforms=transform)
        trainset = CustomDataset.create_dataset(train_path, transforms=transform)
        trainset_1 = CustomDataset.create_dataset(train_path, transforms=Transform())
        # print(testset.targets)

        # testset.targets = [int(t != label_class) for t in testset.targets]
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,
                                  drop_last=False)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2,
                                 drop_last=False)
        return train_loader, test_loader, DataLoader(trainset_1, batch_size=batch_size,
                                                     shuffle=True, num_workers=2, drop_last=False)
    else:
        print('Unsupported Dataset')
        exit()
