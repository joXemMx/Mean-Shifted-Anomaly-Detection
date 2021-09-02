"""
Copyright by Jonas Emmert

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Insitute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany
"""

from torchvision import datasets


def create_dataset(path, transforms):
    # ToTensor() automatically normalizes pixel values to [0,1]
    dataset = datasets.ImageFolder(path, transform=transforms)
    return dataset
