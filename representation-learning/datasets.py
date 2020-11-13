import torch
import torch.utils.data as data

import torchvision.datasets as datasets
import torchvision.transforms as transforms


class CIFAR10(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(data.Dataset).__init__()
        self.cifar10 = datasets.CIFAR10(root, train, download=download)
        self.data = torch.tensor(self.cifar10.data.transpose((0, 3, 1, 2)))
        self.targets = torch.tensor(self.cifar10.targets)
        self.tensor_dataset = data.TensorDataset(self.data, self.targets)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, index):
        img, target = self.tensor_dataset[index]

        img = transforms.ToPILImage()(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(data.Dataset).__init__()
        self.cifar10 = datasets.CIFAR100(root, train, download=download)
        self.data = torch.tensor(self.cifar10.data.transpose((0, 3, 1, 2)))
        self.targets = torch.tensor(self.cifar10.targets)
        self.tensor_dataset = data.TensorDataset(self.data, self.targets)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, index):
        img, target = self.tensor_dataset[index]

        img = transforms.ToPILImage()(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
