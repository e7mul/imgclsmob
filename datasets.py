import torch
import torchvision
import numpy as np
from os import path

from torchvision import transforms
import torch.utils.data as data


def CIFAR100(dataroot):
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=True,
        download=True,
        transform=val_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset


class CacheClassLabel(data.Dataset):
    """
    A dataset wrapper that has a quick access to all labels of data.
    """
    def __init__(self, dataset):
        super(CacheClassLabel, self).__init__()
        self.dataset = dataset
        self.labels = torch.LongTensor(len(dataset)).fill_(-1)
        label_cache_filename = path.join(dataset.root, str(type(dataset))+'_'+str(len(dataset))+'.pth')
        if path.exists(label_cache_filename):
            self.labels = torch.load(label_cache_filename)
        else:
            for i, data in enumerate(dataset):
                self.labels[i] = data[1]
            torch.save(self.labels, label_cache_filename)
        self.number_classes = len(torch.unique(self.labels))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img,target = self.dataset[index]
        return img, target


class Subclass(data.Dataset):
    """
    A dataset wrapper that return the task name and remove the offset of labels (Let the labels start from 0)
    """
    def __init__(self, dataset, class_list, remap=False):
        '''
        :param dataset: (CacheClassLabel)
        :param class_list: (list) A list of integers
        :param remap: (bool) Ex: remap class [2,4,6 ...] to [0,1,2 ...]
        '''
        super(Subclass,self).__init__()
        assert isinstance(dataset, CacheClassLabel), 'dataset must be wrapped by CacheClassLabel'
        self.dataset = dataset
        self.class_list = class_list
        self.remap = remap
        self.indices = []
        for c in class_list:
            self.indices.extend((dataset.labels==c).nonzero().flatten().tolist())
        if remap:
            self.class_mapping = {c: i for i, c in enumerate(class_list)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img,target = self.dataset[self.indices[index]]
        if self.remap:
            raw_target = target.item() if isinstance(target,torch.Tensor) else target
            target = self.class_mapping[raw_target]
        return img, target


def continual_wrapper(dataset, val_dataset, num_tasks=10):
    train_dict, val_dict = {}, {}
    classes = np.random.permutation(np.unique(dataset.labels))
    for task in range(num_tasks):
        train_dict[task] =  Subclass(dataset, classes[task*10:(task+1)*10])
        val_dict[task] =  Subclass(val_dataset, classes[task*10:(task+1)*10])
    return train_dict, val_dict

if __name__ == '__main__':
    pass