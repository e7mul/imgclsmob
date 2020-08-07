import torch
import torch.nn as nn
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from pytorch.pytorchcv.models.seresnet_cifar import seresnet20_cifar100
from datasets import continual_wrapper, CIFAR100
from experiment import learn, test


def main(config):
    acc_dict = {}
    trainset, valset = CIFAR100('data/')
    train_dict, val_dict = continual_wrapper(trainset, valset, num_tasks=10)

    net = seresnet20_cifar100().to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=config.lr, momentum=config.mo)
    for task in range(config.num_tasks):
        trainloader = torch.utils.data.DataLoader(train_dict[task], batch_size=config.bs)
        valloader = torch.utils.data.DataLoader(val_dict[task], batch_size=config.bs)
        learn(net, trainloader, valloader, criterion, optimizer, config)
        acc_dict[task] = {}
        for i in range(task + 1):
            valloader = torch.utils.data.DataLoader(val_dict[i], batch_size=config.bs)
            acc_dict[task][i] = test(net, valloader, criterion, config)[1]
    visualise(acc_dict)


def dict2array(acc_dict):
    acc_array = np.zeros((len(acc_dict), len(acc_dict)))
    for key, val in acc_dict.items():
        for k, v in val.items():
            acc_array[k, key] = v
    return acc_array


def visualise(acc_dict):
    fig, axs = plt.subplots(1,2, figsize=(12, 5))
    val_array = dict2array(acc_dict)
    test_acc = np.mean(val_array, axis = 0)
    axs[0].imshow(val_array, vmin=0, vmax=1)
    for i in range(val_array.shape[0]):
        for j in range(val_array.shape[1]):
            if j >= i:
                axs[0].text(j,i, round(val_array[i,j], 2), va='center', ha='center', c='w')
    axs[0].set_ylabel('Number of task')
    axs[0].set_xlabel('Tasks finished')
    axs[1].plot(test_acc)
    axs[1].set_xlabel('Tasks finished')
    axs[1].set_ylabel('Accuracy %')
    fig.suptitle(f"CIFAR-100", fontsize=15)
    plt.savefig('test.png')

if __name__ == '__main__':

    @dataclass
    class Config:
        num_tasks = 10
        bs = 128
        lr = 1e-2
        mo = 0.9
        epochs = 10
        device = torch.device('cuda:0')

    main(Config)



