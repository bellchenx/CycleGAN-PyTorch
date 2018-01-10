import os
import copy
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def get_loader(config, data_dir):
    root = os.path.join(os.path.abspath(os.curdir), data_dir)
    print('-- Loading images')
    dataset = ImageFolder(
        root=root,
        transform=transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    return loader

def denorm(x):
    out = x * 0.5 + 0.5
    return out.clamp(0, 1)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('-- Total number of parameters: %d' % num_params)

class ItemPool(object):
    def __init__(self, max_num=50):
        self.max_num = max_num
        self.num = 0
        self.items = []

    def __call__(self, in_items):
        """`in_items` is a list of item."""
        if self.max_num <= 0:
            return in_items
        return_items = []
        for in_item in in_items:
            if self.num < self.max_num:
                self.items.append(in_item)
                self.num = self.num + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_num)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items