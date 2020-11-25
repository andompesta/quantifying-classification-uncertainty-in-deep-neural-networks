from torchvision import datasets, transforms
from dynaconf import settings
import os
import torch

def norm_(sample):
    sample = sample / sample.max()
    return sample

def mnist_collate(samples):
    x, y = tuple(zip(*samples))
    return torch.stack(x), torch.LongTensor(y)

def get_mnist_dataset(transformations=[
    transforms.ToTensor(),
    transforms.Lambda(norm_)
]):
    root_path = settings.get("data_dir")

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    train_set = datasets.MNIST(root=root_path, transform=transforms.Compose(transformations), train=True, download=True)
    test_set = datasets.MNIST(root=root_path, transform=transforms.Compose(transformations), train=False, download=True)

    return train_set, test_set


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt

    t, s = get_mnist_dataset()
    for i, l in DataLoader(t, batch_size=2, collate_fn=mnist_collate):
        print(i)
        print(l)
        plt.imshow(i[0].numpy().reshape(28, 28))
        plt.show()
        break
