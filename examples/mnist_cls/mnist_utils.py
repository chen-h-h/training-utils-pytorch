import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from PIL import Image

# model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# data_loader
class MNIST(datasets.MNIST):
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'input': img, 'target': target}
    
def build_dataset(dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = MNIST(dir, train=True, download=True, transform=transform)
    test_dataset = MNIST(dir, train=False, transform=transform)
    return train_dataset, test_dataset

def build_dataloader(is_distributed, dataset_dir, batch_size, test_batch_size):
    train_dataset, test_dataset = build_dataset(dataset_dir)

    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              shuffle=(train_sampler is None))
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)
    return train_loader, test_loader