import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

train_batch_size = 32
test_batch_size = 2
num_workers = 4
train_size_rate = 0.8

data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def make_train_dataloader(data_path):

    dataset = datasets.ImageFolder(root=data_path, transform=data_transforms)
    train_size = int(len(dataset) * train_size_rate)
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=train_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader


def make_test_dataloader(data_path):

    testset = datasets.ImageFolder(root=data_path, transform=data_transforms)
    test_loader = DataLoader(testset, batch_size=test_batch_size, num_workers=num_workers)

    return test_loader