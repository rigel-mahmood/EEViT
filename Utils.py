import torch
import torchvision
import torchvision.transforms as transforms
from randomaug import RandAugment

from MyImageNetDataSet import MyImageNetDataset
from timm.data import create_transform, Mixup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision import datasets

def get_loaders_CIFAR10(size, batch_size, aug=False,N = 2, M = 14): #if aug=False, no data aug
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Add RandAugment with N, M(hyperparameter)
    if aug:  
        N = 2; M = 14;
        transform_train.transforms.insert(0, RandAugment(N, M, 32))

    # Prepare dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    return trainloader, testloader

def get_loaders_CIFAR100(size, batch_size, aug=True,N = 2, M = 14):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Add RandAugment with N, M(hyperparameter)
    if aug:  
        N = 2; M = 14;
        transform_train.transforms.insert(0, RandAugment(N, M, 32))

    # Prepare dataset
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    return trainloader, testloader

def get_loaders_SVHN(size, batch_size, aug=True,N = 2, M = 14):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),  # Optional, not typically recommended for digits
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728),
                         (0.1980, 0.2010, 0.1970))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728),
                         (0.1980, 0.2010, 0.1970))
    ])

    # Add RandAugment with N, M(hyperparameter)
    if aug:  
        N = 2; M = 14;
        transform_train.transforms.insert(0, RandAugment(N, M, 32))

    ## Download datasets from torchvision.datasets module
    train_ds = torchvision.datasets.SVHN(root = './data', split="train", 
                                        download=True, transform=transform_train)
    test_ds = torchvision.datasets.SVHN(root = './data', split="test", 
                                        download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(test_ds, batch_size=100, shuffle=False, num_workers=8)

    return trainloader, testloader

def get_loaders_CIFAR10_224(size, batch_size, aug=True,N = 2, M = 14):
    transform_train = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Add RandAugment with N, M(hyperparameter)
    if aug:  
        N = 2; M = 14;
        transform_train.transforms.insert(0, RandAugment(N, M))

    # Prepare dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    return trainloader, testloader

def get_loaders_imagenet(size,batch_size, aug=True,N = 2, M = 14):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(int(224 / 0.875)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Add RandAugment with N, M(hyperparameter)
    if aug:  
        N = 2; M = 14;
        transform_train.transforms.insert(0, RandAugment(N, M))

    imagenet_val_dataset = MyImageNetDataset(root='C:/ImagenetData/ILSVRC/Data/CLS-LOC', split='val',transform=transform_val)  

    # Create a DataLoader for the validation dataset
    val_loader = DataLoader(imagenet_val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    imagenet_train_dataset = MyImageNetDataset(root='C:/ImagenetData/ILSVRC/Data/CLS-LOC', split='train',transform=transform_train)  
    train_loader = DataLoader(imagenet_train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    return train_loader, val_loader




