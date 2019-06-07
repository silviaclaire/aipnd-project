import os
import torch
from torchvision import datasets, transforms

def line():
    print('\n-----------------------------------\n')

def load_dataset(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir  = os.path.join(data_dir, 'valid')

    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),
        'valid': transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
    }

    # classname (folder name) : idx (integer label used in training)
    class_to_idx = image_datasets['train'].class_to_idx

    return dataloaders, class_to_idx


class Params:
    def __init__(self, dict_):
        for key in dict_:
            setattr(self, key, dict_[key])
