import torch
from torch import nn
from torch import optim
from torchvision import models
from collections import OrderedDict


def create_model(params):
    assert(all([key in dir(params) for key in ['arch', 'hidden_units', 'learning_rate']]))
    model = getattr(models, params.arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(model.fc.in_features, params.hidden_units)),
                            ('relu', nn.ReLU()),
                            ('fc2', nn.Linear(params.hidden_units, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=params.learning_rate)
    return model, criterion, optimizer
