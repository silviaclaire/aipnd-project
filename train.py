import os
import torch
import argparse

from model import create_model
from utils import line, load_dataset


parser = argparse.ArgumentParser(description='Train a new network on a dataset.')

parser.add_argument('data_dir', action='store',
                    help='Directory of the dataset')

parser.add_argument('--save_dir', action='store',
                    dest='save_dir', default='./',
                    help='Set directory to save checkpoints')

parser.add_argument('--arch', action='store',
                    dest='arch', default='resnet18',
                    help='Choose architecture')

parser.add_argument('--learning_rate', action='store',
                    dest='learning_rate', default=0.003, type=float,
                    help='Set learning rate')

parser.add_argument('--hidden_units', action='store',
                    dest='hidden_units', default=256, type=int,
                    help='Set hidden units')

parser.add_argument('--epochs', action='store',
                    dest='epochs', default=3, type=int,
                    help='Set epochs')

parser.add_argument('--gpu', action='store_true',
                    dest='use_gpu', default=False,
                    help='Use GPU for training')

cfg = parser.parse_args()
# cfg = parser.parse_args(['flowers', '--gpu'])

line()
print('Configs:')
print('data_dir         = {!r}'.format(cfg.data_dir))
print('save_dir         = {!r}'.format(cfg.save_dir))
print('arch             = {!r}'.format(cfg.arch))
print('learning_rate    = {!r}'.format(cfg.learning_rate))
print('hidden_units     = {!r}'.format(cfg.hidden_units))
print('epochs           = {!r}'.format(cfg.epochs))
print('use_gpu          = {!r}'.format(cfg.use_gpu))
line()

# get device
if cfg.use_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# load dataset
dataloaders, class_to_idx = load_dataset(cfg.data_dir)
print('dataset loaded.')
line()

# define model, criterion and optimizer
model, criterion, optimizer = create_model(cfg)
model.to(device)
print('model created.')
print(model)
line()

# train network
print('start training...')
train_losses, valid_losses = [], []
steps = 0
running_loss = 0
print_every = 20

for epoch in range(cfg.epochs):
    for inputs, labels in dataloaders['train']:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:

            valid_loss = 0
            accuracy = 0
            model.eval()

            with torch.no_grad():
                for inputs, labels in dataloaders['valid']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    valid_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            train_losses.append(running_loss/len(dataloaders['train']))
            valid_losses.append(valid_loss/len(dataloaders['valid']))

            print(f"Epoch {epoch+1}/{cfg.epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Valid loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                  f"Valid accuracy: {accuracy/len(dataloaders['valid']):.3f}")

            running_loss = 0
            model.train()

print('finished training.')
line()

# save checkpoint
checkpoint = {'epochs': cfg.epochs,
              'arch': cfg.arch,
              'hidden_units': cfg.hidden_units,
              'learning_rate': cfg.learning_rate,
              'class_to_idx': class_to_idx,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()}
save_path = os.path.join(cfg.save_dir, 'checkpoint_cli.pth')
torch.save(checkpoint, save_path)
print(f'checkpoint saved to {save_path}.')
line()
