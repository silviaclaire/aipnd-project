import os
import glob
import json
import torch
import random
import argparse
import numpy as np
from PIL import Image

from model import create_model
from utils import line, Params


parser = argparse.ArgumentParser(description='Train a new network on a dataset.')

parser.add_argument('image_filepath', action='store',
                    help='Filepath of an image')

parser.add_argument('checkpoint_filepath', action='store',
                    help='Filepath of a checkpoint created by train.py')

parser.add_argument('--category_names ', action='store',
                    dest='category_names', default='cat_to_name.json',
                    help='Use a mapping of categories to real names')

parser.add_argument('--top_k', action='store',
                    dest='top_k', default=5, type=int,
                    help='Return top K most likely classes')

parser.add_argument('--gpu', action='store_true',
                    dest='use_gpu', default=False,
                    help='Use GPU for training')

cfg = parser.parse_args()
# random_idx = str(random.randint(1, 102))
# image_list = glob.glob(f'flowers/test/{random_idx}/*.jpg')
# image_path = random.choice(image_list)
# with open('cat_to_name.json', 'r') as f:
#     cat_to_name = json.load(f)
# line()
# print(f'answer: {cat_to_name[random_idx]}')
# cfg = parser.parse_args([image_path, 'checkpoint_cli.pth', '--gpu'])

line()
print('Configs:')
print('image_filepath       = {!r}'.format(cfg.image_filepath))
print('checkpoint_filepath  = {!r}'.format(cfg.checkpoint_filepath))
print('category_names       = {!r}'.format(cfg.category_names))
print('top_k                = {!r}'.format(cfg.top_k))
print('use_gpu              = {!r}'.format(cfg.use_gpu))
line()

# get device
if cfg.use_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def process_image(image_filepath):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image_filepath)

    # resize the images where the shortest side is 256 pixels, keeping the aspect ratio
    if image.width > image.height:
        target_height = 256
        target_width = image.width * 256 / image.height
    else:
        target_width = 256
        target_height = image.height * 256 / image.width
    image.thumbnail((target_width, target_height))

    # center crop 224x224
    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = (image.width + 224) / 2
    bottom = (image.height + 224) / 2
    image = image.crop((left, top, right, bottom))

    # convert to numpy array
    image = np.array(image)

    # Scale all values between 0 and 1
    image = image / 255

    # Normalize based on the preset mean and standard deviation
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    for i in range(2):
        image[i] = (image[i] - mean[i]) / std[i]

    # reorder dimensions
    image = image.transpose((2, 0, 1))

    return image


def predict(image_filepath, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # preprocess image
    image = process_image(image_filepath)
    image = torch.from_numpy(image).float().to(device)

    # add a fourth dimension to the beginning to indicate batch size
    image.unsqueeze_(0)

    # pass the image through our model
    model.eval()
    output = model.forward(image)

    # reverse the log function in our output
    output = torch.exp(output)

    # get the top predicted class, and the output percentage for that class
    probs, idxs = output.topk(topk, dim=1)
    probs = probs.data.cpu().numpy()[0]
    idxs = idxs.data.cpu().numpy()[0]

    # map index to class
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in idxs]

    return probs, classes


# create model from checkpoint
checkpoint = torch.load(cfg.checkpoint_filepath)
model, _, _ = create_model(Params(checkpoint))
model.class_to_idx = checkpoint['class_to_idx']
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

# predict image
probs, classes = predict(cfg.image_filepath, model, topk=cfg.top_k)

# covert classes to names
with open(cfg.category_names, 'r') as f:
    cat_to_name = json.load(f)
labels = [cat_to_name[c] for c in classes]

# output prediction result
top_k_result = {label: prob for label, prob in zip(labels, probs)}
print('prediction:')
print(top_k_result)
line()
