import argparse
from tqdm import tqdm
import os
import PIL.Image as Image
from models import resnet101
import torch
import torch.nn as nn
import numpy as np
from tools import data_transformer

parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='input.png', metavar='D',
                    help="input image file to be processed.")
parser.add_argument('--model', type=str, default='model.pth', metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='input_feature.txt', metavar='D',
                    help="output directory of the images features")

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

state_dict = torch.load(args.model,map_location=lambda storage, loc: storage)
model, input_size = resnet101()
model.load_state_dict(state_dict)
model.eval()

model_feature = nn.Sequential(*list(model.children())[:-1])
for param in model_feature.parameters():
    param.requires_grad = False


if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

data_transforms = data_transformer(input_size)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')



output_file = open(args.outfile, "w")
output_file.write("Path|Categories|Feature\n")
data = data_transforms(pil_loader(args.data))
data = data.view(1, data.size(0), data.size(1), data.size(2))
if use_cuda:
    data = data.cuda()
output = np.reshape(model(data).data, 200)
feature = np.reshape(model_feature(data),(2048))
pred = np.argsort(-output)[:5]
output_file.write("%s|%s|%s\n" % (args.data, str(pred.tolist()), str(feature.tolist())))
output_file.close()

print("Succesfully wrote " + args.outfile + ', you can find input image info in it.')
