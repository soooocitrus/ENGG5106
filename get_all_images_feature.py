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
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--model', type=str, default='model.pth', metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outdir', type=str, default='features', metavar='D',
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

#test_dir = args.data + '/test_images/mistery_category'

def read_file_to_dic(path):
    result = {}
    with open(path, 'r') as f:
        for line in f:
            items = line.split()
            key, value = items[0], items[1]
            result[key] = value
    return result

def read_class_label_to_dic(path):
    result = dict()
    with open(path, 'r') as f:
        for line in f:
            items = line.split()
            image_id, class_label = items[0], items[1]
            if class_label in result:
                result[class_label].append(image_id)
            else:
                result[class_label] = [image_id]
    return result

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

images_dic = read_file_to_dic('data/CUB_200_2011/CUB_200_2011/images.txt')
class_label_dic = read_class_label_to_dic('data/CUB_200_2011/CUB_200_2011/image_class_labels.txt')


total = 0
for categary_id in class_label_dic.keys():
    count = 0
    print("Start calculating category " + categary_id)
    output_file = open(args.outdir+'/'+categary_id+'.txt', "w")
    output_file.write("Id|Path|Categories|Feature\n")
    image_ids = class_label_dic[categary_id]
    for image_id in image_ids:
        count +=1
        image_path = images_dic[image_id]
        data = data_transforms(pil_loader('data/CUB_200_2011/CUB_200_2011/images/'+image_path))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        if use_cuda:
            data = data.cuda()
        output = np.reshape(model(data).data, 200)
        feature = np.reshape(model_feature(data),(2048))
        pred = np.argsort(-output)[:5]
        output_file.write("%s|%s|%s|%s\n" % (image_id, image_path, str(pred.tolist()), str(feature.tolist())))
    total += count
    print("Finish calculating category " + categary_id + ". This category contains " + str(count) + " images. By now totally " + str(total) +" images processed")
    output_file.close()

print("Succesfully wrote " + args.outdir + ', you can find each category features with its label number')
