import os
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from torchvision import datasets
from tools import Parser, data_transformer_with_augment, data_transformer, CNNLayerVisualization, show_images
from models import simple_cnn, alexnet, vgg16, resnet101
import matplotlib.pyplot as plt


model, input_size = resnet101()

# Training settings
args = Parser().parse()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

if args.path != None:
    state_dict = torch.load(args.path)
    model.load_state_dict(state_dict)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

path_to_images = os.path.abspath(os.path.join(os.curdir, 'bird_dataset', 'train_images'))

# Data initialization and loading
data_transforms_augment = data_transformer_with_augment(input_size)
data_transforms = data_transformer(input_size)

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms_augment),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

# Neural network and optimizer
# We define neural net in cnn.py so that it can be reused by the evaluate.py script

# model = SimpleCNN()
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch):
    model.train()
    # cnn_layer = 2
    # filter_pos = 5
    # # Fully connected layer is not needed
    # layer_vis = CNNLayerVisualization(model.layer4, cnn_layer, filter_pos)
    #
    # # Layer visualization with pytorch hooks
    # layer_vis.visualise_layer_with_hooks()
    # ax1.set_title('Channel 1')
    # show_images(chan_1, ax=ax1)
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # show_images(data, 3, min=0, max=0 + 1)
        # plt.show()
        # print(target)
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item()))


def validation(epoch):
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        valid = pred.eq(target.data.view_as(pred)).cpu()
        wrong_images = np.argwhere(valid.numpy() == 0)[:, 0]
        correct += valid.sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


path = args.experiment + '/' + datetime.today().strftime('%Y-%m-%d %H:%M:%S')
os.mkdir(path)
for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation(epoch)
    model_file = path + '/model.pth'
    torch.save(model.state_dict(), model_file)
    if epoch % 2 == 0:
        os.mkdir(path + '/' + str(epoch))
        intermediate_model_file = path + '/' + str(epoch) + '/model.pth'
        torch.save(model.state_dict(), intermediate_model_file)
        print('\nSaved intermideate model to ' + intermediate_model_file + '. You can run `python main.py --load-model '+
              intermediate_model_file + '` to load the unfinished training model')
    print(
        '\nSaved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file +
        '` to generate the Kaggle formatted csv file')
