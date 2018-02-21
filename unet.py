from torch.utils.data.dataset import Dataset
import os
from os.path import isfile, join
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm_notebook as tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn

use_cuda = torch.cuda.is_available()


class CARVANA(Dataset):
    """
        CARVANA dataset that contains car images as .jpg. Each car has 16 images
        taken in different angles and a unique id: id_01.jpg, id_02.jpg, ..., id_16.jpg
        The labels are provided as a .gif image that contains the manually cutout mask
        for each training image
    """

    def __init__(self, root, subset="train", transform=None):
        """

        :param root: it has to be a path to the folder that contains the dataset folders
        :param train: boolean true if you want the train set false for the test one
        :param transform: transform the images and labels
        """

        # initialize variables
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.subset = subset
        self.data_path, self.labels_path = [], []

        def load_images(path):
            """
            returns all the sorted image paths.

            :param path:
            :return: array with all the paths to the images
            """
            images_dir = [join(path, f) for f in os.listdir(path) if isfile(join(path, f))]
            images_dir.sort()

            return images_dir

        # load the data regarding the subset
        if self.subset == "train":
            self.data_path = load_images(self.root + "/train")
            self.labels_path = load_images(self.root + "/train_masks")
        elif self.subset == "val":
            self.data_path = load_images(self.root + "/val")
            self.labels_path = load_images(self.root + "/val_masks")
        elif self.subset == "test":
            self.data_path = load_images(self.root + "/test")
            self.labels_path = None
        else:
            raise RuntimeError('Invalid subset ' + self.subset + ', it must be one of:'
                                                                 ' \'train\', \'val\' or \'test\'')

    def __getitem__(self, index):
        """

        :param index:
        :return: tuple (img, target) with the input data and its label
        """

        # load image and labels
        img = Image.open(self.data_path[index])
        target = Image.open(self.labels_path[index]) if not self.subset == 'test' else None

        # apply transforms to both
        if self.transform is not None:
            img = self.transform(img)
            target = self.transform(target)

        return img, target

    def __len__(self):
        return len(self.data_path)


# create datasets
train_dataset = CARVANA(root='./data',
                              subset="train",
                              transform=transforms.Compose([
                                  transforms.Scale((256,256)),
                                  transforms.ToTensor()])
                              )

val_dataset = CARVANA(root='./data',
                              subset="val",
                              transform=transforms.Compose([
                                  transforms.Scale((256,256)),
                                  transforms.ToTensor()])
                              )


# define the dataloader with the previous dataset
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=8,
                                           shuffle=True,
                                           pin_memory=False, # If True, the data loader will copy tensors into CUDA pinned memory before returning them.
                                           num_workers=0) # change to 1 if run in the server


val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=8,
                                           shuffle=True,
                                           pin_memory=False, # If True, the data loader will copy tensors into CUDA pinned memory before returning them.
                                           num_workers=0) # change to 1 if run in the server


 
class conv_block(nn.Module):
    """
    Define the convolutional - batch norm - relu block to avoid re-writing it
    every time
    """

    def __init__(self, in_size, out_size, kernel_size=3, padding=1, stride=1):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size,
                              padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))



class small_UNET_256(nn.Module):
    """
    Define UNET model that accepts a 256 input and mostly uses 3x3 kernels
    with stride and padding = 1. It reduces the size of the image to 8x8 pixels
    ** It might not work if the input 'x' is not a square.

    detail about nn.conv:
    torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    """

    def __init__(self):
        super(small_UNET_256, self).__init__()


        '''
        more detail about down_1: 
        nn.Conv2d(3, 16, 3) --> nn.Conv2d(16, 32, 3)
        '''

        self.down_1 = nn.Sequential(
            conv_block(3, 16),  # initial colordim = 3
            conv_block(16, 32, stride=2, padding=1))

        self.down_2 = nn.Sequential(
            conv_block(32, 64),
            conv_block(64, 128))

        self.middle = conv_block(128, 128, kernel_size=1, padding= 0)

        self.up_2 = nn.Sequential(
            conv_block(256, 128),
            conv_block(128, 32))

        self.up_1 = nn.Sequential(
            conv_block(64, 64),
            conv_block(64, 32))

        self.output = nn.Sequential(
            conv_block(32, 16),
            conv_block(16, 1, kernel_size=1, padding=0))

    def forward(self, x):
        # 256
        down1 = self.down_1(x) # image: 3*256*256 ---> 16*254*254 ---> 32*252*252
        out = F.max_pool2d(down1, kernel_size=2, stride=2)

        # 64
        down2 = self.down_2(out)
        out = F.max_pool2d(down2, kernel_size=2, stride=2)

        # 8
        out = self.middle(out)

        # 64
        out = F.upsample(out, scale_factor=2) # up-conv 2*2
        out = torch.cat([down2, out], 1)
        out = self.up_2(out)

        # 128
        out = F.upsample(out, scale_factor=2)
        out = torch.cat([down1, out], 1)
        out = self.up_1(out)

        # 256
        out = F.upsample(out, scale_factor=2)
        return self.output(out)


class BCELoss2d(nn.Module):
    #
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = F.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)




def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        if use_cuda:
            data = Variable(data.cuda())
            target = Variable(target.cuda())
        else:
            data = Variable(data)
            target = Variable(target)

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100 * batch_idx / len(train_loader), loss.data[0]))



def evaluate():
    model.eval()
    val_loss = 0
    correct = 0
    for data, target in val_loader:

        if use_cuda:
            data = Variable(data.cuda(), volatile=True)
            target = Variable(target.cuda())
        else:
            data = Variable(data, volatile=True)
            target = Variable(target)


        output = model(data)

        val_loss += criterion(output, target).data[0] # sum up batch loss                                                               
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability                                                                 
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    val_loss /= len(val_loader.dataset) # mean loss
    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100 * correct / len(val_loader.dataset)))


model = small_UNET_256()
criterion = BCELoss2d()

if use_cuda:
    model.cuda()
    criterion.cuda()

optimizer = optim.SGD(model.parameters(),
                      weight_decay=1e-4,
                      lr=1e-4,
                      momentum=0.9,
                      nesterov=True)


for epoch in range(1, 3):
    train(epoch)
    evaluate()



