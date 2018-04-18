from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
import copy


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()


class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        vgg = models.vgg19(pretrianed=True).features
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        ## generate encoder
        for idx, layer in enumerate(vgg):
            self.encoder.add_module(str(idx),layer)
        ## generate decoder by using the exact opposite way of making the encoder
        ## only changing the maxpooling to upsampling nearest neighbor
        self.decoder.add_module(nn.Conv2d(512,256,kernel_size=(3,3), stride=(1,1), padding=(1,1)))
        self.decoder.add_module(nn.ReLU())
        self.decoder.add_module(nn.UpsamplingNearest2d(scale_factor=2))
        self.decoder.add_module(nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
        self.decoder.add_module(nn.ReLU())
        self.decoder.add_module(nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
        self.decoder.add_module(nn.ReLU())
        self.decoder.add_module(nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
        self.decoder.add_module(nn.ReLU())
        self.decoder.add_module(nn.Conv2d(256, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
        self.decoder.add_module(nn.ReLU())
        self.decoder.add_module(nn.UpsamplingNearest2d(scale_factor=2))
        self.decoder.add_module(nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.decoder.add_module(nn.ReLU())
        self.decoder.add_module(nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.decoder.add_module(nn.ReLU())
        self.decoder.add_module(nn.UpsamplingNearest2d(scale_factor=2))
        self.decoder.add_module(nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.decoder.add_module(nn.ReLU())
        self.decoder.add_module(nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.decoder.add_module(nn.ReLU())

    def forward(self, content, style):
        content_encode = self.encoder(content)
        style_encode = self.encoder(style)






class xun:
    def __init__(self, style_img_dir='', content_img_dir='', use_cuda=False, epochs=4, batch=4):
        self.style_img_dir = style_img_dir
        self.content_img_dir = content_img_dir
        self.epochs = epochs
        self.batch = batch
        self.use_cuda = use_cuda


    def generate_model(self):
        vgg = models.vgg19(pretrained=True).features
        encoder = nn.Sequential()
        for idx, layer in enumerate(vgg):
            encoder.add_module(str(idx),layer)
            if idx==20:
                break


vgg = models.vgg19(pretrained=True).features
cut = nn.Sequential()

for idx, layer in enumerate(vgg):
    cut.add_module(str(idx), layer)
    if idx==20:
        break

print(cut)

