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
import numpy as np


class content_loss(nn.Module):
    def __init__(self, target, weight):
        super(content_loss, self).__init__()
        self.target = target.detach()
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target * self.weight)
        return input.clone()

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss



class style_loss(nn.Module):
    def __init__(self, target, weight):
        super(style_loss, self).__init__()
        self.target = target.detach()
        self.weight = weight
        self.criterion = nn.MSELoss()

    def gram_matrix(self, input_tensor):
        batch, channel, width, height = input_tensor.size()
        resize = input_tensor.view(batch * channel, width * height)
        gram = torch.mm(resize, resize.t())
        return gram / (batch * channel * width * height)

    def forward(self, input):
        self.target_gram = self.gram_matrix(self.target)
        self.target_gram.mul_(self.weight)
        self.input_gram = self.gram_matrix(input)
        self.input_gram.mul_(self.weight)
        self.loss = self.criterion(self.input_gram, self.target_gram)
        return input.clone()

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class gatys:

    def __init__(self, imsize=(128,128), style_loss_weight=0, content_loss_weight=1, epochs=300):
        self.imsize = imsize
        self.height, self.width = self.imsize
        self.style_loss_weight = style_loss_weight
        self.content_loss_weight = content_loss_weight
        self.epochs = epochs
        self.dtype = torch.FloatTensor
        self.toPIL = transforms.ToPILImage()
        self.content_img = 0
        self.style_img = 0
        self.vgg = models.vgg16(pretrained=True).features
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1','conv_2','conv_3','conv_4','conv_5']
        self.conv_content_layer_idx = []
        self.conv_style_layer_idx = []
        self.relu_content_layer_idx = []
        self.relu_style_layer_idx = []
        self.pool_content_layer_idx = []
        self.pool_style_layer_idx = []
        self.output = 0
        self.img_preprocess = transforms.Compose([
                                transforms.Resize(self.imsize),
                                transforms.ToTensor()
                            ])

    def load_content_img(self, img_dir):
        img = Image.open(img_dir)
        img = Variable(self.img_preprocess(img))
        img = img.unsqueeze(0)
        self.content_img = img.type(self.dtype)

    def load_style_img(self, img_dir):
        img = Image.open(img_dir)
        img = Variable(self.img_preprocess(img))
        img = img.unsqueeze(0)
        self.style_img = img.type(self.dtype)

    def imshow(self, input_tensor, title=None):
        img = input_tensor.clone().cpu()
        img = img.view(3, self.height, self.width)
        img = self.toPIL(img.data)
        plt.imshow(img)
        plt.title(title)
        plt.pause(0.001)
        plt.show()

    def imsave(self, input_tensor):
        img = input_tensor.clone().cpu()
        img = img.view(3, self.height, self.width)
        img = self.toPIL(img.data)
        img.save('output.jpg')

    def save_img(self, input_tensor, img_dir):
        img = input_tensor.clone().cpu()
        img = img.view(3, self.height, self.width)
        torchvision.utils.save_image(img, img_dir)

    def choose_layers(self, content_layer_list=[], style_layer_list=[]):
        if content_layer_list:
            self.content_layers = content_layer_list
        if style_layer_list:
            self.style_layers = style_layer_list
        for content in self.content_layers:
            if content[0] == 'c':
                self.conv_content_layer_idx.append(int(content.split('_')[-1]))
            if content[0] == 'r':
                self.relu_content_layer_idx.append(int(content.split('_')[-1]))
            if content[0] == 'm':
                self.pool_content_layer_idx.append(int(content.split('_')[-1]))
        for style in self.style_layers:
            if style[0] == 'c':
                self.conv_style_layer_idx.append(int(style.split('_')[-1]))
            if style[0] == 'r':
                self.relu_style_layer_idx.append(int(style.split('_')[-1]))
            if style[0] == 'm':
                self.pool_style_layer_idx.append(int(style.split('_')[-1]))

    def generate_model(self):
        content_losses = []
        style_losses = []
        last_conv_idx = max(self.conv_content_layer_idx + self.conv_style_layer_idx, default=0)
        last_relu_idx = max(self.relu_content_layer_idx + self.relu_style_layer_idx, default=0)
        last_pool_idx = max(self.pool_content_layer_idx + self.pool_style_layer_idx, default=0)
        conv_cnt = [1]
        relu_cnt = [1]
        pool_cnt = [1]
        network = copy.deepcopy(self.vgg)
        new_model = nn.Sequential()

        for idx in range(len(network)):
            layer_name = torch.typename(network[idx]).split('.')[-1]

            if layer_name == 'Conv2d':
                name = layer_name + '_' + str(conv_cnt[0])
                new_model.add_module(name, network[idx])
                if conv_cnt[0] in self.conv_content_layer_idx:
                    target = new_model(self.content_img).clone()
                    con_loss = content_loss(target, self.content_loss_weight)
                    new_model.add_module("content_loss_conv_"+str(conv_cnt[0]), con_loss)
                    content_losses.append(con_loss)
                if conv_cnt[0] in self.conv_style_layer_idx:
                    target = new_model(self.style_img).clone()
                    sty_loss = style_loss(target, self.style_loss_weight)
                    new_model.add_module("style_loss_conv_"+str(conv_cnt[0]), sty_loss)
                    style_losses.append(sty_loss)
                conv_cnt[0] += 1

            if layer_name == 'ReLU':
                name = layer_name + '_' + str(relu_cnt[0])
                new_model.add_module(name, network[idx])
                if relu_cnt[0] in self.relu_content_layer_idx:
                    target = new_model(self.content_img).clone()
                    con_loss = content_loss(target, self.content_loss_weight)
                    new_model.add_module("content_loss_relu_" + str(relu_cnt[0]), con_loss)
                    content_losses.append(con_loss)
                if relu_cnt[0] in self.relu_style_layer_idx:
                    target = new_model(self.style_img).clone()
                    sty_loss = style_loss(target, self.style_loss_weight)
                    new_model.add_module("style_loss_relu_" + str(conv_cnt[0]), sty_loss)
                    style_losses.append(sty_loss)
                relu_cnt[0] += 1

            if layer_name == 'MaxPool2d':
                name = layer_name + '_' + str(pool_cnt[0])
                new_model.add_module(name, network[idx])
                if pool_cnt[0] in self.pool_content_layer_idx:
                    target = new_model(self.content_img).clone()
                    con_loss = content_loss(target, self.content_loss_weight)
                    new_model.add_module("content_loss_pool_" + str(pool_cnt[0]), con_loss)
                    content_losses.append(con_loss)
                if pool_cnt[0] in self.pool_style_layer_idx:
                    target = new_model(self.style_img).clone()
                    sty_loss = style_loss(target, self.style_loss_weight)
                    new_model.add_module("style_loss_pool" + str(pool_cnt[0]), sty_loss)
                    style_losses.append(sty_loss)
                pool_cnt[0] += 1
            if conv_cnt[0] > last_conv_idx and relu_cnt[0] > last_relu_idx and pool_cnt[0] > last_pool_idx:
                break

        return new_model, style_losses, content_losses


    def run_network(self):
        initial_input = Variable(torch.randn(self.content_img.size())).type(self.dtype)
        generated_img = nn.Parameter(initial_input.data)
        optimizer = optim.LBFGS([generated_img])
        ## print('Generating Gatys model...')
        ## print(' ')
        model, style_losses, content_losses = self.generate_model()
        epoch = [0]
        ## print('Optimizing Gatys model...')
        ## print(' ')
        """
        if self.use_tv == True:
            tv_x = torch.sum(torch.abs(output[:,:,:,1:]-output[:,:,:,:-1]))
            tv_y = torch.sum(torch.abs(output[:,:,1:,:]-output[:,:,:-1,:]))
            tv_loss = self.tv_weight * (tv_x + tv_y)
            loss += tv_loss
        ## and update the weights"""
        while epoch[0] < self.epochs:
            def closure():
                generated_img.data.clamp_(0,1)
                optimizer.zero_grad()
                model(generated_img)
                style_loss_sum = 0
                content_loss_sum = 0
                for style_loss_nodes in style_losses:
                    style_loss_sum += style_loss_nodes.backward()
                for content_loss_nodes in content_losses:
                    content_loss_sum += content_loss_nodes.backward()
                total_loss = style_loss_sum + content_loss_sum
                if (epoch[0]+1) % 100 == 0:
                    print('')
                    print('epoch:' + str(epoch[0]+1))
                    print('style_loss: ' + str(style_loss_sum.data[0]))
                    print('content_loss: ' + str(content_loss_sum.data[0]))
                epoch[0] += 1
                return total_loss
            optimizer.step(closure)

        generated_img.data.clamp_(0,1)
        self.output = generated_img


