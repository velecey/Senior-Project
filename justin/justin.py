import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import os
import time
import numpy as np


## this is a class to generate a resblock, it can shift to instance normalization instead of batch normalization
class resblock(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, use_instance=False):
        super(resblock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_instance = use_instance
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        if self.use_instance is True:
            self.in1 = nn.InstanceNorm2d(self.out_channels)
        else:
            self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        if self.use_instance is True:
            self.in2 = nn.InstanceNorm2d(self.out_channels)
        else:
            self.bn2 = nn.BatchNorm2d(self.out_channels)


    def forward(self, input):
        residual = input
        out = self.conv1(input)
        if self.use_instance == True:
            out = self.in1(out)
        else:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.use_instance == True:
            out = self.in2(out)
        else:
            out = self.bn2(out)
        out = out + residual
        return out

class VGG(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG, self).__init__()
        vgg_features = torchvision.models.vgg16(pretrained=True).features
        self.step1 = nn.Sequential()
        self.step2 = nn.Sequential()
        self.step3 = nn.Sequential()
        self.step4 = nn.Sequential()
        for i in range(4):
            self.step1.add_module(str(i),vgg_features[i])
        for i in range(4,9):
            self.step2.add_module(str(i),vgg_features[i])
        for i in range(9,16):
            self.step3.add_module(str(i),vgg_features[i])
        for i in range(16,23):
            self.step4.add_module(str(i), vgg_features[i])
        if not requires_grad:
            for params in self.parameters():
                params.requires_grad = False

    def forward(self, input_tensor):
        out = self.step1(input_tensor)
        relu1_2 = out
        out = self.step2(out)
        relu2_2 = out
        out = self.step3(out)
        relu3_3 = out
        out = self.step4(out)
        relu4_3 = out
        return [relu1_2, relu2_2, relu3_3, relu4_3]

class justin:
    def __init__(self, imsize=(256, 256), style_img_dir='', content_img_dir='', use_instance=False,
                 epochs=2, batch=4,content_layers=['relu_4'],
                 style_layers=['relu_2', 'relu_4', 'relu_7', 'relu_10']):
        self.style_img_dir = style_img_dir
        self.content_img_dir = content_img_dir
        self.imsize = imsize
        self.toPIL = transforms.ToPILImage()
        self.use_instance = use_instance
        self.content_weight = 1.0
        self.style_weight = 5.0
        self.epochs = epochs
        self.batch = batch
        self.img_preprocess = transforms.Compose([
            transforms.Scale(self.imsize),
            transforms.CenterCrop(self.imsize),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.use_cuda = False
        if torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False
        if self.use_cuda == True:
            self.dtype = torch.cuda.FloatTensor
            torch.cuda.set_device(0)
            print("Current device: %d" % torch.cuda.current_device())
        else:
            self.dtype = torch.FloatTensor

    def rgb_to_bgr(self, batch):
        batch = batch.transpose(0,1)
        (r,g,b) = torch.chunk(batch, 3)
        batch = torch.cat((b,g,r))
        batch = batch.transpose(0,1)
        return batch

    def subtract_imgnet_mean(self, batch):
        tensortype = type(batch.data)
        mean = tensortype(batch.data.size())
        mean[:,0,:,:] = 103.939
        mean[:,1,:,:] = 116.779
        mean[:,2,:,:] = 123.68
        batch = batch.sub(Variable(mean))
        return batch

    def load_img(self, img_dir):
        img = Image.open(img_dir)
        img = img.resize(self.imsize, Image.ANTIALIAS)
        img = np.array(img).transpose(2,0,1)
        img = torch.from_numpy(img).float()
        return img

    def gram_matrix(self, input_tensor):
        batch, channel, height, width = input_tensor.size()
        resize = input_tensor.view(batch, channel, width * height)
        resize_t = resize.transpose(1, 2)
        gram = resize.bmm(resize_t)
        return gram / (channel*width*height)

    def imshow(self, input_tensor, title=None):
        img = input_tensor.clone().cpu()
        img = img.view(3, self.height, self.width)
        img = self.toPIL(img)
        plt.imshow(img)
        plt.title(title)
        plt.pause(0.001)
        plt.show()


    def generate_model(self):
        model = nn.Sequential()
        model.add_module('conv_in', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9, stride=1, padding=4))
        model.add_module('conv_in_bn', nn.BatchNorm2d(num_features=32))
        model.add_module('relu_conv_in', nn.ReLU())
        model.add_module('dsample_1_conv', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1))
        model.add_module('dsammple_1_bn', nn.BatchNorm2d(num_features=64))
        model.add_module('dsample_1_relu', nn.ReLU())
        model.add_module('dsample_2_conv', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1))
        model.add_module('dsammple_2_bn', nn.BatchNorm2d(num_features=128))
        model.add_module('dsample_2_relu', nn.ReLU())
        model.add_module('resblock1', resblock(use_instance=self.use_instance))
        model.add_module('resblock2', resblock(use_instance=self.use_instance))
        model.add_module('resblock3', resblock(use_instance=self.use_instance))
        model.add_module('resblock4', resblock(use_instance=self.use_instance))
        model.add_module('resblock5', resblock(use_instance=self.use_instance))
        model.add_module('usample_1_conv', nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1))
        model.add_module('usammple_1_bn', nn.BatchNorm2d(num_features=64))
        model.add_module('usample_1_relu', nn.ReLU())
        model.add_module('usample_2_conv', nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1))
        model.add_module('usammple_2_bn', nn.BatchNorm2d(num_features=32))
        model.add_module('usample_2_relu', nn.ReLU())
        model.add_module('conv_out', nn.Conv2d(in_channels=32, out_channels=3, kernel_size=9, stride=1, padding=4))
        model.add_module('tanh', nn.Tanh())
        return model


    def train_network(self):
        start = time.time()
        print('Generating Justin model...')

        ## generate the model
        model = self.generate_model()
        vgg_model = VGG()
        if self.use_cuda:
            model = model.cuda()
            vgg_model = vgg_model.cuda()

        ## load the style image and feed forward to vgg net and make a list with
        ## size of batch containing same gram matricies
        style_img = self.load_img(self.style_img_dir)
        style_batch = style_img.repeat(self.batch, 1, 1, 1)
        style_batch = self.rgb_to_bgr(style_batch)
        if self.use_cuda:
            style_batch = style_batch.cuda()
        style_batch = Variable(style_batch, volatile=True)
        style_batch = self.subtract_imgnet_mean(style_batch)
        style_feature = vgg_model(style_batch)
        y_s_gram_list = []
        for i in range(4):
            y_s_gram_list.append(self.gram_matrix(style_feature[i]))

        ## set up the optimizer and criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        ## load the training dataset
        train_dataset = datasets.ImageFolder(self.content_img_dir, self.img_preprocess)
        train_loader = DataLoader(train_dataset, batch_size=self.batch)
        model.train()
        for _ in range(self.epochs):
            cnt = 0
            for batch_idx, (input_batch, _) in enumerate(train_loader):
                img_num = len(input_batch)
                cnt += img_num

                optimizer.zero_grad()

                input_batch = self.rgb_to_bgr(input_batch)
                input_batch = Variable(input_batch)
                if self.use_cuda:
                    input_batch = input_batch.cuda()

                ## feed forward the input through the network to train
                content_loss = 0
                y_hat = model(input_batch) * 150.0
                input_batch_c = Variable(input_batch.data.clone(), volatile=True)

                y_hat = self.subtract_imgnet_mean(y_hat)
                input_batch_c = self.subtract_imgnet_mean(input_batch_c)

                y_hat_features = vgg_model(y_hat)
                input_batch_features = vgg_model(input_batch_c)
                input_batch_features = Variable(input_batch_features[1].data, requires_grad=False)

                content_loss = self.content_weight * criterion(y_hat_features[1], input_batch_features)

                ## compute the style loss
                style_loss = 0
                for i in range(len(y_hat_features)):
                    y_s_gram = Variable(y_s_gram_list[i].data, requires_grad=False)
                    y_hat_s_gram = self.gram_matrix(y_hat_features[i])
                    style_loss += self.style_weight * criterion(y_hat_s_gram, y_s_gram[:img_num,:,:])

                ## sum both losses
                loss = content_loss + style_loss

                ## and compute the total variation loss
                """
                if self.use_tv == True:
                    tv_x = torch.sum(torch.abs(y_hat[:,:,:,1:]-y_hat[:,:,:,:-1]))
                    tv_y = torch.sum(torch.abs(y_hat[:,:,1:,:]-y_hat[:,:,:-1,:]))
                    tv_loss = self.tv_weight * (tv_x + tv_y)
                    loss += tv_loss
                """

                ## and update the weights
                loss.backward()
                optimizer.step()
                if cnt:
                    end = time.time()
                    print('{} images done'.format(cnt))
                    print('content loss : ' + str(content_loss.data[0]))
                    print('style loss : ' + str(style_loss.data[0]))
                    ## print('total variation loss : ' + tv_loss)
                    print('{} time passed'.format(end-start))
                    print('')
        print('Optimizing done')
        model.eval()
        model.cpu()
        torch.save(model.state_dict(), 'justin_model.pth')

    ## this is a function to feed forward an arbitrary input to the current model
    def feed_forward_and_save_img(self, img_dir):
        img = self.load_img(img_dir)
        img = img.unsqueeze(0)
        if self.use_cuda:
            img = img.cuda()
        img = img.mul(255)
        img = self.rgb_to_bgr(img)
        img = self.subtract_imgnet_mean(img)
        img = Variable(img, volatile=True)
        model = self.generate_model()
        model.load_state_dict(torch.load('justin_model.pth'))
        if self.use_cuda:
            model = model.cuda()
        output = model(img)
        output = output.squeeze(0)
        output = -output
        output = self.subtract_imgnet_mean(output)
        output = -output
        (b, g, r) = torch.chunk(output, 3)
        output = torch.cat((r, g, b))
        if self.use_cuda:
            img = output.clone().cpu()
        else:
            img = output.clone()
        img = img.clamp(0, 255).numpy()
        img = img.transpose(1,2,0).astype('uint8')
        img = Image.fromarray(img)
        img.save('output.jpg')


