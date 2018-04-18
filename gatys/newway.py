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
import time

## Define the Graph
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1 = nn.Sequential()
        self.conv2 = nn.Sequential()
        self.conv3 = nn.Sequential()
        self.conv4 = nn.Sequential()
        self.conv5 = nn.Sequential()
        self.conv6 = nn.Sequential()
        vgg = models.vgg19(pretrained=True).features
        for i in range(1):
            self.conv1.add_module('conv1_1' + '-' + str(i), vgg[i])
        for i in range(1, 7):
            self.conv2.add_module('conv2_1' + '-' + str(i), vgg[i])
        for i in range(7, 12):
            self.conv3.add_module('conv3_1' + '-' + str(i), vgg[i])
        for i in range(12, 21):
            self.conv4.add_module('conv4_1' + '-' + str(i), vgg[i])
        for i in range(21, 23):
            self.conv5.add_module('conv4_2' + '-' + str(i), vgg[i])
        for i in range(23, 30):
            self.conv6.add_module('conv5_1' + '-' + str(i), vgg[i])

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        conv1_1 = out
        out = self.conv2(out)
        conv2_1 = out
        out = self.conv3(out)
        conv3_1 = out
        out = self.conv4(out)
        conv4_1 = out
        out = self.conv5(out)
        conv4_2 = out
        out = self.conv6(out)
        conv5_1 = out
        return [conv1_1, conv2_1, conv3_1, conv4_1, conv5_1, conv4_2]



class gatys:
    def __init__(self, imsize=(128,128), style_weight=1000, content_weight=1, style_img_dir='', content_img_dir='',
                 epochs = 300, use_cuda=False):
        self.imsize = imsize
        self.width, self.height = imsize
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.style_img_dir = style_img_dir
        self.content_img_dir = content_img_dir
        self.epochs = epochs
        self.toPIL = transforms.ToPILImage()
        self.tv_weight = 0.01
        self.use_cuda = use_cuda
        if torch.cuda.is_available():
            self.use_cuda = True
        if self.use_cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
        self.img_preprocess = transforms.Compose([
                                transforms.Resize(self.imsize),
                                transforms.ToTensor()
                            ])

    def load_img(self, img_dir):
        img = Image.open(img_dir)
        img = Variable(self.img_preprocess(img))
        img = img.unsqueeze(0)
        return img

    def imshow(self, input_tensor, title=None):
        img = input_tensor.clone().cpu()
        img = img.view(3, self.height, self.width)
        img = self.toPIL(img)
        plt.imshow(img)
        plt.title(title)
        plt.pause(0.001)
        plt.show()

    def gram_matrix(self, input_tensor):
        batch, channel, width, height = input_tensor.size()
        resize = input_tensor.view(batch, channel, width * height)
        resize_t = resize.transpose(1,2)
        gram = torch.bmm(resize, resize_t)
        return gram / (channel*width*height)


    def run_network(self):
        vgg = VGG()
        for params in vgg.parameters():
            params.requires_grad = False
        print('Generating model...')
        content_img = self.load_img('Images/yonsei.JPG')
        style_img = self.load_img('mosaic.jpeg')
        random_input = Variable(torch.randn(content_img.size()), requires_grad = True).type(self.dtype)
        generated_img = nn.Parameter(random_input.data)
        optimizer = optim.LBFGS([generated_img])
        criterion = nn.MSELoss()

        y_c = vgg(content_img.clone())[-1]
        y_s = vgg(style_img.clone())[:-2]


        epochs = [0]
        while epochs[0] < self.epochs:
            def closure():
                style_weight = 1000
                content_weight = 1
                wl = 0.2

                generated_img.data.clamp_(0,1)
                optimizer.zero_grad()

                feed_forward = vgg(generated_img)
                y_hat_c = feed_forward[-1]
                y_hat_s = feed_forward[:-1]

                content_loss = criterion(content_weight * y_hat_c, content_weight * y_c) / 2

                style_loss = 0
                for inp, target in zip(y_hat_s, y_s):
                    style_loss += wl * criterion(style_weight * self.gram_matrix(inp), style_weight * self.gram_matrix(target)) / 4


                ## tv_x = torch.sum(torch.abs(generated_img[:,1:,:] - generated_img[:,:-1,:]))/(self.imsize*self.imsize)
                ## tv_y = torch.sum(torch.abs(generated_img[:,:,1:] - generated_img[:,:,:-1]))/(self.imsize*self.imsize)
                ## tv_loss = tv_weight * (tv_x + tv_y)

                loss = content_loss + style_loss
                loss.backward()

                if (epochs[0]+1) % 30 == 0:
                    self.save_img(generated_img.data, 'iter_{}.jpg'.format(epochs[0]+1))
                if (epochs[0]+1) % 10 == 0:
                    print('{},{},{}'.format(epochs[0]+1,content_loss.data[0],style_loss.data[0]))
                epochs[0] += 1
                return loss

            optimizer.step(closure)

        generated_img.data.clamp_(0,1)
        return generated_img.data

    def save_img(self, input_tensor, img_dir):
        img = input_tensor.clone().cpu()
        img = img.view(3, self.height, self.width)
        torchvision.utils.save_image(img, img_dir)


gatys = gatys(imsize=(256,256))
start = time.time()
output = gatys.run_network()
end = time.time()
print(end-start)
gatys.save_img(output, 'asdfasdf.jpg')



