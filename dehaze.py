from __future__ import print_function
import res_cat619_701 as generator
from misc import *
import os
import random
import torch.nn.parallel
from torch.autograd import Variable
from PIL import Image
import read_dict
import time
#input_list = ['/home/yunpengwu/dehaze/data/1.jpg', '/home/yunpengwu/dehaze/data/0.jpg']

class DPR_Dehaze(object):
    def __init__(self):               # load model and  param
        datasetname, input_path, out_path, net_path, size = read_dict.read_dict()
        if datasetname == 'list':
            input_path = input_path.split()
        self.datasetname = datasetname
        self.valdata =  input_path
        self.valbatchsize = 1
        self.realsize = 256
        self.cropsize = 256
        self.net_pth = net_path
        self.out_path = out_path.split(' ')
        self.size = size.split(' ')
        self.manualSeed = random.randint(1, 10000)
        self.Network = generator.res_deahze()
        self.Network.apply(weights_init)
        self.Network.load_state_dict(torch.load(self.net_pth))  # opt.net
        for index_s in range(len(self.size)):
            if not os.path.exists(self.out_path[index_s]):
                os.makedirs(self.out_path[index_s])

    def normal_div(self, img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min)
        return img

    def normal_range(self, im, range):
        if range is not None:
            self.normal_div(im, range[0], range[1])
        else:
            self.normal_div(im, im.min(), im.max())
        return self.normal_div(im, im.min(), im.max())

    def getLoader(self):          ## read images

        valloader, ims = getLoader(self.datasetname,
                                   self.valdata,
                                   self.realsize,  # opt.originalSize,
                                   self.cropsize,
                                   self.valbatchsize,
                                   mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                                   split='val',
                                   shuffle=False,
                                   seed=self.manualSeed)
        return valloader, ims

    def dehaze_data(self):        ##dehaze images
        # Dataloader
        valloader, ori_ims = self.getLoader()
        # print(net)
        Network= self.Network
        Network.train()
        Network.cuda()
        dehimage = []
        for i, data in enumerate(valloader, 0):
            t0 = time.time()
            valdehaze_cpu = data
            valdehaze_cpu = valdehaze_cpu.float().cuda()
            valdehaze = Variable(valdehaze_cpu, volatile=True)
            output = Network(valdehaze)
            output = output.data.cpu()
            output = torch.squeeze(output)
            output = self.normal_range(output, None)
            print(i)
            output = output.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            dehimage.append(Image.fromarray(output))
            t1 = time.time()
            print(t1 - t0)
        return dehimage, ori_ims

    def save_data(self, dehimage, ori_ims):    ## save

        for i in range(len(dehimage)): # save deimages
            for j in range(len(self.size)):     # save images of 3 different size  into 3 files
                w = int(self.size[j].split('*')[0])
                h = int(self.size[j].split('*')[1])
                deh_im = dehimage[i].resize((w, h))
                deh_im.save(self.out_path[j] + str(ori_ims[i].split('/')[-1]))         # str(i)+'.jpg'#########################



DPR = DPR_Dehaze() ####embody class
dehimage, ori_ims = DPR.dehaze_data() ####using method in the class  ----dehaze images
DPR.save_data(dehimage, ori_ims) ####using method in the class  -----save images
