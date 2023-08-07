import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from layers import Conv2dBlock
#from special_layers import *
from special_layers_pytorch import *

class selfdan_module(nn.Module):
    def __init__(self, use_cuda, device_ids, meanshape):
        super().__init__()

        self.stage1 = selfdan_1(meanshape)
        self.stage2 = selfdan_2(meanshape)

        #self.criterion = torch.nn.L1Loss()
        self.criterion = self.landmarkPairErrorNorm

        if use_cuda:
            self.stage1.to(device_ids[0])
            self.stage2.to(device_ids[0])


    # from DAN (originally, a pair is used for computing loss)
    def landmarkPairErrorNorm(self, output, landmarks):

        meanError = torch.mean(torch.sqrt(torch.sum((output - landmarks) ** 2, dim=1)), dim=1)    # final : [N, 1]
        eyeDist = torch.norm(torch.mean(landmarks[:, 36:42], dim=1) - torch.mean(landmarks[:, 42:48], dim=1), dim=1)    # final : [N, 1]
      
        return meanError / eyeDist

    # optimizers : [opt1, opt2]
    def forward(self, input_img, input_label):

        x1_ret, x1_fc1 = self.stage1(input_img)
        x2_ret = self.stage2(input_img, x1_ret, x1_fc1) # hidden fc1

        # backpropagate
        loss_s1 = self.criterion(x1_ret, input_label)
        loss_s2 = self.criterion(x2_ret, input_label)     # x1_fc1 : hidden

        #outputs = torch.cat([x1_ret, x2_ret], dim=1)
        outputs = [x1_ret, x2_ret]
        losses = [loss_s1, loss_s2]
        #losses = [loss_s1.detach(), loss_s2.detach()]

        return outputs, losses


class selfdan_1(nn.Module):
    def __init__(self, meanshape):
        super().__init__()

        self.meanshape = torch.from_numpy(meanshape).view((1, 68, 2)).float().cuda()

        # create
        dim_in = 1
        self.create_layers_s1(dim_in)


    def create_layers_s1(self, dim_in):
        # conv1
        self.s1_conv1_1 = Conv2dBlock(dim_in, 64, 3, 1, 0, activation='relu')
        self.s1_conv1_2 = Conv2dBlock(64, 64, 3, 1, 0, activation='relu')
        self.s1_pool1 = Conv2dBlock(64, 64, 3, 2, 1, activation='relu')

        # conv2
        self.s1_conv2_1 = Conv2dBlock(64, 128, 3, 1, 1, activation='relu')
        self.s1_conv2_2 = Conv2dBlock(128, 128, 3, 1, 1, activation='relu')
        self.s1_pool2 = Conv2dBlock(128, 128, 3, 2, 1, activation='relu')
        
        # conv3
        self.s1_conv3_1 = Conv2dBlock(128, 256, 3, 1, 1, activation='relu')
        self.s1_conv3_2 = Conv2dBlock(256, 256, 3, 1, 1, activation='relu')
        self.s1_pool3 = Conv2dBlock(256, 256, 3, 2, 1, activation='relu')

        # conv4
        self.s1_conv4_1 = Conv2dBlock(256, 512, 3, 1, 1, activation='relu')
        self.s1_conv4_2 = Conv2dBlock(512, 512, 3, 1, 1, activation='relu')
        self.s1_pool4 = Conv2dBlock(512, 512, 3, 2, 1, activation='relu')

        # fc1
        self.s1_drop = nn.Dropout(0.5) # dropout
        self.s1_fc1 = nn.Linear(512*7*7, 256) # fc1
        self.s1_fc2 = nn.Linear(256, 136) # fc2

        self.s1_relu = torch.nn.ReLU()
        self.s1_bn = torch.nn.BatchNorm1d(256)

    def forward(self, x):

        input_img = x.clone()
        
        # stage 1
        x = self.s1_conv1_1(x)
        x = self.s1_conv1_2(x)
        x = self.s1_pool1(x)

        x = self.s1_conv2_1(x)
        x = self.s1_conv2_2(x)
        x = self.s1_pool2(x)

        x = self.s1_conv3_1(x)
        x = self.s1_conv3_2(x)
        x = self.s1_pool3(x)

        x = self.s1_conv4_1(x)
        x = self.s1_conv4_2(x)
        x = self.s1_pool4(x)
        x = torch.flatten(x, start_dim=1)

        x = self.s1_drop(x)
        x = self.s1_fc1(x)  # hidden 
        xfc1 = x.clone()
        
        x = self.s1_relu(x)
        x = self.s1_bn(x)
        x_ret = self.s1_fc2(x)

        x_ret = x_ret.view((-1, 68, 2))

        meanshape = self.meanshape.to(device='cuda:'+str(x.get_device()))
        x_ret = x_ret + meanshape

        return x_ret, xfc1


# -------------------------- STAGE 2 ---------------------
class selfdan_2(nn.Module):
    def __init__(self, meanshape):
        super().__init__()

        cat_in = 3
        #self.meanshape = meanshape.cuda().to(device_ids[0])
        self.meanshape = torch.from_numpy(meanshape).view((1, 68, 2)).float().cuda()

        self.create_layers_s2(cat_in)

        

    def create_layers_s2(self, cat_in):
        # ----
        
        # transform
        # affine
        # landmarktransform
        # landmarkimage

        self.TansformParam = EstimateAffineParams(self.meanshape.squeeze(0))
        self.Image_trafo_layer = AffineImageTransformation(112)
        self.Landmark_trafo_layer = AffineLandmarkTransformation()
        self.HeatMapLayer = HeatMap(112, 16)
        self.hidden_linear = torch.nn.Linear(256, 112 * 112 // 4)
        #self.view_upscale = View([None, -1, 64, 64])
        #self.bilinear_upsample = Upsample(112)


        # conv1
        self.s2_conv1_1 = Conv2dBlock(cat_in, 64, 3, 1, 0, activation='relu')
        self.s2_conv1_2 = Conv2dBlock(64, 64, 3, 1, 0, activation='relu')
        self.s2_pool1 = Conv2dBlock(64, 64, 3, 2, 1, activation='relu')

        # conv2
        self.s2_conv2_1 = Conv2dBlock(64, 128, 3, 1, 1, activation='relu')
        self.s2_conv2_2 = Conv2dBlock(128, 128, 3, 1, 1, activation='relu')
        self.s2_pool2 = Conv2dBlock(128, 128, 3, 2, 1, activation='relu')

        # conv3
        self.s2_conv3_1 = Conv2dBlock(128, 256, 3, 1, 1, activation='relu')
        self.s2_conv3_2 = Conv2dBlock(256, 256, 3, 1, 1, activation='relu')
        self.s2_pool3 = Conv2dBlock(256, 256, 3, 2, 1, activation='relu')

        # conv4
        self.s2_conv4_1 = Conv2dBlock(256, 512, 3, 1, 1, activation='relu')
        self.s2_conv4_2 = Conv2dBlock(512, 512, 3, 1, 1, activation='relu')
        self.s2_pool4 = Conv2dBlock(512, 512, 3, 2, 1, activation='relu')

        # fc1
        self.s2_drop = nn.Dropout(0.5) # dropout
        self.s2_fc1 = nn.Linear(512*7*7, 256) # fc1
        self.s2_fc2 = nn.Linear(256, 136) # fc2

        self.s2_relu = torch.nn.ReLU()
        self.s2_bn = torch.nn.BatchNorm1d(256)

    # input_img : original image
    # x : x_ret (output of stage 1) [N, 136]
    # x1fc1 : hidden feature of previous stage [N, 256]
    # meanshape - >
    def forward(self, input_img, x, x1fc1):

        #print("X : device " + str(x.get_device()))
        #print("Xfc1 : device " + str(x1fc1.get_device()))
        #print("mean : device " + str(self.meanshape.get_device()))

        #meanshape = self.meanshape.to(device='cuda:'+str(x.get_device()))
        #print("after mean : device " + str(meanshape.get_device()))

        # previous layer
        hidden = self.hidden_linear(x1fc1)
        hidden = hidden.view((-1, 1, 112 // 2, 112 // 2))
        hidden = F.interpolate(hidden, scale_factor=2, mode='bilinear')        # binilnear upsample

        # cat
        x_aff_param = self.TansformParam(x)     # take previous lmk (x_ret)
        x_aff_img = self.Image_trafo_layer(input_img, x_aff_param)        # take input image
        x_aff_lms = self.Landmark_trafo_layer(x, x_aff_param)                # take previous lmk
        x_heatmap = self.HeatMapLayer(x_aff_lms)

        #x_fea = self.feature_cat(xfc1).view([-1, 1, 56, 56])
        #x_fea = F.interpolate(x_fea, scale_factor=2, mode='nearest')       # upsample

        x2 = torch.cat([x_aff_img, x_heatmap, hidden], dim=1)


        # stage 2
        x2 = self.s2_conv1_1(x2)
        x2 = self.s2_conv1_2(x2)
        x2 = self.s2_pool1(x2)

        x2 = self.s2_conv2_1(x2)
        x2 = self.s2_conv2_2(x2)
        x2 = self.s2_pool2(x2)

        x2 = self.s2_conv3_1(x2)
        x2 = self.s2_conv3_2(x2)
        x2 = self.s2_pool3(x2)

        x2 = self.s2_conv4_1(x2)
        x2 = self.s2_conv4_2(x2)
        x2 = self.s2_pool4(x2)
        x2 = torch.flatten(x2, start_dim=1)

        x2 = self.s2_drop(x2)
        x2 = self.s2_fc1(x2)
        #x2fc1 = x.clone()
        x2 = self.s2_relu(x2)
        x2 = self.s2_bn(x2)
        x2 = self.s2_fc2(x2)
        x2 = x2.view((-1, 68, 2))
        
        x2_ret = self.Landmark_trafo_layer(x2 + x_aff_lms, x_aff_param, inverse=True)

        return x2_ret

