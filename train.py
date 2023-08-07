import os
import random
import time
import shutil
import numpy as np
import cv2
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torch.nn import Parameter

from model import selfdan_module
from dataset.dataset import ImageServer

import matplotlib.pyplot as plt


def draw_landmarks(img, landmarks, gt_landmarks, name):
    img_rgb_plot = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_rgb_gt = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for i in range(landmarks.shape[0]):
        cv2.circle(img_rgb_plot, (landmarks[i,0], landmarks[i,1]), 1, (0, 0, 255), -1)
        cv2.circle(img_rgb_gt, (gt_landmarks[i,0], gt_landmarks[i,1]), 1, (0, 0, 255), -1)

    img_cat = cv2.hconcat([img_rgb_plot, img_rgb_gt])

    cv2.imwrite(name, img_cat)

class Trainer():
    def __init__(self):
        self.batchsize = 256
        self.landmarkPatchSize = 16
        self.stage = 2
        self.use_cuda = True
        self.resume_iter = 0
        self.ckpt_dir = 'ckpt_eyeDist/'

        #self.meanshape = None

        if os.path.exists(self.ckpt_dir) == False:
            os.makedirs(self.ckpt_dir)

        # temp
        #comset = np.load('../data/commonSet.npz')
        #print(comset['imgs'][0].shape)
        # CUDA configuration
        gpu_ids = [0,1,2,3]
        device_ids = gpu_ids
        orig_gpus = gpu_ids
        if self.use_cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
            device_ids = list(range(len(device_ids)))
            gpu_ids = device_ids
            cudnn.benchmark = True
        

        # dataset
        datasetDir = "../data/"
        trainSet = ImageServer.Load(datasetDir + "dataset_nimgs=60960_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz")
        validationSet = ImageServer.Load(datasetDir + "dataset_nimgs=100_perturbations=[]_size=[112, 112].npz")
        self.loadData(trainSet, validationSet)

        # loader
        self.train_loader = torch.utils.data.DataLoader(dataset=trainSet,
            batch_size=self.batchsize, shuffle=True, num_workers=4, drop_last=False)
        self.val_loader = torch.utils.data.DataLoader(dataset=validationSet,
            batch_size=128, shuffle=False, num_workers=4)

        # network (when first, load initial meanshape)
        meanshape = self.initLandmarks
        #meanshape = torch.from_numpy(self.initLandmarks).view((1, 68, 2)).float()
        self.net_class = selfdan_module(self.use_cuda, device_ids, meanshape)

        # cuda
        if self.use_cuda:
            self.net_class = nn.parallel.DataParallel(self.net_class, device_ids=device_ids)
            self.net = self.net_class.module
        else:
            self.net = self.net_class


        # optimizer
        params_1 = list(self.net.stage1.parameters())
        params_2 = list(self.net.stage2.parameters())
        self.opt_1 = torch.optim.Adam(params_1, lr=0.001)
        self.opt_2 = torch.optim.Adam(params_2, lr=0.001)

        # resume
        if self.resume_iter > 0:
            #meanshape = torch.randint(112 - 1, (1, 68, 2)).float()
            #self.meanshape = torch.load('./ckpt/mean_shape.pth')
            self.resume(checkpoint_dir=self.ckpt_dir, iteration=self.resume_iter, test=False)

            self.train_batches = self.resume_iter
        else:
            self.train_batches = 0

        

    
    
    # from original DAN code
    def loadData(self, trainSet, validationSet):
        self.nSamples = trainSet.gtLandmarks.shape[0]
        self.imageHeight = trainSet.imgSize[0]
        self.imageWidth = trainSet.imgSize[1]
        self.nChannels = trainSet.imgs.shape[1]

        self.Xtrain = trainSet.imgs
        self.Xvalid = validationSet.imgs

        self.Ytrain = trainSet.labels
        self.Yvalid = validationSet.labels

        self.testIdxsTrainSet = range(len(self.Xvalid))
        self.testIdxsValidSet = range(len(self.Xvalid))

        self.meanImg = trainSet.meanImg
        self.stdDevImg = trainSet.stdDevImg
        self.initLandmarks = trainSet.initLandmarks[0].reshape((1, 136))  # for FIRST

        
    def getErrors(self, X, y, loss, idxs, chunkSize=50):
        error = 0

        nImages = len(idxs)
        nChunks = 1 + nImages / chunkSize

        idxs = np.array_split(idxs, nChunks)
        for i in range(len(idxs)):
            error += loss(X[idxs[i]], y[idxs[i]])

        error = error / len(idxs)
        return error

    # from DAN code without Lasagne
    def compute_error(self, pred, landmarks):

        meanError = torch.mean(torch.sqrt(torch.sum((pred - landmarks) ** 2, dim=1)), dim=1)    # final : [N, 1]
        eyeDist = torch.norm(torch.mean(landmarks[:, 36:42], dim=1) - torch.mean(landmarks[:, 42:48], dim=1), dim=1)    # final : [N, 1]
        

        res = meanError / eyeDist
        return torch.mean(res), torch.mean(meanError)

    
    def get_model_list(self, dirname, key, iteration=0):
        if os.path.exists(dirname) is False:
            return None
        gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                    os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
        if gen_models is None:
            return None
        gen_models.sort()
        if iteration == 0:
            last_model_name = gen_models[-1]
        else:
            for model_name in gen_models:
                if '{:0>8d}'.format(iteration) in model_name:
                    return model_name
            raise ValueError('Not found models with this iteration')
        return last_model_name

    # save
    def save_model(self, checkpoint_dir, iteration):
        # Save net, opt
        net_name = os.path.join(checkpoint_dir, 'net_%08d.pt' % iteration)
        opt_name = os.path.join(checkpoint_dir, 'optimizer_%08d.pt' % iteration)
        #meanshape_name = os.path.join(checkpoint_dir, 'meanshape_%08d.npy' % iteration)
        
        torch.save(self.net.state_dict(), net_name)

        torch.save({'s1': self.opt_1.state_dict(), 's2': self.opt_2.state_dict()}, opt_name)

    # resume
    def resume(self, checkpoint_dir, iteration=0, test=False):
        last_model_name = self.get_model_list(checkpoint_dir, "net", iteration=iteration)
        #self.net.load_state_dict(torch.load(last_model_name), strict=False)    # 1
        self.load_my_state_dict(self.net, torch.load(last_model_name))       # 2

        iteration = int(last_model_name[-11:-3])

        print("Resume from {} at iteration {}".format(checkpoint_dir, iteration))

        last_opt_name = self.get_model_list(checkpoint_dir, "optimizer", iteration=iteration)
        
        self.opt_1.load_state_dict(torch.load(last_opt_name)['s1'])
        self.opt_2.load_state_dict(torch.load(last_opt_name)['s2'])  

        return iteration

    def load_my_state_dict(self, our_model, state_dict):
        own_state = our_model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, Parameter):
                param = param.data
            own_state[name].copy_(param)

    def validate(self):
        
        iterable_val_loader = iter(self.val_loader)

        #for batch in self.iterate_minibatches(self.Xtrain, self.Ytrain, self.batchsize, True):                
        if len(self.Xvalid) % self.batchsize == 0:
            ent_batch = (len(self.Xvalid) // self.batchsize)
        else:
            ent_batch = (len(self.Xvalid) // self.batchsize) + 1
        
        MERR = 0.0
        MRES = 0.0

        # batch run
        for batch in range(ent_batch):
            val_imgs, val_labels = iterable_val_loader.next()

            if self.use_cuda:
                val_imgs = val_imgs.cuda()
                val_labels = val_labels.cuda()


            with torch.no_grad():
                x1_ret, x1_fc1 = self.net.stage1(val_imgs)
                x2_ret = self.net.stage2(val_imgs, x1_ret, x1_fc1) # hidden fc1


            res, meanerror = self.compute_error(x2_ret, val_labels)
            MERR = MERR + meanerror.detach()
            MRES = MRES + res.detach()

            # sample save
            sample_img = val_imgs[1].add_(1).mul_(127.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            sample_gt = val_labels[1].detach().cpu().numpy()
            sample_label = x2_ret[1].detach().cpu().numpy()

            draw_landmarks(sample_img, sample_label, \
                sample_gt, self.ckpt_dir + "samples_" + str(batch) + ".png")

        print("VALID : -> [MER %.6f], [RES %.6f]" % ((MERR / len(self.Xvalid)), (MRES / len(self.Xvalid))))

    def run(self, num_epoch = 500):

        # label : [2, 68, 2] (y[:, 0] = imageServer.initLandmarks y[:, 1] = imageServer.gtLandmarks)

        print("Starting training... target epoch " + str(num_epoch))
        print("N : " + str(len(self.Xtrain)) + ", batch : " + str(self.batchsize) + ", batches : " + str(len(self.Xtrain) // self.batchsize))

        train_batches = self.train_batches
        
        start_epoch =  train_batches // (len(self.Xtrain) // self.batchsize)


        for epoch in range(start_epoch, num_epoch):
            print("Starting epoch " + str(epoch))

            train_err = 0
            start_time = time.time()

            iterable_train_loader = iter(self.train_loader)

            #for batch in self.iterate_minibatches(self.Xtrain, self.Ytrain, self.batchsize, True):                
            if len(self.Xtrain) % self.batchsize == 0:
                ent_batch = (len(self.Xtrain) // self.batchsize)
            else:
                ent_batch = (len(self.Xtrain) // self.batchsize) + 1
            
            # batch run
            for batch in range(ent_batch):
                train_imgs, train_labels = iterable_train_loader.next()
               
                #inputs, targets = batch

                if self.use_cuda:
                    train_imgs = train_imgs.cuda()
                    train_labels = train_labels.cuda()

                # ------- run
                outputs, losses = self.net_class(train_imgs, train_labels)

                # ------- backpropagate (if multi, use mean)
                for sidx in range(len(losses)):
                    if not losses[sidx].dim() == 0:
                        losses[sidx] = torch.mean(losses[sidx])

                
                self.opt_1.zero_grad()
                losses[0].backward(retain_graph=True)
                self.opt_1.step()

                self.opt_2.zero_grad()
                losses[1].backward()
                self.opt_2.step()
                
                        
                # outputs : [x1, x2]
                # losses : [loss1, loss2]

                #loss = self.landmarkErrorNorm(output, train_labels)
                
                if train_batches % 2 == 0:
                    merr, _ = self.compute_error(outputs[1], train_labels)
                    message = "[" + str(epoch) + "/" + str(num_epoch) + "] batch " + str(train_batches) + "/" + str(ent_batch) + \
                        " == loss : (1) %.6f" % losses[0] + ", (2) %.6f" % losses[1]

                    message = message + "    [ERR %.6f]" % merr.detach().cpu()
                    print(message)

                if train_batches % 50 == 0:
                    self.save_model(self.ckpt_dir, train_batches)

                if train_batches % 50 == 0:
                    self.net.eval()
                    self.validate()
                    self.net.train()

                train_batches += 1




def main():
    trainer = Trainer() 
    trainer.run()
    

if __name__ == '__main__':
    main()