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


def draw_landmarks(img, landmarks, name):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_orig =cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for i in range(landmarks.shape[0]):
        cv2.circle(img_rgb, (landmarks[i,0], landmarks[i,1]), 1, (0, 0, 255), -1)

    imcat = cv2.hconcat([img_rgb, img_orig])
    cv2.imwrite(name, imcat)

# from DAN code without Lasagne (landmarks : [N, 2, 68])
# !! We need meanError final [N, 1] and eyeDist [N, 1] but...
def compute_error(pred, landmarks):

    meanError = torch.mean(torch.sqrt(torch.sum((pred - landmarks) ** 2, dim=1)), dim=1)    # final : [N, 1]
    eyeDist = torch.norm(torch.mean(landmarks[:, 36:42], dim=1) - torch.mean(landmarks[:, 42:48], dim=1), dim=1)    # final : [N, 1]

    res = meanError / eyeDist
    return torch.mean(res), torch.mean(meanError)

def validate(net, batchsize, val_dataset, val_loader, val_name = "", use_cuda=True):
    
    iterable_val_loader = iter(val_loader)

    #for batch in self.iterate_minibatches(self.Xtrain, self.Ytrain, self.batchsize, True):                
    if len(val_dataset.imgs) % batchsize == 0:
        ent_batch = (len(val_dataset.imgs) // batchsize)
    else:
        ent_batch = (len(val_dataset.imgs) // batchsize) + 1
    
    MERR = 0.0
    MRES = 0.0

    # batch run
    for batch in range(ent_batch):
        val_imgs, val_labels = iterable_val_loader.next()

        if use_cuda:
            val_imgs = val_imgs.cuda()
            val_labels = val_labels.cuda()
        
        val_imgs = val_imgs.mul_(1/127.5).add_(-1).clamp_(-1, 1)


        # run
        # def forward(self, input_img, input_label, optimizers, criterion):
        
        x1_ret, x1_fc1 = net.stage1(val_imgs)
        x2_ret = net.stage2(val_imgs, x1_ret, x1_fc1) # hidden fc1


        # sample one
        #sample_img = val_imgs[0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        sample_img = val_imgs[0].add_(1).mul_(127.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        sample_label = x2_ret[0].detach().cpu().numpy()
        #print(sample_label.shape)
        draw_landmarks(sample_img, sample_label, "samples/sss" + str(batch) + ".png")
    


        res, meanerror = compute_error(x2_ret, val_labels)
        MERR = MERR + meanerror.detach()
        MRES = MRES + res.detach()
    
    #print("<" + val_name + "> VALID : -> [MER %.6f], [RES %.6f]" % ((MERR / len(val_dataset.imgs)), (MRES / len(val_dataset.imgs))))
    print("<" + val_name + "> VALID : -> [MER %.6f], [RES %.6f]" % ((MERR / ent_batch), (MRES / ent_batch)))

    #return (MRES / len(val_dataset.imgs)), (MERR / len(val_dataset.imgs))
    return (MRES / ent_batch), (MERR / ent_batch)


def get_model_list(dirname, key, iteration=0):
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

def resume(model, checkpoint_dir, iteration=0, test=False):
    last_model_name = get_model_list(checkpoint_dir, "net", iteration=iteration)
    
    print("last_model_name : " + last_model_name)
    dan_state_dict = torch.load(last_model_name)
    #self.net.load_state_dict(torch.load(last_model_name), strict=False)    # 1

    own_state = model.state_dict()
    for name, param in dan_state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        own_state[name].copy_(param)

    print("Resume from {} at iteration {}".format(checkpoint_dir, iteration))



# -----------------------
def main():
    
    # for meanshape
    trainSet = ImageServer.Load("../data/dataset_nimgs=60960_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz") 
    validationSet = ImageServer.Load("../data/dataset_nimgs=100_perturbations=[]_size=[112, 112].npz")
    
    commonSet = ImageServer.Load("../data/commonSet.npz")
    challengingSet = ImageServer.Load("../data/challengingSet.npz")
    w300Set = ImageServer.Load("../data/w300Set.npz")

    meanshape = trainSet.initLandmarks[0].reshape((1, 136)) 
    #meanshape = commonSet.initLandmarks[0].reshape((1, 136)) 

    # -------- config
    batchsize = 8
    landmarkPatchSize = 16
    stage = 2
    use_cuda = True

    # CUDA configuration
    gpu_ids = [0]
    device_ids = gpu_ids
    orig_gpus = gpu_ids
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        gpu_ids = device_ids


    # model
    net_class = selfdan_module(use_cuda, device_ids, meanshape)
    # cuda
    if use_cuda:
        net_class = nn.parallel.DataParallel(net_class, device_ids=device_ids)
        net = net_class.module
    else:
        net = net_class

    #net.train()
    # prepare val_set
    val_loader = torch.utils.data.DataLoader(dataset=validationSet,
        batch_size=batchsize, shuffle=False, num_workers=0)

    # common
    common_loader = torch.utils.data.DataLoader(dataset=commonSet,
        batch_size=batchsize, shuffle=False, num_workers=0)
  
    # chall
    chall_loader = torch.utils.data.DataLoader(dataset=challengingSet,
        batch_size=batchsize, shuffle=False, num_workers=0)

    # w300
    w300_loader = torch.utils.data.DataLoader(dataset=w300Set,
        batch_size=batchsize, shuffle=False, num_workers=0)
    
    
    # for batches --- loop
    net.eval()
    record_filedes = open('res_eyeDist_record.txt', 'w')
    for i in range(7400, 7400 + 1):
        if i % 50 != 0:
            continue
        resume_iter = i
        resume(net, checkpoint_dir='./ckpt/', iteration=resume_iter, test=False)

        rcd = "iter [%08d]" % resume_iter

        # 1
        #val_res, val_mre = validate(net, batchsize, validationSet, val_loader, val_name = "VAL", use_cuda=use_cuda)
        #rcd = rcd + "\t VAL : MRE %.6f \t RES %.6f \n" % (val_mre, val_res)

        com_res, com_mre = validate(net, batchsize, commonSet, common_loader, val_name = "COMMON", use_cuda=use_cuda)
        rcd = rcd + "\t COMMON : MRE %.6f \t RES %.6f \n" % (com_mre, com_res)

        chal_res, chal_mre = validate(net, batchsize, challengingSet, chall_loader, val_name = "CHALL", use_cuda=use_cuda)
        rcd = rcd + "\t CHALL : MRE %.6f \t RES %.6f \n" % (chal_mre, chal_res)
        
        w300_res, w300_mre = validate(net, batchsize, w300Set, w300_loader, val_name = "300W", use_cuda=use_cuda)
        rcd = rcd + "\t 300W : MRE %.6f \t RES %.6f \n" % (w300_mre, w300_res)

    
        #rcd = rcd + "MRE \t %.6f \t %.6f \t %.6f \n" % (com_mre, chal_mre, w300_mre)
        #rcd = rcd + "RES \t %.6f \t %.6f \t %.6f \n" % (com_res, chal_res, w300_res)
        red = rcd + "===================================\n"

        print(rcd)
        record_filedes.write(rcd)

    record_filedes.close()

if __name__ == '__main__':
    main()