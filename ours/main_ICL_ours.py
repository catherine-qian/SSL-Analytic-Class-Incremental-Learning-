import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import numpy as np
import torch
from torch.autograd import Variable
import hdf5storage
import argparse
from torch.utils.data import DataLoader
import dataread
import time
import tqdm
import random
import loaddata
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import scale as zscale # Z-score normalizatin: mean-0, std-1
import funcs
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
# import allnoise
from ICLfuncs import *

torch.manual_seed(7)  # For reproducibility across different computers
torch.cuda.manual_seed(7)

print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))       # 打印按指定格式排版的时间


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='xinyuan experiments')
    parser.add_argument('-gpuidx', metavar='gpuidx', type=int, default=0, help='gpu number')
    parser.add_argument('-epoch', metavar='EPOCH', type=int, default=30)
    parser.add_argument('-drop', metavar='drop', type=float, default=0.2)
    parser.add_argument('-lr', metavar='lr', type=float, default=0.001)
    parser.add_argument('-trA', metavar='trA', type=str, default='gcc') # Options:  
    parser.add_argument('-teA', metavar='teA', type=str, default='gcc') # Options: gcc, melgcc
    parser.add_argument('-trV', metavar='trV', type=str, default=None) # Options: 'faceSSLR'
    parser.add_argument('-teV', metavar='teV', type=str, default=None) #  
    parser.add_argument('-model', metavar='model', type=str, default='MLP3') # Options: modelMLP3attentsoftmax
    parser.add_argument('-batch', metavar='batch', type=int, default=2**8)
    parser.add_argument('-train', metavar='train', type=int, default=1)  # whether need training set
    # parser.add_argument('-test', metavar='eval', type=int, default=1)  # whether the evaluation mode
    parser.add_argument('-Vy', metavar='Vy', type=int, default=1)  # whether use the vertical video feature
    parser.add_argument('-VO', metavar='VO', type=int, default=0)  # train and test on frames with face
    parser.add_argument('-datapath', type=str, default='/mntcephfs/lee_dataset/loc/ICASSP2021data')  # train and test on frames with face
    parser.add_argument('-upbound', type=int, default=0)  # whether is the upperbound
    parser.add_argument('-Hidden', default=5000, type=int,help='')
    parser.add_argument('-phaseN', type=int, default=10)  # total incremetnal learning step number
    parser.add_argument('-recurbase', type=int, default=0)  # whether is the upperbound
    parser.add_argument('-full_classes', type=int, default=360)  # whether is the upperbound
    parser.add_argument('-baseclass', type=int, default=0)  # whether is the upperbound
    parser.add_argument('-pretrained', type=str, default=None)  # load pretrained base model
    parser.add_argument('-incremental', action='store_true')
    parser.add_argument('-basetraining', action='store_true')
    parser.add_argument('-upperbound', action='store_true')
    parser.add_argument('-rg', default=1e-1, type=float, help='')
    args = parser.parse_args()

# savemodel=False
BATCH_SIZE = args.batch
print(sys.argv[1:])
print("experiments - xinyuan")

device = torch.device("cuda:{}".format(args.gpuidx) if torch.cuda.is_available() else 'cpu')
args.device = device
print(device)
print(args)


def training(epoch, Xtr, Ztr, Itr, GTtr, phase, phaseN):
    model.train()

    GTstep=360/phaseN
    ICLrange=[phase*GTstep,(phase+1)*GTstep]

    train_loader = ILdata_select(Xtr, Ztr, Itr, GTtr, 0, args)
    for batch_idx, (data, target, num, gtlabel) in tqdm.tqdm(enumerate(train_loader)):
        # data: input feature
        # target: 360-Gaussian distribution label
        inputs, target = Variable(data).type(torch.FloatTensor).to(device), Variable(target).type(torch.FloatTensor).to(device)

        # start training
        y_pred = model.forward(inputs)  # return the predicted angle
        loss = criterion(y_pred.double(), target.double())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (round(train_loader.__len__()/5/100)*100)>0 and batch_idx % (round(train_loader.__len__()/5/100)*100) == 0:
            print("training - epoch%d-batch%d: loss=%.3f" % (epoch, batch_idx, loss.data.item()))
        
    scheduler.step()
    torch.save(model, 'save/checkpoint_ep' + str(epoch))

    torch.cuda.empty_cache()

def testing(epoch, Xte, Yte, Ite, GT, phase, phaseN):  # Xte: feature, Yte: binary flag
    model.eval()

    GTstep=360/phaseN
    ICLrange=[0,(phase+1)*GTstep]

    Xte, Yte, Ite, GT, DoArange=funcs.ICLselect(Xte, Yte, Ite, GT, ICLrange, 'Test')

    Y_pred_t=[]
    for ist in range(0, len(Xte), BATCH_SIZE):
        ied = np.min([ist+BATCH_SIZE, len(Xte)])
        inputs = Variable(torch.from_numpy(Xte[ist:ied])).type(torch.FloatTensor).to(device)
        output = model.forward(inputs)
        Y_pred_t.extend(output.cpu().detach().numpy()) # in CPU

    # ------------ error evaluate   ----------
    MAE1, ACC1, MAE2, ACC2, _,_,_,_, N1, N2 = funcs.MAEeval(Y_pred_t, Yte, Ite)
    # print(DoArange+" ep%1d ph%1d MAE1-2: %.1f, %.1f | ACC1-2: %.1f, %.1f" % (ep, phase, MAE1, MAE2, ACC1, ACC2))

    torch.cuda.empty_cache()
    return MAE1, MAE2, ACC1, ACC2, N1, N2, DoArange

# ############################# load the data and the model ##############################################################
Xtr, Ytr, Itr, Ztr, Xte1, Yte1, Ite1, Xte2, Yte2, Ite2, GT1, GT2, GTtr = dataread.dataread(args) # <--- logger to be added
modelname = args.model  
lossname='MSE'

models, criterion = loaddata.Dataextract(modelname, lossname)
model = models.Model(args, 360).to(device)
optimizer = torch.optim.Adam(model.parameters(), args.lr)
scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

if args.basetraining:
    ckpts_root = './save'
    if not os.path.exists(ckpts_root):
        os.makedirs(ckpts_root)

    for epoch in range(args.epoch):
        training(epoch, Xtr, Ztr, Itr, GTtr, 0, args.phaseN)

        MAEl1, MAEl2, ACCl1, ACCl2, Nl1, Nl2, DoArange = testing(-1, Xte2, Yte2, Ite2, GT2, 0, args.phaseN) 
        MAEh1, MAEh2, ACCh1, ACCh2, Nh1, Nh2, _ = testing(-1, Xte1, Yte1, Ite1, GT1, 0, args.phaseN)

        N=Nl1+Nl2+Nh1+Nh2
        MAEavg = Nl1/N*MAEl1 + Nl2/N*MAEl2 + Nh1/N*MAEh1 + Nh2/N*MAEh2
        ACCavg = Nl1/N*ACCl1 + Nl2/N*ACCl2 + Nh1/N*ACCh1 + Nh2/N*ACCh2
        print("Base training Epoch %01d/%01d:: MAE1:%.2f, ACC1:%.2f, MAE2:%.2f, ACC2:%.2f,| MAEh1:%.2f, ACCh1:%.2f, MAEh2:%.2f, ACCh2:%.2f| Avg.MAE, ACC %.2f %.2f" % (epoch, args.epoch, MAEl1, ACCl1, MAEl2, ACCl2, MAEh1, ACCh1, MAEh2, ACCh2, MAEavg, ACCavg))

    sys.exit()


# print(model)
# h,b,p=plt.hist(GTtr,bins=360)
if args.pretrained:
     model = torch.load(args.pretrained, map_location=device)
     print('Load pretrained base model:' + args.pretrained)

######## Training + Testing #######
EP = args.epoch
MAEl1, MAEl2, ACCl1, ACCl2 = np.zeros(args.phaseN), np.zeros(args.phaseN), np.zeros(args.phaseN), np.zeros(args.phaseN)
MAEh1, MAEh2, ACCh1, ACCh2 = np.zeros(args.phaseN), np.zeros(args.phaseN), np.zeros(args.phaseN), np.zeros(args.phaseN)
MAEavg, ACCavg = np.zeros(args.phaseN), np.zeros(args.phaseN) 

model_dict = {}

# incremental learning
print('Training for base classes...')
args.baseclass=360//args.phaseN
args.num_classes = args.baseclass

train_loader = ILdata_select(Xtr, Ztr, Itr, GTtr, 0, args)
if args.incremental:
    bias_fe = False
    args.num_classes = args.baseclass
    model.fc = nn.Sequential(nn.Linear(model.fc.weight.size(1), args.Hidden, bias=bias_fe), # args.Hidden=5000, 对应文章中expansion到最后的classifier
                                nn.ReLU(),
                                # nn.Linear(args.Hidden, args.num_classes, bias=False)).to(device) # 逐步增加last layer
                                nn.Linear(args.Hidden, 360, bias=False)).to(device)

    if args.recurbase:
        R = ((1 * torch.eye(args.Hidden)).float()).to(device)
        R = IL_align(train_loader, model, args, R, 1)
    else:
        R = cls_align(train_loader, model, args)

phase=0
MAEl1[phase], MAEl2[phase], ACCl1[phase], ACCl2[phase], Nl1, Nl2, DoArange = testing(-1, Xte2, Yte2, Ite2, GT2, phase, args.phaseN) 
MAEh1[phase], MAEh2[phase], ACCh1[phase], ACCh2[phase], Nh1, Nh2, _ = testing(-1, Xte1, Yte1, Ite1, GT1, phase, args.phaseN) 

N=Nl1+Nl2+Nh1+Nh2
MAEavg[phase] = Nl1/N*MAEl1[phase]+Nl2/N*MAEl2[phase]+Nh1/N*MAEh1[phase]+Nh2/N*MAEh2[phase]
ACCavg[phase] = Nl1/N*ACCl1[phase]+Nl2/N*ACCl2[phase]+Nh1/N*ACCh1[phase]+Nh2/N*ACCh2[phase]
print("Base phase %01d/%01d: MAE1-2:%.1f %.2f, ACC1-2:%.2f %.2f| MAEh1-2:%.2f %.2f, ACCh1-2:%.2f %.2f | Avg.MAE, ACC %.2f %.2f" %
    (0, args.phaseN, MAEl1[0], MAEl2[0], ACCl1[0], ACCl2[0],MAEh1[0], MAEh2[0], ACCh1[0], ACCh2[0], MAEavg[0], ACCavg[0]))


# CIL for phases
print('Training for incremental classes with {} phase(s) in total...'.format(args.phaseN))

nc_each = 360//args.phaseN
ep=-1 # no need to train
frate1, frate2,frateh1, frateh2 = np.zeros(args.phaseN), np.zeros(args.phaseN),np.zeros(args.phaseN), np.zeros(args.phaseN)
for phase in range(1, args.phaseN):
    if args.incremental:
        args.num_classes = args.baseclass+nc_each*phase

        # matrix update
        W = model.fc[-1].weight
        # W = torch.cat([W, torch.zeros(args.num_classes-W.shape[0], args.Hidden).to(args.device)], dim=0)  # 逐步增加last layer
        # model.fc[-1] = nn.Linear(args.Hidden, args.num_classes, bias=False) # 逐步增加last layer
        model.fc[-1] = nn.Linear(args.Hidden, 360, bias=False) # 
        model.fc[-1].weight = torch.nn.parameter.Parameter(W.float()) # initialize

        train_loader_phase = ILdata_select(Xtr, Ztr, Itr, GTtr, phase, args)
        R = IL_align(train_loader_phase, model, args, R, repeat=1)
        print('Incremental Learning for Phase {}/{}'.format(phase + 1, args.phaseN))

    # testing
    MAEl1[phase], MAEl2[phase], ACCl1[phase], ACCl2[phase], Nl1, Nl2, DoArange = testing(ep, Xte2, Yte2, Ite2, GT2, phase, args.phaseN) # loud speaker
    MAEh1[phase], MAEh2[phase], ACCh1[phase], ACCh2[phase], Nh1, Nh2, DoArange = testing(ep, Xte1, Yte1, Ite1, GT1, phase, args.phaseN) # human set

    frate1[phase], frate2[phase] = ACCl1[0]-ACCl1[phase], ACCl2[0]-ACCl2[phase]
    frateh1[phase], frateh2[phase] = ACCh1[0]-ACCh1[phase], ACCh2[0]-ACCh2[phase]

    N=Nl1+Nl2+Nh1+Nh2
    MAEavg[phase] = Nl1/N*MAEl1[phase]+Nl2/N*MAEl2[phase]+Nh1/N*MAEh1[phase]+Nh2/N*MAEh2[phase]
    ACCavg[phase] = Nl1/N*ACCl1[phase]+Nl2/N*ACCl2[phase]+Nh1/N*ACCh1[phase]+Nh2/N*ACCh2[phase]
    print(DoArange+"phase %01d/%01d: MAE1-2:%.2f %.2f, ACC1-2:%.2f %.2f| MAEh1-2:%.2f %.2f, ACCh1-2:%.2f %.2f | Avg.MAE, ACC %.2f %.2f" %
    (phase, args.phaseN, MAEl1[phase], MAEl2[phase], ACCl1[phase], ACCl2[phase],MAEh1[phase], MAEh2[phase], ACCh1[phase], ACCh2[phase], MAEavg[phase], ACCavg[phase]))

    print("Forgeting rate for Phase %01d/%01d: ACC1 %.2f ACC2 %.2f | ACCh1 %.2f ACCh2 %.2f" % (phase, args.phaseN, frate1[phase], frate2[phase],frateh1[phase], frateh2[phase]))


print("finish all! average test MAE1-2:%.2f %.2f, ACC1-2:%.2f %.2f| MAEh1-2:%.2f %.2f, ACCh1-2:%.2f %.2f | Avg.MAE %.2f ACC%.2f" % (np.mean(MAEl1), np.mean(MAEl2), np.mean(ACCl1), np.mean(ACCl2),np.mean(MAEh1[MAEh1>0]), np.mean(MAEh2[MAEh2>0]), np.mean(ACCh1[ACCh1>0]), np.mean(ACCh2[ACCh2>0]), np.mean(MAEavg), np.mean(ACCavg)))
print("finish all! paper rst: loadspeaker: %.2f %.2f, %.2f %.2f | human: %.2f %.2f, %.2f %.2f | Avg. %.2f, %.2f" % (np.mean(MAEl1),np.mean(ACCl1),  np.mean(MAEl2), np.mean(ACCl2), np.mean(MAEh1[MAEh1>0]), np.mean(ACCh1[ACCh1>0]), np.mean(MAEh2[MAEh2>0]), np.mean(ACCh2[ACCh2>0]), np.mean(MAEavg), np.mean(ACCavg)))
