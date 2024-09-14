import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

from dataloader_finetuning import IcreLoader
from torch.utils.data import Dataset, DataLoader
import argparse
import tqdm
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import numpy as np
from datetime import datetime
import random

import dataread
import loaddata
import funcs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def testing(args, model, epoch, Xte, Yte, Ite, GT, phase, phaseN):  # Xte: feature, Yte: binary flag
    model.eval()

    GTstep=360/phaseN
    ICLrange=[0,(phase+1)*GTstep]

    Xte, Yte, Ite, GT, DoArange=funcs.ICLselect(Xte, Yte, Ite, GT, ICLrange, 'Test')

    all_val_out = torch.Tensor([])
    all_val_labels = torch.Tensor([])
    Y_pred_t=[]
    for ist in range(0, len(Xte), args.infer_batch_size):
        ied = np.min([ist + args.infer_batch_size, len(Xte)])
        inputs = Variable(torch.from_numpy(Xte[ist:ied])).type(torch.FloatTensor).to(device)
        output = model.forward(inputs)
        Y_pred_t.extend(output.cpu().detach().numpy()) # in CPU

    # ------------ error evaluate   ----------
    MAE1, ACC1, MAE2, ACC2, _,_,_,_, N1, N2 = funcs.MAEeval(Y_pred_t, Yte, Ite)
    # print(DoArange+" ep%1d ph%1d MAE1-2: %.1f, %.1f | ACC1-2: %.1f, %.1f" % (ep, phase, MAE1, MAE2, ACC1, ACC2))

    return MAE1, MAE2, ACC1, ACC2, N1, N2, DoArange


def train(args, step, total_incremental_steps, train_data_set):
    train_loader = DataLoader(train_data_set, batch_size=args.train_batch_size, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, shuffle=True)
    
    step_out_class_num = (step + 1) * args.class_num_per_step

    models, criterion = loaddata.Dataextract(args.model, 'MSE')
    if step == 0 or args.upperbound:
        model = models.Model(args, 360)
    else:
        model = torch.load('./save/step_{}_best_model.pkl'.format(step-1))
    
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    train_loss_list = []
    best_val_res = 0.0
    best_exemplar_class_vids = None
    best_exemplar_class_mean = None
    best_val_res = 0.0

    for epoch in range(args.max_epoches):
        train_loss = 0.0
        num_steps = 0
        model.train()
        for batch_idx, (data, labels, _) in tqdm.tqdm(enumerate(train_loader)):
            # data: input feature
            # target: 360-Gaussian distribution label
            inputs, labels = Variable(data).type(torch.FloatTensor).to(device), Variable(labels).type(torch.FloatTensor).to(device)

            # start training
            out = model.forward(inputs)  # return the predicted angle
            loss = criterion(out.double(), labels.double())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_steps += 1
        train_loss /= num_steps
        train_loss_list.append(train_loss)
        print('Epoch:{} train_loss:{:.5f}'.format(epoch, train_loss), flush=True)

        #--------------------------TEST-----------------------
        global Xte1, Yte1, Ite1, Xte2, Yte2, Ite2, GT1, GT2
        MAEl1, MAEl2, ACCl1, ACCl2, Nl1, Nl2, DoArange = testing(args, model, -1, Xte2, Yte2, Ite2, GT2, step, total_incremental_steps) 
        MAEh1, MAEh2, ACCh1, ACCh2, Nh1, Nh2, _ = testing(args, model, -1, Xte1, Yte1, Ite1, GT1, step, total_incremental_steps)

        N=Nl1+Nl2+Nh1+Nh2
        MAEavg = Nl1/N*MAEl1 + Nl2/N*MAEl2 + Nh1/N*MAEh1 + Nh2/N*MAEh2
        ACCavg = Nl1/N*ACCl1 + Nl2/N*ACCl2 + Nh1/N*ACCh1 + Nh2/N*ACCh2
        print("Epoch %01d/%01d:: MAE1:%.2f, ACC1:%.2f, MAE2:%.2f, ACC2:%.2f,| MAEh1:%.2f, ACCh1:%.2f, MAEh2:%.2f, ACCh2:%.2f| Avg.MAE, ACC %.2f %.2f" % (epoch, args.max_epoches, MAEl1, ACCl1, MAEl2, ACCl2, MAEh1, ACCh1, MAEh2, ACCh2, MAEavg, ACCavg), flush=True)

        if ACCavg > best_val_res:
            best_val_res = ACCavg
            print('Saving best model at Epoch {}'.format(epoch), flush=True)
            torch.save(model, './save/step_{}_best_model.pkl'.format(step))

        # scheduler.step()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', metavar='model', type=str, default='MLP3') # Options: modelMLP3attentsoftmax
    parser.add_argument('--drop', metavar='drop', type=float, default=0.2)
    parser.add_argument('--train', metavar='train', type=int, default=1)  # whether need training set
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--infer_batch_size', type=int, default=128)
    parser.add_argument('--gen_exem_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_epoches', type=int, default=30)
    parser.add_argument('--num_classes', type=int, default=360)

    parser.add_argument('-trA', metavar='trA', type=str, default='gcc') # Options:  
    parser.add_argument('-teA', metavar='teA', type=str, default='gcc') # Options: gcc, melgcc
    parser.add_argument('-trV', metavar='trV', type=str, default=None) # Options: 'faceSSLR'
    parser.add_argument('-teV', metavar='teV', type=str, default=None) #  
    parser.add_argument('-datapath', type=str, default='/mntcephfs/lee_dataset/loc/ICASSP2021data')  # train and test on frames with face

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=boolean_string, default=False)
    parser.add_argument("--milestones", type=int, default=[500], nargs='+', help="")
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--class_num_per_step', type=int, default=36)
    parser.add_argument('--memory_size', type=int, default=500)
    parser.add_argument('--upperbound', action='store_true')
    args = parser.parse_args()
    print(args)

    total_incremental_steps = args.num_classes // args.class_num_per_step

    setup_seed(args.seed)
    
    print('Training start time: {}'.format(datetime.now()))

    train_set = IcreLoader(args=args, mode='train')
    Xtr, Ytr, Itr, Ztr, Xte1, Yte1, Ite1, Xte2, Yte2, Ite2, GT1, GT2, GTtr = dataread.dataread(args)

    ckpts_root = './save'
    if not os.path.exists(ckpts_root):
        os.makedirs(ckpts_root)

    for step in range(total_incremental_steps):

        train_set.set_incremental_step(step)

        print('-----------------------------------Incremental step: {}-----------------------------------------------'.format(step))
        train(args, step, total_incremental_steps, train_set)
        print('-----------------------------------End Incremental step: {}-----------------------------------------------'.format(step))