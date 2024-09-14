import torch
import torch.nn as nn
import numpy as np
import tqdm
import torch.nn.functional as F
import funcs
from torch.utils.data import DataLoader

def cls_align(train_loader, model, args):

    new_model = torch.nn.Sequential(model.MLP3, model.fc[:2])
    model.eval()

    auto_cor = torch.zeros(model.fc[-1].weight.size(1), model.fc[-1].weight.size(1)).to(args.device)
    # crs_cor = torch.zeros(model.fc[-1].weight.size(1), args.num_classes).to(args.device)
    crs_cor = torch.zeros(model.fc[-1].weight.size(1), 360).to(args.device)

    with torch.no_grad():
        for epoch in range(1):
            pbar = tqdm.tqdm(enumerate(train_loader), desc='Re-Alignment Base', total=len(train_loader), unit='batch')
            for i, (images, target,_,_) in pbar:
                images = images.float().to(args.device)
                target = target.to(args.device)

                new_activation = new_model(images)
                # label_onehot = F.one_hot(target, args.num_classes).float()
                label_onehot = target.float()

                auto_cor += torch.t(new_activation) @ new_activation
                crs_cor += torch.t(new_activation) @ (label_onehot)
                # pre_output = model(images)
                # pre_loss = F.cross_entropy(pre_output, labels)
                # softmax = nn.Softmax(dim=1)
    # R = torch.pinverse(auto_cor + args.rg * torch.eye(model.fc[-1].weight.size(1)).cuda(args.gpu, non_blocking=True))
    print('numpy inverse')
    R = np.mat(auto_cor.cpu().numpy() + args.rg * np.eye(model.fc[-1].weight.size(1))).I
    R = torch.tensor(R).float().to(args.device)
    Delta = R @ crs_cor
    model.fc[-1].weight = torch.nn.parameter.Parameter(torch.t(0.9*Delta.float()))
    return R



def IL_align(train_loader, model, args, R, repeat):

    new_model = torch.nn.Sequential(model.MLP3, model.fc[:2])
    model.eval()

    W = (model.fc[-1].weight.t()).double()
    R = R.double()
    with torch.no_grad():
        for epoch in range(repeat):
            pbar = tqdm.tqdm(enumerate(train_loader), desc='Re-Alignment', total=len(train_loader), unit='batch')
            for i, (images, target,_,_) in pbar:
                images = images.float().to(args.device)
                target = target.to(args.device)
 
                new_activation = new_model(images)
                new_activation = new_activation.double()
                # label_onehot = F.one_hot(target, args.num_classes).double()
                label_onehot = target.float()

                R = R - R@new_activation.t()@torch.pinverse(torch.eye(images.size(0)).to(args.device) +
                                                                    new_activation@R@new_activation.t())@new_activation@R
                W = W + R @ new_activation.t() @ (label_onehot - new_activation @ W)

    #Delta = torch.pinverse(auto_cor + args.rg * torch.eye(model.fc[-1].weight.size(1)).cuda(args.gpu, non_blocking=True)) @ crs_cor
    model.fc[-1].weight = torch.nn.parameter.Parameter(torch.t(W.float()))
    return R


def ILdata_select(Xtr, Ztr, Itr, GTtr, phase, args):
    phaseN= args.phaseN

    GTstep=360/phaseN
    ICLrange=[0,(phase+1)*GTstep] if args.upperbound else [phase*GTstep,(phase+1)*GTstep]

    Xtr, Ztr, Itr, GTtr, DoArange=funcs.ICLselect(Xtr, Ztr, Itr, GTtr, ICLrange, 'Train')
    # Ztr = Ztr[:,:args.num_classes]
    print(len(Xtr))
    train_loader_obj = funcs.MyDataloaderClass(Xtr, Ztr, Itr, GTtr)  # Xtr-data feature, Ztr-Gaussian-format label
    train_loader = DataLoader(dataset=train_loader_obj, batch_size=args.batch, shuffle=True, num_workers=1,drop_last=True)
    print('phase'+str(phase)+' select '+str(ICLrange[0])+'~'+str(ICLrange[1])+' DoA'+str(Ztr.shape[1]))
    return train_loader