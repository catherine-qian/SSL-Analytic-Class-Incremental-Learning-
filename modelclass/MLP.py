import torch
import torch.nn.functional as F
import torch.nn as nn
from .layers import LSCLinear, SplitLSCLinear

print('here new model')

class Model(nn.Module):
    def __init__(self, args, step_out_class_num, LSC=False):
        super(Model, self).__init__()
        self.drop = args.drop

        self.MLP = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=args.drop),
        )
        
        if LSC:
            self.classifier = LSCLinear(1000, step_out_class_num)
        else:
            self.fc = nn.Linear(1000, step_out_class_num)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y_hidden = self.MLP(x[:,:306]) # only the audio parts
        y_pred = self.fc(y_hidden)
        y_pred = self.sig(y_pred)

        return y_pred
    

    def incremental_classifier(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_features = self.fc.in_features
        out_features = self.fc.out_features

        self.fc = nn.Linear(in_features, numclass, bias=True)
        self.fc.weight.data[:out_features] = weight
        self.fc.bias.data[:out_features] = bias
