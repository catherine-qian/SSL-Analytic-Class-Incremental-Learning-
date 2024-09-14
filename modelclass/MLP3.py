import torch
import torch.nn.functional as F
import torch.nn as nn
from .layers import LSCLinear, SplitLSCLinear

print('here new model')

class Model(nn.Module):
    def __init__(self, args, step_out_class_num, LSC=False):
        super(Model, self).__init__()
        self.drop = args.drop

        self.MLP3 = nn.Sequential(
            nn.Linear(306, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=args.drop),

            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=args.drop),

            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=args.drop),

        )
        
        if LSC:
            self.fc = LSCLinear(1000, step_out_class_num)
        else:
            self.fc = nn.Linear(1000, step_out_class_num)
        self.sig = nn.Sigmoid()

    def forward(self, x, out_logits=True, out_features=False, out_features_norm=False, AFC_train_out=False):
        y_hidden = self.MLP3(x[:,:306]) # only the audio parts
        y_pred = self.fc(y_hidden)
        y_pred = self.sig(y_pred)

        # outputs = ()
        # if AFC_train_out:
        #     y_hidden.retain_grad()
        #     outputs += (y_pred, y_hidden)
        #     return outputs
        # else:
        #     if out_logits:
        #         outputs += (y_pred,)
        #     if out_features:
        #         outputs += (F.normalize(y_hidden),)
        #     if len(outputs) == 1:
        #         return outputs[0]
        #     else:
        #         return outputs
        return y_pred
    

    def incremental_classifier(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_features = self.fc.in_features
        out_features = self.fc.out_features

        self.fc = nn.Linear(in_features, numclass, bias=True)
        self.fc.weight.data[:out_features] = weight
        self.fc.bias.data[:out_features] = bias
