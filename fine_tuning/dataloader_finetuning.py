import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import librosa
import random
import h5py
import dataread

class IcreLoader(Dataset):
    def __init__(self, args, mode='train', incremental_step=0):
        self.mode = mode
        self.args = args
        self.data_root = '/mntnfs/lee_data1/xianghuyue/projects/SPL2024_CIL_SSL/data'

        self.all_features_dict = np.load(
            os.path.join(self.data_root, 'all_features_dict.npy'), allow_pickle=True
        ).item()
        self.all_id_category_dict = np.load(
            os.path.join(self.data_root, 'all_id_category_dict.npy'), allow_pickle=True
        ).item()
        self.categoty_encode_dict = np.load(
            os.path.join(self.data_root, 'category_encode_dict.npy'), allow_pickle=True
        ).item()
        self.all_classId_vid_dict = np.load(
            os.path.join(self.data_root, 'all_classId_vid_dict.npy'), allow_pickle=True
        ).item()

        self.incremental_step = incremental_step
        self.current_step_class = self.set_current_step_classes()
        self.all_current_data_vids = self.current_step_data()

        self.exemplar_class_vids = None

    def set_current_step_classes(self):
        if self.mode == 'train':
            if self.args.upperbound:
                current_step_class = np.array(range(0, self.args.class_num_per_step * (self.incremental_step + 1)))
            else:
                current_step_class = np.array(range(self.args.class_num_per_step * self.incremental_step, self.args.class_num_per_step * (self.incremental_step + 1)))
        else:
            current_step_class = np.array(range(0, self.args.class_num_per_step * (self.incremental_step + 1)))
        return current_step_class

    def current_step_data(self):
        all_current_data_vids = []
        for class_idx in self.current_step_class:
            all_current_data_vids += self.all_classId_vid_dict[class_idx]
        return all_current_data_vids

    def set_incremental_step(self, step):
        self.incremental_step = step
        self.current_step_class = self.set_current_step_classes()
        self.all_current_data_vids = self.current_step_data()

    def __getitem__(self, index):
        vid = self.all_current_data_vids[index]
        category = self.all_id_category_dict[vid]
        category_id = self.categoty_encode_dict[category]
        feature, label = self.all_features_dict[vid]

        return feature, label, category_id
    
        
    def __len__(self):
        return len(self.all_current_data_vids)
