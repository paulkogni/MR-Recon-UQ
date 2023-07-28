import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pdb
import torch
import pathlib
import random
import h5py
import json


class SKMTEA(Dataset):

    def __init__(self, root, split_file):
        """

        Args:
            root (str): Path to the dataset
            split_file (str): Path to the split 
        """

        root_4x = os.path.join(root, 'skm-tea-4x')
        root_8x = os.path.join(root, 'skm-tea-8x')
        root_16x = os.path.join(root, 'skm-tea-16x')
        self.examples = []
        with open(split_file, 'r') as f:
            data = json.load(f)
        file_names = [data['images'][i]['file_name'] for i in range(len(data['images']))]



        files_4x = [os.path.join(root_4x, file_name) for file_name in file_names]
        files_8x = [os.path.join(root_8x, file_name) for file_name in file_names]
        files_16x = [os.path.join(root_16x, file_name) for file_name in file_names]
        self.examples = files_4x + files_8x + files_16x
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):

        fname = self.examples[index]

        
        # load the images
        with h5py.File(fname, 'r') as data:
            img_us = data['img_us']
            img_fs = data['img_fs']

            # extract slice and put axes in correct order
            slice = random.randrange(0, img_us.shape[2])
            img_us = np.transpose(img_us[:,:,slice,:], axes=(2,0,1))
            img_fs = np.transpose(img_fs[:,:,slice,:], axes=(2,0,1))

            # convert to torch tensors
            img_us = torch.Tensor(img_us)
            img_fs = torch.Tensor(img_fs)

            # normalize
            mean = img_us.mean()
            std = img_us.std()
            eps = 1e-5
            img_us = (img_us - mean) / (std + eps)
            img_fs = (img_fs - mean) / (std + eps)

            return img_us, img_fs, mean, std


def load_data_into_loader(batch_size, path, path_split):
    """

    Args:
        batch_size (int)
        path (str): path to the preprocessed dataset
        path_split (str): path to where the skmtea split files are 
    """
    train_path = os.path.join(path_split, 'train.json')
    val_path = os.path.join(path_split, 'val.json')

    dataset_train = SKMTEA(path, train_path)
    dataset_val = SKMTEA(path, val_path)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, drop_last=True, shuffle=True)
    validation_loader = DataLoader(dataset_val, batch_size=1, drop_last=True, shuffle=True)

    l_train = len(train_loader.dataset.examples)
    l_val = len(validation_loader.dataset.examples)
    print("Number of training/validation data:", (l_train, l_val))

    return train_loader, validation_loader