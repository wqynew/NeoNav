import collections
import copy
import json
import os
import time
import gym
from gym.envs.registration import register
import gym.spaces
import networkx as nx
import numpy as np
import scipy.io as sio
from absl import logging
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, models, transforms
import time
import pickle
import torch.nn.functional as F

import cv2
from pathlib import Path
import shutil

_MAX_DEPTH_VALUE = 12102

def read_cached_data(output_size):
    load_start = time.time()
    result_data = {}
    depth_image_path = './depth_imgs.npy'
    logging.info('loading depth: %s', depth_image_path)
    depth_data = dict(np.load(depth_image_path,encoding="latin1").item())
    logging.info('processing depth')
    for home_id in depth_data:
        images = depth_data[home_id]
        for image_id in images:
            depth = images[image_id]
            
            depth = cv2.resize(
                depth / _MAX_DEPTH_VALUE, (output_size, output_size),
                interpolation=cv2.INTER_NEAREST)
            depth_mask = (depth > 0).astype(np.float32)
            depth = np.dstack((depth, depth_mask))
            images[image_id] = depth
    result_data['depth'] = depth_data
    return result_data

worlds=['Home_011_1','Home_013_1' ,'Home_016_1' ]
#worlds= ['Home_001_1','Home_002_1','Home_003_1','Home_004_1','Home_006_1','Home_010_1', 'Home_014_1', 'Home_015_1']

class Dataset(object):
    def __init__(self):
        self.depth_data=read_cached_data(64)['depth']
        self.label=[]
        self.strname=[]
        for world in worlds:
            keys=self.depth_data[world].keys()
            for ele in keys:
                t=[world,ele]
                self.label.append(t)
                self.strname.append(world+'/'+ele) 
    def __getitem__(self, index):
        world=self.label[index][0]
        imid=self.label[index][1]
        return (torch.from_numpy(self.depth_data[world][imid]).float(),
                self.strname[index])
    def __len__(self):
        return len(self.label)

def create_iterator():
    ret = Dataset()
    return ret
