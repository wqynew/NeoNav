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
#import matplotlib.pyplot as plt
import time
import pickle
import torch.nn.functional as F

import cv2
from pathlib import Path
import shutil

gDataIdxList=[]
CHECK_DIR='./checkpoint'

_MAX_DEPTH_VALUE = 12102

ACTIONS=['right', 'rotate_cw', 'rotate_ccw', 'forward', 'left', 'backward', 'stop']

print("hello1")

def read_cached_data(output_size):
    load_start = time.time()
    result_data = {}

    depth_image_path = './depth_imgs.npy'
    logging.info('loading depth: %s', depth_image_path)
    depth_data = dict(np.load(depth_image_path,encoding="latin1").item())

    logging.info('processing depth')
    for home_id in depth_data:
        #print(home_id)
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


def checkaction(pre,nex):
    a=(pre, nex)
    fallist=[(0,4),(4,0),(1,2),(2,1),(3,5),(5,3)]
    if a in fallist:
        return False
    else:
        return True 
        
def save_checkpoint(state, global_t):
    filename = 'checkpoint-{}.ckpt'.format(global_t)
    checkpoint_path = os.path.join(CHECK_DIR, filename)
    best_path = os.path.join(CHECK_DIR, 'best.ckpt')
    torch.save(state, best_path)
    torch.save(state, checkpoint_path)
    print('--- checkpoint saved to %s ---'.format(checkpoint_path))

def gau_kl(q_mu, q_sigma, p_mu, p_sigma):
    # https://github.com/openai/baselines/blob/f2729693253c0ef4d4086231d36e0a4307ec1cb3/baselines/acktr/utils.py
    num = (q_mu - p_mu)**2 + q_sigma**2 - p_sigma**2
    den = 2 * (p_sigma**2) + 1e-8
    kl = torch.mean(num/den + torch.log(p_sigma) - torch.log(q_sigma))
    #print(kl)
    return kl

# def gau_kl(pm, pv, qm, qv):
#     """
#     Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
#     Also computes KL divergence from a single Gaussian pm,pv to a set
#     of Gaussians qm,qv.
#     Diagonal covariances are assumed.  Divergence is expressed in nats.
#     """
#     # Determinants of diagonal covariances pv, qv
#     div=torch.clamp(torch.prod(qv/pv,1),min=1e-100)
#     # Inverse of diagonal covariance qv
#     iqv = 1./torch.clamp(qv,min=1e-7)
#     # Difference between means pm, qm
#     diff = qm - pm
#     result=0.5 *(torch.log(div)            # log |\Sigma_q| / |\Sigma_p|
#              + torch.sum(iqv * pv, 1)          # + tr(\Sigma_q^{-1} * \Sigma_p)
#              + torch.sum(diff * iqv * diff, 1)) -200 # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
#     del pm,pv,qm,qv,div,diff,iqv
#     return result

class Dataset(object):
    def __init__(self, data,labeld,Nextobs,phasestr,preact,pre_labeld):
        labeld,preact, pre_labeld=np.array(labeld,dtype=np.float32),np.array(preact,dtype=np.float32),np.array(pre_labeld,dtype=np.float32) 
        Nextobs=np.array(Nextobs)
        #==========================================================
        self.size = labeld.shape[0]
        self.data=data
        #============================================================
        self.Nextobs=Nextobs
        self.labeld = torch.from_numpy(labeld)
        self.pre_act = torch.from_numpy(preact)
        self.pre_labeld= torch.from_numpy(pre_labeld)
        self.phasestr=phasestr
        self.depth_data=read_cached_data(64)['depth']
        #================================================================
        

       
    def __getitem__(self, index):
        home_id=self.data[index][0]
        x1=self.depth_data[home_id][self.data[index][1]]
        x2=self.depth_data[home_id][self.data[index][2]]
        x3=self.depth_data[home_id][self.data[index][3]]
        x4=self.depth_data[home_id][self.data[index][4]]
        x5=self.depth_data[home_id][self.data[index][5]]
        x6=self.depth_data[home_id][self.data[index][6]]
        nextx=self.depth_data[home_id][self.data[index][7]]
        g=self.depth_data[home_id][self.data[index][8]]
        
        #print(home_id)
        if self.phasestr=='train':
            return (x1,x2,x3,x4,x5,x6,nextx,g,
                self.labeld[index],
                self.Nextobs[index],self.pre_act[index],self.pre_labeld[index])
        else:
            return (x1,x2,x3,x4,x5,x6,nextx,g,
                self.labeld[index],
                index,self.pre_act[index],self.pre_labeld[index])

    def __len__(self):
        return self.size

#==========================================================
def create_pairs(data,phasestr,istrain=True):
    labeld = []
    pre_act=[]
    Nextobs=[]
    TrainData=[]

    if istrain:
        with open ('./data/inquiry_train_data', 'rb') as ft:
            Inquiry_data=pickle.load(ft)
    else:
        with open ('./data/inquiry_test_data', 'rb') as ft:#testinquiry_data51
            Inquiry_data=pickle.load(ft)
    if len(Inquiry_data)!=len(data):
        print ("please check!!!!")

    for i in range(len(data)):
        inqui=[data[i][6],data[i][8]]
        indices = [idx for idx, ele in enumerate(Inquiry_data) if ele == inqui]
        tmp_action=[0,0,0,0,0,0,0]
        tmp_obs=[i]*7
        for ele in indices:
            tmp_action[ACTIONS.index(data[ele][9])]=1
            tmp_obs[ACTIONS.index(data[ele][9])]=ele
        #print("tmp_action:",tmp_action)
        tmp_action=np.array(tmp_action,dtype=np.float32)
        labeld.append(tmp_action)
        tmp_pre=[0.]*7
        if data[i][10]!='zero':  
            tmp_pre[ACTIONS.index(data[i][10])]=1.
        tmp_pre=np.array(tmp_pre,dtype=np.float32)
        pre_act.append(tmp_pre)
        Nextobs.append(tmp_obs)
        if i%100==0:
            print(i)
    print("stop reading!!")
    if istrain:
        with open('./data/roundtrainaction', 'wb') as fp:
            pickle.dump(labeld, fp)
        with open('./data/roundpretrainaction', 'wb') as fp:
            pickle.dump(pre_act, fp)
        with open('./data/roundtrainobs', 'wb') as fp:
            pickle.dump(Nextobs, fp)
    else:
        with open('./data/roundtestaction', 'wb') as fp:
            pickle.dump(labeld, fp)

        with open('./data/roundpretestaction', 'wb') as fp:
            pickle.dump(pre_act, fp)
    return labeld,Nextobs,pre_act

def create_label(data,phasestr,istrain=True):
    pre_labeld = []
    for i in range(len(data)):
        tmp_action=[0,0,0,0,0,0,0]
        tmp_action[ACTIONS.index(data[i][9])]=1
        tmp_action=np.array(tmp_action,dtype=np.float32)
        pre_labeld.append(tmp_action)       
    print("stop reading!!")
    if istrain:
        with open('./data/roundtrainoneaction', 'wb') as fp:
            pickle.dump(pre_labeld, fp)
    else:
        with open('./data/roundtestoneaction', 'wb') as fp:
            pickle.dump(pre_labeld, fp)
    return pre_labeld

def create_iterator(data, batchsize, istrain=True):
    Nextobs=[]
    if istrain:
        my_file = Path("./data/roundtrainaction")
        if my_file.is_file():
            with open ('./data/roundtrainaction', 'rb') as ft:
                labeld=pickle.load(ft)
            with open ('./data/roundtrainobs', 'rb') as ft:
                Nextobs=pickle.load(ft)
            with open ('./data/roundpretrainaction', 'rb') as ft:
                pre_act=pickle.load(ft)
        else:
            labeld,Nextobs,pre_act= create_pairs(data,'train',istrain) 
        #=========================================================================================     
        my_file = Path("./data/roundtrainoneaction")
        if my_file.is_file():
            print("right!!")
            with open ('./data/roundtrainoneaction', 'rb') as ft:
                pre_labeld=pickle.load(ft)
        else:
            pre_labeld= create_label(data,'train',istrain) 
        #=============================================================================================
        phasestr='train'
    else:
        my_file = Path("./data/roundtestaction")
        if my_file.is_file():
            with open ('./data/roundtestaction', 'rb') as ft:
                labeld=pickle.load(ft)
            with open ('./data/roundpretestaction', 'rb') as ft:
                pre_act=pickle.load(ft)
        else:
            labeld,_,pre_act= create_pairs(data,'eva',istrain)
        #=========================================================================================     
        my_file = Path("./data/roundtestoneaction")
        if my_file.is_file():
            print("right!!")
            with open ('./data/roundtestoneaction', 'rb') as ft:
                pre_labeld=pickle.load(ft)
        else:
            pre_labeld= create_label(data,'eva',istrain) 
        #=============================================================================================
        phasestr='eva'    
    ret = Dataset(data,labeld, Nextobs,phasestr,pre_act,pre_labeld)
    return ret
#==========================================================
