from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import pickle
import torch.nn.functional as F
import network
import cv2
from pathlib import Path
import shutil
import torch.nn.functional as F
import testenv as scene_loader

import multiprocessing as mp
import threading as td
import time
import sys
mode='eva'
CHECK_DIR='./checkpoint'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open ('updateevafeaturedes', 'rb') as ft:
    wholeData=pickle.load(ft)
#=====================================================================================================
net = network.create_model(pretrained=False)
best_path = os.path.join(CHECK_DIR, 'best_53.ckpt') #
if os.path.isfile(best_path):      
    checkpoint = torch.load(best_path)
    global_t=checkpoint['global_t']
    net.load_state_dict(checkpoint['state_dict'],strict=False)
    print("=> loaded checkpoint '{}' (global_t {})"
        .format(best_path, checkpoint['global_t']))
else:
    print("can't evaluate!!!!!")

worlds= ['Home_011_1']
ACTIONS=['right', 'rotate_cw', 'rotate_ccw', 'forward', 'left', 'backward', 'stop']
#======================================================================================================
env =scene_loader.ActiveVisionDatasetEnv(world=worlds[0])
def getImagedes(world,startid):
    filename=world+'/'+startid
    imageDes=wholeData[filename]
    return imageDes

def collisionPreact(idx1,idx2):
    com=[(0,4),(4,0),(1,2),(2,1),(3,5),(5,3)]
    a=(idx1,idx2)
    if a in com:
        return False
    return True
    
def evaluate(data):
    global env
    sr_list=[0,0,0,0,0]
    spl_list=[0,0,0,0,0]
    spl=0
    successtime=0
    count=0
    pre_sr=0
    pre_spl=0
    for ele in data:
        count+=1
        if count%200==0 and count<1000:
            print("sr:",successtime)
            print("spl:",spl)
            idx=int(count/200)-1
            sr_list[idx]=(successtime-pre_sr)/200
            spl_list[idx]=(spl-pre_spl)/200
            pre_sr=successtime
            pre_spl=spl
        world=ele[0]
        startid=ele[1]
        endid=ele[2]
        preact=[0.]*7
        preact=np.array(preact,dtype=np.float32)
        preact=torch.from_numpy(preact)
        if world!=env._cur_world:
            env =scene_loader.ActiveVisionDatasetEnv(world=world)
        goal_vertex=env.id_to_index[endid]
        env._cur_world=world
        step=0
        while step<100:
            env._cur_image_id=startid
            reco=[]   
            for jdx in range(10):         
                env.step(2)
                if jdx%2==0:
                    continue
                else:
                    reco.append(env._cur_image_id)
            x0=getImagedes(world,reco[4])
            x1=getImagedes(world,reco[3])
            x2=getImagedes(world,reco[2])
            x3=getImagedes(world,reco[1])
            x4=getImagedes(world,reco[0])
            current=getImagedes(world,startid)
            goal=getImagedes(world,endid)
            net.eval()
            act_prob=net.chooseActionx(x0.unsqueeze(0),x1.unsqueeze(0),x2.unsqueeze(0),x3.unsqueeze(0),x4.unsqueeze(0),current.unsqueeze(0),goal.unsqueeze(0),preact.unsqueeze(0))
            _, pred = torch.max(act_prob, 1)
            action = env._actions[pred[0]]
            step+=1
            if action=='stop':
                next_image_id=False
            else:
                next_image_id = env._all_graph[world][startid][action]
            if not next_image_id:
                sortv,idx= torch.sort(act_prob[0])
                midi=1     
                while not next_image_id:
                    check=collisionPreact(torch.argmax(preact),idx[midi])
                    if not check:
                        midi+=1
                    action=env._actions[idx[midi]]
                    if action=='stop':
                        midi+=1
                        action=env._actions[idx[midi]]
                    pred[0]=idx[midi]
            
                    next_image_id = env._all_graph[world][startid][action]
                    midi+=1
                startid=next_image_id
            else:
                startid=next_image_id
            #======================================================
            pos1=env.pos[startid]
            pos2=env.pos[endid]
            path_len=(pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2
            l1=np.sqrt(pos1[2]*pos1[2]+pos1[3]*pos1[3])
            l2=np.sqrt(pos2[2]*pos2[2]+pos2[3]*pos2[3])
            v=(pos1[2]*pos2[2]+pos1[3]*pos2[3])/(l1*l2)
            if v>1:
                v=1
            if v<-1:
                v=-1
            path_ang=np.arccos(v)*180/3.14159
            if path_len<=1 and path_ang<90:
                vex=env.id_to_index[ele[1]]
                slen=len(env.shortest_path(vex,goal_vertex))
                mylen=step
                spl+=slen/max(mylen,slen)
                successtime+=1
                print(step)
                break
            #======================================================================
            preact=[0.]*7
            preact[pred[0]]=1.0
            preact=np.array(preact,dtype=np.float32)
            preact=torch.from_numpy(preact)
    sr_list[-1]=(successtime-pre_sr)/200
    spl_list[-1]=(spl-pre_spl)/200
    print("spl_list:",spl_list)   
    print("sr_list:",sr_list)  
    print("spl:",spl)   
    print("successtime:",successtime)    

if __name__=='__main__':
    evaluateData=[]
    with open ('test_data/testdata', 'rb') as ft:
        data=pickle.load(ft)
    evaluateData.extend(data)
    print("length:",len(evaluateData))
    evaluate(evaluateData)


 
   
   



    








    




