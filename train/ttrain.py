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
import tnetwork as network
import cv2
from pathlib import Path
import shutil
import tutil as util
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=128
CHECKPOINT=True
CHECK_DIR='./checkpoint'
with open ('./data/round_traindata', 'rb') as ft:
    data=pickle.load(ft)
print("data:",len(data))


with open ('./data/round_testdata', 'rb') as ft:
    test_data=pickle.load(ft)
depthdata=util.read_cached_data(64)['depth']
train_iter = util.create_iterator(
        data,
        BATCH_SIZE)
test_iter=util.create_iterator(test_data,BATCH_SIZE,istrain=False)

train_loader = torch.utils.data.DataLoader(
        train_iter,
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)

test_loader = torch.utils.data.DataLoader(
        test_iter,
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
#==========================================================================

net = network.create_model()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

best_path = os.path.join(CHECK_DIR, 'best.ckpt')
if os.path.isfile(best_path):      
    checkpoint = torch.load(best_path,map_location=device)
    global_t=checkpoint['global_t']
    net.load_state_dict(checkpoint['state_dict'],strict=False)
    net.train()
    print("=> loaded checkpoint '{}' (global_t {})"
        .format(best_path, checkpoint['global_t']))
else:
    global_t=0
    net.train()
    print("=> no checkpoint found at '{}'".format(best_path))
net= net.to(device)
criterion = nn.MSELoss()
criterionx = nn.CrossEntropyLoss()
testacc={}
testacc[0]=0.
if not os.path.exists('out5/'):
    os.makedirs('out5/')
for epoch in range(1500): 
    print("ok!!")
    running_loss = 0.0
    running_corrects = 0
    i=0
    scheduler.step()
    for x0,x1,x2,x3,x4,x5, xnext,g,labels,Nextobs,preact,pre_labeld in train_loader:
        net.train()    
        x0= x0.to(device)
        x1= x1.to(device)
        x2= x2.to(device)
        x3= x3.to(device)
        x4= x4.to(device)
        x5= x5.to(device)
        xnext= xnext.to(device)
        preact=preact.to(device)
        g=g.to(device)
        labels = labels.to(device)
        pre_labeld=pre_labeld.to(device)         
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            i+=1 
            act_prob=net(x5,xnext,preact)
            probs = F.softmax(act_prob,dim=1)
            preds = probs.multinomial(1,replacement=False).view(-1).data#
            _, a=torch.max(pre_labeld,1)
            class_loss = criterionx(act_prob,a)
            loss=class_loss
            loss.backward()
            optimizer.step()
        global_t+=1
        running_loss += loss.item()* labels.size(0)
        running_corrects += torch.sum(preds.data.cpu() == a.data.cpu())
        if i % 10== 0 and i>0:   
            print('epoch:', epoch, '| global_t:', global_t,'| class_loss:', class_loss.item(),'| all_loss:',loss.item())            
            print("prediction action:",preds.data.cpu())
            print("ground truth action:",a.data.cpu())
            print("acc:",running_corrects.double() / (i*BATCH_SIZE))

        if i % 200== 0:
            util.save_checkpoint({'global_t': global_t,'state_dict': net.state_dict(),'epoch_loss':None,'epoch_act_acc':None,'epoch_sce_acc':None}, global_t) 
            print("testacc：",testacc)       #
        del x0,x1,x2,x3,x4,x5, xnext,g,labels

    epoch_loss = running_loss / (i*BATCH_SIZE)
    epoch_acc = running_corrects.double() / (i*BATCH_SIZE)

    print('{} Loss: {:.4f} act_Acc: {:.4f}'.format(
                'train', epoch_loss, epoch_acc))
    util.save_checkpoint({'global_t': global_t,'state_dict': net.state_dict(),'epoch_loss':epoch_loss,'epoch_act_acc':epoch_acc}, global_t)  
    running_corrects = 0
    #==================================for test============================================================================================
    j=0
    for x0,x1,x2,x3,x4,x5, xnext,g,labels,strname,preact,pre_labeld  in test_loader:
        j+=1   
        x0= x0.to(device)
        x1= x1.to(device)
        x2= x2.to(device)
        x3= x3.to(device)
        x4= x4.to(device)
        x5= x5.to(device)
        xnext= xnext.to(device)
        g=g.to(device)
        labels = labels.to(device)
    
        preact = preact.to(device)
        pre_labeld = pre_labeld.to(device)
        net.eval()      
        optimizer.zero_grad()    
        with torch.set_grad_enabled(False):
            act_prob=net(x5, xnext, preact)        
            _, preds = torch.max(act_prob, 1)
            _, apre_labeld=torch.max(pre_labeld,1)
        running_corrects += torch.sum(preds.data.cpu() == apre_labeld.data.cpu())
        del x0,x1,x2,x3,x4,x5, g, act_prob,

    epoch_acc = running_corrects.double() / (j*BATCH_SIZE)
    testacc[global_t]=epoch_acc
    print('{} act_Acc: {:.4f}'.format(
                'test', epoch_acc))

    





