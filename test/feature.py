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
import network
import feautil as futil
import pickle
CHECK_DIR='./checkpoint'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=50
train_iter = futil.create_iterator()

train_loader = torch.utils.data.DataLoader(
        train_iter,
        batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
net = network.create_model()
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
    print("=> no checkpoint found at '{}'".format(best_path))
#=======================================================
net= net.to(device)
optimizer = optim.SGD(net.parameters(), lr=0.0000001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
feades={}
for inputs, labels in train_loader:  
    net.eval()
    inputs = inputs.to(device)
    optimizer.zero_grad()
    with torch.set_grad_enabled(False):
        fdes=net.forward_once(inputs)
    for i in range(fdes.shape[0]):
        feades[labels[i]]=fdes[i].cpu().data
with open('updateevafeaturedes', 'wb') as fp:
    pickle.dump(feades, fp)

print(len(feades))
print("the end!")


        




