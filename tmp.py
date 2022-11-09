import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import csv
import os
import Dataset
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import sys
import random


##################実験設定#######################################
EPOCHS = 10 #最大学習エポック数
mag = '40x' #使用倍率 '40x' or '20x' or '10x' or '5x'
cv_test = 1
device='cuda:0'
################################################################

# 訓練用と検証用に症例を分割
random.seed(0)
num_cv = 5

train_all = []
valid_all = []
test_all = []
train_label = []
valid_label = []
test_label = []

list_subtype = [['DLBCL'],['FL'],['Reactive']]
# list_subtype = [['DLBCL', 'FL'],['AITL', 'ATLL'],['CHL'],['Reactive']] ####複数のサブタイプを1つのクラスとして扱う場合

list_use = ['DLBCL','FL','Reactive']

num_data = []
#max_main = 300 # 各クラスの最大症例数
#max_R = 300 # Reactiveクラスの最大症例数

label = 0
for group_subtype in list_subtype:
    train_group = []
    for subtype in group_subtype:
        if subtype in list_use:
            #if subtype == 'Reactive':
            #    max_sample = max_R
            #else:
            #    max_sample = max_main
            list_id = np.loadtxt(f'/ml/slide_ID/{subtype}.txt', delimiter=',', dtype='str').tolist()
            random.seed(0)
            random.shuffle(list_id)
            #list_id = list_id[0:max_sample]
            num_e = len(list_id) // num_cv
            num_r = len(list_id) % num_cv
            tmp_all = []
            for cv in range(num_cv):
                tmp = []
                for i in range(num_e):
                    tmp.append(list_id.pop(0))
                if cv < num_r:
                    tmp.append(list_id.pop(0))
                tmp_all.append(tmp)
            train_tmp = tmp_all[cv_test%5] + tmp_all[(cv_test+1)%5] + tmp_all[(cv_test+2)%5]
            #train_tmp = train_tmp[0:100] # train dataを減らす
            train_group += train_tmp
            train_all += train_tmp
            valid_all += tmp_all[(cv_test+3)%5]
            test_all += tmp_all[(cv_test+4)%5]
            train_tmp = [label] * len(train_tmp)
            valid_tmp = [label] * len(tmp_all[(cv_test+3)%5])
            test_tmp = [label] * len(tmp_all[(cv_test+4)%5])
            train_label += train_tmp
            valid_label += valid_tmp
            test_label += test_tmp
    num_data.append(len(train_group))
    label += 1
    

train_dataset = []
for i,slideID in enumerate(train_all):
    class_label = train_label[i]
    train_dataset.append([slideID, class_label])

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])

patches_train = Dataset.MyDataset(
    dataset=train_dataset,
    mag=mag,
    transform=transform,
    bag_max_size=10000,
    seed=0
)

train_loader = torch.utils.data.DataLoader(
    patches_train,
    batch_size=1,
    shuffle=False,
    num_workers=os.cpu_count(),
    pin_memory=True,
)

print('start dataloader')

for i,(bag, slideID, label) in enumerate(train_loader):
    
    bag = bag.to(device) 
    label = label.to(device) 
    
    print(i,'th ', slideID, ' : ', bag.size())
