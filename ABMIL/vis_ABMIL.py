import numpy as np
from PIL import Image, ImageStat, ImageDraw
import argparse
import os, re, shutil, sys, time
import numpy as np
import csv
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import random

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def str_to_float(l):
    return float(l)

def draw_heatmap(slideID, att_dir, save_dir):
    b_size = 224    #ブロック画像のサイズ
    t_size = 4 #サムネイル画像中の1ブロックのサイズ
    img = cv2.imread(f'./thumb_JMR/{slideID}_thumb.tif')
    #thumb = Image.open(f'../../../../nvme/Data/DataKurume/thumb/{slideID}_thumb.tif')
    w, h = img.shape[1], img.shape[0]
    w_num = w // t_size
    h_num = h // t_size

    att_file = f'{att_dir}/{slideID}.csv'
    att_data = np.loadtxt(att_file, delimiter=',')

    att = []
    pos_x = []
    pos_y = []
    # 症例内のアテンション最大値と最小値を取得
    att_list = att_data[:,2].astype(np.float32).tolist()
    att_max = max(att_list)
    att_min = min(att_list)
    for j in range (len(att_list)):
        att_list[j] = (att_list[j] - att_min) / (att_max - att_min) #attentionを症例で正規化
    att = att_list
    pos_x = att_data[:,0].astype(np.int).tolist()
    pos_y = att_data[:,1].astype(np.int).tolist()

    cmap = plt.get_cmap('jet')

    for i in range (len(att)):
        cval = cmap(float(att[i]))
        cv2.rectangle(img, (int(pos_x[i]*4/224), int(pos_y[i]*4/224)), (int(((pos_x[i]/224)+1)*4-1), int(((pos_y[i]/224)+1)*4-1)), (cval[2] * 255, cval[1] * 255, cval[0] * 255), thickness=-1)
        #cv2.rectangle(img_att, (cb_w*4, cb_h*4), ((cb_w+1)*4, (cb_h+1)*4), (cval[2] * 255, cval[1] * 255, cval[0] * 255), thickness=-1)
    cv2.imwrite(f'{save_dir}/{slideID}_att.tif', img)

if __name__ == "__main__":
    #args = sys.argv
    att_dir = f'./test_att'
    save_dir = f'./vis_att'
    makedir(save_dir)

    list_subtype = ['DLBCL', 'FL', 'Reactive']

    max_sample = 300

    data_all = []

    for subtype in list_subtype:
        list_id = np.loadtxt(f'./slideID_JMR/{subtype}.txt', delimiter=',', dtype='str').tolist()
        random.seed(0)
        random.shuffle(list_id)
        list_id = list_id[0:max_sample]
        data_all += list_id

    for slideID in data_all:
        draw_heatmap(slideID, att_dir, save_dir)
