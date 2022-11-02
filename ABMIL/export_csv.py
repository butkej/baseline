import os
import glob
#import torch
#from torchvision import transforms
#from torchvision.transforms import functional as tvf
import random
from PIL import Image, ImageStat
import numpy as np
import cv2
import csv
import sys
import math
import openslide

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

args = sys.argv
part = int(args[1])

#DATA_PATH = '../../../../nvme/Data/DataJMR'
DATA_PATH = './DataJMR'

list_subtype = ['all']
#list_subtype = ['DLBCL','FL','Reactive']
train_all = []
for subtype in list_subtype:
    tmp_id = np.loadtxt(f'./slideID_JMR/{subtype}.txt', delimiter=',', dtype='str').tolist()
    train_all = train_all + tmp_id

print(len(train_all))

makedir('csv_JMR')

svs_list = os.listdir(f'{DATA_PATH}/svs')

t_size = 4 #サムネイル中の1パッチのサイズ
b_size = 224

for slideID in train_all[40*part:40*(part+1)]:

    svs_fn = [s for s in svs_list if slideID in s]
    svs = openslide.OpenSlide(f'{DATA_PATH}/svs/{svs_fn[0]}')
    width,height = svs.dimensions
    b_w = width // b_size # x方向のパッチ枚数
    b_h = height // b_size # y方向のパッチ枚数

    thumb = Image.new('RGB',(b_w * t_size, b_h * t_size))   #標本サムネイル
    thumb_s = Image.new('L',(b_w, b_h)) #彩度分布画像

    for h in range(b_h):
        for w in range(b_w):
            #サムネイル作成
            b_img = svs.read_region((w*b_size,h*b_size),0,(b_size,b_size)).convert('RGB')
            r_img = b_img.resize((t_size, t_size), Image.BILINEAR)  #サムネイル用に縮小
            thumb.paste(r_img, (w * t_size, h * t_size))

            b_array = np.array(b_img)

            #彩度画像作成
            R_b, G_b, B_b = cv2.split(b_array)
            Max_b = np.maximum(np.maximum(R_b, G_b), B_b)
            Min_b = np.minimum(np.minimum(R_b, G_b), B_b)
            Sat_b = Max_b - Min_b
            img_g = b_img.convert('L')
            s_img = Image.fromarray(Sat_b)
            statS = ImageStat.Stat(s_img)    #彩度画像の統計量取得
            statV = ImageStat.Stat(img_g)
            hsv = cv2.cvtColor(b_array, cv2.COLOR_BGR2HSV)
            hue, s, v = cv2.split(hsv)
            b_ratio = B_b / R_b
            if statV.mean[0] < 230 and np.count_nonzero(G_b > 230) < 224*224 / 2 and statS.mean[0] > 0 and np.mean(b_ratio) > 0.9 and np.count_nonzero(b_ratio > 1) > 224*224 / 16:
            #if statV.mean[0] < 230 and statV.mean[0] > 100 and np.count_nonzero(G_b > 230) < 224*224 / 2 and np.count_nonzero(G_b < 50) < 100 and statS.mean[0] > 0 and np.mean(b_ratio) > 0.9 and np.var(hue) > 25 and np.count_nonzero(b_ratio > 1) > 224*224 / 16:
                #s_pix[w,h] = round(statS.mean[0])
                thumb_s.putpixel((w,h), round(statS.mean[0]))
                #thumb_s.putpixel((w,h), 255)
            else:
                #s_pix[w,h] = 0
                thumb_s.putpixel((w,h), 0)

    makedir(f'./thumb_JMR')
    makedir(f'./thumb_JMRs')
    thumb.save(f'./thumb_JMR/{slideID}_thumb.tif')    #標本サムネイル保存
    thumb_s.save(f'./thumb_JMRs/{slideID}_sat.tif')    #彩度分布画像保存

    s_array = np.asarray(thumb_s)   #cv形式に変換
    ret, s_mask = cv2.threshold(s_array, 0, 255, cv2.THRESH_OTSU) #判別分析法で二値化
    #s_mask = Image.fromarray(s_mask)    #PIL形式に変換

    num_i = np.count_nonzero(s_mask)
    pos = np.zeros((num_i,2))
    i = 0
    for h in range(b_h):
        for w in range(b_w):
            if not s_mask[h,w] == 0:
                pos[i][0] = w * b_size
                pos[i][1] = h * b_size
                i = i + 1

    np.savetxt(f'./csv_JMR/{slideID}.csv', pos, delimiter=',', fmt='%d')
