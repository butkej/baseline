import os
import glob
import torch
from torchvision import transforms
from torchvision.transforms import functional as tvf
import random
from PIL import Image
import numpy as np

import openslide

DATA_PATH = './DataJMR'
csv_PATH = './csv_JMR'

class SingleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, mag='40x', train = True, transform = None, bag_num=50, bag_size=100, epoch=0):

        '''
        初期値はバッグあたりのパッチ数 = 100, 1症例の最大バッグ数 = 50
        '''

        self.transform = transform
        self.train = train
        self.mag = mag

        self.bag_list = []
        for slide_data in dataset:
            slideID = slide_data[0]
            label = slide_data[1]

            pos = np.loadtxt(f'{csv_PATH}/{slideID}.csv', delimiter=',', dtype='int')
            if not self.train:# == 'False':
                np.random.seed(seed=int(slideID.replace('JMR',''))) # テスト時は症例番号をseedにする
            else:
                np.random.seed(seed=epoch) # 学習時はエポック数をseedにする
            np.random.shuffle(pos)
            if pos.shape[0] > bag_num*bag_size: # 最大バッグ数を作成可能なとき
                pos = pos[0:(bag_num*bag_size),:]
                for i in range(bag_num):
                    patches = pos[i*bag_size:(i+1)*bag_size,:].tolist()
                    self.bag_list.append([patches, slideID, label])
            else: # 最大バッグ数を作成できない
                for i in range(pos.shape[0]//bag_size):
                    patches = pos[i*bag_size:(i+1)*bag_size,:].tolist()
                    self.bag_list.append([patches, slideID, label])

        if self.train:
            random.seed(epoch)
            random.shuffle(self.bag_list)
        self.data_num = len(self.bag_list)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        pos_list = self.bag_list[idx][0]
        patch_len = len(pos_list)

        b_size = 224

        svs_list = os.listdir(f'{DATA_PATH}')
        svs_fn = [s for s in svs_list if self.bag_list[idx][1] in s]
        svs = openslide.OpenSlide(f'{DATA_PATH}/{svs_fn[0]}')

        bag = torch.empty(patch_len, 3, 224, 224, dtype=torch.float)
        i = 0
        # 画像読み込み

        for pos in pos_list:
            if self.transform:
                if self.mag == '40x': # 指定した領域(224 x 224)の画素を取得
                    img = svs.read_region((pos[0],pos[1]),0,(b_size,b_size)).convert('RGB')
                elif self.mag == '20x': # 指定した領域(448 x 448)
                    img = svs.read_region((pos[0]-(int(b_size/2)),pos[1]-(int(b_size/2))),0,(b_size*2,b_size*2)).convert('RGB')
                elif self.mag == '10x': # 指定した領域(224 x 224)
                    img = svs.read_region((pos[0]-(int(b_size*3/2)),pos[1]-(int(b_size*3/2))),1,(b_size,b_size)).convert('RGB')
                elif self.mag == '5x': # 指定した領域(448 x 448)
                    img = svs.read_region((pos[0]-(int(b_size*7/2)),pos[1]-(int(b_size*7/2))),1,(b_size*2,b_size*2)).convert('RGB')
                img = self.transform(img) # 224 x 224 pixelにリサイズ
                bag[i] = img
            i += 1

        slideID = self.bag_list[idx][1]
        label = self.bag_list[idx][2]
        label = torch.LongTensor([label])
        # バッグとラベルを返す
        if self.train:
            return bag, slideID, label
        else:
            return bag, slideID, label, pos_list
