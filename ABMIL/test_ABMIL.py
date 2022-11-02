# -*- coding: utf-8 -*-
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

num_gpu = 8

#正誤確認関数(正解:ans=1, 不正解:ans=0)
def eval_ans(y_hat, label):
    true_label = int(label)
    if(y_hat == true_label):
        ans = 1
    if(y_hat != true_label):
        ans = 0
    return ans

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

# 3エポック目以降で最も損失が小さいモデルを使用
def select_epoch(log_file):

    train_log = np.loadtxt(log_file, delimiter=',', dtype='str')
    valid_loss = train_log[1:,3].astype(np.float32)
    loss_list = []
    total_epoch = valid_loss.shape[0]/num_gpu
    for i in range(int(total_epoch)):
    #for i in range(10):
        tmp = valid_loss[i*num_gpu:(i+1)*num_gpu]
        if i < 2:
            loss_list.append(1000000)
        else:
            loss_list.append(np.sum(tmp))
    return loss_list.index(min(loss_list))

def test(model, device, test_loader, output_file):

    model.eval() #テストモードに変更

    for (input_tensor, slideID, class_label, pos_list) in test_loader:

        input_tensor = input_tensor.to(device) # 入力をGPUへ

        for bag_num in range(input_tensor.shape[0]):

            with torch.no_grad():
                class_prob, class_hat, A = model(input_tensor[bag_num]) # モデルからクラス確率、予測クラス、アテンションを取得

            # softmax
            class_softmax = F.softmax(class_prob, dim=1).squeeze(0)
            class_softmax = class_softmax.tolist() # listに変換

            # bagの分類結果と各パッチのattention_weightを出力
            f = open(output_file, 'a')
            f_writer = csv.writer(f, lineterminator='\n')
            slideid_tlabel_plabel = [slideID[bag_num], int(class_label[bag_num]), class_hat] + class_softmax # [slideID, 真のラベル, 予測ラベル] + [y_prob]
            f_writer.writerow(slideid_tlabel_plabel)
            pos_x = []
            pos_y = []
            for pos in pos_list:
                pos_x.append(int(pos[0]))
                pos_y.append(int(pos[1]))
            f_writer.writerow(pos_x) # 座標書き込み
            f_writer.writerow(pos_y) # 座標書き込み
            attention_weights = A.cpu().squeeze(0)
            attention_weights_list = attention_weights.tolist()
            att_list = []
            for att in attention_weights_list:
                att_list.append(float(att[0]))
            f_writer.writerow(att_list) # 各instanceのattention_weight書き込み
            f.close()

def test_model(cv_test):

    ##################実験設定#######################################
    EPOCHS = 10 #総エポック数
    mag = '40x' #使用倍率 '40x' or '20x' or '10x' or '5x'
    device = 'cuda:0' #conflictなど起きないようにsingle GPUで動作
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
    max_main = 300 # 各クラスの最大症例数
    max_R = 300 # Reactiveクラスの最大症例数

    label = 0
    for group_subtype in list_subtype:
        train_group = []
        for subtype in group_subtype:
            if subtype in list_use:
                if subtype == 'Reactive':
                    max_sample = max_R
                else:
                    max_sample = max_main
                list_id = np.loadtxt(f'./slideID_JMR/{subtype}.txt', delimiter=',', dtype='str').tolist()
                random.seed(0)
                random.shuffle(list_id)
                list_id = list_id[0:max_sample]
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

    n_class = len(num_data) # クラス数
    num_all = sum(num_data) # 症例数

    # データセット作成(症例IDとクラスラベルのペア)
    i = 0
    test_dataset = []
    for slideID in test_all:
        class_label = test_label[i]
        test_dataset.append([slideID, class_label])
        i += 1

    log = f'./train_log/MIL_log_cv-{cv_test}.csv'
    epoch_m = select_epoch(log) # 検証データの最良エポックを取得
    makedir('test_result')
    result = f'test_result/MIL_test_cv-{cv_test}.csv' # 出力ファイル
    makedir('model_params')
    model_params = f'./model_params/MIL_cv-{cv_test}_epoch-{epoch_m}.pth' # 最良モデルパラメータ

    f = open(result, 'w')
    f.close()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # model読み込み
    from model import feature_extractor, MLP_attention, fc_label, MIL
    # 各ブロック宣言
    feature_extractor = feature_extractor()
    MLP_attention = MLP_attention()
    fc_label = fc_label(n_class=n_class)
    # MIL構築
    model = MIL(feature_extractor, MLP_attention, fc_label, n_class)
    model.load_state_dict(torch.load(model_params,map_location='cpu'))
    model = model.to(device)

    # 前処理
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])

    JMR_test = Dataset.SingleDataset(
        train=False,
        transform=transform,
        dataset=test_dataset,
        mag=mag,
        bag_num=50,
        bag_size=100
    )

    test_loader = torch.utils.data.DataLoader(
        JMR_test,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        num_workers=1,
    )

    test(model, device, test_loader, result)

if __name__ == '__main__':

    args = sys.argv
    cv_test = int(args[1]) # testに使用するfold (5fold CVのうち1~5)

    test_model(cv_test)
