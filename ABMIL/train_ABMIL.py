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

file_PATH = './svs_JMR' #### WSIが保存されているディレクトリを指定

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' #適当な数字で設定すればいいらしいがよくわかっていない

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

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

def train(model, rank, loss_fn, optimizer, train_loader):

    model.train() #訓練モードに変更
    train_class_loss = 0.0
    correct_num = 0

    for (input_tensor, slideID, class_label) in train_loader:

        # MILとバッチ学習のギャップを吸収
        input_tensor = input_tensor.to(rank, non_blocking=True) # 入力をGPUへ
        class_label = class_label.to(rank, non_blocking=True) #ラベルをGPUへ

        for bag_num in range(input_tensor.shape[0]):

            optimizer.zero_grad() #勾配初期化
            class_prob, class_hat, A = model(input_tensor[bag_num]) #モデルからクラス確率、予測クラス、アテンションを取得

            # 各loss計算
            class_loss = loss_fn(class_prob, class_label[bag_num]) # クロスエントロピーを計算
            train_class_loss += class_loss.item()

            class_loss.backward() #逆伝播
            optimizer.step() #パラメータ更新
            correct_num += eval_ans(class_hat, class_label[bag_num])

    return train_class_loss, correct_num # 学習ロスと正答数を返す

def valid(model, rank, loss_fn, test_loader):

    model.eval() # 評価モードに変更
    test_class_loss = 0.0
    correct_num = 0

    for (input_tensor, slideID, class_label) in test_loader:

        input_tensor = input_tensor.to(rank, non_blocking=True) # 入力をGPUへ
        class_label = class_label.to(rank, non_blocking=True) # ラベルをGPUへ

        for bag_num in range(input_tensor.shape[0]):

            with torch.no_grad():
                class_prob, class_hat, A = model(input_tensor[bag_num]) # クラス確率、予測クラス、アテンションを取得

            # 各loss計算
            class_loss = loss_fn(class_prob, class_label[bag_num]) # クロスエントロピーを計算
            test_class_loss += class_loss.item()
            correct_num += eval_ans(class_hat, class_label[bag_num])

    return test_class_loss, correct_num #検証ロスと正答数を返す

#if __name__ == "__main__":
#マルチプロセス (GPU) で実行される関数
#rank : mp.spawnで呼び出すと勝手に追加される引数で, GPUが割り当てられている
#world_size : mp.spawnの引数num_gpuに相当
def train_model(rank, world_size, cv_test):
    setup(rank, world_size)

    ##################実験設定#######################################
    EPOCHS = 10 #最大学習エポック数
    mag = '40x' #使用倍率 '40x' or '20x' or '10x' or '5x'
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

    # 症例数に応じて損失の重みを計算
    n_class = len(num_data) # クラス数
    num_all = sum(num_data) # 症例数
    for i in range(len(num_data)):
        num_data[i] = num_all/(num_data[i]*n_class)
    weights = torch.tensor(num_data).to(rank, non_blocking=True)

    # データセット作成(症例IDとクラスラベルのペア)
    i = 0
    train_dataset = []
    for slideID in train_all:
        class_label = train_label[i]
        train_dataset.append([slideID, class_label])
        i += 1
    i = 0
    valid_dataset = []
    for slideID in valid_all:
        class_label = valid_label[i]
        valid_dataset.append([slideID, class_label])
        i += 1

    # 学習のログ出力ファイル
    log = f'./train_log/MIL_log_cv-{cv_test}.csv'

    if rank == 0:
        #ログヘッダー書き込み
        f = open(log, 'w')
        f_writer = csv.writer(f, lineterminator='\n')
        csv_header = ["epoch", "train_loss", "train_acc", "valid_loss", "valid_acc", "time"]
        f_writer.writerow(csv_header)
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
    model = model.to(rank)
    process_group = torch.distributed.new_group([i for i in range(world_size)])
    #modelのBatchNormをSyncBatchNormに変更してくれる
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
    #modelをmulti GPU対応させる
    ddp_model = DDP(model, device_ids=[rank])

    # クロスエントロピー損失関数使用
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    # SGDmomentum法使用
    lr = 0.0001
    optimizer = optim.SGD(ddp_model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # 前処理
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])

    # 訓練開始
    for epoch in range(EPOCHS):

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        start_t = time.time()

        JMR_train = Dataset.SingleDataset(
            train=True,
            transform=transform,
            dataset=train_dataset,
            mag=mag,
            bag_num=50,
            bag_size=100,
            epoch=epoch
        )
        #Datasetをmulti GPU対応させる
        #下のDataLoaderでbatch_sizeで設定したbatch_sizeで各GPUに分配
        train_sampler = torch.utils.data.distributed.DistributedSampler(JMR_train, rank=rank)

        #pin_memory=Trueの方が早くなるらしいが, pin_memory=Trueにすると劇遅になるケースがあり原因不明
        train_loader = torch.utils.data.DataLoader(
            JMR_train,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=4,
            sampler=train_sampler
        )

        class_loss, acc = train(ddp_model, rank, loss_fn, optimizer, train_loader)　#学習

        scheduler.step()

        train_loss += class_loss
        train_acc += acc

        JMR_valid = Dataset.SingleDataset(
            train=True,
            transform=transform,
            dataset=valid_dataset,
            mag=mag,
            bag_num=50,
            bag_size=100,
            epoch=epoch
        )

        valid_sampler = torch.utils.data.distributed.DistributedSampler(JMR_valid, rank=rank)

        valid_loader = torch.utils.data.DataLoader(
            JMR_valid,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=4,
            sampler=valid_sampler
        )

        class_loss, acc = valid(ddp_model, rank, loss_fn, valid_loader)　#検証データの予測

        valid_loss += class_loss
        valid_acc += acc

        train_loss /= float(len(train_loader.dataset))
        train_acc /= float(len(train_loader.dataset))
        valid_loss /= float(len(valid_loader.dataset))
        valid_acc /= float(len(valid_loader.dataset))
        elapsed_t = time.time() - start_t

        # ログに書き込み
        f = open(log, 'a')
        f_writer = csv.writer(f, lineterminator='\n')
        f_writer.writerow([epoch, train_loss, train_acc, valid_loss, valid_acc, elapsed_t])
        f.close()

        # epochごとにmodelのパラメータ保存
        if rank == 0 and epoch > 1:
            model_params = f'./model_params/MIL_cv-{cv_test}_epoch-{epoch}.pth'
            torch.save(ddp_model.module.state_dict(), model_params)

if __name__ == '__main__':

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    num_gpu = 8 # GPU数

    args = sys.argv
    cv_test = int(args[1]) # testに使用するfold (5fold CVのうち1~5)

    #マルチプロセスで実行するために呼び出す
    #train_model : マルチプロセスで実行する関数
    #args : train_modelの引数
    #nprocs : プロセス (GPU) の数
    mp.spawn(train_model, args=(num_gpu, cv_test), nprocs=num_gpu, join=True)
