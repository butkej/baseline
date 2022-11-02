import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# 特徴量抽出器
class feature_extractor(nn.Module):  # 特徴抽出器（ResNet50）
    def __init__(self, pre=True):
        super(feature_extractor, self).__init__()
        # 学習済みResNet50
        res50 = models.resnet50(pretrained=pre)
        self.feature_ex = nn.Sequential(*list(res50.children())[:-1])

    def forward(self, input):
        x = input.squeeze(0)
        feature = self.feature_ex(x)
        feature = feature.squeeze(2).squeeze(2)
        return feature


class fc_label(nn.Module):  # 最終の分類層
    def __init__(self, n_class):
        super(fc_label, self).__init__()
        # 次元圧縮
        self.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_class),
        )

    def forward(self, input):
        x = input.squeeze(0)
        z = self.fc(x)
        return z


class MLP_attention(nn.Module):  # アテンションネットワーク
    def __init__(self):
        super(MLP_attention, self).__init__()
        # attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(2048, 512), nn.Tanh(), nn.Linear(512, 1)
        )

    def forward(self, input):
        A = self.attention(input)
        return A


class MIL(nn.Module):
    def __init__(self, feature_ex, attention, fc_label, n_class):
        super(MIL, self).__init__()
        self.feature_ex = feature_ex
        self.attention = attention
        self.fc_label = fc_label
        self.n_class = n_class

    def forward(self, input):
        x = input.squeeze(0)
        feature = self.feature_ex(x)  # 各パッチの特徴ベクトルを計算
        A = self.attention(feature)  # 特徴ベクトルからアテンション重みを計算
        A_t = torch.transpose(A, 1, 0)
        A_n = F.softmax(A_t, dim=1)  # バッグ内でアテンションのsoftmaxをとる
        M = torch.mm(A_n, feature)  # 特徴ベクトルをアテンションの重み付き和を計算
        class_prob = self.fc_label(M).reshape(1, self.n_class)  # 全結合層により各クラスの出力を取得
        class_softmax = F.softmax(class_prob, dim=1)  # softmax
        class_hat = int(torch.argmax(class_softmax, 1))  # 予測クラス
        return class_prob, class_hat, A
