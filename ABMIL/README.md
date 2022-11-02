# Attention-based Multiple Instance Learning CNN

This document explains how to use the source code for attention-based multiple-instance leaning (ABMIL) CNN in [1]. This program can directly use svs file by using a WSI library OpenSlide.

## Environmental Requirement
We confirmed that the source code was running with the following environment.

- python 3.6
- numpy 1.18.1
- opencv-python 4.2.0.32
- pillow 6.1.0
- pytorch 1.4.0
- CUDA 10
- NVIDIA Quadro RTX 5000

Please install a library OpenSlide into your anaconda environment.

    $ conda install -c bioconda openslide-python

## Structure of directories and experimental dataset

We assume the following structure of directories.

```
ABMIL/
　　├ train_ABMIL.py
　　├ test_ABMIL.py
　　├ eval_ABMIL.py
　　├ att_ABMIL.py
　　├ vis_ABMIL.py
　　├ model.py
　　├ Dataset.py
　　├ slideID_JMR/
　　├ DataJMR/
　　├ thumb_JMR/
　　└ csv_JMR/
```

- train_ABMIL.py: Attention-based MILの学習プログラム
- test_ABMIL.py: Attention-based MILのテストプログラム
- eval_ABMIL.py: テストプログラムの出力を評価するプログラム
- att_ABMIL.py: 可視化用のアテンション出力プログラム
- vis_ABMIL.py: アテンション可視化プログラム
- model.py: モデル
- Dataset.py: データローダー
- slideID_JMR: 各病型の症例ID
- DataJMR: 全症例のsvsファイル
- thumb_JMR: 全症例のサムネイル（export_csv.pyにより生成）
- csv_JMR: 各症例の組織領域の座標ファイル（export_csv.pyにより生成）

Other directories in the code are automatically generated.

## 使用方法
### 学習と分類
モデルの学習は以下のコードで実行可能です．

    $ python train_ABMIL.py 1

各引数は以下の通り．
- 第1引数: 5-foldのうちテストに使用するfoldの指定

その他のパラメータはコード内で指定しています．
また複数GPUによる並列計算を想定しているため，シングルGPUで実行する場合にはコードの書き換えが必要になります．


テストも同様のコードで実行可能です．

    $ python test_ABMIL.py 1

引数は学習時と同様です．テストでは各バッグに対して
- 1行目: 症例ID，正解ラベル，予測ラベル，各クラスの予測確率
- 2, 3行目: 各パッチのx座標とy座標
- 4行目: 2, 3行目の座標のパッチに対応したアテンション重みの値

が順に書き込まれたファイルが出力されます．
test_resultに保存された出力のファイルを用いて，以下のコードで5-fold CVの結果をまとめます．

    $ python eval_ABMIL.py MIL 3

test_ABMILの出力例だとMIL_test_cv-[1,2,3,4,5].csvと保存されるので，'MIL'の箇所を引数として指定します．また分類するクラス数を第2引数として指定します．ディレクトリtest_predictに分類結果のまとめと混同行列が保存されます．


### アテンションの可視化
可視化のためのアテンションを以下のコードで出力します．

    $ python att_ABMIL.py 1

ディレクトリtest_attが生成され，各パッチのアテンションが症例ごとに別のcsvファイルで保存されます．これらを用いて以下のコードで可視化結果を出力します．

    $ python vis_ABMIL.py

可視化結果はディレクトリvis_attに保存されます．


---
## Reference

[1] N. Hashimoto, D. Fukushima, R. Koga, Y. Takagi, K. Ko, K. Kohno, M. Nakaguro, S. Nakamura, H. Hontani and I. Takeuchi, "Multi-scale domain-adversarial multiple-instance CNN for cancer subtype classification with unannotated histopathological images," Proc. IEEE Conference on Computer Vision and Pattern Recognition, pp. 3852-3861, June 2020.

Last update: July 14, 2022
