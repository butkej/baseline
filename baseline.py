import os
import argparse
import random
import numpy as np
import torch
import torchvision

import pytorch_lightning as pl

from sklearn.model_selection import StratifiedKFold
from src import utils, custom_model, dataset


# Functions & Classes


def k_fold_cross_val(X, y, args, k: int = 5):
    """k-fold cross validation for any number of RUNS where each run
    splits the data into the same amount of SPLITS."""
    KF = StratifiedKFold(n_splits=k, shuffle=False)

    KF.get_n_splits()

    fold = 1
    for train_index, val_index in KF.split(np.zeros(len(X)), y):

        print("FOLD Nr. " + str(fold))

        # Model initiliazation or reinit if fold > 1
        model, input_size = utils.lightning_mode(args)

        print("Start of training for " + str(args.epochs) + " epochs.")
        print("TRAIN:", train_index, "VAL:", val_index)
        X_train, X_val = np.array(X)[train_index], np.array(X)[val_index]
        y_train, y_val = np.array(y)[train_index], np.array(y)[val_index]

        if args.baseline == "classic":
            train_dataset = dataset.convert_to_tile_dataset(X_train, y_train)
            val_dataset = dataset.convert_to_tile_dataset(X_val, y_val)

            classic(args, model, train_dataset, val_dataset)
        elif args.baseline == "clam":
            pass
        elif args.baseline == "vit":
            pass
        else:
            print("Error! Choosen baseline strategy is unclear")

        fold += 1


def classic(args, model, train_ds, val_ds):
    # pass model, dataloader/set and epochs, performs training/validation loops
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs
    )
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_acc", min_delta=0.00, patience=3, verbose=True, mode="max"
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.num_gpus,
        default_root_dir=EXPERIMENT_DIR,
        enable_progress_bar=True,
        callbacks=[early_stop_callback],
    )
    trainer.fit(model, train_dl, val_dl)
    # eval after full training fold
    # model.eval()
    # predict with the model
    # y_hat = model(x)


def clam(args):
    pass


def vit(args):
    pass


######

if __name__ == "__main__":

    # Config
    EXPERIMENT_DIR = "/home/butkej/work/experiments/baseline-prototype"
    # DATA_DIR = "/ml/wsi/"
    DATA_DIR = "/mnt/crest/wsi/"
    SLIDE_INFO_DIR = "slide_ID/"
    PATCH_INFO_DIR = "csv_JMR/"
    SUBTYPES = ["DLBCL", "FL", "Reactive"]
    MAGNIFICATION = "40x"  # or "20x" or "10x"
    # SUBTYPES = [['DLBCL', 'FL'],['AITL', 'ATLL'],['CHL'],['Reactive']]

    # Setup
    utils.seed_everything(3407)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = utils.parse_args()
    args_dict = vars(args)

    print("Called with args:")
    print(args)

    # Data loading
    loader_kwargs = {}
    if torch.cuda.is_available():
        loader_kwargs = {"num_workers": 0, "pin_memory": True}

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor()]
    )

    paths, labels = dataset.load_data_paths(
        subtypes=SUBTYPES, path_to_slide_info=SLIDE_INFO_DIR
    )  # slides is list of all slides and their labels

    X, y = [], []
    for slide, target in zip(paths, labels):
        wsi_info = dataset.get_wsi_info(
            slide, target, number_of_patches=1000, patch_info_path=PATCH_INFO_DIR
        )
        patches, label = dataset.patch_wsi(
            args,
            wsi_info,
            transform,
            path_to_data=DATA_DIR,
            magnification=MAGNIFICATION,
        )
        X.append(patches)
        y.append(label)
    assert len(X) == len(y)

    # y = np.asarray(y)
    # if len(np.unique(y)) > 2:
    #    y = dataset.one_hot_encode_labels(y)

    # Run
    print("\nStart of K-FOLD CROSSVALIDATION with " + str(args.folds) + " folds.")
    k_fold_cross_val(X, y, args)
