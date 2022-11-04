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


def k_fold_cross_val(X, y, args, model, optim, k: int = 5):
    """k-fold cross validation for any number of RUNS where each run
    splits the data into the same amount of SPLITS."""
    KF = StratifiedKFold(n_splits=k, shuffle=True)

    KF.get_n_splits()

    fold = 1
    for train_index, val_index in KF.split(slideIDs, labels):

        print("FOLD Nr. " + str(fold))
        print("Start of training for " + str(args.epochs) + " epochs.")
        if args.baseline == "classic":
            pass
        elif args.baseline == "clam":
            pass
        elif args.baseline == "vit":
            pass
        else:
            print("Error! Choosen baseline strategy is unclear")

        fold += 1


def classic(args):
    # pass model, dataloader/set and epochs, performs training/validation loops
    train_dl = torch.utils.DataLoader()
    val_dl = torch.utils.DataLoader()
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.num_gpus,
        default_root_dir=EXPERIMENT_DIR,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_dl, val_dl)


def clam(args):
    pass


def vit(args):
    pass


######

if __name__ == "__main__":
    # Config
    EXPERIMENT_DIR = "/home/butkej/experiments/"
    # Setup
    utils.seed_everything(3407)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = utils.parse_args()
    args_dict = vars(args)

    print("Called with args:")
    print(args)

    # Model initiliazation
    model, input_size = utils.lightning_mode(args)

    loader_kwargs = {}
    if torch.cuda.is_available():
        loader_kwargs = {"num_workers": 4, "pin_memory": True}

    # optim = utils.choose_optimizer(args.optimizer, model)
    print("\nSuccessfully build and compiled the chosen model!")

    # Run
    print("\nStart of K-FOLD CROSSVALIDATION with " + args.folds + " folds.")
    k_fold_cross_val(X, y, args, model, optim)
