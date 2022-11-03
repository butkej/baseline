import os
import argparse
import random
import numpy as np
import torch
import torchvision

from sklearn.model_selection import StratifiedKFold
from src import utils, custom_model


# Functions & Classes


def k_fold_cross_val(X, y, model, optim, k: int = 5):
    """k-fold cross validation for any number of RUNS where each run splits the data into the same amount of SPLITS."""
    KF = StratifiedKFold(n_splits=k, shuffle=True)

    KF.get_n_splits()

    fold = 1
    for train_index, val_index in KF.split(slideIDs, labels):
        pass


def classic(args):
    # pass model, dataloader/set and epochs, performs training/validation loops
    pass


def clam(args):
    pass


def vit(args):
    pass


######

if __name__ == "__main__":
    # Setup
    utils.seed_everything(3407)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = utils.parse_args()
    args_dict = vars(args)

    print("Called with args:")
    print(args)

    # Model initiliazation
    model, input_size = utils.choose_model(args.model)

    loader_kwargs = {}
    if torch.cuda.is_available():
        #    #model.cuda()
        loader_kwargs = {"num_workers": 4, "pin_memory": True}

    optim = utils.choose_optimizer(args.optimizer, model)
    print("\nSuccessfully build and compiled the chosen model!")
