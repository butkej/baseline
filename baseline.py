import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision

import pytorch_lightning as pl

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from src import utils, custom_model, dataset


# Functions & Classes


def eval_patientwise(model, data, labels):
    """performs evalutation with a model and a given X,y dataset split (e.g. val or test)
    Computes patientwise (bag or collection of tiles)
    accuracy, AUROC, classification_report and confusion matrix
    """
    model.eval()
    acc = 0
    true_slide_labels = []
    y_probs_slide = []
    y_preds_slide = []
    softmax = torch.nn.Softmax(dim=1)

    for wsi, wsi_label in zip(data, labels):
        y_probs = []
        preds = []
        true_slide_labels.append(wsi_label)
        for img in wsi:
            with torch.no_grad():
                pred = model(img.unsqueeze(dim=0))
            preds.append(pred.argmax(dim=-1).detach().cpu().numpy())

            softmaxxed = softmax(pred)
            y_probs.append(softmaxxed.detach().cpu().numpy())

        y_probs_slide.append(np.mean(y_probs, axis=0)[0])

        if np.round(np.mean(preds)) == wsi_label:
            acc += 1
        y_preds_slide.append(np.round(np.mean(preds)))

    print("Eval Accuracy patient wise is:")
    acc_patientwise = acc / len(labels)
    print(str(acc_patientwise))

    print("Patientwise AUROC is:")
    auc = roc_auc_score(
        y_true=true_slide_labels,
        y_score=y_probs_slide,
        multi_class="ovr",
    )
    print(str(auc))

    print("Classification report:")
    cr = classification_report(
        y_true=true_slide_labels, y_pred=y_preds_slide, target_names=SUBTYPES
    )
    print(cr)

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true=true_slide_labels, y_pred=y_preds_slide)
    print(cm)
    return acc_patientwise, auc, cr, cm


def eval_bagwise():
    pass


def k_fold_cross_val(X, y, args, k: int = 5):
    """k-fold cross validation for any number of RUNS where each run
    splits the data into the same amount of SPLITS."""
    results = {}
    KF = StratifiedKFold(n_splits=k, shuffle=True)

    print(KF.get_n_splits())

    fold = 1
    for train_index, val_index in KF.split(np.zeros(len(X)), y):

        print("FOLD Nr. " + str(fold))

        print("Start of training for " + str(args.epochs) + " epochs.")
        print("TRAIN:", train_index, "VAL:", val_index)
        # X_train, X_val = np.array(X)[train_index], np.array(X)[val_index]
        # y_train, y_val = np.array(y)[train_index], np.array(y)[val_index]
        X_train = [X[index] for index in train_index]
        y_train = [y[index] for index in train_index]
        X_val = [X[index] for index in val_index]
        y_val = [y[index] for index in val_index]

        if args.baseline == "classic":
            # Model initilization or reinit if fold > 1
            model, input_size = utils.lightning_mode(args)
            if args.freeze:
                custom_model.freeze_model_layers(model, freeze_ratio=0.5)
            if fold == 1:
                print(model)

            train_dataset = dataset.convert_to_tile_dataset(X_train, y_train)
            del X_train, y_train
            val_dataset = dataset.convert_to_tile_dataset(X_val, y_val)

            model.train()

            classic(args, model, train_dataset, val_dataset)
            acc, auc, cr, cm = eval_patientwise(model, X_val, y_val)
            results["Accuracy for Fold {}".format(fold)] = acc
            results["ROC-AUC for Fold {}".format(fold)] = auc
            results["Classification Report for Fold {}".format(fold)] = cr
            results["Conf Matrix for Fold {}".format(fold)] = cm
            del train_dataset, val_dataset, X_val, y_val, model

        elif args.baseline == "clam":
            feature_extractor = custom_model.Resnet50_baseline(
                pretrained=args.pretrained
            )
            X_train_features = dataset.feature_extract_bag(feature_extractor, X_train)
            del X_train
            X_val_features = dataset.feature_extract_bag(feature_extractor, X_val)
            del X_val

            train_dataset = dataset.convert_to_bag_dataset(X_train_features, y_train)
            del X_train_features, y_train
            val_dataset = dataset.convert_to_bag_dataset(X_val_features, y_val)

            print(val_dataset[2][0].shape)
            print(val_dataset[2][1])

            model_kwargs = {
                "gate": True,
                "size_arg": "small",
                "dropout": False,
                "k_sample": 8,
                "n_classes": len(SUBTYPES),
                "instance_loss_fn": nn.CrossEntropyLoss(),
                "subtyping": True,
            }

            model = custom_model.CLAM_Lightning(model_kwargs)
            clam(args, model, train_dataset, val_dataset)

        elif args.baseline == "vit":
            # Model initilization or reinit if fold > 1
            model = utils.lightning_mode(args)
            if args.freeze:
                custom_model.freeze_model_layers(model, freeze_ratio=0.5)
            if fold == 1:
                print(model)

            train_dataset = dataset.convert_to_tile_dataset(X_train, y_train)
            # del X_train, y_train
            val_dataset = dataset.convert_to_tile_dataset(X_val, y_val)

            model.train()

            classic(args, model, train_dataset, val_dataset)
            acc, auc, cr, cm = eval_patientwise(model, X_val, y_val)
            results["Accuracy for Fold {}".format(fold)] = acc
            results["ROC-AUC for Fold {}".format(fold)] = auc
            results["Classification Report for Fold {}".format(fold)] = cr
            results["Conf Matrix for Fold {}".format(fold)] = cm
            # del train_dataset, val_dataset, X_val, y_val, model

        else:
            print("Error! Choosen baseline strategy is unclear")

        fold += 1
    return results


def classic(args, model, train_ds, val_ds):
    # pass model, dataloader/set and epochs, performs training/validation loops
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs
    )
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_acc", min_delta=0.001, patience=5, verbose=True, mode="max"
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


def clam(args, model, train_ds, val_ds):
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=1, shuffle=True, **loader_kwargs
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, **loader_kwargs
    )
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_acc", min_delta=0.001, patience=5, verbose=True, mode="max"
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
    VIT_PRETRAINED_PATH = "src/ViT_B_16_imagenet1k_pretrained.pth"
    # SUBTYPES = [['DLBCL', 'FL'],['AITL', 'ATLL'],['CHL'],['Reactive']]

    # Setup
    utils.seed_everything(3407)

    args = utils.parse_args()
    args_dict = vars(args)

    print("Called with args:")
    print(args)

    # Data loading
    loader_kwargs = {}
    if torch.cuda.is_available():
        loader_kwargs = {"num_workers": 4, "pin_memory": False}

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ]
    )

    paths, labels = dataset.load_data_paths(
        subtypes=SUBTYPES, path_to_slide_info=SLIDE_INFO_DIR
    )  # slides is list of all slides and their labels

    X, y = [], []
    for slide, target in zip(paths, labels):
        wsi_info = dataset.get_wsi_info(
            slide, target, number_of_patches=500, patch_info_path=PATCH_INFO_DIR
        )
        patches, label = dataset.patch_wsi(
            wsi_info,
            transform,
            path_to_data=DATA_DIR,
            magnification=MAGNIFICATION,
        )

        X.append(patches)
        y.append(label)
    utils.check_data(X, "X")
    utils.check_data(y, "y")
    del patches, label
    assert len(X) == len(y)

    # y = np.asarray(y)
    # if len(np.unique(y)) > 2:
    #    y = dataset.one_hot_encode_labels(y)

    # Run
    print("\nStart of K-FOLD CROSSVALIDATION with " + str(args.folds) + " folds.")
    results = k_fold_cross_val(X, y, args)

    print("{}-fold summarized results:".format(args.folds))
    overall_acc = 0
    overall_auroc = 0
    overall_cm = np.zeros((len(SUBTYPES), len(SUBTYPES)))
    for key in results.keys():
        if "Accuracy" in key:
            overall_acc += results[key]
        elif "AUC" in key:
            overall_auroc += results[key]
        elif "Conf" in key:
            overall_cm += results[key]

    print("Accuracy: {}".format(str(overall_acc / args.folds)))
    print("AUROC: {}".format(str(overall_auroc / args.folds)))
    print(overall_cm)
