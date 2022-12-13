import os
import argparse
import random
import numpy as np
import torch
from torchvision import models

from src import custom_model

VIT_PRETRAINED_PATH = "./src/ViT_B_16_imagenet1k_pretrained.pth"

# Functions & Classes


def parse_args():
    """Parse input arguments.
    Returns:
        args: argparser.Namespace class object
        An argparse.Namespace class object contains experimental hyper-parameters.
    """
    parser = argparse.ArgumentParser(
        prog="Baseline model training for Tsukuba CREST Lymphoma Project",
        description="Trains a deep learning with k-fold cross validation",
    )

    # General
    parser.add_argument(
        "-b",
        "--baseline",
        dest="baseline",
        help="choice of baseline strategy",
        default="classic",
        type=str,
    )

    parser.add_argument(
        "-l",
        "--lightning",
        dest="lightning",
        help="use pytorch-lightning [bool]",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        help="choice of model [string]",
        default="resnet18",
        type=str,
    )

    parser.add_argument(
        "-e",
        "--epochs",
        dest="epochs",
        help="amount of epochs to train for [integer]",
        default=100,
        type=int,
    )

    parser.add_argument(
        "-k",
        "--folds",
        dest="folds",
        help="amount of folds in k-fold cross val",
        default=5,
        type=int,
    )

    # Model-related settings
    parser.add_argument(
        "-g",
        "--gpu",
        dest="num_gpus",
        help="choice of gpu amount [integer]",
        default=1,
        type=int,
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        dest="batch_size",
        help="choice of batch size [integer]",
        default=32,
        type=int,
    )

    parser.add_argument(
        "-o",
        "--optimizer",
        dest="optimizer",
        help="choice of optimizer [string]",
        default="adam",
        type=str,
    )

    parser.add_argument(
        "-c",
        "--num_classes",
        dest="num_classes",
        help="set the number of target output classes",
        default=3,
        type=int,
    )

    parser.add_argument(
        "-fe",
        "--feature_extract",
        dest="feature_extract",
        help="",  # TODO
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "-p",
        "--pretrained",
        dest="pretrained",
        help="choice of pretrained model or not [bool]",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "-f",
        "--freeze",
        dest="freeze",
        help="choice of model layer freezin or not [bool]",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    return args


def seed_everything(seed: int = 3407) -> None:
    """Function to pass a seed argument to all common packages for ML/DL
    lol @ https://arxiv.org/pdf/2109.08203.pdf"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def lightning_mode(args):
    """If lightning flag is set, load ResNet baseline from pytorch lightning"""
    if args.lightning and not args.model == "vit":
        model = custom_model.ClassicBaseline(
            model_name=args.model,
            optimizer_name=args.optimizer,
            num_classes=args.num_classes,
            input_size=224,
            feature_extract=args.feature_extract,
            use_pretrained=args.pretrained,
        )
        return model, model.input_size

    elif args.lightning and args.model == "vit":
        model_kwargs = {
            "embed_dim": 768,
            "hidden_dim": 512,
            "num_heads": 8,
            "num_layers": 6,
            "patch_size": 16,
            "num_channels": 3,
            "num_patches": 768,
            "num_classes": args.num_classes,
            "dropout": 0.2,
        }
        model = custom_model.ViT(model_kwargs, lr=1e-4)
        if args.pretrained:
            model.load_model(VIT_PRETRAINED_PATH)
        return model, None

    else:
        model, input_size = choose_model(
            args.model, args.num_classes, args.feature_extract, args.pretrained
        )
        return model, input_size


def choose_model(
    model_name: str,
    num_classes: int,
    feature_extract: bool,
    use_pretrained: bool = False,
):
    """Chooses a model from custom_model.py according to the string specifed in
    the model CLI argument and build with specified args"""
    model = None
    input_size = 0

    if model_name == "resnet18":
        if use_pretrained:
            model = models.resnet18(weights="IMAGENET1K_V2")
        else: model = models.resnet18()
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        if use_pretrained:
            model = models.resnet50(weights="IMAGENET1K_V2")
        else: model = models.resnet50()
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50-bottlenecked":
        model = custom_model.Resnet50_baseline(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Error! Choosen model is unclear.")

    return model, input_size


def choose_optimizer(selection, model):
    """Chooses an optimizer according to the string specifed in the model CLI argument and build with specified args"""
    if str(selection) == "adam":
        optim = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
        )
    elif str(selection) == "adadelta":
        optim = torch.optim.Adadelta(
            model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0
        )
    elif str(selection) == "momentum":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            dampening=0,
            weight_decay=0,
            nesterov=False,
        )
    else:
        print("Error! Choosen optimizer or its parameters are unclear")

    return optim


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def check_data(data, varname: str):
    """Prints characteristic image/array values that are often
    useful to checkout.
    """
    print("\n")
    print("Data Check for variable: {}".format(varname))
    print("-----------")
    try:
        print("dtype: {}".format(data.dtype))
    except:
        print("type: {}".format(type(data)))
    try:
        print("max: {}".format(data.max()))
        print("min: {}".format(data.min()))
        print("mean: {}".format(np.round(data.mean(), decimals=5)))
        print("std: {}".format(np.round(data.std(), decimals=5)))
    except:
        pass
    print("-----------")
    print("\n")

    if type(data) is list:
        print("Length of list: {}".format(len(data)))
        try:
            print("Shape of first element: {}".format(data[0].shape))
        except:
            print("First element: {}".format(data[0]))


def round_sum_to_one(x):
    """round each value in a list/array to a nearest acceptable value,
    while still ensuring that the sum of values sums to 1"""
    track_pos_low = 0
    track_pos_high = 0
    for pos, number in enumerate(x):
        rounded_num = np.round(number, decimals=4)
        # TODO
