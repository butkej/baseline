import os
import random
import numpy as np
import torch
from torchvision import models

# Functions & Classes
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
        "-m",
        "--model",
        dest="model",
        help="choice of model [string]",
        default="resnet50",
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
        "-o",
        "--optimizer",
        dest="optimizer",
        help="choice of optimizer [string]",
        default="adam",
        type=str,
    )

    args = parser.parse_args()
    return args


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
        model = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        model = models.resnet50(pretrained=use_pretrained)
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

    # elif model_name == "vit":
    #    model = ViT("B_32_imagenet1k", pretrained=True)
    #    set_parameter_requires_grad(model, feature_extract)
    #    num_ftrs = model.fc.in_features
    #    model.fc = nn.Linear(num_ftrs, num_classes)
    #    input_size = 384  # TODO ?
    # TODO VIT and CLAM selection
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
