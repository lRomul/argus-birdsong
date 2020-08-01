import torch

from timm import create_model
from argus import Model

from src.metrics import F1score
from src.losses import SoftBCEWithLogitsLoss


class BirdsongModel(Model):
    nn_module = {
        'timm': create_model,
    }
    loss = {
        'SoftBCEWithLogitsLoss': SoftBCEWithLogitsLoss
    }
    prediction_transform = torch.nn.Sigmoid
