import torch
import numpy as np
from sklearn.metrics import f1_score

from argus.metrics import Metric


class F1score(Metric):
    name = 'f1_score'
    better = 'max'

    def __init__(self):
        self.predictions = []
        self.targets = []

    def reset(self):
        self.predictions = []
        self.targets = []

    @torch.no_grad()
    def update(self, step_output: dict):
        pred = step_output['prediction']
        target = step_output['target']

        pred = torch.argmax(pred, dim=1)
        target = torch.argmax(target, dim=1)

        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        self.predictions.append(pred)
        self.targets.append(target)

    def compute(self):
        y_true = np.concatenate(self.targets, axis=0)
        y_pred = np.concatenate(self.predictions, axis=0)
        score = f1_score(y_true, y_pred, average='micro')
        return score
