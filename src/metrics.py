import torch
import numpy as np
from sklearn.metrics import f1_score

from argus.metrics import Metric

from src.utils import nocall_prediction


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

        target = torch.argmax(target, dim=1)

        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        self.predictions.append(pred)
        self.targets.append(target)

    def compute(self):
        y_true = np.concatenate(self.targets, axis=0)
        y_pred = np.concatenate(self.predictions, axis=0)

        max_score = 0
        max_threshold = 0
        for threshold in np.arange(0, 1, 0.01):
            pred = nocall_prediction(y_pred, threshold=threshold)
            score = f1_score(y_true, pred, average='micro')

            if score > max_score:
                max_score = score
                max_threshold = threshold

        return max_score, max_threshold

    def epoch_complete(self, state, name_prefix=''):
        with torch.no_grad():
            max_score, max_threshold = self.compute()
        state.metrics[name_prefix + self.name] = max_score
        state.metrics[name_prefix + 'max_threshold'] = max_threshold
