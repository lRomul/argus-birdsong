import torch.nn as nn
import resnest.torch as resnest_torch


class ResNeSt(nn.Module):
    def __init__(self,
                 model_name='resnest50_fast_1s1x64d',
                 **kwargs):
        super().__init__()
        num_classes = kwargs['num_classes']
        del kwargs['num_classes']

        self.model = getattr(resnest_torch, model_name)(**kwargs)
        self.model.conv1[0].stride = (1, 1)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.model(x)
