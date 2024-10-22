import torch

from timm import create_model
from argus import Model
from argus.utils import deep_to, deep_detach, deep_chunk
from argus.model.build import choose_attribute_from_dict, cast_nn_module
from argus.loss import pytorch_losses

from src.metrics import F1score
from src.losses import SoftBCEWithLogitsLoss
from src.models.skip_attenstion import SkipAttention
from src.models.resnest import ResNeSt


class BirdsongModel(Model):
    nn_module = {
        'timm': create_model,
        'SkipAttention': SkipAttention,
        'ResNeSt': ResNeSt
    }
    loss = {
        'SoftBCEWithLogitsLoss': SoftBCEWithLogitsLoss,
        **pytorch_losses
    }
    prediction_transform = torch.nn.Sigmoid

    def __init__(self, params):
        super().__init__(params)
        self.amp = None

        if 'iter_size' not in self.params:
            self.params['iter_size'] = 1
        self.iter_size = self.params['iter_size']

    def train_step(self, batch, state) -> dict:
        self.train()
        self.optimizer.zero_grad()

        # Gradient accumulation
        for i, chunk_batch in enumerate(deep_chunk(batch, self.iter_size)):
            input, target = deep_to(chunk_batch, self.device, non_blocking=True)
            prediction = self.nn_module(input)
            loss = self.loss(prediction, target)
            if self.amp is not None:
                delay_unscale = i != (self.iter_size - 1)
                # Mixed precision
                with self.amp.scale_loss(loss, self.optimizer,
                                         delay_unscale=delay_unscale) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

        self.optimizer.step()
        torch.cuda.synchronize()

        prediction = deep_detach(prediction)
        target = deep_detach(target)
        prediction = self.prediction_transform(prediction)
        return {
            'prediction': prediction,
            'target': target,
            'loss': loss.item()
        }
