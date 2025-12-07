import torch
import numpy as np


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.scores = []

    def update(self, val):
        self.scores.append(val)

    def show(self):
        out = torch.mean(
            torch.stack(self.scores[np.maximum(len(self.scores) - self.num, 0) :])
        )
        return out


class EMA(object):
    def __init__(self, beta=0.9):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new

        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=1024):
        self.step += 1

        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            return None

        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
