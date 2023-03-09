import math, copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Unet

class ConsistencyModel(nn.Module):
    """
    This is ridiculous Unet structure, hey but it works!
    """

    def __init__(self, 
        n_channel: int, 
        eps: float = 0.002, 
        D: int = 128,
        device: str="cuda:0") -> None:
        super(ConsistencyModel, self).__init__()

        self.eps = eps
        self.model = Unet(n_channel, D=256).to(device)

    def forward(self, x, t) -> torch.Tensor:
        if isinstance(t, float):
            t = (
                torch.tensor([t] * x.shape[0], dtype=torch.float32)
                .to(x.device)
                .unsqueeze(1)
            )  # (batch, 1)
        x_ori = x  # (batch, C, H, W)
        x = self.model(x, t)

        t = t - self.eps
        c_skip_t = 0.25 / (t.pow(2) + 0.25)  # (batch, 1)
        c_out_t = 0.25 * t / ((t + self.eps).pow(2) + 0.25).pow(0.5) # (batch, 1)
        return c_skip_t[:, :, None, None] * x_ori + c_out_t[:, :, None, None] * x

    def loss(self, x, z, t1, t2, ema_model):
        x2 = x + z * t2[:, :, None, None]  # x, z: (batch, C, H, W)
        x2 = self(x2, t2)  # (batch, C, H, W)

        with torch.no_grad():
            x1 = x + z * t1[:, :, None, None]
            x1 = ema_model(x1, t1)

        return F.mse_loss(x1, x2)

    @torch.no_grad()
    def sample(self, x, ts: List[float]):
        x = self(x, ts[0])

        for t in ts[1:]:
            z = torch.randn_like(x)
            x = x + math.sqrt(t**2 - self.eps**2) * z
            x = self(x, t)

        return x

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

