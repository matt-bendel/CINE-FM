# utils/ema.py  (drop-in)
import torch
import torch.nn as nn
from typing import Dict

class Ema:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        # keep a FP32 master copy (only float params)
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().float().clone().to(v.device)
            for k, v in model.state_dict().items()
            if v.is_floating_point()
        }
        self._backup = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if not v.is_floating_point():  # ignore non-float buffers
                continue
            s = self.shadow.get(k, None)
            v32 = v.detach().float()
            if s is None:
                self.shadow[k] = v32.clone()
            else:
                s.mul_(self.decay).add_(v32, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        # backup model’s current (typed) weights
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        # load EMA -> cast to each param’s dtype/device
        to_load = {}
        for k, v in model.state_dict().items():
            if k in self.shadow and v.is_floating_point():
                to_load[k] = self.shadow[k].to(dtype=v.dtype, device=v.device)
        model.load_state_dict({**model.state_dict(), **to_load}, strict=False)

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if self._backup is not None:
            model.load_state_dict(self._backup, strict=False)
            self._backup = None

    def load_shadow(self, sd: Dict[str, torch.Tensor]):
        self.shadow = {k: v.clone().float() for k, v in sd.items()}
