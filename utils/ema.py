import torch
import torch.nn as nn
from typing import Dict


class Ema:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self._backup = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
            else:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if self._backup is not None:
            model.load_state_dict(self._backup, strict=False)
            self._backup = None

    def load_shadow(self, sd: Dict[str, torch.Tensor]):
        self.shadow = {k: v.clone() for k, v in sd.items()}
