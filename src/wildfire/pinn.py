from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


@dataclass
class MLPConfig:
    layers: Iterable[int] = (3, 64, 64, 64, 1)
    activation: str = "tanh"


class PINN(nn.Module):
    def __init__(self, config: MLPConfig | None = None):
        super().__init__()
        self.config = config or MLPConfig()
        self.model = self._build()

    def _build(self) -> nn.Sequential:
        acts = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        if self.config.activation not in acts:
            raise ValueError(f"Unsupported activation: {self.config.activation}")

        layers = list(self.config.layers)
        modules: list[nn.Module] = []
        for in_f, out_f in zip(layers[:-2], layers[1:-1]):
            modules.append(nn.Linear(in_f, out_f))
            modules.append(acts[self.config.activation]())
        modules.append(nn.Linear(layers[-2], layers[-1]))
        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
