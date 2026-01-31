from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import torch


@dataclass
class Domain:
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    def sample_interior(self, n: int, device: torch.device) -> torch.Tensor:
        x = torch.rand(n, 1, device=device) * (self.x_max - self.x_min) + self.x_min
        y = torch.rand(n, 1, device=device) * (self.y_max - self.y_min) + self.y_min
        t = torch.rand(n, 1, device=device) * (self.t_max - self.t_min) + self.t_min
        return torch.cat([x, y, t], dim=1)

    def sample_initial(self, n: int, device: torch.device) -> torch.Tensor:
        x = torch.rand(n, 1, device=device) * (self.x_max - self.x_min) + self.x_min
        y = torch.rand(n, 1, device=device) * (self.y_max - self.y_min) + self.y_min
        t = torch.zeros(n, 1, device=device) + self.t_min
        return torch.cat([x, y, t], dim=1)

    def sample_boundary(self, n: int, device: torch.device) -> torch.Tensor:
        n_side = n // 4
        t = torch.rand(n_side, 1, device=device) * (self.t_max - self.t_min) + self.t_min
        x0 = torch.full((n_side, 1), self.x_min, device=device)
        x1 = torch.full((n_side, 1), self.x_max, device=device)
        y0 = torch.full((n_side, 1), self.y_min, device=device)
        y1 = torch.full((n_side, 1), self.y_max, device=device)
        y = torch.rand(n_side, 1, device=device) * (self.y_max - self.y_min) + self.y_min
        x = torch.rand(n_side, 1, device=device) * (self.x_max - self.x_min) + self.x_min

        left = torch.cat([x0, y, t], dim=1)
        right = torch.cat([x1, y, t], dim=1)
        bottom = torch.cat([x, y0, t], dim=1)
        top = torch.cat([x, y1, t], dim=1)
        return torch.cat([left, right, bottom, top], dim=0)


@dataclass
class TrainConfig:
    n_interior: int = 10_000
    n_boundary: int = 2_000
    n_initial: int = 2_000
    epochs_adam: int = 5000  # If set, use Adam for this many epochs
    epochs_lbfgs: int = 1000  # If set, use LBFGS for this many epochs after Adam
    lr_adam: float = 1e-1  # Learning rate for Adam, falls back to lr if not set
    lr_lbfgs: float = 1.0  # Learning rate for LBFGS
    weight_pde: float = 1.0
    weight_bc: float = 1.0
    weight_ic: float = 1.0


def gradient(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def split_xyz_t(xyzt: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    if xyzt.shape[1] == 3:
        x = xyzt[:, 0:1]
        y = xyzt[:, 1:2]
        t = xyzt[:, 2:3]
        return x, y, t
    elif xyzt.shape[1] == 4:
        x = xyzt[:, 0:1]
        y = xyzt[:, 1:2]
        z = xyzt[:, 2:3]
        t = xyzt[:, 3:4]
        return x, y, z, t
    else:
        raise ValueError("Input tensor must have shape (N, 3) or (N, 4)")
