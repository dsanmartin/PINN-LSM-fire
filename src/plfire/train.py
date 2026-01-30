from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .lsm import LevelSetConfig, LevelSetPDE
from .pinn import MLPConfig, PINN
from .utils import Domain, TrainConfig


@dataclass
class ExperimentConfig:
    domain: Domain = field(default_factory=Domain)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: MLPConfig = field(default_factory=MLPConfig)
    lsm: LevelSetConfig = field(default_factory=LevelSetConfig)


def loss_terms(model: PINN, pde: LevelSetPDE, domain: Domain, cfg: TrainConfig, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    xyt_interior = domain.sample_interior(cfg.n_interior, device)
    xyt_boundary = domain.sample_boundary(cfg.n_boundary, device)
    xyt_initial = domain.sample_initial(cfg.n_initial, device)

    res = pde.residual(xyt_interior, model)
    loss_pde = torch.mean(res**2)

    phi_b = model(xyt_boundary)
    target_b = pde.boundary_condition(xyt_boundary)
    loss_bc = torch.mean((phi_b - target_b) ** 2)

    phi_i = model(xyt_initial)
    target_i = pde.initial_condition(xyt_initial)
    loss_ic = torch.mean((phi_i - target_i) ** 2)

    loss = cfg.weight_pde * loss_pde + cfg.weight_bc * loss_bc + cfg.weight_ic * loss_ic
    metrics = {
        "loss": float(loss.detach().cpu()),
        "loss_pde": float(loss_pde.detach().cpu()),
        "loss_bc": float(loss_bc.detach().cpu()),
        "loss_ic": float(loss_ic.detach().cpu()),
    }
    return loss, metrics


def train(cfg: ExperimentConfig, device: torch.device | None = None) -> PINN:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PINN(cfg.model).to(device)
    pde = LevelSetPDE(cfg.lsm)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    for epoch in range(1, cfg.train.epochs + 1):
        optimizer.zero_grad()
        loss, metrics = loss_terms(model, pde, cfg.domain, cfg.train, device)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:05d} | loss={metrics['loss']:.4e} "
                f"pde={metrics['loss_pde']:.4e} bc={metrics['loss_bc']:.4e} ic={metrics['loss_ic']:.4e}"
            )

    return model


def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)
