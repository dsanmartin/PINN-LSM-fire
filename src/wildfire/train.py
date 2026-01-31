from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .pde import PDE, PDEConfig
from .lsm import LSMConfig
from .pinn import MLPConfig, PINN
from .utils import Domain, TrainConfig


@dataclass
class ExperimentConfig:
    domain: Domain = field(default_factory=Domain)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: MLPConfig = field(default_factory=MLPConfig)
    pde: PDEConfig = field(default_factory=PDEConfig)


def loss_terms(model: PINN, pde: PDE, domain: Domain, cfg: TrainConfig, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    xyt_interior = domain.sample_interior(cfg.n_interior, device)
    xyt_boundary = domain.sample_boundary(cfg.n_boundary, device)
    xyt_initial = domain.sample_initial(cfg.n_initial, device)

    res = pde.residual(xyt_interior, model)
    loss_pde = torch.mean(res**2)

    bc_type = pde.model.config.bc_type.lower().strip() if hasattr(pde.model.config, "bc_type") else "dirichlet"
    if bc_type == "neumann":
        xyt_boundary.requires_grad_(True)
        phi_b = model(xyt_boundary)
        grads_b = torch.autograd.grad(
            phi_b,
            xyt_boundary,
            grad_outputs=torch.ones_like(phi_b),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        n_side = xyt_boundary.shape[0] // 4
        nx = torch.cat(
            [
                torch.full((n_side, 1), -1.0, device=device),
                torch.full((n_side, 1), 1.0, device=device),
                torch.zeros((n_side, 1), device=device),
                torch.zeros((n_side, 1), device=device),
            ],
            dim=0,
        )
        ny = torch.cat(
            [
                torch.zeros((n_side, 1), device=device),
                torch.zeros((n_side, 1), device=device),
                torch.full((n_side, 1), -1.0, device=device),
                torch.full((n_side, 1), 1.0, device=device),
            ],
            dim=0,
        )
        dn = grads_b[:, 0:1] * nx + grads_b[:, 1:2] * ny
        loss_bc = torch.mean(dn**2)
    else:
        phi_b = model(xyt_boundary)
        target_b = pde.boundary_condition(xyt_boundary, model)
        loss_bc = torch.mean((phi_b - target_b) ** 2)

    phi_i = model(xyt_initial)
    target_i = pde.initial_condition(xyt_initial)
    loss_ic = torch.mean((phi_i - target_i) ** 2)
    
    # Separate loss tracking for Asensio model (temperature and fuel)
    metrics = {
        "loss": 0.0,
        "loss_pde": float(loss_pde.detach().cpu()),
        "loss_bc": float(loss_bc.detach().cpu()),
        "loss_ic": float(loss_ic.detach().cpu()),
    }
    
    if pde.config.model_type.lower() == "asensio" and phi_i.shape[1] == 2:
        # Asensio model has temperature (col 0) and fuel (col 1)
        T_pred = phi_i[:, 0:1]
        Y_pred = phi_i[:, 1:2]
        T_target = target_i[:, 0:1]
        Y_target = target_i[:, 1:2]
        
        loss_ic_temp = torch.mean((T_pred - T_target) ** 2)
        loss_ic_fuel = torch.mean((Y_pred - Y_target) ** 2)
        
        metrics["loss_ic_temp"] = float(loss_ic_temp.detach().cpu())
        metrics["loss_ic_fuel"] = float(loss_ic_fuel.detach().cpu())
        metrics["T_pred_mean"] = float(T_pred.mean().detach().cpu())
        metrics["T_target_mean"] = float(T_target.mean().detach().cpu())
        metrics["Y_pred_mean"] = float(Y_pred.mean().detach().cpu())
        metrics["Y_target_mean"] = float(Y_target.mean().detach().cpu())

    loss = cfg.weight_pde * loss_pde + cfg.weight_bc * loss_bc + cfg.weight_ic * loss_ic
    metrics["loss"] = float(loss.detach().cpu())
    
    return loss, metrics


def train(cfg: ExperimentConfig, device: torch.device | None = None) -> PINN:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PINN(cfg.model).to(device)
    pde = PDE(cfg.pde)
    
    # Get training parameters for each optimizer
    n_adam = cfg.train.epochs_adam
    n_lbfgs = cfg.train.epochs_lbfgs
    lr_adam = cfg.train.lr_adam
    lr_lbfgs = cfg.train.lr_lbfgs
    
    # Phase 1: Adam optimizer
    if n_adam > 0:
        print("\n=== Adam Training ===")
        optimizer_adam = torch.optim.Adam(model.parameters(), lr=lr_adam)
        
        for epoch in range(1, n_adam + 1):
            optimizer_adam.zero_grad()
            loss, metrics = loss_terms(model, pde, cfg.domain, cfg.train, device)
            loss.backward()
            optimizer_adam.step()

            if epoch % 500 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:05d} | loss={metrics['loss']:.4e} "
                    f"pde={metrics['loss_pde']:.4e} bc={metrics['loss_bc']:.4e} ic={metrics['loss_ic']:.4e}"
                )
    
    # Phase 2: L-BFGS optimizer
    if n_lbfgs > 0:
        print("\n=== L-BFGS Training ===")
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=lr_lbfgs,
            max_iter=20,
            history_size=50,
            line_search_fn="strong_wolfe"
        )
        
        for epoch in range(1, n_lbfgs + 1):
            def closure():
                optimizer_lbfgs.zero_grad()
                loss, _ = loss_terms(model, pde, cfg.domain, cfg.train, device)
                loss.backward()
                return loss
            
            optimizer_lbfgs.step(closure)
            
            if epoch % 100 == 0 or epoch == 1:
                # Compute metrics for display (gradients needed for residual computation)
                _, metrics = loss_terms(model, pde, cfg.domain, cfg.train, device)
                print(
                    f"Epoch {n_adam + epoch:05d} | loss={metrics['loss']:.4e} "
                    f"pde={metrics['loss_pde']:.4e} bc={metrics['loss_bc']:.4e} ic={metrics['loss_ic']:.4e}"
                )

    return model


def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)
