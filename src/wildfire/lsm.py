from __future__ import annotations

from dataclasses import dataclass

import torch

from .utils import gradient, split_xyz_t


@dataclass
class LevelSetConfig:
    speed: float = 1.0
    epsilon: float = 1e-6
    center: tuple[float, float] = (0.5, 0.5)
    radius: float = 0.15
    vx: float = 0.0
    vy: float = 0.0
    r0: float = 0.165 
    cf: float = 3.24 
    bc_type: str = "dirichlet"


class LevelSetPDE:
    """Level Set equation: phi_t + F * |grad(phi)| = 0.

    This version uses constant speed F; extend with a callable speed field if needed.
    """

    def __init__(self, config: LevelSetConfig | None = None):
        self.config = config or LevelSetConfig()

    def residual_v1(self, xyt: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        xyt.requires_grad_(True)
        phi = phi(xyt)
        grads = gradient(phi, xyt)
        phi_t = grads[:, 2:3]
        grad_phi = grads[:, 0:2]
        grad_norm = torch.sqrt(torch.sum(grad_phi**2, dim=1, keepdim=True) + self.config.epsilon)
        advect = self.config.vx * grad_phi[:, 0:1] + self.config.vy * grad_phi[:, 1:2]
        return phi_t + advect + self.config.speed * grad_norm
    
    def residual(self, xyt: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        xyt.requires_grad_(True)
        phi = phi(xyt)
        grads = gradient(phi, xyt)
        phi_t = grads[:, 2:3]
        grad_phi = grads[:, 0:2]
        grad_norm = torch.sqrt(torch.sum(grad_phi**2, dim=1, keepdim=True) + self.config.epsilon)
        nx = grad_phi[:, 0:1] / grad_norm
        ny = grad_phi[:, 1:2] / grad_norm
        coef = self.config.r0 * (1 + self.config.cf * (self.config.vx * nx + self.config.vy * ny))
        Ux = coef * nx
        Uy = coef * ny
        advect = Ux * grad_phi[:, 0:1] + Uy * grad_phi[:, 1:2]
        return phi_t + advect

    def initial_condition(self, xyt: torch.Tensor) -> torch.Tensor:
        x, y, _ = split_xyz_t(xyt)
        cx, cy = self.config.center
        dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return torch.where(dist <= self.config.radius, torch.full_like(dist, -1.0), torch.full_like(dist, 1.0))

    def boundary_condition(self, xyt: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        # Zero Neumann boundary condition via mirror can be enforced by PDE penalty only.
        # Here we simply set phi to be the initial condition for boundary points.

        bc_type = self.config.bc_type.lower().strip()
        if bc_type == "periodic":
            # Periodic boundary condition: enforce phi(x_min, y, t) == phi(x_max, y, t)
            # and phi(x, y_min, t) == phi(x, y_max, t).
            x, y, t = split_xyz_t(xyt)
            x_min = x.min()
            x_max = x.max()
            y_min = y.min()
            y_max = y.max()
            x_periodic = torch.where(
                torch.isclose(x, x_min),
                x_max,
                torch.where(torch.isclose(x, x_max), x_min, x),
            )
            y_periodic = torch.where(
                torch.isclose(y, y_min),
                y_max,
                torch.where(torch.isclose(y, y_max), y_min, y),
            )
            xyt_periodic = torch.cat([x_periodic, y_periodic, t], dim=1)
            return phi(xyt_periodic)

        if bc_type == "neumann":
            # Zero Neumann boundary condition is enforced in the loss via normal derivatives.
            # Return zeros here to keep API consistent when called directly.
            return torch.zeros((xyt.shape[0], 1), device=xyt.device)

        # Dirichlet boundary condition: use initial condition values on boundary.
        return self.initial_condition(xyt)
