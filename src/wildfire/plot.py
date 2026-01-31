from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from .pde import PDE, PDEConfig
from .utils import Domain


@dataclass
class PlotConfig:
    resolution: int = 200
    cmap: str = "coolwarm"


def _grid(domain: Domain, resolution: int, t: float, device: torch.device) -> torch.Tensor:
    xs = torch.linspace(domain.x_min, domain.x_max, resolution, device=device)
    ys = torch.linspace(domain.y_min, domain.y_max, resolution, device=device)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    tt = torch.full_like(xx, t)
    xyt = torch.stack([xx, yy, tt], dim=-1).reshape(-1, 3)
    return xyt


def _plot_field(
    field: np.ndarray,
    domain: Domain,
    title: str,
    path: Path | None,
    cfg: PlotConfig,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        field,
        origin="lower",
        extent=[domain.x_min, domain.x_max, domain.y_min, domain.y_max],
        cmap=cfg.cmap,
        aspect="auto",
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def _plot_zero_level_set(
    field: np.ndarray,
    domain: Domain,
    title: str,
    path: Path | None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.linspace(domain.x_min, domain.x_max, field.shape[1])
    y = np.linspace(domain.y_min, domain.y_max, field.shape[0])
    xx, yy = np.meshgrid(x, y, indexing="xy")
    ax.contour(xx, yy, field, levels=[0.0], colors="black", linewidths=2.0)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(title)
    ax.set_xlim(domain.x_min, domain.x_max)
    ax.set_ylim(domain.y_min, domain.y_max)
    ax.set_aspect("equal", adjustable="box")

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_initial_condition(
    domain: Domain,
    path: str | Path | None = None,
    cfg: PlotConfig | None = None,
    lsm: PDEConfig | None = None,
) -> None:
    cfg = cfg or PlotConfig()
    lsm = lsm or PDEConfig()
    pde = PDE(lsm)
    device = torch.device("cpu")
    xyt = _grid(domain, cfg.resolution, domain.t_min, device)
    phi = pde.initial_condition(xyt).reshape(cfg.resolution, cfg.resolution)
    field = phi.detach().cpu().numpy()
    _plot_field(field, domain, "Initial condition", Path(path) if path else None, cfg)


def plot_final_result(
    model: torch.nn.Module,
    domain: Domain,
    path: str | Path | None = None,
    cfg: PlotConfig | None = None,
) -> None:
    cfg = cfg or PlotConfig()
    device = next(model.parameters()).device
    xyt = _grid(domain, cfg.resolution, domain.t_max, device)
    with torch.no_grad():
        phi = model(xyt).reshape(cfg.resolution, cfg.resolution)
    field = phi.detach().cpu().numpy()
    _plot_field(field, domain, "Final result", Path(path) if path else None, cfg)


def plot_initial_perimeter(
    domain: Domain,
    path: str | Path | None = None,
    cfg: PlotConfig | None = None,
    lsm: PDEConfig | None = None,
) -> None:
    cfg = cfg or PlotConfig()
    lsm = lsm or PDEConfig()
    pde = PDE(lsm)
    device = torch.device("cpu")
    xyt = _grid(domain, cfg.resolution, domain.t_min, device)
    phi = pde.initial_condition(xyt).reshape(cfg.resolution, cfg.resolution)
    field = phi.detach().cpu().numpy()
    _plot_zero_level_set(field, domain, "Initial perimeter (phi=0)", Path(path) if path else None)


def plot_final_perimeter(
    model: torch.nn.Module,
    domain: Domain,
    path: str | Path | None = None,
    cfg: PlotConfig | None = None,
) -> None:
    cfg = cfg or PlotConfig()
    device = next(model.parameters()).device
    xyt = _grid(domain, cfg.resolution, domain.t_max, device)
    with torch.no_grad():
        phi = model(xyt).reshape(cfg.resolution, cfg.resolution)
    field = phi.detach().cpu().numpy()
    _plot_zero_level_set(field, domain, "Final perimeter (phi=0)", Path(path) if path else None)


def plot_perimeter_comparison(
    model: torch.nn.Module,
    domain: Domain,
    path: str | Path | None = None,
    cfg: PlotConfig | None = None,
    lsm: PDEConfig | None = None,
) -> None:
    cfg = cfg or PlotConfig()
    lsm = lsm or PDEConfig()
    pde = PDE(lsm)
    device = next(model.parameters()).device

    xyt_init = _grid(domain, cfg.resolution, domain.t_min, torch.device("cpu"))
    phi_init = pde.initial_condition(xyt_init).reshape(cfg.resolution, cfg.resolution)

    xyt_final = _grid(domain, cfg.resolution, domain.t_max, device)
    with torch.no_grad():
        phi_final = model(xyt_final).reshape(cfg.resolution, cfg.resolution)

    field_init = phi_init.detach().cpu().numpy()
    field_final = phi_final.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.linspace(domain.x_min, domain.x_max, field_init.shape[1])
    y = np.linspace(domain.y_min, domain.y_max, field_init.shape[0])
    xx, yy = np.meshgrid(x, y, indexing="xy")
    ax.contour(xx, yy, field_init, levels=[0.0], colors="blue", linewidths=2.0)
    ax.contour(xx, yy, field_final, levels=[0.0], colors="red", linewidths=2.0)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(r"Perimeter comparison ($\phi=0$)")
    ax.set_xlim(domain.x_min, domain.x_max)
    ax.set_ylim(domain.y_min, domain.y_max)
    ax.set_aspect("equal", adjustable="box")
    handles = [
        Line2D([0], [0], color="blue", lw=2, label="Initial"),
        Line2D([0], [0], color="red", lw=2, label="Final"),
    ]
    ax.legend(handles=handles, loc="upper right")

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_phi_comparison(
    model: torch.nn.Module,
    domain: Domain,
    path: str | Path | None = None,
    cfg: PlotConfig | None = None,
    lsm: PDEConfig | None = None,
) -> None:
    cfg = cfg or PlotConfig()
    lsm = lsm or PDEConfig()
    pde = PDE(lsm)
    device = next(model.parameters()).device

    xyt_init = _grid(domain, cfg.resolution, domain.t_min, torch.device("cpu"))
    phi_init = pde.initial_condition(xyt_init).reshape(cfg.resolution, cfg.resolution)

    xyt_final = _grid(domain, cfg.resolution, domain.t_max, device)
    with torch.no_grad():
        phi_final = model(xyt_final).reshape(cfg.resolution, cfg.resolution)

    field_init = phi_init.detach().cpu().numpy()
    field_final = phi_final.detach().cpu().numpy()
    vmin = float(min(field_init.min(), field_final.min()))
    vmax = float(max(field_init.max(), field_final.max()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    im0 = axes[0].imshow(
        field_init,
        origin="lower",
        extent=[domain.x_min, domain.x_max, domain.y_min, domain.y_max],
        cmap=cfg.cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title(r"$\phi(x,y,0)$")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$y$")

    im1 = axes[1].imshow(
        field_final,
        origin="lower",
        extent=[domain.x_min, domain.x_max, domain.y_min, domain.y_max],
        cmap=cfg.cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(r"$\phi(x,y,t_{\text{max}})$")
    axes[1].set_xlabel(r"$x$")

    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.85)

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
