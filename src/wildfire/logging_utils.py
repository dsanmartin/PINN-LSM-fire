from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .train import ExperimentConfig


def build_config_log(cfg: ExperimentConfig) -> str:
    # Get config values from LSMConfig or AsensioConfig
    model_type = cfg.pde.model_type.lower() if hasattr(cfg.pde, "model_type") else "lsm"
    
    if model_type == "asensio" and cfg.pde.asensio_config:
        asensio = cfg.pde.asensio_config
        return f"""{'='*60}
Experiment Configuration
{'='*60}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Domain:
  x: [{cfg.domain.x_min}, {cfg.domain.x_max}]
  y: [{cfg.domain.y_min}, {cfg.domain.y_max}]
  t: [{cfg.domain.t_min}, {cfg.domain.t_max}]

Training:
  n_interior: {cfg.train.n_interior}
  n_boundary: {cfg.train.n_boundary}
  n_initial: {cfg.train.n_initial}
  lr_adam: {cfg.train.lr_adam}
  epochs_adam: {cfg.train.epochs_adam}
  epochs_lbfgs: {cfg.train.epochs_lbfgs}
  lr_lbfgs: {cfg.train.lr_lbfgs}
  weight_pde: {cfg.train.weight_pde}
  weight_bc: {cfg.train.weight_bc}
  weight_ic: {cfg.train.weight_ic}

Initial condition:
  center: ({asensio.center[0]}, {asensio.center[1]})
  radius: {asensio.radius}

Asensio Model:
  T0: {asensio.T0}
  T_inf: {asensio.T_inf}
  T_ign: {asensio.T_ign}
  velocity: ({asensio.vx}, {asensio.vy})
  E: {asensio.E}
  R: {asensio.R}
  Y_f: {asensio.Y_f}
  SIGMA: {asensio.SIGMA}
  delta: {asensio.delta}
  H_R: {asensio.H_R}
  boundary_conditions: {asensio.bc_type}

PINN Model:
  layers: {cfg.model.layers}
  activation: {cfg.model.activation}
{'='*60}
"""
    else:
        lsm = cfg.pde.lsm_config if cfg.pde.lsm_config else None
        if lsm:
            return f"""{'='*60}
Experiment Configuration
{'='*60}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Domain:
  x: [{cfg.domain.x_min}, {cfg.domain.x_max}]
  y: [{cfg.domain.y_min}, {cfg.domain.y_max}]
  t: [{cfg.domain.t_min}, {cfg.domain.t_max}]

Training:
  n_interior: {cfg.train.n_interior}
  n_boundary: {cfg.train.n_boundary}
  n_initial: {cfg.train.n_initial}
  lr_adam: {cfg.train.lr_adam}
  epochs_adam: {cfg.train.epochs_adam}
  epochs_lbfgs: {cfg.train.epochs_lbfgs}
  lr_lbfgs: {cfg.train.lr_lbfgs}
  weight_pde: {cfg.train.weight_pde}
  weight_bc: {cfg.train.weight_bc}
  weight_ic: {cfg.train.weight_ic}

Initial condition:
  center: ({lsm.center[0]}, {lsm.center[1]})
  radius: {lsm.radius}

PDE:
  speed: {lsm.speed}
  velocity: ({lsm.vx}, {lsm.vy})
  epsilon: {lsm.epsilon}
  r0: {lsm.r0}
  cf: {lsm.cf}
  boundary_conditions: {lsm.bc_type}

PINN Model:
  layers: {cfg.model.layers}
  activation: {cfg.model.activation}
{'='*60}
"""
        else:
            return f"""{'='*60}
Experiment Configuration
{'='*60}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Domain:
  x: [{cfg.domain.x_min}, {cfg.domain.x_max}]
  y: [{cfg.domain.y_min}, {cfg.domain.y_max}]
  t: [{cfg.domain.t_min}, {cfg.domain.t_max}]

Training:
  n_interior: {cfg.train.n_interior}
  n_boundary: {cfg.train.n_boundary}
  n_initial: {cfg.train.n_initial}
  lr: {cfg.train.lr}
  epochs: {cfg.train.epochs}
  weight_pde: {cfg.train.weight_pde}
  weight_bc: {cfg.train.weight_bc}
  weight_ic: {cfg.train.weight_ic}

PINN Model:
  layers: {cfg.model.layers}
  activation: {cfg.model.activation}
{'='*60}
"""


def write_config_log(cfg: ExperimentConfig, out_dir: Path) -> Path:
    config_text = build_config_log(cfg)
    print(config_text)
    log_path = out_dir / "config.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(config_text)
    print(f"Configuration saved to: {log_path}\n")
    return log_path
