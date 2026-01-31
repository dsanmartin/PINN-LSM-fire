from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .train import ExperimentConfig


def build_config_log(cfg: ExperimentConfig) -> str:
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

Initial condition:
  center: ({cfg.lsm.center[0]}, {cfg.lsm.center[1]})
  radius: {cfg.lsm.radius}

Level Set Method:
  speed: {cfg.lsm.speed}
  velocity: ({cfg.lsm.vx}, {cfg.lsm.vy})
  epsilon: {cfg.lsm.epsilon}
  r0: {cfg.lsm.r0}
  cf: {cfg.lsm.cf}
  boundary_conditions: {cfg.lsm.bc_type}

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
