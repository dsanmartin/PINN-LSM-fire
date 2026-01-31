from __future__ import annotations

from argparse import ArgumentParser
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path

from wildfire.plot import plot_perimeter_comparison, plot_phi_comparison
from wildfire.logging_utils import write_config_log
from wildfire.pde import PDEConfig
from wildfire.train import ExperimentConfig, train
from wildfire.utils import Domain, TrainConfig


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train PINN for Level Set Method and plot perimeters.")

    parser.add_argument("--config", type=str, default=None, help="Path to INI config file.")

    # Domain parameters
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--t-min", type=float, default=None)
    parser.add_argument("--t-max", type=float, default=None)

    # Training parameters
    parser.add_argument("--n-interior", type=int, default=None)
    parser.add_argument("--n-boundary", type=int, default=None)
    parser.add_argument("--n-initial", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--weight-pde", type=float, default=None)
    parser.add_argument("--weight-bc", type=float, default=None)
    parser.add_argument("--weight-ic", type=float, default=None)

    # Initial condition parameters
    parser.add_argument("--center-x", type=float, default=None)
    parser.add_argument("--center-y", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None)

    # Velocity parameters
    parser.add_argument("--vx", type=float, default=None)
    parser.add_argument("--vy", type=float, default=None)
    return parser


def _read_config(path: str | None) -> ConfigParser | None:
    if not path:
        return None
    cfg = ConfigParser()
    cfg.read(path)
    return cfg


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    base_domain = Domain()
    base_train = TrainConfig()
    base_lsm = PDEConfig()
    cfg_file = _read_config(args.config)

    if cfg_file is not None and cfg_file.has_section("domain"):
        base_domain = Domain(
            x_min=cfg_file.getfloat("domain", "x_min", fallback=base_domain.x_min),
            x_max=cfg_file.getfloat("domain", "x_max", fallback=base_domain.x_max),
            y_min=cfg_file.getfloat("domain", "y_min", fallback=base_domain.y_min),
            y_max=cfg_file.getfloat("domain", "y_max", fallback=base_domain.y_max),
            t_min=cfg_file.getfloat("domain", "t_min", fallback=base_domain.t_min),
            t_max=cfg_file.getfloat("domain", "t_max", fallback=base_domain.t_max),
        )

    if cfg_file is not None and cfg_file.has_section("train"):
        base_train = TrainConfig(
            n_interior=cfg_file.getint("train", "n_interior", fallback=base_train.n_interior),
            n_boundary=cfg_file.getint("train", "n_boundary", fallback=base_train.n_boundary),
            n_initial=cfg_file.getint("train", "n_initial", fallback=base_train.n_initial),
            lr=cfg_file.getfloat("train", "lr", fallback=base_train.lr),
            epochs=cfg_file.getint("train", "epochs", fallback=base_train.epochs),
            weight_pde=cfg_file.getfloat("train", "weight_pde", fallback=base_train.weight_pde),
            weight_bc=cfg_file.getfloat("train", "weight_bc", fallback=base_train.weight_bc),
            weight_ic=cfg_file.getfloat("train", "weight_ic", fallback=base_train.weight_ic),
        )

    if cfg_file is not None and cfg_file.has_section("initial"):
        base_lsm = PDEConfig(
            center=(
                cfg_file.getfloat("initial", "center_x", fallback=base_lsm.center[0]),
                cfg_file.getfloat("initial", "center_y", fallback=base_lsm.center[1]),
            ),
            radius=cfg_file.getfloat("initial", "radius", fallback=base_lsm.radius),
            speed=base_lsm.speed,
            epsilon=base_lsm.epsilon,
            vx=base_lsm.vx,
            vy=base_lsm.vy,
            r0=base_lsm.r0,
            cf=base_lsm.cf,
            bc_type=base_lsm.bc_type,
        )

    if cfg_file is not None and cfg_file.has_section("pde"):
        base_lsm = PDEConfig(
            center=base_lsm.center,
            radius=base_lsm.radius,
            speed=cfg_file.getfloat("pde", "speed", fallback=base_lsm.speed),
            epsilon=cfg_file.getfloat("pde", "epsilon", fallback=base_lsm.epsilon),
            vx=cfg_file.getfloat("pde", "vx", fallback=base_lsm.vx),
            vy=cfg_file.getfloat("pde", "vy", fallback=base_lsm.vy),
            r0=cfg_file.getfloat("pde", "r0", fallback=base_lsm.r0),
            cf=cfg_file.getfloat("pde", "cf", fallback=base_lsm.cf),
            bc_type=cfg_file.get("pde", "bc_type", fallback=base_lsm.bc_type),
        )

    domain = Domain(
        x_min=base_domain.x_min if args.x_min is None else args.x_min,
        x_max=base_domain.x_max if args.x_max is None else args.x_max,
        y_min=base_domain.y_min if args.y_min is None else args.y_min,
        y_max=base_domain.y_max if args.y_max is None else args.y_max,
        t_min=base_domain.t_min if args.t_min is None else args.t_min,
        t_max=base_domain.t_max if args.t_max is None else args.t_max,
    )
    train_cfg = TrainConfig(
        n_interior=base_train.n_interior if args.n_interior is None else args.n_interior,
        n_boundary=base_train.n_boundary if args.n_boundary is None else args.n_boundary,
        n_initial=base_train.n_initial if args.n_initial is None else args.n_initial,
        lr=base_train.lr if args.lr is None else args.lr,
        epochs=base_train.epochs if args.epochs is None else args.epochs,
        weight_pde=base_train.weight_pde if args.weight_pde is None else args.weight_pde,
        weight_bc=base_train.weight_bc if args.weight_bc is None else args.weight_bc,
        weight_ic=base_train.weight_ic if args.weight_ic is None else args.weight_ic,
    )
    lsm_cfg = PDEConfig(
        center=(
            base_lsm.center[0] if args.center_x is None else args.center_x,
            base_lsm.center[1] if args.center_y is None else args.center_y,
        ),
        radius=base_lsm.radius if args.radius is None else args.radius,
        speed=base_lsm.speed,
        epsilon=base_lsm.epsilon,
        vx=base_lsm.vx if args.vx is None else args.vx,
        vy=base_lsm.vy if args.vy is None else args.vy,
        r0=base_lsm.r0,
        cf=base_lsm.cf,
        bc_type=base_lsm.bc_type,
    )

    cfg = ExperimentConfig(domain=domain, train=train_cfg, lsm=lsm_cfg)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    base_out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir = base_out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    write_config_log(cfg, out_dir)

    model = train(cfg)
    plot_phi_comparison(model, cfg.domain, out_dir / "phi_comparison.png", lsm=cfg.lsm)
    plot_perimeter_comparison(model, cfg.domain, out_dir / "perimeter_comparison.png", lsm=cfg.lsm)
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
