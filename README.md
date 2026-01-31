# Physics-Informed Neural Networks for wildfire simulation

At the moment, we have the following models implemented:
- Level Set method (LSM) to track the fire perimeter.
- Asensio & Ferragut mode. Energy equation + simplified fuel.

## Mathematical Formulation

### Leve Set Method

The Level Set Method represents the fire front as the zero level set of a function $\phi(x, y, t)$:
- $\phi < 0$: burning region
- $\phi = 0$: fire front (interface)
- $\phi > 0$: unburned region

### Level Set Equation

The implementation solves the following PDE:

$$\frac{\partial \phi}{\partial t} + v_x \frac{\partial \phi}{\partial x} + v_y \frac{\partial \phi}{\partial y} = 0$$

where:
- $v_x, v_y$: advection velocity components (e.g., wind)


### Asensio & Ferragut model

The Asensio & Ferragut model simulates wildfire propagation using coupled temperature and fuel equations with radiative heating effects.

**Temperature equation:**
$$\frac{\partial T}{\partial t} + \mathbf{V} \cdot \nabla T = \nabla \cdot \left((k + 4\delta\sigma T^3) \nabla T\right) + S(T, Y)$$

**Fuel equation:**
$$\frac{\partial Y}{\partial t} = -Y_{\text{f}} Y H(T - T_{\text{ign}}) \exp\left(-\frac{T_{\text{act}}}{T}\right)$$

where:
- $T(x, y, t)$: Temperature field
- $Y(x, y, t)$: Fuel fraction (0 = no fuel, 1 = fully fueled)
- $\mathbf{V} = (v_x, v_y)$: Wind velocity field
- $k$: Thermal conductivity
- $\sigma$: Stefan-Boltzmann constant
- $\delta$: Thickness of the combustible layer
- $T_{\text{ign}}$: Ignition temperature threshold
- $T_{\text{act}}$: Activation temperature
- $Y_{\text{f}}$: Fuel consumption rate
- $H(\cdot)$: Heaviside step function (approximated with sigmoid)
- $S(T, Y)$: Heat source term from fuel combustion

If you set $\delta=0$, $S(x,y,t)=0$, $Y(x,y,t)=0$, $\forall x,y,t$, a diffusion-convection equation is solved.

### Physics-Informed Neural Network (PINN)

A neural network $\phi_\theta(x, y, t)$ approximates the level set function, trained to minimize:

$$\mathcal{L} = w_{\text{PDE}} \mathcal{L}_{\text{PDE}} + w_{\text{BC}} \mathcal{L}_{\text{BC}} + w_{\text{IC}} \mathcal{L}_{\text{IC}} + w_{\text{Data}} \mathcal{L}_{\text{Data}}$$

where:
- $\mathcal{L}_{\text{PDE}}$: PDE residual at interior collocation points
- $\mathcal{L}_{\text{BC}}$: boundary condition loss
- $\mathcal{L}_{\text{IC}}$: initial condition loss
- $\mathcal{L}_{\text{Data}}$: data loss
- $w_{\ast}$: represent a weight for each loss

### TO DO

Add conservation equations and data (simulated + real). 

## Package structure

```
src/
  wildfire/
    __init__.py     # Main exports
    pde.py          # PDE and configuration
    pinn.py         # Neural network architecture
    train.py        # Training loop and loss computation
    utils.py        # Domain sampling and utilities
    plot.py         # Visualization tools
examples/
  main.py           # Training script with CLI
  config.ini        # Configuration file example
```

## Installation

```bash
pip install -e .
```

## Usage

### Command-line interface

Run with configuration file:
```bash
python examples/main.py --config examples/config.ini
```

Override specific parameters:
```bash
python examples/main.py --config examples/config.ini --epochs 10000 --vx 1.0 --vy 0.5
```

### Configuration file format

Create a `.ini` file with the following sections:

```ini
[domain]
x_min = 0.0
x_max = 10.0
y_min = 0.0
y_max = 10.0
t_min = 0.0
t_max = 1.0

[train]
n_interior = 10000    # Interior collocation points
n_boundary = 2000     # Boundary collocation points
n_initial = 2000      # Initial condition points
lr = 1e-3             # Learning rate
epochs = 5000         # Training epochs
weight_pde = 1.0      # PDE loss weight
weight_bc = 1.0       # Boundary condition loss weight
weight_ic = 1.0       # Initial condition loss weight

[initial]
center_x = 5.0        # Initial fire center x-coordinate
center_y = 5.0        # Initial fire center y-coordinate
radius = 2.0          # Initial fire radius

[pde]
vx = 1.0              # x-component of advection velocity
vy = 1.0              # y-component of advection velocity
```

### Programmatic usage

```python
from wildfire.train import ExperimentConfig, train
from wildfire.utils import Domain, TrainConfig
from wildfire.pde import PDEConfig

cfg = ExperimentConfig(
    domain=Domain(x_min=0, x_max=10, y_min=0, y_max=10, t_min=0, t_max=1),
    train=TrainConfig(epochs=5000, lr=1e-3),
    lsm=PDEConfig(speed=1.0, vx=1.0, vy=0.5, center=(5, 5), radius=2)
)

model = train(cfg)
```

### Visualization

```python
from wildfire.plot import plot_perimeter_comparison, plot_phi_comparison

# Plot initial and final fire perimeters
plot_perimeter_comparison(model, cfg.domain, "perimeter.png", lsm=cfg.lsm)

# Plot initial and final phi fields side-by-side
plot_phi_comparison(model, cfg.domain, "phi.png", lsm=cfg.lsm)
```

## Output

Each experiment creates a timestamped folder (format `YYYYMMDDHHmm`) in `examples/outputs/` containing:
- `config.log`: Complete configuration used for the experiment
- `phi_comparison.png`: Initial and final level set function
- `perimeter_comparison.png`: Initial and final fire fronts (φ=0 contours)

## Examples

See [examples/EXAMPLES.md](examples/EXAMPLES.md) for detailed simulation scenarios with different wind conditions (no wind, south wind, east wind, northeast wind) and corresponding results.
