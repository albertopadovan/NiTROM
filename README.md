# NiTROM

**N**on-**i**ntrusive **T**rajectory-based **R**educed-**O**rder **M**odelling — a Python
package for learning low-dimensional models of high-dimensional dynamical systems
directly from trajectory data.

NiTROM learns a reduced-order model (ROM) as a pair of objects that are trained
**together**:

* a **projection** between the ambient state space $\mathbb{R}^N$ and a latent space
  $\mathbb{R}^r$ (linear/oblique, or a polynomial manifold), and
* a **latent-space dynamics model** $\dot{z} = f(z, u)$ (polynomial, optionally
  constrained to be globally asymptotically stable).

Unlike intrusive projection-based ROMs, no access to the full-order operators is
required — only trajectory data. Unlike plain operator inference, the cost is the
*trajectory* mismatch obtained by actually time-marching the ROM, and the bases are
optimized alongside the operators using **analytic adjoint gradients** on matrix
manifolds.

---

## What's in the box

| Capability | Where |
| --- | --- |
| NiTROM trajectory-matching training (adjoint gradients, learned bases) | [NitromModule](src/nitrom/optimization/modules/nitrom_module.py) |
| Operator inference (gradient-based **and** closed-form least squares) | [OpInfModule](src/nitrom/optimization/modules/opinf_module.py), [solve_opinf](src/nitrom/optimization/opinf_solver.py) |
| Polynomial-manifold (nonlinear decoder) inference | [PolyManifoldInfModule](src/nitrom/optimization/modules/poly_manifold_module.py) |
| Stability-guaranteed (GAS) polynomial models | [GasPolynomialModel](src/nitrom/latent_space_models/gas_polynomial_model.py) |
| Riemannian optimization on Grassmann / Stiefel manifolds | [manifold_optimization.py](src/nitrom/optimization/manifold_optimization.py) |
| Batched RK2/RK4/backward-Euler/RK45 integrators + discrete adjoint | [time_steppers/](src/nitrom/time_steppers/time_stepper.py) |
| NumPy **or** PyTorch backend, MPI / `torchrun` trajectory parallelism | [backend.py](src/nitrom/backend.py) |
| POD, trajectory loading/sharding, quadratic interpolation | [utils.py](src/nitrom/utils.py), [training_data.py](src/nitrom/training_data.py) |

---

## Installation

```bash
git clone https://github.com/albertopadovan/NiTROM.git
cd NiTROM
pip install -e .              # runtime deps: numpy, scipy, numba, matplotlib, torch, dill
pip install -e ".[dev]"       # + pytest, ruff
```

Requires Python ≥ 3.10. `mpi4py` is optional and only needed for MPI-parallel training
on the NumPy backend; `torch` is imported lazily, so a NumPy-only session never pulls
it in.

---

## Quick start

```python
import numpy as np
from nitrom.backend import set_backend
from nitrom.training_data import TrainingPool, TrainingData
from nitrom.utils import compute_POD
from nitrom.projections.linear_projection import LinearProjection
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.roms.param_registry import ParamRegistry
from nitrom.optimization import NitromModule, train

set_backend("numpy")          # or "torch" for autograd / GPU

# 1. Load trajectories from disk (sharded across ranks under mpiexec/torchrun)
pool = TrainingPool(
    n_traj=4,
    fname_traj="./trajectories/traj_%03d.npy",
    fname_time="./trajectories/time.npy",
    fname_weights="./trajectories/weight_%03d.npy",
    fname_forcing="./trajectories/forcing_%03d.pkl",
    fname_derivs="./trajectories/deriv_%03d.npy",
    dtype=np.float64,
)
data = TrainingData(pool, which_trajs=[0, 1, 2, 3],
                    percent_time_length=1.0, leggauss_deg=5, nsave_rom=15)

# 2. Initial bases from POD; oblique projection (Psi = Phi here => orthogonal)
U, _, _ = compute_POD(pool, normalize=True)
Phi = U[:, :2]
projection = LinearProjection([Phi, Phi])

# 3. Latent quadratic model  zdot = A z + H:zz^T (+ B u)
model = PolynomialModel(r=2, poly_comp=[1, 2], dtype=np.float64)

# 4. Train bases and operators jointly, on their natural manifolds
registry = ParamRegistry(model, projection)
rom = NitromModule(data, registry, fom=my_fom, n_substeps=15)
rom.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])
train(rom, n_epochs=400, lr=1.0, optimizer_type="lbfgs", tol=1e-14)
```

Run in parallel over trajectories with `mpiexec -n 4 python train.py` (NumPy backend)
or `torchrun --nproc_per_node=4 train.py` (PyTorch backend) — the cost and gradient are
all-reduced automatically and every rank runs the identical optimizer in lockstep.

---

## Core concepts

### Data

[`TrainingPool`](src/nitrom/training_data.py) loads trajectory snapshots, per-trajectory
weights, time derivatives, and forcing callables from disk and shards them across
ranks (auto-detecting `MPI.COMM_WORLD` or the `torch.distributed` process group).
[`TrainingData`](src/nitrom/training_data.py) is the view handed to an optimizer: it
selects a subset of trajectories, truncates them to a fraction of their length, sets up
Gauss–Legendre quadrature for the adjoint integrals, and can expand each physical
trajectory into several time-shifted "virtual" trajectories.

### Projections

[`Projection`](src/nitrom/projections/projection.py) defines `encode`/`decode` plus the
VJPs the adjoint needs.

* [`LinearProjection`](src/nitrom/projections/linear_projection.py) — oblique projection
  with trial basis $\Phi$ and test basis $\Psi$: $\mathrm{encode}(q)=\Psi^\top q$,
  $\mathrm{decode}(z)=\Phi(\Psi^\top\Phi)^{-1}z$.
* [`PolynomialProjection`](src/nitrom/projections/polynomial_projection.py) — linear
  encoder with a **polynomial-manifold decoder**
  $\mathrm{decode}(z)=\Phi S z + \mathbb{P}\sum_k A_k z^{\otimes k}$, where the
  complementary projector $\mathbb{P} = I - \Phi S\Psi^\top$ guarantees
  $\mathrm{encode}(\mathrm{decode}(z))=z$.

### Latent-space models

* [`PolynomialModel`](src/nitrom/latent_space_models/polynomial_model.py) —
  $f(z,u) = A_1 z + A_2 : zz^\top + \dots + Bu$ for any set of polynomial degrees, with
  analytic Jacobians, adjoint RHS, and parameter VJPs.
* [`GasPolynomialModel`](src/nitrom/latent_space_models/gas_polynomial_model.py) —
  the same dynamics reparameterized through free variables $(K, R, Q, S)$ so that the
  assembled operators are **globally asymptotically stable by construction**:
  $A = ((K-K^\top) - R^{-1}R^{-\top})\tilde{Q}$ and
  $H_{ijk} = (S_{ilk}-S_{lik})\tilde{Q}_{lj}$ with $\tilde{Q}=Q^{-1}Q^{-\top}$.
  `retract_general_tensors_to_gas_tensors` maps an existing (possibly unstable) $(A,H)$
  onto this manifold via a Lyapunov solve, so a Galerkin or OpInf model can be used as
  a warm start.

### ParamRegistry

[`ParamRegistry`](src/nitrom/roms/param_registry.py) unifies the parameters of the model
and the projection into a single optimization vector. Parameters appearing in *both*
(e.g. a basis that also enters the latent dynamics) are registered once and marked
shared, so a single variable feeds both sites and their gradients are summed.

### Inference modules and training

Every trainable object subclasses
[`InferenceModule`](src/nitrom/optimization/modules/base.py), a minimal `nn.Module`-like
container that returns a scalar cost from `forward()` and its **analytic** gradient from
`gradient()`. That single contract lets one training loop drive all of them:

| Module | Cost |
| --- | --- |
| `OpInfModule` | $J = \sum_i w_i\lVert \dot z_i - f(t_i,z_i)\rVert^2 + \lambda\sum_k\lVert A_k\rVert^2$ — least-squares fit of the projected derivatives |
| `PolyManifoldInfModule` | $J = \sum_i w_i\lVert x_i - \mathrm{decode}(z_i)\rVert^2 + \lambda\sum_k\lVert A_k\rVert^2$ — nonlinear-manifold reconstruction error |
| `NitromModule` | $J = \sum_j \alpha_j^{-1}\sum_i \lVert y^{(j)}(t_i) - \hat y^{(j)}(t_i)\rVert^2$ — output mismatch of the **time-marched** ROM, differentiated by the adjoint |

Common controls on any module:

* `set_unlearnable("B")` / `set_learnable(...)` — freeze parameters (their gradients are
  zeroed), e.g. to keep a fixed input operator $B = \Phi^\top B_{\text{fom}}$.
* `set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])` — optimize bases on
  matrix manifolds instead of Euclidean space.

[`train`](src/nitrom/optimization/train.py) runs Adam, SGD, or L-BFGS with optional
restarts and multi-start perturbations, and records `loss_history` / `gradnorm_history`.
On the PyTorch backend it uses `torch.optim` (plus LR schedulers and distributed
all-reduce); on the NumPy backend every optimizer supplies only a search *direction* and
the step comes from a strong-Wolfe line search with Armijo backtracking. Manifold-typed
parameters are retracted and their gradients tangent-projected on both backends, with
momentum buffers vector-transported between tangent spaces.

For the pure operator-inference problem,
[`solve_opinf`](src/nitrom/optimization/opinf_solver.py) skips iteration entirely and
returns the exact regularized weighted-least-squares solution, honouring frozen
parameters.

### Time steppers

[`solve_ivp`](src/nitrom/time_steppers/time_stepper.py) integrates batched IVPs with
`rk2`, `rk4`, `backward_euler` (Newton, with batched Jacobians), or adaptive `rk45`
(Dormand–Prince 5(4)), returning the solution interpolated onto the requested evaluation
times. `solve_adjoint_ivp_discrete` provides the matching discrete adjoint, so NiTROM
gradients can be taken either **discretely** (exactly consistent with the forward
integrator) or **continuously** — selectable via `adjoint_method` on `NitromModule`.

### Backends

All math is written against a thin [`Backend`](src/nitrom/backend.py) shim, so the same
code runs on NumPy (light, CPU) or PyTorch (autograd, GPU). Select it once per process:

```python
from nitrom.backend import set_backend
set_backend("numpy")   # or "torch"
```

Objects bind to whichever backend is active *at construction time*.
`setup_distributed()` / `cleanup_distributed()` wrap `torch.distributed` process-group
setup (NCCL on CUDA, gloo otherwise) for `torchrun` jobs.

---

## Examples

Each directory under [examples/](examples/) is self-contained: generate data, train, then
post-process.

* **[examples/toymodel/](examples/toymodel/)** — a 3-state cubic ODE reduced to $r=2$.
  The fastest way to see the whole workflow: `generate_data.py` → `train_opinf.py` →
  `train_nitrom.py` → `read_results.py`. Also includes `check_gradients.py`
  (finite-difference validation of the adjoint gradients) and `train_opinf_sweep.py`
  (regularization sweep).
* **[examples/cavity/](examples/cavity/)** — 2D lid-driven cavity flow at $Re = 8300$ on
  a $100\times100$ grid (Numba-accelerated FOM), reduced to $r=50$.
* **[examples/cavity_30/](examples/cavity_30/)** — the same cavity case at $r=30$.

The training scripts compare Galerkin projection, OpInf, GAS-OpInf, NiTROM, and
GAS-NiTROM side by side, and save self-contained ROM checkpoints (bases + operators) as
pickles.

[`nitrom.plotting.set_plot_style`](src/nitrom/plotting.py) applies the shared
publication figure style used by the post-processing scripts (requires a working LaTeX
installation, since `text.usetex` is enabled).

---

## Tests

```bash
pytest                        # full suite
pytest tests/optimization -q  # just the optimization tests
```

The suite covers the backend shim, training-data sharding, projections, latent-space
models, time steppers and their adjoints, manifold geometry, the Riemannian L-BFGS, and
finite-difference gradient checks for every inference module
([tests/optimization/modules/_gradcheck.py](tests/optimization/modules/_gradcheck.py)).
Linting uses `ruff` with the configuration in [pyproject.toml](pyproject.toml).

---

## License

MIT — see [LICENSE](LICENSE).
