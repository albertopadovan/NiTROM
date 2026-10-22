# NiTROM

NiTROM is a Python package for learning reduced-order models (ROMs) of dynamical
systems from data. It provides several modeling algorithms, including operator
inference, polynomial-manifold inference, and joint learning of projections and
reduced dynamics from trajectories.

The package's central contribution is **N**on-**i**ntrusive **T**rajectory-based
**R**educed-**O**rder **M**odelling (NiTROM): a method that jointly optimizes a
projection into a low-dimensional latent space and the operators governing the
latent dynamics by minimizing the output error of the time-integrated ROM.

NiTROM training uses analytic adjoint gradients and supports optimization of trial
and test bases on Grassmann and Stiefel manifolds. Full-order governing operators
are not required; NiTROM needs trajectories, input signals for forced systems, and
an output map with its derivative or adjoint action. The other inference methods
can be used independently or to initialize a NiTROM fit.

## Features

| Capability | Public API |
| --- | --- |
| Joint learning of projections and dynamics from trajectory errors | `NitromModule`, `ParamRegistry`, `train` |
| Operator inference from projected time derivatives | `OpInfModule`, `solve_opinf` |
| Polynomial-manifold decoder inference | `PolynomialProjection`, `PolyManifoldInfModule` |
| Polynomial latent dynamics, including constant and input terms | `PolynomialModel` |
| Globally asymptotically stable unforced dynamics | `GasPolynomialModel` |
| Dynamics with an attracting trapping region | `AtrPolynomialModel` |
| POD and data-driven balancing from trajectories | `compute_pod`, `compute_data_driven_balancing` |
| Batched integration and discrete adjoints | `step`, `solve_ivp`, `solve_fixed_ivp`, `solve_adjoint_ivp_discrete` |
| CPU/GPU arrays and trajectory parallelism | NumPy/PyTorch backends, MPI/PyTorch distributed training |

The APIs above are exported from `nitrom`. Numerical helpers for interpolation,
quadrature, and finite-difference derivatives live in `nitrom.numerics`.

## Installation

Requires Python 3.10 or newer. From a source checkout:

```bash
git clone https://github.com/albertopadovan/NiTROM.git
cd NiTROM
python -m pip install -e .
```

The base installation requires NumPy, SciPy, and dill. Install optional features
with extras:

```bash
python -m pip install -e '.[plot]'                 # examples and plotting
python -m pip install -e '.[torch]'                # PyTorch arrays and optimizers
python -m pip install -e '.[mpi]'                  # NumPy trajectory parallelism
python -m pip install -e '.[dev,torch,mpi,plot]'    # development with all extras
```

MPI runs also need an MPI runtime and launcher. Select a PyTorch installation
suited to your hardware for GPU use. PyTorch is imported only when needed; the
NumPy backend is the default.

## Setting up a case

The projection, latent dynamics, and fitting objective are separate choices. For
example, the same polynomial dynamics can be fitted with operator inference and
then refined together with the bases using NiTROM.

| Decision | What to choose |
| --- | --- |
| Data and loss | Training trajectories, input signals, time windows, normalization weights, and outputs to match. OpInf needs state derivatives; NiTROM needs the output map and its derivative or adjoint action. |
| Reduced coordinates | Latent dimension `r`; an orthogonal or oblique linear projection, or a polynomial-manifold decoder; and initial bases from POD, data-driven balancing, or supplied bases. |
| Latent dynamics | Polynomial degrees, constant and input terms, and whether to use unconstrained dynamics (`PolynomialModel`), enforce global asymptotic stability of the unforced origin (`GasPolynomialModel`), or enforce an attracting trapping region (`AtrPolynomialModel`). |
| Fitting algorithm | `OpInfModule` for derivative matching, `PolyManifoldInfModule` for decoder reconstruction, or `NitromModule` for integrated output matching. Ordinary polynomial OpInf supports a direct `solve_opinf` fit; constrained models and NiTROM use iterative `train`. |
| Optimization | Initial operators, fixed versus learned parameters, basis manifold constraints, regularization, optimizer, and stopping criteria. Decoder inference holds the bases fixed; NiTROM can learn them jointly with the dynamics. |
| Time integration | Integration method and resolution or adaptive tolerances. For NiTROM, also choose discrete or continuous adjoints and the corresponding quadrature controls. |
| Validation | Held-out initial conditions and inputs, prediction horizon, output-error measures, and checks of rollout stability and sensitivity to latent dimension and integration resolution. |

## Quick start

This self-contained NumPy example generates four trajectories of a three-state
linear system, fits a quadratic two-state ROM with operator inference, then
refines its bases and dynamics with NiTROM. It needs only the base installation.
Temporary trajectory files are removed after loading.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from nitrom import (
    LinearProjection,
    NitromModule,
    OpInfModule,
    ParamRegistry,
    PolynomialModel,
    TrainingData,
    TrainingPool,
    compute_pod,
    set_backend,
    solve_ivp,
    solve_opinf,
    train,
)

set_backend("numpy")
time = np.linspace(0.0, 4.0, 51)
rates = np.array([-0.3, -0.8, -1.5])
rng = np.random.default_rng(0)

with TemporaryDirectory() as directory:
    root = Path(directory)
    np.save(root / "time.npy", time)
    for j in range(4):
        x0 = rng.standard_normal(3)
        X = x0[:, None] * np.exp(rates[:, None] * time)
        np.save(root / f"traj_{j:03d}.npy", X)
        np.save(root / f"deriv_{j:03d}.npy", rates[:, None] * X)

    pool = TrainingPool(
        n_traj=4,
        fname_traj=str(root / "traj_%03d.npy"),
        fname_time=str(root / "time.npy"),
        fname_derivs=str(root / "deriv_%03d.npy"),
        dtype=np.float64,
    )

data = TrainingData(pool, which_trajs=list(range(4)))
U, _, _ = compute_pod(pool)
Phi = U[:, :2].copy()
projection = LinearProjection([Phi.copy(), Phi.copy()])
model = PolynomialModel(r=2, poly_comp=[1, 2], dtype=np.float64)

# Fit dynamics on the initial POD coordinates.
opinf = OpInfModule(data, model, projection, reg={2: 1e-6})
solve_opinf(opinf, solver="lstsq")

# NiTROM needs an output map, not the full-order governing operators.
class StateOutput:
    output_is_state = True

    def compute_output(self, q):
        return q

    def compute_output_derivative(self, q):
        return np.eye(3)


rom = NitromModule(
    data, ParamRegistry(model, projection), fom=StateOutput(), n_substeps=2
)
rom.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])
train(rom, n_epochs=20, lr=1.0, optimizer_type="lbfgs", print_every=0)

# Predict from an ambient initial condition using the trained components.
z0 = projection.encode(pool.X[0, :, 0])
Z = solve_ivp(model.evaluate_rhs, z0, time[0], time[-1], 0.04, time)
X_rom = projection.decode(Z.T).T
print(X_rom.shape)  # (3, 51)
```

This demonstrates the API on a small system. The [examples](examples/README.md)
provide forced systems, model comparisons, saved predictions, and plotting.

## Data and coordinates

[`TrainingPool`](src/nitrom/training_data.py) loads `.npy` trajectories and shards
them across ranks in a distributed run. Each trajectory file has shape `(N, nt)`;
the shared time file has shape `(nt,)`. Loaded arrays have shape
`(local_ntraj, N, nt)`.

Optional file patterns supply one scalar normalization weight, one dill-serialized
input callable `u(t)`, and one derivative array of shape `(N, nt)` per trajectory.
Missing derivatives are estimated with finite differences and cached to disk,
even when only NiTROM training is planned. The default stencil has five points.
Weights default to one; larger weights reduce a trajectory's contribution to the
loss. Specify `dtype` explicitly when needed: pools default to `float32`, while
latent models default to `float64`.

[`TrainingData`](src/nitrom/training_data.py) selects trajectories with
`which_trajs` and a fraction of their snapshots with `percent_time_length`. It
scales normalization weights so losses average over the selected trajectories
and snapshots. `num_shifts` and `shift_start_times` support multiple time windows
per physical trajectory. Inference modules can also consume per-sample
`data.time_weights` of shape `(nt,)` or `(local_ntraj, nt)` for OpInf and NiTROM.

[`compute_pod`](src/nitrom/bases/pod.py) returns `(U, singular_values, V)`.
`normalize=True` divides each trajectory by the square root of its pool weight.
[`compute_data_driven_balancing`](src/nitrom/bases/balancing.py) builds trial and
test bases from empirical reachability and observability factors; see the
[CGL example](examples/cgl/README.md) for its use with impulse-response data.

### Projections

[`LinearProjection`](src/nitrom/projections/linear_projection.py) takes two bases
`[Phi, Psi]`, each of shape `(N, r)`, with invertible $\Psi^\top\Phi$. Its default
coordinate convention is

$$
z = \Psi^\top x, \qquad
\hat{x} = \Phi(\Psi^\top\Phi)^{-1}z.
$$

With `biorthogonalize="encoder"`, the inverse instead appears in the encoder:
$z=(\Psi^\top\Phi)^{-1}\Psi^\top x$ and $\hat{x}=\Phi z$. Operators must be expressed
in the coordinates used by the projection.

[`PolynomialProjection`](src/nitrom/projections/polynomial_projection.py) adds
polynomial decoder terms in the complementary space, preserving
`encode(decode(z)) == z` up to roundoff. Projection methods take a vector `(N,)`
or `(r,)`, or a batch with the coordinate axis last: `(batch, N)` or `(batch, r)`.
Transpose a single trajectory to encode or decode it, as in the quick start.

### Latent dynamics

[`PolynomialModel`](src/nitrom/latent_space_models/polynomial_model.py) represents

$$
\dot{z} = \sum_{k\in\mathcal{K}} A_k z^{\otimes k} + Bu.
$$

`poly_comp` selects the degrees: `[1, 2]` gives linear and quadratic terms;
`[0, 1, 2]` also includes a constant. The dense degree-`k` tensor has shape
`(r,) * (k + 1)`. Operators default to zero or can be supplied through `tensors`.
For input terms, use `forcing_config={"forcing_exists": True, "m": input_dim}`
and supply input callables with the training data. An optional `"B"` entry
initializes the input operator; call `module.set_unlearnable("B")` to keep it
fixed during fitting.

[`GasPolynomialModel`](src/nitrom/latent_space_models/gas_polynomial_model.py)
parameterizes linear/quadratic operators so the **unforced** origin is globally
asymptotically stable. Its `retract_general_tensors_to_gas_tensors` method provides
a constrained initialization from ordinary operators. External forcing changes
the equilibrium behavior; the unforced guarantee does not imply convergence to
the origin under sustained input.

[`AtrPolynomialModel`](src/nitrom/latent_space_models/atr_polynomial_model.py)
adds a learned shift and constant term to obtain an attracting trapping region
for autonomous dynamics. It supports bounded behavior beyond convergence to a
fixed origin, including models intended to describe oscillations. Neither
constraint guarantees stability of a numerical rollout at an arbitrary step size.

## Inference and training

| Module | Learns | Objective |
| --- | --- | --- |
| `OpInfModule(data, model, projection, reg=...)` | Latent operators with a fixed projection | Weighted projected-derivative error plus operator penalties |
| `PolyManifoldInfModule(data, nonlin_poly_comp, Phi, Psi, reg=...)` | Nonlinear decoder tensors with fixed bases | Weighted state reconstruction error plus decoder penalties |
| `NitromModule(data, registry, fom=..., ...)` | Dynamics and projection parameters | Weighted output error of the integrated ROM plus regularization |

[`ParamRegistry`](src/nitrom/roms/param_registry.py) combines model and projection
parameters for NiTROM. Parameters with the same name in both components share one
optimization variable and accumulate contributions from both.

All inference modules expose a scalar `forward()`/`module()` and analytic
`gradient()`. [`train`](src/nitrom/optimization/train.py) supports `"adam"`,
`"sgd"`, and `"lbfgs"` on both backends and records `loss_history` and
`gradnorm_history`. NumPy uses manifold search directions with a strong-Wolfe
line search and Armijo fallback; PyTorch uses `torch.optim` with analytic gradients.
Use `set_unlearnable("B")` / `set_learnable("B")` to freeze or unfreeze a parameter,
and `set_manifold_types(...)` to select Euclidean, Grassmann, or Stiefel geometry.
NumPy training also accepts `callback(module, iteration, loss, gradnorm)` for
checkpointing; PyTorch training supports `scheduler_creator` for LR scheduling.

### Operator inference

`OpInfModule` accepts nonnegative regularization weights by polynomial degree:
`reg={2: 0.01}` penalizes the quadratic tensor and `reg={3: 0.01}` the cubic tensor.
Omitted degrees and the input operator are unpenalized; `None` or `{}` disables
regularization. Scalar weights are not accepted. Penalties use the Frobenius
norms of the physical dense operator tensors, including for constrained models.

For an ordinary `PolynomialModel`, [`solve_opinf`](src/nitrom/optimization/opinf_solver.py)
fits the operators directly and honors frozen parameters. `solver="lstsq"` solves
the augmented weighted least-squares system with SVD; its optional `rcond` sets
the relative singular-value cutoff. The default `solver="normal"` uses normal
equations. Both support NumPy, PyTorch, and distributed data; distributed `lstsq`
combines local QR factors. GAS/ATR parameterizations require iterative `train`.

### Trajectory matching

The `fom` argument to `NitromModule` supplies output evaluation through
`compute_output(q)` and its Jacobian through `compute_output_derivative(q)`.
For a constant linear output, return its `(n_outputs, N)` matrix. Alternatively,
provide `apply_output_adjoint(e, q)` to apply the output Jacobian transpose
without materializing it. Output evaluation operates on trajectory arrays of
shape `(local_ntraj, N, nt)` and returns `(local_ntraj, n_outputs, nt)`.

For `LinearProjection` and `fom.output_is_state = True`, NiTROM automatically
uses a Gram formulation of the state error that avoids storing the full
reconstruction and residual. Other output/decoder combinations use the general
path. Set `gram=False` when cancellation affects accuracy at very small errors;
`gram=True` requires the linear-decoder/state-output combination.

## Integration and adjoints

[`solve_ivp`](src/nitrom/time_steppers/integration.py) supports `rk2`, `rk4`,
`backward_euler` with Newton solves, and adaptive Dormand–Prince `rk45`. It takes
initial states `(r,)` or `(batch, r)` and returns `(r, nt)` or `(batch, r, nt)` at
`t_eval`. `step` advances once; `solve_fixed_ivp` returns a `FixedStepSolution`
with times `t`, states `X`, and optional Runge–Kutta stage states.

`NitromModule` defaults to `time_stepper="rk4"` and `adjoint_method="discrete"`.
For fixed-step training, `n_substeps` divides each measurement interval into
that many steps, including on nonuniform measurement grids. Discrete adjoints
differentiate the discretized loss. `adjoint_method="continuous"` instead uses
continuous adjoint integration and `n_leggauss` quadrature nodes per interval
for parameter-gradient integrals.

Adaptive RK45 uses `atol` and `rtol`; its adjoint reconstructs measurement intervals
and does not differentiate step acceptance or the step-size controller. Gradient
agreement is therefore limited by adaptive solver tolerances.

## Backends and parallel training

Select `set_backend("numpy")` or `set_backend("torch")` before constructing data,
models, and projections. Objects retain the backend selected at construction;
changing the default does not convert them. Models and projections also accept
an explicit `backend=` adapter. Use matching backend arrays, dtypes, and devices
across components; Torch allocations use `device=` and Torch dtypes such as
`torch.float64`. GPU training uses the same analytic-gradient interface.

For a NumPy training script using `TrainingPool`, launch trajectory parallelism
with MPI:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpiexec -n 4 python train_rom.py
```

For PyTorch, initialize the process group **before** constructing the pool:

```python
from nitrom import cleanup_distributed, set_backend, setup_distributed

set_backend("torch")
device, rank, world_size = setup_distributed()
try:
    # Construct the pool and model with device=device, then train on all ranks.
    ...
finally:
    cleanup_distributed()
```

Launch that script with `torchrun --standalone --nproc_per_node=4 train_rom.py`.
`setup_distributed` selects NCCL on CUDA or gloo on CPU. Data are sharded across
ranks, while training reduces costs and gradients and runs optimizers in
lockstep. Each rank must own at least one pool trajectory or virtual window,
and initial parameters must agree across ranks. Explicit `comm=` groups are
preserved through sharding and the associated collectives. A serial NumPy session
does not initialize MPI.

## Examples

The [examples guide](examples/README.md) covers two workflows:

- [Forced three-state ODE](examples/toymodel/README.md): POD-Galerkin, OpInf,
  GAS-OpInf, NiTROM, and GAS-NiTROM.
- [Complex Ginzburg–Landau equation](examples/cgl/README.md): POD and empirically
  balanced cubic OpInf models, followed by NiTROM refinement of each.

After installing `.[plot]`, run a small complete workflow from the repository root:

```bash
python examples/toymodel/generate_data.py --smoke
python examples/toymodel/train_models.py --smoke
python examples/toymodel/plot_results.py --smoke
```

Replace `toymodel` with `cgl` for the CGL workflow. Omit `--smoke` from **all three**
commands for the full settings. Outputs go to each example's ignored
`outputs/smoke/` or `outputs/full/`; use the same `--output-dir` across stages to
choose another location.

Matching NiTROM checkpoints continue from saved parameters automatically;
`train_models.py --fresh` restarts the fits. Optimizer state is not restored.
Plotting reads saved predictions and works without LaTeX; `--tex` enables
publication styling. The smoke profiles exercise every model and figure but do
not establish full-profile scientific accuracy.

## Development and tests

Install `.[dev,plot]` for CPU development and the example workflows. Add `torch`
for backend agreement checks and `mpi` for MPI tests.

```bash
python -m pytest -q                              # default CPU checks
python -m pytest -q tests/examples               # tutorial workflow checks
python -m pytest -q -m 'not distributed and not gpu'  # include slow CPU checks
python -m pytest -q -m distributed               # MPI/Torch communication tests
python -m pytest -q -m gpu                       # CUDA checks
python -m pytest -q -m ''                        # all applicable checks
ruff check src tests
ruff format --check src tests
```

`slow`, `distributed`, and `gpu` tests are excluded by default. Optional test
requirements depend on the suite; CUDA and communication checks need their
corresponding hardware/runtime. Pytest imports this checkout's `src` tree.
Parallel test execution is opt-in with `python -m pytest -n auto`; limit BLAS
threads when using many workers.

## License

MIT; see [LICENSE](LICENSE).
