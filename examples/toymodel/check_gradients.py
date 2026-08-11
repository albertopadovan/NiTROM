import os
import sys
import pickle
import numpy as np

# Resolve paths so we can run this from any directory
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, "../../"))
sys.path.insert(0, os.path.join(repo_root, "src"))
sys.path.append(script_dir)

from nitrom.backend import set_backend
set_backend("numpy")

import fom_class
from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import NitromModule
from nitrom.projections.linear_projection import LinearProjection
from nitrom.roms.param_registry import ParamRegistry
from nitrom.training_data import TrainingData, TrainingPool
from nitrom.utils import compute_POD

dtype = np.float64
traj_path = os.path.join(script_dir, "trajectories/")
n_traj = 4
r = 2  # reduced dimension
poly_comp = [1, 2]

# %% Load the trajectories into a TrainingPool
pool = TrainingPool(
    n_traj=n_traj,
    fname_traj=os.path.join(traj_path, "traj_%03d.npy"),
    fname_time=os.path.join(traj_path, "time.npy"),
    dtype=dtype,
    fname_weights=os.path.join(traj_path, "weight_%03d.npy"),
    fname_forcing=os.path.join(traj_path, "forcing_%03d.pkl"),
    fname_derivs=os.path.join(traj_path, "deriv_%03d.npy"),
)

training_data = TrainingData(
    pool,
    which_trajs=list(range(n_traj)),
    percent_time_length=1.0,
    leggauss_deg=5,
    nsave_rom=15,
)

# %% Setup the FOM
beta = 20.0
A2 = np.diag(np.array([-1.0, -2.0, -5.0], dtype=dtype))
A3 = np.zeros((3, 3, 3), dtype=dtype)
A3[:, :, -1] = np.diag(np.array([beta, beta, 0.0], dtype=dtype))
B_op = np.ones((3, 1), dtype=dtype)
C = np.ones((1, 3), dtype=dtype)
fom = fom_class.full_order_model(A2, A3, B_op, C, dtype=dtype)

# Compute POD basis and projections
U, _, _ = compute_POD(pool, normalize=True)
Phi = U[:, :r]  # (N, r)
B_fom = np.ones((Phi.shape[0], 1), dtype=dtype)
B_r = Phi.T @ B_fom
forcing_config = {"forcing_exists": True, "B": B_r, "m": B_fom.shape[1]}

(A2r, A3r), (Br, _) = fom.assemble_petrov_galerkin_tensors(Phi, Phi)


def numpy_finite_diff_grad(module, eps=1e-6):
    """Central finite-difference gradient of module w.r.t. each parameter."""
    fd = []
    # Store original parameters
    orig_params = {}
    for name in module.param_names:
        orig_params[name] = np.copy(getattr(module, name))
    
    for name in module.param_names:
        p = orig_params[name]
        g = np.zeros_like(p)
        flat_g = g.ravel()
        
        # We perturb each element of the parameter
        for i in range(p.size):
            # Pos perturbation
            p_pos = np.copy(p)
            p_pos.ravel()[i] += eps
            setattr(module, name, p_pos)
            loss_pos = float(module())
            
            # Neg perturbation
            p_neg = np.copy(p)
            p_neg.ravel()[i] -= eps
            setattr(module, name, p_neg)
            loss_neg = float(module())
            
            # Set back to original
            setattr(module, name, p)
            
            # Central difference
            flat_g[i] = (loss_pos - loss_neg) / (2.0 * eps)
            
        # Zero out the finite difference gradient if the parameter is not learnable
        if not module.is_learnable[name]:
            g.fill(0.0)
            
        fd.append(g)
        
    return fd


def run_gradient_check(module_name, module, adjoint_method, rtol=1e-4, atol=1e-6):
    print(f"\n--- Checking {module_name} with {adjoint_method} adjoint ---")
    
    # Compute analytic gradient
    grad_analytic = module.gradient()
    
    # Compute finite difference gradient
    grad_numeric = numpy_finite_diff_grad(module, eps=1e-6)
    
    all_passed = True
    for name, a, n in zip(module.param_names, grad_analytic, grad_numeric, strict=True):
        max_diff = np.max(np.abs(a - n))
        a_norm = np.linalg.norm(a)
        n_norm = np.linalg.norm(n)
        rel_diff = np.linalg.norm(a - n) / (n_norm + 1e-15)
        
        # Check if they are close
        passed = np.allclose(a, n, rtol=rtol, atol=atol)
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
            
        print(f"  Parameter '{name}': {status}")
        print(f"    Analytic Norm: {a_norm:.6e}, Numeric Norm: {n_norm:.6e}")
        print(f"    Max Abs Diff:  {max_diff:.6e}, Rel Diff:      {rel_diff:.6e}")
        
    return all_passed


def main():
    success = True
    
    # Check standard NiTROM
    print("================== STANDARD NiTROM GRADIENT CHECK ==================")
    nitrom_ckpt_found = False
    
    # Try to load discrete or continuous adjoint solution
    nitrom_ckpt_path = os.path.join(script_dir, "models_discrete_adjoint", "nitrom_model.pkl")
    if not os.path.exists(nitrom_ckpt_path):
        nitrom_ckpt_path = os.path.join(script_dir, "models_continuous_adjoint", "nitrom_model.pkl")
        
    if os.path.exists(nitrom_ckpt_path):
        print(f"Loading trained NiTROM model from: {nitrom_ckpt_path}")
        with open(nitrom_ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        init_Phi = np.asarray(ckpt["Phi"], dtype=dtype)
        init_Psi = np.asarray(ckpt.get("Psi", ckpt["Phi"]), dtype=dtype)
        tensors = [np.asarray(t, dtype=dtype) for t in ckpt["tensors"]]
        nitrom_model = PolynomialModel(
            ckpt["r"], ckpt["poly_comp"], dtype=dtype, forcing_config=ckpt["forcing_config"], tensors=tensors
        )
        projection = LinearProjection([init_Phi, init_Psi])
        nitrom_ckpt_found = True
    else:
        print("Trained NiTROM model not found. Initializing from Galerkin projection...")
        nitrom_model = PolynomialModel(
            r, poly_comp, dtype=dtype, forcing_config=forcing_config, tensors=(A2r, A3r, Br)
        )
        projection = LinearProjection([Phi, Phi])
        
    # Check for both adjoint methods
    for method in ["discrete", "continuous"]:
        registry = ParamRegistry(nitrom_model, projection)
        nitrom_module = NitromModule(
            training_data, registry, fom=fom, n_substeps=15, adjoint_method=method
        )
        nitrom_module.set_unlearnable("B")
        nitrom_module.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])
        
        # Continuous adjoint may have slightly larger discrepancy due to discretization mismatch
        atol = 1e-5 if method == "continuous" else 1e-6
        rtol = 1e-3 if method == "continuous" else 1e-4
        
        passed = run_gradient_check("NiTROM", nitrom_module, method, rtol=rtol, atol=atol)
        if not passed:
            success = False

    # Check GAS-NiTROM
    print("\n================== GAS-NiTROM GRADIENT CHECK ==================")
    gas_nitrom_ckpt_found = False
    
    # Try to load discrete or continuous adjoint solution
    gas_ckpt_path = os.path.join(script_dir, "models_discrete_adjoint", "gas_nitrom_model.pkl")
    if not os.path.exists(gas_ckpt_path):
        gas_ckpt_path = os.path.join(script_dir, "models_continuous_adjoint", "gas_nitrom_model.pkl")
        
    if os.path.exists(gas_ckpt_path):
        print(f"Loading trained GAS-NiTROM model from: {gas_ckpt_path}")
        with open(gas_ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        init_Phi = np.asarray(ckpt["Phi"], dtype=dtype)
        init_Psi = np.asarray(ckpt.get("Psi", ckpt["Phi"]), dtype=dtype)
        gas_params = [np.asarray(t, dtype=dtype) for t in ckpt["gas_params"]]
        gas_model = GasPolynomialModel(
            ckpt["r"], ckpt["poly_comp"], dtype=dtype, gas_params=gas_params, forcing_config=ckpt["forcing_config"]
        )
        projection_gas = LinearProjection([init_Phi, init_Psi])
        gas_nitrom_ckpt_found = True
    else:
        print("Trained GAS-NiTROM model not found. Initializing from Galerkin retraction...")
        seed = GasPolynomialModel(r, poly_comp, dtype=dtype)
        seed.retract_general_tensors_to_gas_tensors([A2r, A3r], optimize_F=True, F_cond_penalty=1e-2)
        gas_init = [*seed.get_params(), np.copy(Br)]
        gas_model = GasPolynomialModel(
            r, poly_comp, dtype=dtype, gas_params=gas_init, forcing_config=forcing_config
        )
        projection_gas = LinearProjection([Phi, Phi])

    # Check for both adjoint methods
    for method in ["discrete", "continuous"]:
        registry_gas = ParamRegistry(gas_model, projection_gas)
        gas_nitrom_module = NitromModule(
            training_data, registry_gas, fom=fom, n_substeps=15, adjoint_method=method
        )
        gas_nitrom_module.set_unlearnable("B")
        gas_nitrom_module.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])
        
        atol = 1e-5 if method == "continuous" else 1e-6
        rtol = 1e-3 if method == "continuous" else 1e-4
        
        passed = run_gradient_check("GAS-NiTROM", gas_nitrom_module, method, rtol=rtol, atol=atol)
        if not passed:
            success = False

    if success:
        print("\n>>> All gradient checks passed successfully! <<<")
        sys.exit(0)
    else:
        print("\n>>> Some gradient checks failed! <<<")
        sys.exit(1)


if __name__ == "__main__":
    main()
