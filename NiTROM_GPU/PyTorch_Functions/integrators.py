import torch
import time

def rk4_step(fun, t, x, dt, args=()):
    """
    Performs a single integration step using the fourth-order Runge-Kutta (RK4) method.
    """
    
    k1 = fun(t, x, *args)
    k2 = fun(t + 0.5 * dt, x + 0.5 * dt * k1, *args)
    k3 = fun(t + 0.5 * dt, x + 0.5 * dt * k2, *args)
    k4 = fun(t + dt, x + dt * k3, *args)
    x_new = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_new


def my_rk4_adaptive(fun, t_vec, x0, args=(), *, atol=1e-6, rtol=1e-3, safety_factor=0.8, fac_min=0.1, fac_max=5.0):
    """
    Integrates a system of ordinary differential equations using an adaptive fourth-order Runge-Kutta (RK4) method.
    Compatible with pyTorch tensors.

    Args:
        fun (callable): The function that computes the derivatives, with signature `fun(t, x, *args)`.
        t_vec (torch.Tensor): 1D tensor or array of time points at which to solve for the state.
        x0 (torch.Tensor): Initial state vector.
        args (tuple, optional): Additional arguments to pass to `fun`.

    Returns:
        xs: A tensor containing the state at each time point in `t_vec`. Each column corresponds to the state at a time step.
    """

    x = x0.clone().detach()
    dt = (t_vec[1] - t_vec[0])/10
    xs = torch.zeros((len(x0), len(t_vec)), dtype=x0.dtype, device=x0.device)
    xs[:, 0] = x0
    t = t_vec[0]
    for i, T in enumerate(t_vec[1:], start=1):
        while t < T:
            dt_trial = min(dt, T - t)
            x_full = rk4_step(fun, t, x, dt_trial, args)
            x_half1 = rk4_step(fun, t, x, dt_trial / 2, args)
            x_half2 = rk4_step(fun, t + dt_trial / 2, x_half1, dt_trial / 2, args)

            dx = x_half2 - x_full
            scale = atol + rtol * torch.max(torch.abs(x_full), torch.abs(x_half2))
            err_vec = torch.abs(dx)/scale
            err = torch.max(err_vec)

            if err <= 1.0:
                t = t + dt_trial
                x = x_half2
            
            exponent = 1.0/(4.0+1.0)
            dt_new = dt_trial * safety_factor * (1.0/err)**exponent
            dt = torch.clamp(dt_new, min=dt*fac_min, max=dt*fac_max)
            dt = (t_vec[1] - t_vec[0])/10
        xs[:, i] = x

    return xs


def my_rk4(fun, t_vec, x0, args=()):
    """
    Integrates a system of ordinary differential equations using a fourth-order Runge-Kutta (RK4) method.
    Compatible with pyTorch tensors.

    Args:
        fun (callable): The function that computes the derivatives, with signature `fun(t, x, *args)`.
        t_vec (torch.Tensor): 1D tensor or array of time points at which to solve for the state.
        x0 (torch.Tensor): Initial state vector.
        args (tuple, optional): Additional arguments to pass to `fun`.

    Returns:
        xs: A tensor containing the state at each time point in `t_vec`. Each column corresponds to the state at a time step.
    """

    x = x0.clone()
    dt = (t_vec[1] - t_vec[0])/100
    xs = torch.zeros((len(x0), len(t_vec)), dtype=x0.dtype, device=x0.device)
    xs[:, 0] = x0
    t = t_vec[0].clone()
    for i, T in enumerate(t_vec[1:], start=1):
        while t < T:
            dt_trial = min(dt, T - t)
            x = rk4_step(fun, t, x, dt_trial, args)
            t += dt_trial
        xs[:, i] = x

    return xs


def etdrk4_setup(linop, dt):
    """
    Prepares the ETDRK4 matrices for a given linear operator and time step.
    
    Args:
        linop (tuple): A tuple containing the eigenvalues and eigenvectors of the linear operator.
        dt (float): The time step for the integration.
    
    Returns:
        E, E2, phi, L_inv3, L_sq: Matrices used in the ETDRK4 method.
    """

    n = linop[0].shape[0]  # Assuming linop is a tuple (V, D, V_inv)
    
    V, D, V_inv = linop
    L = V @ torch.diag(D) @ V_inv
    E = V @ torch.diag(torch.exp(D * dt)) @ V_inv
    E2 = V @ torch.diag(torch.exp(D * dt/2)) @ V_inv
    phi = V @ torch.diag(D**(-1) * (torch.exp(D * dt/2) - 1)) @ V_inv
    L_inv3 = V @ torch.diag(D**(-3)) @ V_inv
    L_sq = V @ torch.diag(D**2) @ V_inv

    I = torch.eye(n, device=V.device, dtype=V.dtype)

    coef1 = -4*I - L*dt + E @ (4*I - 3*L*dt + L_sq * dt**2)
    coef2 = 2 * (2*I + L*dt + E @ (-2*I + L*dt))
    coef3 = -4*I - 3*L*dt - L_sq * dt**2 + E @ (4*I - L*dt)
    
    return E, E2, phi, L_inv3, coef1, coef2, coef3


def my_etdrk4(etdrk4_coefs, fun_nonlinear, t_vec, x0, args=()):
    internal_steps = 1
    dtype = torch.complex128
    x0 = x0.to(dtype)
    args = tuple(arg.to(dtype) if isinstance(arg, torch.Tensor) else arg for arg in args)

    n = len(x0)
    dt = (t_vec[1] - t_vec[0]) / internal_steps

    E, E2, phi, L_inv3, coef1, coef2, coef3 = etdrk4_coefs

    n_outputs = len(t_vec)
    n_steps = (n_outputs - 1) * internal_steps
    t_vec_internal = torch.arange(n_steps+1, device=x0.device) * dt + t_vec[0]
    xs = torch.zeros((n, n_outputs), device=x0.device, dtype=x0.dtype)
    xs[:, 0] = x0
    x = x0.clone().detach()

    for m in range(1, n_steps + 1):
        if torch.linalg.vector_norm(x) > 1e6:
            if m % internal_steps == 0:
                idx = m // internal_steps
                xs[:, idx] = x.real
        else:
            t = t_vec_internal[m-1]
            N1 = fun_nonlinear(t, x, *args)
            an = E2 @ x + phi @ N1
            N2 = fun_nonlinear(t + dt/2, an, *args)
            bn = E2 @ x + phi @ N2
            N3 = fun_nonlinear(t + dt/2, bn, *args)
            cn = E2 @ an + phi @ (2 * N3 - N1)
            N4 = fun_nonlinear(t + dt, cn, *args)

            x = E @ x + dt**(-2) * L_inv3 @ (coef1 @ N1 + coef2 @ (N2 + N3) + coef3 @ N4)
            if m % internal_steps == 0:
                idx = m // internal_steps
                xs[:, idx] = x.real

    return xs.real


def my_cnab2(linop, fun_nonlinear, t_vec, x0, args=()):
    dt = (t_vec[1] - t_vec[0])/70
    dt_half = 0.5 * dt
    x_curr = x0.clone().detach()
    t_curr = t_vec[0].clone().detach()
    N_prev = fun_nonlinear(t_curr, x_curr, *args)

    I = torch.eye(len(x0), dtype=x0.dtype, device=x0.device)
    A = I - dt_half * linop
    B = I + dt_half * linop
    LU, pivots, _ = torch.linalg.lu_factor_ex(A)

    xs = torch.zeros((len(x0), len(t_vec)), dtype=x0.dtype, device=x0.device)
    xs[:, 0] = x_curr

    for i, t_next in enumerate(t_vec[1:], start=1):
        while t_curr < t_next:
            dt_trial = min(dt, t_next - t_curr)
            t_next2 = t_curr + dt_trial
            N_curr = fun_nonlinear(t_curr, x_curr, *args)
            rhs = B @ x_curr + dt_half * (3 * N_curr - N_prev)
            rhs = rhs.unsqueeze(-1)
            x_next = torch.linalg.lu_solve(LU, pivots, rhs).squeeze(-1)

            t_curr = t_next2
            x_curr = x_next
            N_prev = N_curr
        xs[:, i] = x_curr

    return xs


def my_bdf4_ab4(linop, fun_nonlinear, t_vec, x0, args=()):
    dt = (t_vec[1] - t_vec[0])
    t_curr = t_vec[0].clone().detach()

    I = torch.eye(len(x0), dtype=x0.dtype, device=x0.device)
    A = 25 * I - 12 * dt * linop
    LU, pivots = torch.linalg.lu_factor(A)

    xs = torch.zeros((len(x0), len(t_vec)), dtype=x0.dtype, device=x0.device)
    xs[:, 0] = x0.clone().detach()
    x_hist = [x0.clone().detach()]
    N_hist = [fun_nonlinear(t_curr, x0, *args)]
    

    for i, t_target in enumerate(t_vec[1:], start=1):
        n = len(x_hist) - 1
        i0 = n; i1 = max(n-1, 0); i2 = max(n-2, 0); i3 = max(n-3, 0)
        xn0 = x_hist[i0]; xn1 = x_hist[i1]; xn2 = x_hist[i2]; xn3 = x_hist[i3]
        Nn0 = N_hist[i0]; Nn1 = N_hist[i1]; Nn2 = N_hist[i2]; Nn3 = N_hist[i3]
        rhs = 48*xn0 - 36*xn1 + 16*xn2 - 3*xn3 + dt * (48*Nn0 - 72*Nn1 + 48*Nn2 - 12*Nn3)
        x_next = torch.linalg.lu_solve(LU, pivots, rhs.unsqueeze(-1)).squeeze(-1)
        N_next = fun_nonlinear(t_target, x_next, *args)

        x_hist.append(x_next.clone().detach())
        N_hist.append(N_next.clone().detach())
        xs[:, i] = x_hist[-1].clone().detach()

    return xs