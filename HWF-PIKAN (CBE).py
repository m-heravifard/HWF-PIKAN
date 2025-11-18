#!/usr/bin/env python3
"""
Authors: Mohammad E. Heravifard & Prof. Kazem Hejranfar
HWF-PIKAN-Spline PINN for the collisionless Boltzmann equation (no force term):

    ∂f/∂t + v * ∂f/∂x = 0

with Gaussian initial condition in (x, v):

    f(0, x, v) = exp( - (x^2 + v^2) / sigma^2 )

Domain:
    t ∈ [0, T]
    x ∈ [-Lx, Lx]
    v ∈ [-Lv, Lv]

Periodic boundary condition in x:
    f(t, -Lx, v) = f(t, +Lx, v)

Architecture:
    - Characteristic coordinate: y = x - v t (wrapped into [-Lx, Lx]),
      so f(t,x,v) = Net(y,v).
    - Hybrid Wavelet–Fourier (HWF) embedding on (y,v)
      (Fourier sin/cos + multiscale Ricker wavelets + LayerNorm).
    - KAN/B-spline core (KA): inner scalar maps -> s_q ∈ [0,1],
      univariate phi_q(s_q) represented by cubic B-splines,
      outer linear combination yields f.
    - Physics-informed losses: PDE residual, IC, periodic BCs.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ======================
# Configuration
# ======================

class Config:
    # Domain
    T = 1.0
    Lx = 1.0
    Lv = 1.0

    # Initial condition width
    sigma = 0.25

    # Training set sizes
    N_int = 30000   # interior collocation points for PDE
    N_ic  = 8000    # initial condition points (t=0)
    N_bc  = 8000    # boundary condition points (periodic in x)

    # HWF embedding hyperparams for (y,v)
    M_fourier = 32      # number of Fourier frequency vectors
    sigma_B   = 1.0     # std for frequency sampling
    J_wavelet = 3       # number of wavelet scales
    train_omega = False # whether to train Fourier frequencies

    # KAN / B-spline core hyperparams
    Q = 16              # number of Kolmogorov inner terms
    bspline_degree = 3  # cubic B-splines
    n_control = 25      # control points per spline

    # PINN scheduling
    lr = 5e-4

    # Stage 1: IC-only training
    n_iters_ic = 1000

    # Stage 2: full PINN (IC + BC + PDE) with Adam
    n_iters_adam = 1000
    print_every = 100

    # Stage 3: LBFGS refinement
    use_lbfgs = True
    lbfgs_max_iter = 20      # max internal LBFGS iterations per epoch
    lbfgs_epochs = 50        # LBFGS outer epochs
    lbfgs_print_every = 5

    # Loss weights
    w_pde = 0.1   # small, PDE is almost satisfied by characteristic ansatz
    w_ic  = 50.0  # strong IC
    w_bc  = 1.0   # periodic BC


cfg = Config()

# (Optional) fixed seed for more stable behaviour
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# Utility: Initial Condition & Analytic Solution
# ======================

def gaussian_ic(x, v, sigma):
    """
    Gaussian initial condition in (x, v):

    f(0, x, v) = exp( - (x^2 + v^2) / sigma^2 )
    """
    return torch.exp(-(x**2 + v**2) / sigma**2)


def analytic_solution_periodic(t, x, v, sigma, Lx):
    """
    Analytic solution with periodicity in x:

        f(t, x, v) = exp( - [ (x0)^2 + v^2 ] / sigma^2 )

    where x0 = (x - v t) wrapped back into [-Lx, Lx].
    t: scalar (float)
    x, v: numpy arrays
    """
    x_phys = x - v * t
    L = 2.0 * Lx
    x_wrap = ((x_phys + Lx) % L) - Lx
    return np.exp(-((x_wrap**2 + v**2) / sigma**2))


# ======================
# B-spline basis (Cox–de Boor) for KAN core
# ======================

def make_open_uniform_knots(num_control: int, degree: int, device=device):
    """
    Open uniform knot vector on [0,1] for B-splines.
    """
    n_knots = num_control + degree + 1
    if n_knots <= 2 * (degree + 1):
        knots = torch.linspace(0.0, 1.0, steps=n_knots, device=device)
    else:
        n_internal = n_knots - 2 * (degree + 1)
        internal = torch.linspace(0.0, 1.0, steps=n_internal + 2,
                                  device=device)[1:-1]
        knots = torch.cat([
            torch.zeros(degree + 1, device=device),
            internal,
            torch.ones(degree + 1, device=device)
        ])
    return knots


def bspline_basis_batch(x: torch.Tensor, degree: int, knots: torch.Tensor):
    """
    Evaluate all B-spline basis functions N_i^degree(x) for x ∈ [0,1].
    x: (M,)   ->   returns (M, n_basis)
    """
    x = x.clamp(min=knots[0].item(), max=knots[-1].item())
    n_knots = knots.shape[0]
    n_basis = n_knots - degree - 1
    M = x.shape[0]

    N = torch.zeros((M, n_basis), dtype=x.dtype, device=x.device)
    for i in range(n_basis):
        left = knots[i]
        right = knots[i + 1]
        if i == n_basis - 1:
            mask = (x >= left) & (x <= right)
        else:
            mask = (x >= left) & (x < right)
        N[:, i] = mask.to(dtype=x.dtype)

    for p in range(1, degree + 1):
        Np = torch.zeros_like(N)
        for i in range(n_basis):
            denom1 = knots[i + p] - knots[i]
            denom2 = knots[i + p + 1] - knots[i + 1]

            term1 = torch.zeros(M, dtype=x.dtype, device=x.device)
            term2 = torch.zeros(M, dtype=x.dtype, device=x.device)

            if denom1 != 0:
                coeff1 = (x - knots[i]) / denom1
                term1 = coeff1 * N[:, i]

            if i + 1 < n_basis and denom2 != 0:
                coeff2 = (knots[i + p + 1] - x) / denom2
                term2 = coeff2 * N[:, i + 1]

            Np[:, i] = term1 + term2
        N = Np
    return N  # (M, n_basis)


# ======================
# Wavelet features (Ricker / Mexican hat) 1D
# ======================

def ricker_wavelet(s: torch.Tensor, scale: float):
    """
    1D Ricker (Mexican-hat) wavelet at scale.
    s: (N, K), scale: float
    """
    u = s / scale
    return (1.0 - u * u) * torch.exp(-0.5 * u * u)


class HWF_Basis1D(nn.Module):
    """
    1D Hybrid Wavelet-Fourier basis on normalized input u ∈ [0,1].

    Fourier part:
        sin(2π k u), cos(2π k u), k = 1..n_fourier  → 2*n_fourier features

    Wavelet part (Ricker):
        For scales j=0..J-1:
            num_k = 2^j centers c_{j,k} in [0,1]
            ψ_{j,k}(u) = Ricker((u - c_{j,k})/scale_j),
        where scale_j ~ 0.5/2^j.

    Total features = 2*n_fourier + n_wavelet,
    where n_wavelet = sum_j 2^j = 2^J - 1.
    """
    def __init__(self, n_fourier: int, J_wavelet: int):
        super().__init__()
        self.n_fourier = n_fourier
        self.J_wavelet = J_wavelet

        # Fourier frequencies
        if n_fourier > 0:
            freqs = torch.arange(1, n_fourier + 1, dtype=torch.float32)
        else:
            freqs = torch.zeros(1, dtype=torch.float32)
        self.register_buffer("freqs", freqs)

        # Precompute wavelet centers and scales
        centers_list = []
        scales_list = []
        for j in range(J_wavelet):
            num_k = 2 ** j
            centers = torch.linspace(0.0, 1.0, num_k + 2)[1:-1]  # avoid exact 0,1
            scale = max(0.4 / num_k, 1e-3)
            centers_list.append(centers)
            scales_list.append(torch.full_like(centers, scale))
        if centers_list:
            all_centers = torch.cat(centers_list)
            all_scales  = torch.cat(scales_list)
        else:
            all_centers = torch.zeros(1, dtype=torch.float32)
            all_scales  = torch.ones(1, dtype=torch.float32)

        self.register_buffer("wavelet_centers", all_centers)  # (n_wavelet,)
        self.register_buffer("wavelet_scales",  all_scales)   # (n_wavelet,)

        self.n_wavelet = 0 if J_wavelet == 0 else all_centers.shape[0]
        self.inner_width = 2 * n_fourier + self.n_wavelet

        if self.inner_width == 0:
            raise ValueError("HWF_Basis1D: need at least one feature.")

    def forward(self, u):
        """
        u: (N,1) or (N,) in [0,1]
        returns: (N, inner_width)
        """
        u = u.view(-1, 1)  # (N,1)
        feats = []

        # Fourier part
        if self.n_fourier > 0:
            freqs = self.freqs.view(1, -1)  # (1, n_fourier)
            arg = 2.0 * np.pi * (u @ freqs)  # (N, n_fourier)
            sinp = torch.sin(arg)
            cosp = torch.cos(arg)
            scale = 1.0 / (np.sqrt(self.n_fourier) + 1e-12)
            feats.append(sinp * scale)
            feats.append(cosp * scale)

        # Wavelet part
        if self.J_wavelet > 0 and self.n_wavelet > 0:
            c = self.wavelet_centers.view(1, -1)  # (1, n_wavelet)
            s = self.wavelet_scales.view(1, -1)   # (1, n_wavelet)
            du = (u - c) / s                      # (N, n_wavelet)
            psi = ricker_wavelet(du, 1.0)         # scale already in du
            feats.append(psi)

        return torch.cat(feats, dim=1)  # (N, inner_width)


# ======================
# KA-BSpline Core (Kolmogorov–Arnold B-spline network)
# ======================

class KA_BSpline_Core(nn.Module):
    """
    KAN/B-spline core on top of an embedding.

    emb: (N, D_emb)
    Steps:
        1) u_inner = emb W^T + b_inner  → (N, Q)
        2) s = 0.5 (tanh(inner_scale * u_inner) + 1) in [0,1]
        3) For each q: phi_q(s_q) = sum_i coeffs[q,i] B_i(s_q)
        4) u_out = sum_q a_q phi_q + b_out  → scalar f.
    """
    def __init__(self, input_dim: int, Q: int, degree: int, n_control: int):
        super().__init__()
        self.Q = Q
        self.degree = degree
        self.n_control = n_control

        # inner linear maps (small init)
        self.W = nn.Parameter(torch.randn(Q, input_dim) * 0.05)
        self.b_inner = nn.Parameter(torch.zeros(Q))
        self.inner_scale = nn.Parameter(torch.tensor(0.5))

        # spline coefficients per q
        self.coeffs = nn.Parameter(torch.randn(Q, n_control) * 0.05)

        # outer scalars
        self.a = nn.Parameter(torch.randn(Q, 1) * 0.05)
        self.b_out = nn.Parameter(torch.zeros(1))

        knots = make_open_uniform_knots(n_control, degree, device=device)
        self.register_buffer("knots", knots)

    def forward(self, emb: torch.Tensor):
        N = emb.shape[0]
        u_inner = emb @ self.W.t() + self.b_inner  # (N, Q)
        s = 0.5 * (torch.tanh(self.inner_scale * u_inner) + 1.0)  # (N, Q) in (0,1)

        s_flat = s.reshape(-1)  # (N*Q,)
        B_flat = bspline_basis_batch(s_flat, self.degree, self.knots)  # (N*Q, n_control)
        B = B_flat.reshape(N, self.Q, self.n_control)  # (N, Q, n_control)

        coeffs = self.coeffs.unsqueeze(0).expand(N, -1, -1)  # (N, Q, n_control)
        phi = (B * coeffs).sum(dim=-1)  # (N, Q)

        a = self.a.view(1, self.Q)
        u_out = (phi * a).sum(dim=1, keepdim=True) + self.b_out  # (N,1)
        return u_out


# ======================
# HWF-PIKAN model with characteristic coordinate y = x - v t
# ======================

class HWF_PIKAN_Spline_CBE(nn.Module):
    """
    HWF-PIKAN-Spline for CBE:

        f(t,x,v) = Net(y,v),
        y = x - v t (wrapped into [-Lx,Lx])

    - Input physical (t,x,v).
    - Transform to (y,v).
    - Normalize (y,v) to [0,1]^2.
    - HWF_Basis1D on y and v separately -> concatenated embedding.
    - KA_BSpline_Core on the embedding -> f.
    """
    def __init__(self, cfg):
        super().__init__()
        self.Lx = cfg.Lx
        self.Lv = cfg.Lv

        self.n_fourier = cfg.M_fourier
        self.J_wavelet = cfg.J_wavelet
        self.Q = cfg.Q
        self.degree = cfg.bspline_degree
        self.n_control = cfg.n_control

        # HWF bases for y and v
        self.basis_y = HWF_Basis1D(self.n_fourier, self.J_wavelet)
        self.basis_v = HWF_Basis1D(self.n_fourier, self.J_wavelet)

        emb_dim = self.basis_y.inner_width + self.basis_v.inner_width

        # LayerNorm on combined embedding
        self.norm = nn.LayerNorm(emb_dim)

        # KAN/B-spline core
        self.core = KA_BSpline_Core(emb_dim, self.Q, self.degree, self.n_control)

    def _wrap_y(self, y):
        """
        Wrap y into [-Lx, Lx] periodically.
        """
        Lx = self.Lx
        L = 2.0 * Lx
        return ((y + Lx) % L) - Lx

    def _normalize_yv(self, y, v):
        """
        Normalize y,v to [0,1]:
            y ∈ [-Lx, Lx] → u_y = (y + Lx)/(2Lx)
            v ∈ [-Lv, Lv] → u_v = (v + Lv)/(2Lv)
        """
        u_y = (y + self.Lx) / (2.0 * self.Lx)
        u_v = (v + self.Lv) / (2.0 * self.Lv)
        return u_y, u_v

    def forward(self, x):
        """
        x: (N,3) with columns (t, x, v)
        returns: (N,1) f(t,x,v)
        """
        t = x[:, 0:1]
        xpos = x[:, 1:2]
        v = x[:, 2:3]

        # characteristic coordinate
        y_raw = xpos - v * t
        y = self._wrap_y(y_raw)

        # normalize
        u_y, u_v = self._normalize_yv(y, v)

        # HWF embeddings
        Phi_y = self.basis_y(u_y)  # (N, Ny)
        Phi_v = self.basis_v(u_v)  # (N, Nv)
        emb = torch.cat([Phi_y, Phi_v], dim=1)  # (N, Ny+Nv)

        emb = self.norm(emb)

        out = self.core(emb)      # (N,1)
        return out


model = HWF_PIKAN_Spline_CBE(cfg).to(device)

# ======================
# Training Data
# ======================

def sample_training_points(cfg, device):
    # Interior points
    t_int = torch.rand(cfg.N_int, 1) * cfg.T
    x_int = (torch.rand(cfg.N_int, 1) * 2.0 - 1.0) * cfg.Lx
    v_int = (torch.rand(cfg.N_int, 1) * 2.0 - 1.0) * cfg.Lv

    # IC points: t=0
    t_ic = torch.zeros(cfg.N_ic, 1)
    x_ic = (torch.rand(cfg.N_ic, 1) * 2.0 - 1.0) * cfg.Lx
    v_ic = (torch.rand(cfg.N_ic, 1) * 2.0 - 1.0) * cfg.Lv

    # Periodic BC in x
    t_bc = torch.rand(cfg.N_bc, 1) * cfg.T
    v_bc = (torch.rand(cfg.N_bc, 1) * 2.0 - 1.0) * cfg.Lv
    x_bc_left  = -cfg.Lx * torch.ones_like(t_bc)
    x_bc_right =  cfg.Lx * torch.ones_like(t_bc)

    data = {
        "t_int": t_int.to(device),
        "x_int": x_int.to(device),
        "v_int": v_int.to(device),

        "t_ic": t_ic.to(device),
        "x_ic": x_ic.to(device),
        "v_ic": v_ic.to(device),

        "t_bc": t_bc.to(device),
        "x_bc_left": x_bc_left.to(device),
        "x_bc_right": x_bc_right.to(device),
        "v_bc": v_bc.to(device),
    }
    return data


data = sample_training_points(cfg, device)

# ======================
# PINN Loss Construction
# ======================

mse_loss = nn.MSELoss()

def pde_residual(model, t, x, v):
    """
    Compute PDE residual for:
        f_t + v * f_x = 0
    """
    t_ = t.clone().detach().requires_grad_(True)
    x_ = x.clone().detach().requires_grad_(True)
    v_ = v.clone().detach()  # v is parameter here

    inp = torch.cat([t_, x_, v_], dim=1)
    f = model(inp)

    grads = torch.autograd.grad(
        outputs=f,
        inputs=(t_, x_),
        grad_outputs=torch.ones_like(f),
        create_graph=True,
        retain_graph=True,
    )
    f_t = grads[0]
    f_x = grads[1]

    res = f_t + v_ * f_x
    return res


def pinn_loss(model, data, cfg):
    # PDE interior loss
    res_int = pde_residual(model, data["t_int"], data["x_int"], data["v_int"])
    loss_pde = mse_loss(res_int, torch.zeros_like(res_int))

    # Initial condition loss
    t_ic = data["t_ic"]
    x_ic = data["x_ic"]
    v_ic = data["v_ic"]

    inp_ic = torch.cat([t_ic, x_ic, v_ic], dim=1)
    f_ic_pred = model(inp_ic)
    f_ic_true = gaussian_ic(x_ic, v_ic, cfg.sigma)
    loss_ic = mse_loss(f_ic_pred, f_ic_true)

    # Periodic boundary in x
    t_bc = data["t_bc"]
    v_bc = data["v_bc"]
    x_bc_left  = data["x_bc_left"]
    x_bc_right = data["x_bc_right"]

    inp_left  = torch.cat([t_bc, x_bc_left,  v_bc], dim=1)
    inp_right = torch.cat([t_bc, x_bc_right, v_bc], dim=1)

    f_left  = model(inp_left)
    f_right = model(inp_right)
    loss_bc = mse_loss(f_left, f_right)

    # Total loss (weighted)
    loss = cfg.w_pde * loss_pde + cfg.w_ic * loss_ic + cfg.w_bc * loss_bc
    return loss, loss_pde.item(), loss_ic.item(), loss_bc.item()


def ic_loss_only(model, data, cfg):
    """
    Stage 1: IC-only loss.
    """
    t_ic = data["t_ic"]
    x_ic = data["x_ic"]
    v_ic = data["v_ic"]

    inp_ic = torch.cat([t_ic, x_ic, v_ic], dim=1)
    f_ic_pred = model(inp_ic)
    f_ic_true = gaussian_ic(x_ic, v_ic, cfg.sigma)
    loss_ic = mse_loss(f_ic_pred, f_ic_true)
    return loss_ic

# ======================
# Training Loop: IC-only + Adam full PINN + optional LBFGS
# ======================

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

print("=== Stage 1: IC-only (HWF-PIKAN-Spline CBE) ===")
for it in range(1, cfg.n_iters_ic + 1):
    optimizer.zero_grad()
    loss_ic = ic_loss_only(model, data, cfg)
    loss_ic.backward()
    optimizer.step()

    if it % cfg.print_every == 0 or it == 1:
        print(f"[IC-only] Iter {it:6d} | IC loss {loss_ic.item():.4e}")

print("=== Stage 2: Full PINN with Adam ===")
for it in range(1, cfg.n_iters_adam + 1):
    optimizer.zero_grad()
    loss, lpde, lic, lbc = pinn_loss(model, data, cfg)
    loss.backward()
    optimizer.step()

    if it % cfg.print_every == 0 or it == 1:
        print(
            f"[Adam] Iter {it:6d} | "
            f"Total {loss.item():.4e} | "
            f"PDE {lpde:.4e} | IC {lic:.4e} | BC {lbc:.4e}"
        )

if cfg.use_lbfgs:
    print("=== Stage 3: LBFGS refinement ===")
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=cfg.lbfgs_max_iter,
        history_size=50,
        line_search_fn="strong_wolfe"
    )

    for epoch in range(1, cfg.lbfgs_epochs + 1):
        def closure():
            lbfgs.zero_grad()
            loss, lpde, lic, lbc = pinn_loss(model, data, cfg)
            loss.backward()
            return loss

        loss = lbfgs.step(closure)

        # Recompute for logging
        loss_val, lpde, lic, lbc = pinn_loss(model, data, cfg)

        if epoch % cfg.lbfgs_print_every == 0 or epoch == 1:
            print(
                f"[LBFGS] Epoch {epoch:4d}/{cfg.lbfgs_epochs} | "
                f"Total {loss_val.item():.4e} | "
                f"PDE {lpde:.4e} | IC {lic:.4e} | BC {lbc:.4e}"
            )

print("Training finished.")

# ======================
# Post-processing / Visualization
# ======================

model.eval()

times_to_plot = [0.0, 0.25, 0.5, 1.0]

Nx_plot = 150
Nv_plot = 150

x_vals = torch.linspace(-cfg.Lx, cfg.Lx, Nx_plot)
v_vals = torch.linspace(-cfg.Lv, cfg.Lv, Nv_plot)
X, V = torch.meshgrid(x_vals, v_vals, indexing="ij")

X_np = X.numpy()
V_np = V.numpy()

fig, axes = plt.subplots(
    nrows=len(times_to_plot),
    ncols=3,  # HWF-PIKAN-Spline, analytic, error
    figsize=(15, 16),
    constrained_layout=True
)

for row_idx, t_val in enumerate(times_to_plot):
    with torch.no_grad():
        t_plot = t_val * torch.ones_like(X)
        inp_plot = torch.stack(
            [t_plot.reshape(-1), X.reshape(-1), V.reshape(-1)], dim=1
        ).to(device)

        f_plot = model(inp_plot).cpu().numpy()
        F_pinn = f_plot.reshape(Nx_plot, Nv_plot)

    # Analytical solution with periodic wrap in x
    F_analytic = analytic_solution_periodic(
        t_val, X_np, V_np, sigma=cfg.sigma, Lx=cfg.Lx
    )

    # Absolute error
    F_err = np.abs(F_pinn - F_analytic)

    # --- HWF-PIKAN-Spline plot ---
    ax_pinn = axes[row_idx, 0]
    im1 = ax_pinn.imshow(
        F_pinn.T,
        extent=[-cfg.Lx, cfg.Lx, -cfg.Lv, cfg.Lv],
        origin="lower",
        aspect="auto"
    )
    ax_pinn.set_title(f"HWF-PIKAN-Spline, t = {t_val}")
    ax_pinn.set_xlabel("x")
    ax_pinn.set_ylabel("v")
    fig.colorbar(im1, ax=ax_pinn, fraction=0.046, pad=0.04)

    # --- Analytic plot ---
    ax_an = axes[row_idx, 1]
    im2 = ax_an.imshow(
        F_analytic.T,
        extent=[-cfg.Lx, cfg.Lx, -cfg.Lv, cfg.Lv],
        origin="lower",
        aspect="auto"
    )
    ax_an.set_title(f"Analytic (periodic), t = {t_val}")
    ax_an.set_xlabel("x")
    ax_an.set_ylabel("v")
    fig.colorbar(im2, ax=ax_an, fraction=0.046, pad=0.04)

    # --- Error plot ---
    ax_err = axes[row_idx, 2]
    im3 = ax_err.imshow(
        F_err.T,
        extent=[-cfg.Lx, cfg.Lx, -cfg.Lv, cfg.Lv],
        origin="lower",
        aspect="auto"
    )
    ax_err.set_title(f"|Error|, t = {t_val}")
    ax_err.set_xlabel("x")
    ax_err.set_ylabel("v")
    fig.colorbar(im3, ax=ax_err, fraction=0.046, pad=0.04)

plt.show()

