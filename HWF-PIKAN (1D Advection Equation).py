"""
Authors: Mohammad E. Heravifard & Prof. Kazem Hejranfar
HWF-PIKAN solver for 1D linear advection using PyTorch (fixed/trainability improvements).

To avoid model collapse:
- Fourier features scaled by 1/sqrt(M) and default sigma_B reduced.
- LayerNorm applied to hybrid embedding (Fourier + wavelets).
- Inner linear weights initialized smaller and a small learnable inner_scale reduces tanh saturation.
- Smaller initializations for outer coefficients.
- Everything remains differentiable for PDE residual computation (u_t + c u_x).

Usage:
    python hwf_pikan_advection_fixed.py
"""
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------
# Configuration / Hyperparameters
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(12345)
np.random.seed(12345)

# PDE parameters
c = 1.0  # advection speed

# Domain
x_min, x_max = 0.0, 1.0
t_min, t_max = 0.0, 0.5

# Training data sizes
N_ic = 200      # initial condition points
N_bc = 200      # boundary condition points (periodic pairing)
N_f = 8000      # collocation points for PDE residual

# Training hyperparams
learning_rate = 5e-4
epochs_adam = 3000
use_lbfgs = True

# HWF embedding hyperparams (more conservative defaults)
M_fourier = 32       # number of Fourier frequency vectors (reduced)
sigma_B = 1.0        # variance for frequency sampling (reduced from 10)
train_omega = False  # whether to make the frequency matrix trainable

# Wavelet embedding hyperparams
J_wavelet = 3        # number of scales (0 .. J_wavelet-1); translations per scale = 2**j

# PIKAN / B-spline architecture
Q = 12               # number of inner units (Kolmogorov terms)
bspline_degree = 3   # degree of B-spline (cubic)
n_control = 25       # number of control coefficients per phi_q

# Misc
print_interval = 250

# ---------------------
# Utilities
# ---------------------
def analytic_solution(x: np.ndarray, t: np.ndarray, c_speed: float = c) -> np.ndarray:
    """Analytic solution for IC sin(2*pi*x) with advection speed c (periodic)."""
    return np.sin(2.0 * np.pi * (x - c_speed * t))


def to_tensor(arr, dtype=torch.float32, device=device, requires_grad=False):
    return torch.tensor(arr, dtype=dtype, device=device, requires_grad=requires_grad)


# ---------------------
# B-spline basis (Cox-de Boor) implemented in PyTorch
# ---------------------
def make_open_uniform_knots(num_control: int, degree: int, device=device):
    n_knots = num_control + degree + 1
    if n_knots <= 2 * (degree + 1):
        knots = torch.linspace(0.0, 1.0, steps=n_knots, device=device)
    else:
        n_internal = n_knots - 2 * (degree + 1)
        internal = torch.linspace(0.0, 1.0, steps=n_internal + 2, device=device)[1:-1]
        knots = torch.cat([
            torch.zeros(degree + 1, device=device),
            internal,
            torch.ones(degree + 1, device=device)
        ])
    return knots


def bspline_basis_batch(x: torch.Tensor, degree: int, knots: torch.Tensor):
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


# ---------------------
# Wavelet feature (Ricker/Mexican-hat) implementation
# ---------------------
def ricker_wavelet(s: torch.Tensor, scale: float):
    u = s / scale
    return (1.0 - u * u) * torch.exp(-0.5 * u * u)


def wavelet_features_2d(z: torch.Tensor, J: int):
    assert z.shape[1] == 2, "z must be (N,2) with columns [x,t]"
    x = z[:, 0]
    t = z[:, 1]
    features = []
    for j in range(J):
        num_k = 2 ** j
        # choose scale proportional to domain division by num_k (avoid extremely tiny scales)
        scale = max(0.5 / num_k, 1e-3)
        centers = torch.tensor([(k + 0.5) / num_k for k in range(num_k)], device=z.device, dtype=z.dtype)
        ux = (x.unsqueeze(1) - centers.unsqueeze(0))
        ut = (t.unsqueeze(1) - centers.unsqueeze(0))
        psi_x = ricker_wavelet(ux, scale)  # (N, num_k)
        psi_t = ricker_wavelet(ut, scale)
        features.append(psi_x)
        features.append(psi_t)
    if len(features) == 0:
        return torch.zeros((z.shape[0], 0), device=z.device, dtype=z.dtype)
    F = torch.cat(features, dim=1)  # (N, total)
    return F


# ---------------------
# Fourier feature mapping (scaled)
# ---------------------
class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, M: int, sigma: float = 1.0, trainable: bool = False):
        super().__init__()
        B = torch.randn(M, in_dim) * sigma
        if trainable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)
        self.two_pi = 2.0 * np.pi
        self.M = M

    def forward(self, z: torch.Tensor):
        if hasattr(self, "B"):
            B = self.B
        else:
            B = self.B
        proj = self.two_pi * (z @ B.t())  # (N, M)
        sinp = torch.sin(proj)
        cosp = torch.cos(proj)
        # scale outputs to keep variance stable and avoid huge embeddings
        scale = 1.0 / (np.sqrt(self.M) + 1e-12)
        return torch.cat([sinp * scale, cosp * scale], dim=1)  # (N, 2M)


# ---------------------
# HWF Embedding (Fourier + Wavelet) with normalization
# ---------------------
class HWFEmbedding(nn.Module):
    def __init__(self, in_dim: int, M_fourier: int, sigma_B: float, J_wavelet: int, trainable: bool = False):
        super().__init__()
        self.fourier = FourierFeatures(in_dim, M_fourier, sigma=sigma_B, trainable=trainable)
        self.J = J_wavelet
        self.wavelet_size = 0 if J_wavelet == 0 else 2 * ((2 ** J_wavelet) - 1)
        self.out_dim = 2 * M_fourier + self.wavelet_size
        # normalize combined embedding to avoid feature-scale imbalance
        if self.out_dim > 0:
            self.norm = nn.LayerNorm(self.out_dim)
        else:
            self.norm = None

    def forward(self, z: torch.Tensor):
        F = self.fourier(z)  # (N, 2M)
        if self.J > 0:
            W = wavelet_features_2d(z, self.J)  # (N, wavelet_size)
            emb = torch.cat([F, W], dim=1)
        else:
            emb = F
        if self.norm is not None:
            emb = self.norm(emb)
        return emb


# ---------------------
# KA-BSpline PIKAN core (accepts embedding input) with better init & inner_scale
# ---------------------
class KA_BSpline_Core(nn.Module):
    def __init__(self, input_dim: int, Q: int, degree: int, n_control: int):
        super().__init__()
        self.Q = Q
        self.degree = degree
        self.n_control = n_control

        # inner linear maps: W shape (Q, input_dim), biases b_inner shape (Q,)
        # initialize smaller to avoid large pre-activations
        self.W = nn.Parameter(torch.randn(Q, input_dim) * 0.05)
        self.b_inner = nn.Parameter(torch.zeros(Q))

        # small learnable scale to prevent tanh saturation early
        self.inner_scale = nn.Parameter(torch.tensor(0.5))

        # control coefficients for each phi_q: shape (Q, n_control)
        self.coeffs = nn.Parameter(torch.randn(Q, n_control) * 0.05)

        # outer linear scalars a_q and final bias (small)
        self.a = nn.Parameter(torch.randn(Q, 1) * 0.05)
        self.b_out = nn.Parameter(torch.zeros(1))

        # knots (fixed, open uniform) on [0,1]
        knots = make_open_uniform_knots(n_control, degree, device=device)
        self.register_buffer("knots", knots)

    def forward(self, emb: torch.Tensor):
        N = emb.shape[0]
        u_inner = emb @ self.W.t() + self.b_inner  # (N, Q)
        # apply small scale to reduce risk of saturation
        s = 0.5 * (torch.tanh(self.inner_scale * u_inner) + 1.0)      # (N, Q)

        s_flat = s.reshape(-1)  # (N*Q,)
        B_flat = bspline_basis_batch(s_flat, self.degree, self.knots)  # (N*Q, n_control)
        B = B_flat.reshape(N, self.Q, self.n_control)  # (N, Q, n_control)

        coeffs = self.coeffs.unsqueeze(0).expand(N, -1, -1)  # (N, Q, n_control)
        phi = (B * coeffs).sum(dim=-1)  # (N, Q)

        a = self.a.view(1, self.Q)  # (1, Q)
        u_out = (phi * a).sum(dim=1, keepdim=True) + self.b_out  # (N,1)
        return u_out


# ---------------------
# Full HWF-PIKAN model (embedding + core)
# ---------------------
class HWF_PIKAN(nn.Module):
    def __init__(self,
                 in_dim: int,
                 M_fourier: int,
                 sigma_B: float,
                 J_wavelet: int,
                 train_omega: bool,
                 Q: int,
                 degree: int,
                 n_control: int):
        super().__init__()
        self.embedding = HWFEmbedding(in_dim, M_fourier, sigma_B, J_wavelet, trainable=train_omega)
        emb_dim = self.embedding.out_dim
        self.core = KA_BSpline_Core(emb_dim, Q, degree, n_control)

    def forward(self, xt: torch.Tensor):
        emb = self.embedding(xt)
        return self.core(emb)


# ---------------------
# PDE residual and losses
# ---------------------
def pde_residual(model: nn.Module, x_f: torch.Tensor, t_f: torch.Tensor) -> torch.Tensor:
    x_f = x_f.reshape(-1, 1)
    t_f = t_f.reshape(-1, 1)
    xt = torch.cat([x_f, t_f], dim=1)
    xt.requires_grad_(True)

    u = model(xt)  # (N,1)

    grads = torch.autograd.grad(
        outputs=u,
        inputs=xt,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]  # (N,2) columns: du/dx, du/dt

    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]

    r = u_t + c * u_x
    return r


# ---------------------
# Data samplers
# ---------------------
def sampler_ic(n_ic: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_ic = np.random.rand(n_ic, 1) * (x_max - x_min) + x_min
    t_ic = np.zeros_like(x_ic)
    u_ic = analytic_solution(x_ic, t_ic)
    return to_tensor(x_ic), to_tensor(t_ic), to_tensor(u_ic)


def sampler_bc(n_bc: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t_bc = np.random.rand(n_bc, 1) * (t_max - t_min) + t_min
    x0 = np.zeros_like(t_bc) + x_min
    x1 = np.zeros_like(t_bc) + x_max
    return to_tensor(x0), to_tensor(x1), to_tensor(t_bc)


def sampler_collocation(n_f: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x_f = np.random.rand(n_f, 1) * (x_max - x_min) + x_min
    t_f = np.random.rand(n_f, 1) * (t_max - t_min) + t_min
    return to_tensor(x_f), to_tensor(t_f)


# ---------------------
# Training
# ---------------------
def train():
    model = HWF_PIKAN(in_dim=2,
                      M_fourier=M_fourier,
                      sigma_B=sigma_B,
                      J_wavelet=J_wavelet,
                      train_omega=train_omega,
                      Q=Q,
                      degree=bspline_degree,
                      n_control=n_control).to(device)

    mse_loss = nn.MSELoss()

    # Prepare data
    x_ic, t_ic, u_ic = sampler_ic(N_ic)
    x0_bc, x1_bc, t_bc = sampler_bc(N_bc)
    x_f, t_f = sampler_collocation(N_f)

    # Move to device
    x_ic, t_ic, u_ic = x_ic.to(device), t_ic.to(device), u_ic.to(device)
    x0_bc, x1_bc, t_bc = x0_bc.to(device), x1_bc.to(device), t_bc.to(device)
    x_f, t_f = x_f.to(device), t_f.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()
    print(f"Training HWF-PIKAN on device={device} | Q={Q}, M_fourier={M_fourier}, J_wavelet={J_wavelet}, n_control={n_control}")
    for ep in range(1, epochs_adam + 1):
        model.train()
        optimizer.zero_grad()

        # IC loss
        xt_ic = torch.cat([x_ic, t_ic], dim=1)
        u_pred_ic = model(xt_ic)
        loss_ic = mse_loss(u_pred_ic, u_ic)

        # BC loss (periodic)
        xt0 = torch.cat([x0_bc, t_bc], dim=1)
        xt1 = torch.cat([x1_bc, t_bc], dim=1)
        u0 = model(xt0)
        u1 = model(xt1)
        loss_bc = mse_loss(u0, u1)

        # PDE residual loss
        r = pde_residual(model, x_f, t_f)
        loss_pde = mse_loss(r, torch.zeros_like(r))

        loss = loss_ic + loss_bc + loss_pde
        loss.backward()
        optimizer.step()

        if ep % print_interval == 0 or ep == 1 or ep == epochs_adam:
            elapsed = time.time() - start_time
            print(f"Epoch {ep}/{epochs_adam} | loss={loss.item():.3e} "
                  f"(ic={loss_ic.item():.3e}, bc={loss_bc.item():.3e}, pde={loss_pde.item():.3e}) | time={elapsed:.1f}s")

    # LBFGS fine-tuning
    if use_lbfgs:
        print("Starting LBFGS fine-tuning...")
        optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=0.8, max_iter=300, max_eval=500,
                                      history_size=50, tolerance_grad=1e-9, tolerance_change=1e-9)

        def closure():
            optimizer_lbfgs.zero_grad()
            xt_ic = torch.cat([x_ic, t_ic], dim=1)
            u_pred_ic = model(xt_ic)
            loss_ic_ = mse_loss(u_pred_ic, u_ic)

            xt0 = torch.cat([x0_bc, t_bc], dim=1)
            xt1 = torch.cat([x1_bc, t_bc], dim=1)
            u0_ = model(xt0)
            u1_ = model(xt1)
            loss_bc_ = mse_loss(u0_, u1_)

            r_ = pde_residual(model, x_f, t_f)
            loss_pde_ = mse_loss(r_, torch.zeros_like(r_))

            loss_ = loss_ic_ + loss_bc_ + loss_pde_
            loss_.backward()
            return loss_

        optimizer_lbfgs.step(closure)
        print("LBFGS finished.")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s")
    return model


# ---------------------
# Evaluation & plotting
# ---------------------
def evaluate_and_plot(model: nn.Module):
    model.eval()
    nx, nt = 256, 200
    x = np.linspace(x_min, x_max, nx)
    t = np.linspace(t_min, t_max, nt)
    X, T = np.meshgrid(x, t)
    xt = np.vstack([X.flatten(), T.flatten()]).T
    xt_tensor = to_tensor(xt).to(device)

    with torch.no_grad():
        u_pred = model(xt_tensor).cpu().numpy().reshape(nt, nx)

    u_exact = analytic_solution(X, T)
    error_l2 = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    print(f"Relative L2 error on grid: {error_l2:.3e}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    times_to_plot = [0.0, t_max * 0.33, t_max * 0.66, t_max]
    for ax, tt in zip(axes.flatten(), times_to_plot):
        idx = np.argmin(np.abs(t - tt))
        ax.plot(x, u_exact[idx, :], 'k-', label=f"Exact t={t[idx]:.3f}")
        ax.plot(x, u_pred[idx, :], 'r--', label="HWF-PIKAN")
        ax.set_xlim(x_min, x_max)
        ax.legend()
        ax.grid(True)
    plt.suptitle("1D Advection: HWF-PIKAN vs Exact")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Residual heatmap
    xg = to_tensor(X.flatten()[:, None]).to(device)
    tg = to_tensor(T.flatten()[:, None]).to(device)
    r = pde_residual(model, xg, tg).detach().cpu().numpy().reshape(nt, nx)
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(x, t, np.abs(r), shading='auto')
    plt.colorbar(label='|residual|')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Absolute PDE residual |u_t + c u_x|')
    plt.tight_layout()
    plt.show()


# ---------------------
# Run
# ---------------------
if __name__ == "__main__":
    model_trained = train()
    evaluate_and_plot(model_trained)
