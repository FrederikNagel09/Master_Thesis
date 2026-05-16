import matplotlib.pyplot as plt
import numpy as np
import torch

# Constants
T = 1000
beta_1 = 1e-4
beta_T = 2e-2  # noqa: N816

# Noise schedule
beta = torch.linspace(beta_1, beta_T, T)
alpha = 1.0 - beta
alpha_cumprod = torch.cumprod(alpha, dim=0)

x_values = np.arange(1, T + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Noise Schedule  (β₁={beta_1}, β_T={beta_T}, T={T})", fontsize=13)

# --- Left: sqrt(ᾱ_t) and sqrt(1 - ᾱ_t) ---
axes[0].plot(x_values, alpha_cumprod.sqrt().numpy(), label=r"$\sqrt{\bar{\alpha}_t}$", color="royalblue")
axes[0].plot(x_values, (1 - alpha_cumprod).sqrt().numpy(), label=r"$\sqrt{1 - \bar{\alpha}_t}$", color="tomato")
axes[0].set_title("Signal vs. Noise Coefficients")
axes[0].set_xlabel("Timestep (t)")
axes[0].grid(True, linestyle="--", alpha=0.7)
axes[0].legend(fontsize=14)

# --- Right: β_t / (2_t(1 - ᾱ_t)) ---
y_values = beta / (2 * alpha * (1 - alpha_cumprod))
axes[1].plot(x_values, y_values.numpy(), label=r"$\frac{\beta_t}{2\alpha_t(1-\bar{\alpha}_t)}$", color="royalblue")
axes[1].set_title("Noise Schedule Function")
axes[1].set_xlabel("Timestep (t)")
axes[1].grid(True, linestyle="--", alpha=0.7)
axes[1].legend(fontsize=14)

plt.tight_layout()
plt.savefig("src/results/noise_schedule_plot.png")
