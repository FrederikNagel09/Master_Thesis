import matplotlib.pyplot as plt
import numpy as np
import torch

# Constants
T = 1000
beta_1 = 1e-4
beta_T = 0.004  # noqa: N816

# Calculations using Torch
beta = torch.linspace(beta_1, beta_T, T)
alpha = 1.0 - beta
alpha_cumprod = torch.cumprod(alpha, dim=0)

# Function to plot: beta / (2 * alpha * (1 - alpha_cumprod))
y_values = beta / (2 * alpha * (1 - alpha_cumprod))
x_values = np.arange(1, T + 1)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values.numpy(), label=r"$\frac{\beta_t}{2\alpha_t(1-\bar{\alpha}_t)}$", color="royalblue")
plt.title("Visualization of the Noise Schedule Function")
plt.xlabel("Timestep (T)")
plt.ylabel("Value")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(fontsize=14)
plt.tight_layout()

plt.savefig("src/results/noise_schedule_plot.png")
