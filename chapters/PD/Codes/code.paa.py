import os
import numpy as np
import pandas as pd
import cv2
from pyts.approximation import PiecewiseAggregateApproximation

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

out_dir = "outputs/"
create_dir(out_dir)
# ------------------------------------------------------------------
# Visualize the signal before and after PAA
# ------------------------------------------------------------------
import matplotlib.pyplot as plt

print("Generating PAA visualization...")

# Create a new synthetic time series
n_timestamps = 200
t = np.linspace(0, 4 * np.pi, n_timestamps)
X = np.sin(t) + 0.2 * np.random.randn(n_timestamps)

# PAA parameters
window_size = 10
paa = PiecewiseAggregateApproximation(window_size=window_size)

# PAA requires 2D input: (n_samples, n_timestamps)
X_paa = paa.transform(X.reshape(1, -1))[0]

# Plot
plt.figure(figsize=(10, 4))
plt.plot(X, 'o--', ms=3, label='Original')
plt.plot(
    np.arange(window_size // 2, n_timestamps + window_size // 2, window_size),
    X_paa,
    'o--',
    ms=4,
    label='PAA'
)

# Draw vertical segment lines
plt.vlines(
    np.arange(0, n_timestamps, window_size) - 0.5,
    X.min(),
    X.max(),
    colors='g',
    linestyles='--',
    linewidth=0.8
)

plt.legend(loc='best', fontsize=10)
plt.xlabel('Time', fontsize=12)
plt.title('Piecewise Aggregate Approximation (PAA)', fontsize=15)
plt.tight_layout()

# Save figure
paa_fig_path = os.path.join(out_dir, "paa_visualization.pdf")
plt.savefig(paa_fig_path, dpi=200)
plt.close()

print(f"PAA visualization saved to: {paa_fig_path}")
