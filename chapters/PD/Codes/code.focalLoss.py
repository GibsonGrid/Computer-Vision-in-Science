
# ---------------------------------------------------------
# Focal Loss function for positive class (y = 1)
# ---------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt

def focal_loss(p, gamma):
    return - (1 - p)**gamma * np.log(p + 1e-8)


plt.figure(figsize=(8, 6))
# Different gamma values
gammas = [0, 1, 2, 3, 4, 5]
# Probabilities from 0 to 1
p = np.linspace(1e-4, 1 - 1e-4, 500)
# Plot curves for each gamma
for g in gammas:
    if g == 0:
        plt.plot(p, focal_loss(p, g),
                 label=rf"$\gamma$ = 0 (FL = CE = -log(p))")
    else:
        plt.plot(p, focal_loss(p, g),
                 label=rf"$\gamma$ = {g}")

# Labels
plt.xlabel("Predicted probability $p$", fontsize=12)
plt.ylabel("Focal Loss $FL(p)$", fontsize=12)
plt.title(rf"Focal Loss for Different Values of $\gamma$: " + r"$FL(p) = -(1-p)^{\gamma}\log(p)$", fontsize=14)
plt.axvline(0.6, linestyle='--', color='red', linewidth=1.5)

eq_text = (
    rf"Well-classified examples"
)
plt.text(
    0.6, 0.2, eq_text,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top'
)

plt.legend(fontsize=14, ncols=1)
plt.grid(True)
plt.tight_layout()

# Save as PDF
plt.savefig("focal_loss_plot.pdf", dpi=300, bbox_inches='tight')

plt.show()
