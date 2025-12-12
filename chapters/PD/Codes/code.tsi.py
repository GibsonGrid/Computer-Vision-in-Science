import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
import os




def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def generate_random_signal(T=1000, seed=42):
    np.random.seed(seed)
    raw_noise = np.random.randn(T)
    x = np.cumsum(raw_noise)
    x = (x - np.mean(x)) / np.std(x)
    x = np.convolve(x, np.ones(20)/20, mode='same')
    return x

def remove_spines(ax):
    """Remove borders and ticks from an axis."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_tsi_with_layout(x, img, title):
    T = len(x)
    time_points = np.arange(T)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=(2, 7),
        height_ratios=(2, 7),
        left=0.05, right=0.95, bottom=0.05, top=0.95,
        wspace=0.02, hspace=0.02
    )

    # ---- Left vertical signal ----
    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.plot(x, time_points, color='black', linewidth=1)
    ax_left.invert_xaxis()
    remove_spines(ax_left)

    # ---- Top horizontal signal ----
    ax_top = fig.add_subplot(gs[0, 1])
    ax_top.plot(time_points, x, color='black', linewidth=1)
    ax_top.xaxis.tick_top()
    remove_spines(ax_top)

    # ---- TSI Image ----
    ax_img = fig.add_subplot(gs[1, 1])
    ax_img.imshow(img, cmap='rainbow', origin='lower')
    ax_img.set_title(title, fontsize=12, pad=8)
    remove_spines(ax_img)

    plt.savefig(out_dir + title + ".pdf")
    # plt.show()
    plt.close(fig)


out_dir = "outputs/"
create_dir(out_dir)
# ------------------------------------------------------
# Compute representations and plot (as before)
# ------------------------------------------------------
x = generate_random_signal()
X = x.reshape(1, -1)

gasf = GramianAngularField(method='summation')
gadf = GramianAngularField(method='difference')
mtf  = MarkovTransitionField()
rp   = RecurrencePlot(threshold='point', percentage=10)

img_gasf = gasf.fit_transform(X)[0]
img_gadf = gadf.fit_transform(X)[0]
img_mtf  = mtf.fit_transform(X)[0]
img_rp   = rp.fit_transform(X)[0]

plot_tsi_with_layout(x, img_gasf, "GASF")
plot_tsi_with_layout(x, img_gadf, "GADF")
plot_tsi_with_layout(x, img_mtf,  "MTF")
plot_tsi_with_layout(x, img_rp,   "RP")

