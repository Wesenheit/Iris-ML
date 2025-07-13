from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sys import argv

name = argv[1]
print(name)
a = pd.read_csv(name)
init = len(a)

id = np.logical_and((a["leng"]).values > 10, a["DEC"].values > -90)
a = a.iloc[id, :]
print(len(a) / init, len(a))
x = a["T_50th"]
y = a["T_true"]
metal_1 = a["[M/H]_50th"].values
metal_2 = a["[M/H]_true"].values
id = np.logical_not(np.isnan(metal_2))
print("RMSE metal", np.sqrt(np.mean((metal_1[id] - metal_2[id]) ** 2)))
print(len(x), len(y))
print("RMSE T: ", np.sqrt(np.mean((x - y) ** 2)))
print("MAD T:", np.mean(np.abs(x - y)))

# z = np.clip(a["leng"],0,20)
z = a["leng"].values
N = 9
xmin = np.round(np.min(a["T_true"]), -3)
xmax = 8000  # np.round(np.max(a["T_true"]),-3)
X = np.linspace(xmin, xmax, 100)
fig = plt.figure(1, figsize=(7, 7))
gs = plt.GridSpec(3, 2, height_ratios=[0.05, 1, 0.2], width_ratios=[1, 0.2])
gs.update(left=0.12, right=0.95, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)

ax1 = fig.add_subplot(gs[1, 0])  # place it where it should be.

# The plot itself
plt1 = ax1.scatter(
    x,
    y,
    c=z,
    marker=".",
    s=20,
    edgecolor="none",
    alpha=1,
    cmap="magma_r",
    vmin=np.min(z),
    vmax=np.max(z),
    rasterized=True,
)
ax1.plot(X, X, color="black")
ax1.grid(True)
ax1.set_xlim(xmin, xmax)
ax1.set_ylim(xmin, xmax)
ax1.set_xlabel(r" ")  # Force this empty !
ax1.set_xticks(
    np.linspace(xmin, xmax, N)
)  # Force this to what I want - for consistency with histogram below !
ax1.set_yticks(np.linspace(xmin, xmax, N))
ax1.set_xticklabels([])  # Force this empty !
ax1.set_ylabel(r"Measured Temperature [K]")

cbax = fig.add_subplot(gs[0, 0])
cbax.set_yticklabels([])
cbax.set_xticklabels([])
cbax.set_axis_off()
cb = fig.colorbar(
    ax=cbax,
    mappable=plt1,
    orientation="horizontal",
    ticklocation="top",
    label=r"$N$",
    fraction=2,
)
# cb.ax.xaxis.set_ticks_position('top')
cb.ax.xaxis.set_label_position("top")
ax1v = fig.add_subplot(gs[1, 1])
bins = np.linspace(xmin, xmax, 30)
ax1v.hist(y, bins=bins, orientation="horizontal", color="k", edgecolor="w")
ax1v.set_yticks(np.linspace(xmin, xmax, N))
ax1v.set_xticklabels([])
ax1v.set_yticklabels([])
ax1v.set_ylim(xmin, xmax)
ax1v.grid(True)
ax1h = fig.add_subplot(gs[2, 0])
bins = np.linspace(xmin, xmax, 30)
ax1h.hist(x, bins=bins, orientation="vertical", color="k", edgecolor="w")
ax1h.set_xticks(np.linspace(xmin, xmax, N))  #
ax1h.set_yticklabels([])
ax1h.set_xlim(xmin, xmax)
ax1h.set_xlabel(r"Predicted temperature [K]")
ax1h.grid(True)
# plt.tight_layout()
name_to_save = name.split("/")[-1]
plt.savefig("T_pred_measured_{}.pdf".format(name_to_save[:-4]))


fig, ax = plt.subplots(figsize=(5, 5))
print(np.median(a["AV_50th"].values), np.quantile(a["AV_50th"].values, 0.84))
ax.hist(a["AV_50th"].values, bins=20)
plt.savefig("hist_{}.pdf".format(name_to_save[:-4]))
