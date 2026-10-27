"""
Produce the figures and numbers used in the README.

Three checks, in order:
  1. the spacing the swarm settles at, against the Lennard-Jones minimum
  2. energy drift of the integrator with damping switched off
  3. the swarm holding formation while following a moving target

Run:  PYTHONPATH=src python make_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, str(Path(__file__).parent / "src"))

from swarm.config import SwarmConfig
from swarm.engine import SwarmEngine

SEED = 20261026
OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

ACCENT = "#0072B2"
GREY = "#6B6B6B"
GRID = "#E6E6E6"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GREY,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "svg.fonttype": "none",
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> results/figures/{name}.png and .svg")


def lj_potential(r, eps, sigma):
    sr6 = (sigma / r) ** 6
    return 4.0 * eps * (sr6**2 - sr6)


def nearest_neighbour_distances(pos):
    d = squareform(pdist(pos))
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


# ── 1. settled spacing against the Lennard-Jones minimum ──────────────────
print("\n[1] spacing at rest")

cfg = SwarmConfig(
    num_agents=60, epsilon=150.0, sigma=8.0, lj_cutoff=60.0,
    target_gain=0.0, damping=3.0, dt=0.05, steps=4000,
    world_size=400.0, init_spread=40.0, init_seed=SEED,
    max_velocity=15.0, max_force=400.0,
)
eng = SwarmEngine(cfg)
for _ in range(cfg.steps):
    eng.step()

nn = nearest_neighbour_distances(eng.pos)
r_min_theory = 2.0 ** (1.0 / 6.0) * cfg.sigma
med = float(np.median(nn))
lo, hi = np.percentile(nn, [2.5, 97.5])
print(f"  theory  r_min = 2^(1/6) sigma = {r_min_theory:.3f} m")
print(f"  measured median nearest-neighbour = {med:.3f} m")
print(f"  95% of agents between {lo:.3f} and {hi:.3f} m")
print(f"  difference from theory = {100*(med-r_min_theory)/r_min_theory:+.2f} %")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(eng.pos[:, 0], eng.pos[:, 1], s=28, color=ACCENT, zorder=3)
ax1.set_aspect("equal")
ax1.set_xlabel("x (metres)")
ax1.set_ylabel("y (metres)")
ax1.set_title(f"{cfg.num_agents} agents after {cfg.steps} steps")

ax2.hist(nn, bins=24, color=ACCENT, alpha=0.85)
ax2.axvline(r_min_theory, color="#D55E00", lw=1.6)
ax2.annotate(
    f"Lennard-Jones minimum\n$2^{{1/6}}\\sigma$ = {r_min_theory:.2f} m",
    xy=(r_min_theory, ax2.get_ylim()[1] * 0.78),
    xytext=(r_min_theory * 1.08, ax2.get_ylim()[1] * 0.80),
    color="#D55E00", fontsize=9.5,
)
ax2.set_xlabel("distance to nearest neighbour (metres)")
ax2.set_ylabel("agents")
ax2.set_title("Spacing the swarm settles at")
ax2.yaxis.grid(True)
ax2.set_axisbelow(True)
save(fig, "01_spacing_vs_theory")

# ── 2. energy drift with damping off ──────────────────────────────────────
print("\n[2] energy drift, damping off")

cfg2 = SwarmConfig(
    num_agents=40, epsilon=120.0, sigma=10.0, lj_cutoff=1e6,
    target_gain=0.0, damping=0.0, dt=0.01, steps=20000,
    world_size=4000.0, init_spread=35.0, init_seed=SEED,
    max_velocity=1e6, max_force=1e9,
)
eng2 = SwarmEngine(cfg2)

# Start from a relaxed lattice. A random cloud puts some agents almost on top
# of each other, and the r^-12 term makes the starting energy meaningless.
relax = SwarmConfig(
    num_agents=cfg2.num_agents, epsilon=cfg2.epsilon, sigma=cfg2.sigma,
    lj_cutoff=1e6, target_gain=0.0, damping=4.0, dt=0.01, steps=6000,
    world_size=cfg2.world_size, init_spread=cfg2.init_spread, init_seed=SEED,
    max_velocity=50.0, max_force=1e6,
)
warm = SwarmEngine(relax)
for _ in range(relax.steps):
    warm.step()
eng2.pos = warm.pos.copy()
eng2.vel = np.zeros_like(warm.vel)

def total_energy(e, c):
    ke = 0.5 * c.agent_mass * float(np.sum(e.vel**2))
    r = pdist(e.pos)
    pe = float(np.sum(lj_potential(np.maximum(r, 1e-9), c.epsilon, c.sigma)))
    return ke + pe

times, energies = [], []
e0 = total_energy(eng2, cfg2)
for i in range(cfg2.steps):
    eng2.step()
    if i % 50 == 0:
        times.append(eng2.t)
        energies.append(total_energy(eng2, cfg2))

energies = np.array(energies)
rel = (energies - e0) / abs(e0)
print(f"  initial energy = {e0:.4f}")
print(f"  final energy   = {energies[-1]:.4f}")
print(f"  relative drift over {cfg2.steps} steps = {rel[-1]*100:+.4f} %")
print(f"  largest excursion = {np.max(np.abs(rel))*100:.4f} %")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(times, rel * 100, color=ACCENT, lw=1.4)
ax.axhline(0, color=GREY, lw=0.8)
ax.set_xlabel("simulated time (seconds)")
ax.set_ylabel("energy change from start (%)")
ax.set_title("Total energy, damping switched off")
ax.yaxis.grid(True)
ax.set_axisbelow(True)
save(fig, "02_energy_drift")

# ── 3. holding formation while following a target ─────────────────────────
print("\n[3] following a moving target")

cfg3 = SwarmConfig(
    num_agents=50, epsilon=80.0, sigma=12.0, lj_cutoff=50.0,
    target_gain=1.2, damping=2.0, dt=0.08, steps=1200,
    world_size=400.0, init_spread=60.0, init_seed=SEED,
)
eng3 = SwarmEngine(cfg3)
targets, centroids, spreads = [], [], []
for _ in range(cfg3.steps):
    eng3.step()
    targets.append(eng3.target_position())
    centroids.append(eng3.centroid)
    spreads.append(float(np.median(nearest_neighbour_distances(eng3.pos))))

targets = np.array(targets)
centroids = np.array(centroids)
lag = np.linalg.norm(centroids - targets, axis=1)
settle = cfg3.steps // 4
print(f"  median distance centroid to target after settling = {np.median(lag[settle:]):.2f} m")
print(f"  median nearest-neighbour spacing while moving = {np.median(spreads[settle:]):.2f} m")
print(f"  Lennard-Jones minimum for sigma={cfg3.sigma} is {2**(1/6)*cfg3.sigma:.2f} m")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(targets[:, 0], targets[:, 1], color="#D55E00", lw=1.4, label="target")
ax1.plot(centroids[:, 0], centroids[:, 1], color=ACCENT, lw=1.4, label="swarm centre")
ax1.scatter(eng3.pos[:, 0], eng3.pos[:, 1], s=16, color=ACCENT, alpha=0.5, zorder=3)
ax1.set_aspect("equal")
ax1.set_xlabel("x (metres)")
ax1.set_ylabel("y (metres)")
ax1.set_title("Swarm following a moving target")
ax1.legend(frameon=False, fontsize=9.5)

t_axis = np.arange(cfg3.steps) * cfg3.dt
ax2.plot(t_axis, spreads, color=ACCENT, lw=1.3)
ax2.axhline(2 ** (1 / 6) * cfg3.sigma, color="#D55E00", lw=1.4)
ax2.annotate("Lennard-Jones minimum", xy=(t_axis[-1] * 0.45, 2 ** (1 / 6) * cfg3.sigma * 1.03),
             color="#D55E00", fontsize=9.5)
ax2.set_xlabel("simulated time (seconds)")
ax2.set_ylabel("median nearest-neighbour distance (metres)")
ax2.set_title("Spacing held while the target moves")
ax2.yaxis.grid(True)
ax2.set_axisbelow(True)
save(fig, "03_target_following")

print("\ndone.")
