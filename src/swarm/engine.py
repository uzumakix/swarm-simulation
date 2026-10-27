"""
bio-mimetic-swarm · engine.py
==============================
Core physics simulation for the decentralized multi-agent system.

Physics model
─────────────
  Double-integrator point masses:  ẍ_i = u_i − k_d · ẋ_i
  Lennard-Jones inter-agent potential (12-6)
  Global attractive potential towards a moving target
  Control law: u_i = −∇U_total(x_i)

Numerical integration
─────────────────────
  Semi-Implicit (Symplectic) Euler:
    v_{n+1} = v_n + a_n · Δt
    x_{n+1} = x_n + v_{n+1} · Δt   ← uses updated velocity
  Preserves symplectic structure → no artificial energy drift.

Performance
───────────
  All pairwise forces are fully vectorised via NumPy broadcasting
  and scipy.spatial.distance.pdist. Zero Python-level loops over agents.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform

from .config import SwarmConfig

logger = logging.getLogger(__name__)

# ── Named constants (no magic numbers in physics code) ─────────────────────

INIT_VELOCITY_SCALE = 0.3       # std-dev of initial velocity perturbation [m/s]
DISTANCE_EPSILON = 1e-8         # minimum distance to avoid div-by-zero [m]
SATURATION_EPSILON = 1e-9       # numerical guard in force/velocity clipping
BOUNDARY_RESTITUTION = 0.5      # velocity damping on boundary reflection


class SwarmEngine:
    """
    N-agent swarm simulation with Lennard-Jones artificial potential fields.

    Parameters
    ----------
    cfg : SwarmConfig
        Frozen configuration dataclass. Injected, never imported globally.

    Attributes
    ----------
    pos : ndarray (N, 2)   - agent positions [m]
    vel : ndarray (N, 2)   - agent velocities [m/s]
    t   : float            - elapsed simulation time [s]
    history : deque         - circular buffer of past positions for rendering
    """

    def __init__(self, cfg: SwarmConfig) -> None:
        self.cfg = cfg
        self.N = cfg.num_agents
        self.t: float = 0.0
        self._step_count: int = 0

        self._rng = np.random.default_rng(cfg.init_seed)

        # Initial positions: Gaussian cloud centred at origin
        self.pos: np.ndarray = self._rng.normal(
            loc=0.0, scale=cfg.init_spread, size=(self.N, 2)
        ).astype(np.float64)

        # Initial velocities: small random perturbation
        self.vel: np.ndarray = self._rng.normal(
            0.0, INIT_VELOCITY_SCALE, size=(self.N, 2)
        ).astype(np.float64)

        # Position history - deque with maxlen for O(1) append/evict
        self.history: deque[np.ndarray] = deque(maxlen=cfg.tail_length)

        # Diagnostic cache
        self._last_forces: np.ndarray = np.zeros((self.N, 2))

        logger.debug(
            "SwarmEngine initialised: N=%d, σ=%.1f, ε=%.1f",
            self.N, cfg.sigma, cfg.epsilon,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Target trajectory
    # ══════════════════════════════════════════════════════════════════════

    def target_position(self, t: Optional[float] = None) -> np.ndarray:
        """
        Lissajous trajectory: a smooth, slowly drifting target.

        x(t) = A_x · sin(ω_x · t)
        y(t) = A_y · sin(ω_y · t + φ)
        """
        if t is None:
            t = self.t
        c = self.cfg
        return np.array([
            c.target_amp_x * np.sin(c.target_freq_x * t),
            c.target_amp_y * np.sin(c.target_freq_y * t + c.target_phase),
        ], dtype=np.float64)

    # ══════════════════════════════════════════════════════════════════════
    # Force computation
    # ══════════════════════════════════════════════════════════════════════

    def _pairwise_distances(self, pos: np.ndarray) -> np.ndarray:
        """
        Compute (N, N) pairwise distance matrix with optional sensor noise.

        If cfg.sensor_noise > 0, adds symmetric Gaussian perturbation to
        each measured distance, simulating imperfect range sensors.
        """
        dist_mat = squareform(pdist(pos, metric="euclidean"))

        if self.cfg.sensor_noise > 0.0:
            noise = self._rng.normal(0.0, self.cfg.sensor_noise, size=dist_mat.shape)
            noise = (noise + noise.T) * 0.5  # ensure symmetry
            dist_mat = np.maximum(dist_mat + noise, 0.0)
            np.fill_diagonal(dist_mat, 0.0)

        return dist_mat

    def _dropout_mask(self, base_mask: np.ndarray) -> np.ndarray:
        """
        Apply communication dropout. Randomly disable pairwise links.

        Dropout is symmetric: if i cannot sense j, j also cannot sense i.
        """
        if self.cfg.dropout_rate <= 0.0:
            return base_mask

        drop = self._rng.random(base_mask.shape) < self.cfg.dropout_rate
        drop = drop | drop.T  # symmetric dropout
        return base_mask & ~drop

    def _lj_forces(self, pos: np.ndarray) -> np.ndarray:
        """
        Vectorised Lennard-Jones 12-6 pairwise forces.

        F_ij = (24ε / r) · [ 2(σ/r)¹² − (σ/r)⁶ ] · r̂_ij

        Implementation: scipy pdist → squareform → NumPy broadcasting.
        Zero Python-level loops over agent pairs.
        """
        cfg = self.cfg

        dist_mat = self._pairwise_distances(pos)

        # Interaction mask: off-diagonal, within cutoff
        mask = (dist_mat > DISTANCE_EPSILON) & (dist_mat < cfg.lj_cutoff)
        mask = self._dropout_mask(mask)

        # Safe distances - replace inactive entries with 1.0 to avoid div-by-zero
        safe_dist = np.where(mask, dist_mat, 1.0)

        # LJ scalar force magnitude
        inv_r = cfg.sigma / safe_dist
        inv_r6 = inv_r ** 6
        inv_r12 = inv_r6 * inv_r6
        f_scalar = (24.0 * cfg.epsilon / safe_dist) * (2.0 * inv_r12 - inv_r6)
        f_scalar = np.where(mask, f_scalar, 0.0)

        # Displacement vectors: disp[i, j] = pos[i] − pos[j]  (from j to i)
        disp = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        unit_vec = disp / safe_dist[:, :, np.newaxis]

        return np.sum(f_scalar[:, :, np.newaxis] * unit_vec, axis=1)

    def _attractive_forces(self, pos: np.ndarray) -> np.ndarray:
        """
        Navigation force: F_att = k_att · (x_target − x_i).

        Gradient of quadratic attractive potential towards moving target.
        """
        target = self.target_position()
        delta = target[np.newaxis, :] - pos
        return self.cfg.target_gain * delta

    def compute_forces(self, pos: np.ndarray) -> np.ndarray:
        """
        Total force = LJ inter-agent + navigation, with saturation.

        Returns (N, 2) clipped force vectors.
        """
        total = self._lj_forces(pos) + self._attractive_forces(pos)

        # Force saturation - prevents instability in dense configurations
        norms = np.linalg.norm(total, axis=1, keepdims=True)
        scale = np.where(
            norms > self.cfg.max_force,
            self.cfg.max_force / (norms + SATURATION_EPSILON),
            1.0,
        )
        total = total * scale

        self._last_forces = total
        return total

    # ══════════════════════════════════════════════════════════════════════
    # Integration
    # ══════════════════════════════════════════════════════════════════════

    def step(self, dt: Optional[float] = None) -> None:
        """
        Advance one timestep via Semi-Implicit (Symplectic) Euler.

        v_{n+1} = v_n + a_n · Δt
        x_{n+1} = x_n + v_{n+1} · Δt   ← uses updated velocity
        """
        if dt is None:
            dt = self.cfg.dt
        cfg = self.cfg

        # Snapshot for trail rendering BEFORE physics update
        self.history.append(self.pos.copy())

        # Acceleration: control input + viscous damping
        forces = self.compute_forces(self.pos)
        accel = (forces / cfg.agent_mass) - (cfg.damping * self.vel)

        # Symplectic Euler: update velocity first, then use it for position
        self.vel += accel * dt

        # Velocity saturation
        v_norms = np.linalg.norm(self.vel, axis=1, keepdims=True)
        v_scale = np.where(
            v_norms > cfg.max_velocity,
            cfg.max_velocity / (v_norms + SATURATION_EPSILON),
            1.0,
        )
        self.vel *= v_scale

        # Position update using already-updated velocity (symplectic property)
        self.pos += self.vel * dt

        # Soft boundary: damped reflection at world edges (fully vectorised)
        half = cfg.world_size / 2.0
        over_max = self.pos > half
        over_min = self.pos < -half
        self.pos = np.where(over_max, half, self.pos)
        self.pos = np.where(over_min, -half, self.pos)
        self.vel = np.where(over_max | over_min, -BOUNDARY_RESTITUTION * self.vel, self.vel)

        self.t += dt
        self._step_count += 1

    # ══════════════════════════════════════════════════════════════════════
    # Diagnostics & metrics
    # ══════════════════════════════════════════════════════════════════════

    @property
    def mean_speed(self) -> float:
        """Mean scalar speed across the swarm [m/s]."""
        return float(np.mean(np.linalg.norm(self.vel, axis=1)))

    @property
    def centroid(self) -> np.ndarray:
        """Swarm centroid position [m]."""
        return self.pos.mean(axis=0)

    @property
    def spread(self) -> float:
        """RMS distance of agents from centroid [m]."""
        diffs = self.pos - self.centroid
        return float(np.sqrt(np.mean(np.sum(diffs ** 2, axis=1))))

    @property
    def kinetic_energy(self) -> float:
        """Total kinetic energy: Σ ½mv² [J]."""
        return float(0.5 * self.cfg.agent_mass * np.sum(self.vel ** 2))

    @property
    def lj_potential_energy(self) -> float:
        """Total LJ potential energy across all unique pairs [J]."""
        dist_mat = squareform(pdist(self.pos))
        cfg = self.cfg

        mask = (dist_mat > DISTANCE_EPSILON) & (dist_mat < cfg.lj_cutoff)
        safe_dist = np.where(mask, dist_mat, 1.0)

        inv_r = cfg.sigma / safe_dist
        inv_r6 = inv_r ** 6
        inv_r12 = inv_r6 * inv_r6
        u_lj = 4.0 * cfg.epsilon * (inv_r12 - inv_r6)
        u_lj = np.where(mask, u_lj, 0.0)

        # Full matrix double-counts each pair
        return float(np.sum(u_lj) / 2.0)

    @property
    def total_energy(self) -> float:
        """Total mechanical energy (kinetic + LJ potential) [J]."""
        return self.kinetic_energy + self.lj_potential_energy

    def convergence_status(self) -> str:
        """Human-readable convergence indicator for HUD."""
        dist = float(np.linalg.norm(self.centroid - self.target_position()))
        if dist < 20.0:
            return "CONVERGED"
        elif dist < 50.0:
            return "TRACKING"
        else:
            return "DEPLOYING"

    def snapshot(self) -> dict[str, object]:
        """Capture full state for a single animation frame."""
        return {
            "pos": self.pos.copy(),
            "target": self.target_position(),
            "history": list(self.history),
            "t": self.t,
            "spread": self.spread,
            "speed": self.mean_speed,
            "status": self.convergence_status(),
            "energy": self.total_energy,
        }
