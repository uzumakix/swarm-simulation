"""
bio-mimetic-swarm · config.py
==============================
Central configuration for the swarm simulation.

Design decisions
────────────────
  frozen=True → immutable after creation. Critical for reproducibility.
  No module-level singleton → engine/renderer receive config via DI.
  Validation in __post_init__ → fail fast on obviously wrong parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SwarmConfig:
    """Immutable, validated configuration for a single simulation run."""

    # ── Scenario metadata ──────────────────────────────────────────────────
    scenario_name: str = "drifting-target"
    scenario_description: str = "Follow a slowly drifting target"

    # ── Agent count ────────────────────────────────────────────────────────
    num_agents: int = 50

    # ── World ──────────────────────────────────────────────────────────────
    world_size: float = 200.0          # metres; domain is [-W/2, W/2]²

    # ── Lennard-Jones Potential ────────────────────────────────────────────
    #   U_LJ(r) = 4ε [ (σ/r)¹² − (σ/r)⁶ ]
    epsilon: float = 80.0              # potential well depth  [N·m]
    sigma: float = 12.0                # equilibrium separation base  [m]
    lj_cutoff: float = 50.0           # hard cutoff distance  [m]

    # ── Navigation ─────────────────────────────────────────────────────────
    #   F_att = k_att · (x_target − x_i)
    target_gain: float = 1.2          # k_att  [1/s²]

    # ── Agent dynamics ─────────────────────────────────────────────────────
    #   ẍ_i = u_i − k_d · ẋ_i
    damping: float = 2.0              # k_d  [1/s]
    agent_mass: float = 1.0           # normalised  [kg]

    # ── Saturation ─────────────────────────────────────────────────────────
    max_force: float = 120.0          # clip magnitude  [N]
    max_velocity: float = 15.0        # clip magnitude  [m/s]

    # ── Target trajectory (Lissajous) ──────────────────────────────────────
    target_amp_x: float = 70.0        # metres
    target_amp_y: float = 50.0        # metres
    target_freq_x: float = 0.004      # rad/s
    target_freq_y: float = 0.006      # rad/s
    target_phase: float = 0.7854      # π/4 offset

    # ── Real-world realism ─────────────────────────────────────────────────
    sensor_noise: float = 0.0         # std-dev of Gaussian noise on distances [m]
    dropout_rate: float = 0.0         # fraction of pairwise links dropped per step

    # ── Numerics ───────────────────────────────────────────────────────────
    dt: float = 0.08                  # timestep  [s]
    steps: int = 600                  # total simulation steps

    # ── Visualisation ──────────────────────────────────────────────────────
    tail_length: int = 30
    anim_interval_ms: int = 30
    anim_fps: int = 30
    output_format: str = "gif"        # "gif" or "mp4"
    output_file: str = ""             # auto-generated if empty

    # ── Colour palette (dark-mode blueprint) ───────────────────────────────
    color_bg: str = "#0e1117"
    color_panel: str = "#161b22"
    color_grid: str = "#1f2937"
    color_agents: str = "#06b6d4"
    color_tail: str = "#0e7490"
    color_target: str = "#ef4444"
    color_text: str = "#e2e8f0"
    color_accent: str = "#22d3ee"

    # ── Initial deployment ─────────────────────────────────────────────────
    init_spread: float = 60.0         # std-dev of Gaussian cloud  [m]
    init_seed: int = 42

    def __post_init__(self) -> None:
        """Validate parameters. Fail fast on obviously broken configs."""
        errors: list[str] = []

        if self.num_agents < 1:
            errors.append(f"num_agents must be >= 1, got {self.num_agents}")
        if self.epsilon <= 0:
            errors.append(f"epsilon must be > 0, got {self.epsilon}")
        if self.sigma <= 0:
            errors.append(f"sigma must be > 0, got {self.sigma}")
        if self.dt <= 0:
            errors.append(f"dt must be > 0, got {self.dt}")
        if self.steps < 1:
            errors.append(f"steps must be >= 1, got {self.steps}")
        if self.agent_mass <= 0:
            errors.append(f"agent_mass must be > 0, got {self.agent_mass}")
        if self.max_force <= 0:
            errors.append(f"max_force must be > 0, got {self.max_force}")
        if self.max_velocity <= 0:
            errors.append(f"max_velocity must be > 0, got {self.max_velocity}")
        if not (0.0 <= self.dropout_rate < 1.0):
            errors.append(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")
        if self.sensor_noise < 0:
            errors.append(f"sensor_noise must be >= 0, got {self.sensor_noise}")
        if self.output_format not in ("gif", "mp4"):
            errors.append(f"output_format must be 'gif' or 'mp4', got '{self.output_format}'")

        if errors:
            raise ValueError(
                "Invalid SwarmConfig:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    # ── Derived properties ─────────────────────────────────────────────────

    @property
    def lj_r_eq(self) -> float:
        """Lennard-Jones equilibrium distance: r_eq = 2^(1/6) · σ."""
        return (2.0 ** (1.0 / 6.0)) * self.sigma

    @property
    def resolved_output_file(self) -> str:
        """Output filename. Auto-generated from scenario if not set."""
        if self.output_file:
            return self.output_file
        ext = "gif" if self.output_format == "gif" else "mp4"
        return f"swarm_{self.scenario_name}.{ext}"

    def summary(self) -> str:
        """One-line summary for console output."""
        return (
            f"Scenario: {self.scenario_name}  ·  "
            f"{self.num_agents} agents  ·  "
            f"ε={self.epsilon}  σ={self.sigma}m  ·  "
            f"k_att={self.target_gain}  k_d={self.damping}  ·  "
            f"Δt={self.dt}s × {self.steps} steps"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Scenario loading
# ══════════════════════════════════════════════════════════════════════════════

def _scenarios_dir() -> Path:
    """Locate the scenarios/ directory relative to the package root."""
    return Path(__file__).resolve().parent.parent.parent / "configs"


def list_scenarios() -> list[str]:
    """Return names of available scenario YAML files."""
    d = _scenarios_dir()
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.yaml"))


def load_scenario(name: str) -> SwarmConfig:
    """
    Load a named scenario from ``scenarios/<name>.yaml``.

    YAML keys map directly to SwarmConfig field names. Unknown keys
    are silently ignored for forward-compatibility.
    """
    try:
        import yaml  # lazy import - only needed when using scenarios
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for scenario loading: pip install 'bio-mimetic-swarm[scenarios]'"
        ) from exc

    path = _scenarios_dir() / f"{name}.yaml"
    if not path.exists():
        available = list_scenarios()
        raise FileNotFoundError(
            f"Scenario '{name}' not found at {path}\n"
            f"Available: {', '.join(available) or '(none)'}"
        )

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    valid_names = {f.name for f in fields(SwarmConfig)}
    filtered = {k: v for k, v in data.items() if k in valid_names}
    return SwarmConfig(**filtered)


def build_config(
    scenario: Optional[str] = None,
    **cli_overrides: object,
) -> SwarmConfig:
    """
    Build a SwarmConfig from optional scenario + CLI overrides.

    Priority: CLI flags > scenario YAML > defaults.
    """
    if scenario:
        cfg = load_scenario(scenario)
        if cli_overrides:
            current = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
            current.update({k: v for k, v in cli_overrides.items() if v is not None})
            return SwarmConfig(**current)
        return cfg

    overrides = {k: v for k, v in cli_overrides.items() if v is not None}
    return SwarmConfig(**overrides)
