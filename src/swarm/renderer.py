"""
bio-mimetic-swarm · renderer.py
================================
Dark-mode animation renderer for the swarm simulation.

Produces a two-panel output:
  Left  - Live spatial view: agent positions, tails, target, centroid
  Right - Time-series telemetry: swarm spread + mean speed

Supports GIF (Pillow) and MP4 (ffmpeg) export.
"""

from __future__ import annotations

import sys
import warnings
from typing import Any

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

from .config import SwarmConfig
from .engine import SwarmEngine

# ══════════════════════════════════════════════════════════════════════════════
# Layout constants
# ══════════════════════════════════════════════════════════════════════════════

FIGURE_SIZE = (14, 7)
FIGURE_DPI = 130
PANEL_RATIO = [1.6, 1]

AGENT_DOT_SIZE = 18
TARGET_MARKER_SIZE = 16
TARGET_RING_SIZE = 22
CENTROID_MARKER_SIZE = 10

TAIL_LINE_WIDTH = 0.6
TAIL_ALPHA_MIN = 0.04
TAIL_ALPHA_MAX = 0.45

HUD_FONT_SIZE = 7.5
TITLE_FONT_SIZE = 10.5
SUBTITLE_FONT_SIZE = 7.5
AXIS_LABEL_SIZE = 8
AXIS_TITLE_SIZE = 9
TICK_SIZE = 7.5
WATERMARK_SIZE = 6

SUBTITLE_COLOR = "#64748b"
LOG_INTERVAL = 50  # steps between progress prints


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _apply_dark_axes(ax: plt.Axes, cfg: SwarmConfig) -> None:
    """Apply consistent dark-mode styling to an Axes."""
    ax.set_facecolor(cfg.color_panel)
    ax.tick_params(colors=cfg.color_text, labelsize=TICK_SIZE)
    ax.xaxis.label.set_color(cfg.color_text)
    ax.yaxis.label.set_color(cfg.color_text)
    ax.title.set_color(cfg.color_accent)
    for spine in ax.spines.values():
        spine.set_edgecolor(cfg.color_grid)
    ax.grid(color=cfg.color_grid, linewidth=0.4, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)


def _make_tail_alpha(length: int) -> np.ndarray:
    """Linearly increasing alpha values for tail fade effect."""
    return np.linspace(TAIL_ALPHA_MIN, TAIL_ALPHA_MAX, max(length - 1, 1))


# ══════════════════════════════════════════════════════════════════════════════
# Simulation runner
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(
    engine: SwarmEngine,
    cfg: SwarmConfig,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Advance the engine for ``cfg.steps`` iterations, capturing snapshots.

    Returns list of frame dicts from ``engine.snapshot()``.
    """
    frames: list[dict[str, Any]] = []
    if verbose:
        print("[1/3] Deploying swarm from staging area ...")

    for step_idx in range(cfg.steps):
        engine.step()
        snap = engine.snapshot()

        if verbose and step_idx % LOG_INTERVAL == 0:
            pct = 100 * step_idx / cfg.steps
            print(
                f"      {pct:5.1f}%  |  t={snap['t']:6.1f}s  "
                f"|  spread={snap['spread']:5.1f}m  "
                f"|  v\u0304={snap['speed']:4.2f} m/s  "
                f"|  {snap['status']}"
            )

        frames.append(snap)

    if verbose:
        print("      100.0%  \u2713  Mission complete.")
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# Figure builder
# ══════════════════════════════════════════════════════════════════════════════

def build_figure(cfg: SwarmConfig) -> tuple[
    plt.Figure, plt.Axes, plt.Axes, dict[str, Any]
]:
    """Construct the two-panel figure layout. Returns (fig, ax_sim, ax_tele, artists)."""
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, facecolor=cfg.color_bg)

    gs = GridSpec(
        1, 2, width_ratios=PANEL_RATIO,
        left=0.05, right=0.97, top=0.88, bottom=0.09, wspace=0.14,
    )

    ax_sim = fig.add_subplot(gs[0])
    ax_tele = fig.add_subplot(gs[1])
    _apply_dark_axes(ax_sim, cfg)
    _apply_dark_axes(ax_tele, cfg)

    # ── Spatial view ──────────────────────────────────────────────────────
    half = cfg.world_size / 2.0
    ax_sim.set_xlim(-half, half)
    ax_sim.set_ylim(-half, half)
    ax_sim.set_aspect("equal")
    ax_sim.set_xlabel("East  [m]", fontsize=AXIS_LABEL_SIZE)
    ax_sim.set_ylabel("North  [m]", fontsize=AXIS_LABEL_SIZE)
    ax_sim.set_title(
        "LENNARD-JONES SWARM",
        fontsize=AXIS_TITLE_SIZE, fontfamily="monospace", pad=6,
    )

    # ── Telemetry panel ───────────────────────────────────────────────────
    ax_tele.set_xlim(0, cfg.steps * cfg.dt)
    ax_tele.set_ylim(0, cfg.world_size * 0.6)
    ax_tele.set_xlabel("Time  [s]", fontsize=AXIS_LABEL_SIZE)
    ax_tele.set_ylabel("Metric  [m  or  m/s \u00d7 5]", fontsize=AXIS_LABEL_SIZE)
    ax_tele.set_title("SWARM TELEMETRY", fontsize=AXIS_TITLE_SIZE, fontfamily="monospace", pad=6)

    # ── Artists ───────────────────────────────────────────────────────────
    scat = ax_sim.scatter(
        [], [], s=AGENT_DOT_SIZE, c=cfg.color_agents,
        alpha=0.90, zorder=5, edgecolors="none",
    )
    (tgt_h,) = ax_sim.plot(
        [], [], "+", color=cfg.color_target,
        markersize=TARGET_MARKER_SIZE, markeredgewidth=2.2, zorder=6,
    )
    (tgt_ring,) = ax_sim.plot(
        [], [], "o", color=cfg.color_target,
        markersize=TARGET_RING_SIZE, markerfacecolor="none",
        markeredgewidth=1.0, alpha=0.5, zorder=4,
    )
    (ctr_h,) = ax_sim.plot(
        [], [], "x", color=cfg.color_accent,
        markersize=CENTROID_MARKER_SIZE, markeredgewidth=1.5, alpha=0.7, zorder=5,
    )

    tail_lines = []
    for _ in range(cfg.num_agents):
        (ln,) = ax_sim.plot(
            [], [], "-", color=cfg.color_tail,
            linewidth=TAIL_LINE_WIDTH, alpha=0.3, zorder=2,
        )
        tail_lines.append(ln)

    (tele_spread,) = ax_tele.plot(
        [], [], color=cfg.color_agents, linewidth=1.2, label="Spread [m]",
    )
    (tele_speed,) = ax_tele.plot(
        [], [], color=cfg.color_target, linewidth=1.2, label="Speed\u00d75 [m/s]",
    )
    ax_tele.legend(
        loc="upper right", framealpha=0,
        fontsize=HUD_FONT_SIZE, labelcolor=cfg.color_text,
    )

    tele_t: list[float] = []
    tele_s: list[float] = []
    tele_v: list[float] = []

    # HUD overlay
    hud_text = ax_sim.text(
        0.02, 0.04, "", transform=ax_sim.transAxes,
        fontsize=HUD_FONT_SIZE, color=cfg.color_text,
        verticalalignment="bottom", fontfamily="monospace", zorder=10,
        bbox={
            "boxstyle": "round,pad=0.55", "facecolor": cfg.color_panel,
            "edgecolor": cfg.color_accent, "linewidth": 1.2, "alpha": 0.92,
        },
    )
    status_text = ax_sim.text(
        0.02, 0.96, "", transform=ax_sim.transAxes,
        fontsize=8, color=cfg.color_accent,
        verticalalignment="top", fontfamily="monospace", fontweight="bold", zorder=10,
    )

    # ── Title bar ─────────────────────────────────────────────────────────
    noise_tag = ""
    if cfg.sensor_noise > 0:
        noise_tag += f"  \u00b7  noise={cfg.sensor_noise}m"
    if cfg.dropout_rate > 0:
        noise_tag += f"  \u00b7  dropout={cfg.dropout_rate:.0%}"

    fig.text(
        0.05, 0.935,
        f"BIO-MIMETIC SWARM  //  {cfg.scenario_name.upper().replace('-', ' ')}",
        color=cfg.color_text, fontsize=TITLE_FONT_SIZE,
        fontweight="bold", fontfamily="monospace",
    )
    fig.text(
        0.05, 0.912,
        f"N={cfg.num_agents} agents  \u00b7  \u03b5={cfg.epsilon}  \u00b7  "
        f"\u03c3={cfg.sigma}m  \u00b7  k_att={cfg.target_gain}  \u00b7  "
        f"k_d={cfg.damping}  \u00b7  \u0394t={cfg.dt}s{noise_tag}",
        color=SUBTITLE_COLOR, fontsize=SUBTITLE_FONT_SIZE, fontfamily="monospace",
    )
    fig.add_artist(plt.Line2D(
        [0.05, 0.97], [0.905, 0.905],
        transform=fig.transFigure, color=cfg.color_grid, linewidth=0.8,
    ))
    fig.text(
        0.97, 0.01, "bio-mimetic-swarm  \u00b7  APF Control",
        color="#1f2937", fontsize=WATERMARK_SIZE, ha="right", fontfamily="monospace",
    )

    artists: dict[str, Any] = {
        "scat": scat, "tgt_h": tgt_h, "tgt_ring": tgt_ring, "ctr_h": ctr_h,
        "tail_lines": tail_lines,
        "tele_spread": tele_spread, "tele_speed": tele_speed,
        "hud_text": hud_text, "status_text": status_text,
        "tele_t": tele_t, "tele_s": tele_s, "tele_v": tele_v,
    }
    return fig, ax_sim, ax_tele, artists


# ══════════════════════════════════════════════════════════════════════════════
# Animation
# ══════════════════════════════════════════════════════════════════════════════

def make_update_fn(
    frames: list[dict[str, Any]],
    artists: dict[str, Any],
    cfg: SwarmConfig,
) -> Any:
    """Return the per-frame update callable for FuncAnimation."""
    tail_alphas = _make_tail_alpha(cfg.tail_length)

    def update(frame_idx: int) -> list[Any]:
        fr = frames[frame_idx]
        pos = fr["pos"]
        target = fr["target"]
        hist = fr["history"]

        artists["scat"].set_offsets(pos)
        artists["tgt_h"].set_data([target[0]], [target[1]])
        artists["tgt_ring"].set_data([target[0]], [target[1]])

        c = pos.mean(axis=0)
        artists["ctr_h"].set_data([c[0]], [c[1]])

        # Tails
        n_hist = len(hist)
        for agent_idx, ln in enumerate(artists["tail_lines"]):
            if n_hist < 2:
                ln.set_data([], [])
                continue
            xs = [h[agent_idx, 0] for h in hist]
            ys = [h[agent_idx, 1] for h in hist]
            ln.set_data(xs, ys)
            fade_idx = min(n_hist - 2, len(tail_alphas) - 1)
            ln.set_alpha(float(tail_alphas[fade_idx]))

        # Telemetry accumulators
        artists["tele_t"].append(fr["t"])
        artists["tele_s"].append(fr["spread"])
        artists["tele_v"].append(fr["speed"] * 5.0)
        artists["tele_spread"].set_data(artists["tele_t"], artists["tele_s"])
        artists["tele_speed"].set_data(artists["tele_t"], artists["tele_v"])

        # HUD
        dist_to_target = float(np.linalg.norm(c - target))
        artists["hud_text"].set_text(
            f"CONTROL: LENNARD-JONES APF\n"
            f"t    = {fr['t']:6.1f} s\n"
            f"N    = {cfg.num_agents} agents\n"
            f"\u03c3    = {cfg.sigma} m  |  \u03b5 = {cfg.epsilon}\n"
            f"\u0394    = {dist_to_target:5.1f} m  (centroid \u2192 target)\n"
            f"v\u0304    = {fr['speed']:4.2f} m/s\n"
            f"STATUS: {fr['status']}"
        )
        artists["status_text"].set_text(f"\u25b6 {fr['status']}")

        return [
            artists["scat"], artists["tgt_h"], artists["tgt_ring"],
            artists["ctr_h"], artists["hud_text"], artists["status_text"],
            artists["tele_spread"], artists["tele_speed"],
            *artists["tail_lines"],
        ]

    return update


def render_animation(
    frames: list[dict[str, Any]],
    cfg: SwarmConfig,
    verbose: bool = True,
) -> str:
    """Build and export the animation. Returns the output filepath."""
    if verbose:
        print("[2/3] Rendering animation ...")

    fig, _ax_sim, _ax_tele, artists = build_figure(cfg)
    update_fn = make_update_fn(frames, artists, cfg)

    anim = animation.FuncAnimation(
        fig, func=update_fn, frames=len(frames),
        interval=cfg.anim_interval_ms, blit=False, repeat=True,
    )

    output_path = cfg.resolved_output_file
    if verbose:
        print(f"[3/3] Saving \u2192 {output_path}  "
              f"({len(frames)} frames, {cfg.anim_fps} fps) ...")

    try:
        writer: animation.AbstractMovieWriter
        if cfg.output_format == "mp4":
            writer = animation.FFMpegWriter(fps=cfg.anim_fps)
        else:
            writer = animation.PillowWriter(fps=cfg.anim_fps)
        anim.save(output_path, writer=writer)
    except Exception as exc:
        print(f"      \u2717  Export failed: {exc}", file=sys.stderr)
        raise

    plt.close(fig)
    return output_path
