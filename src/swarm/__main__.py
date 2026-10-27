"""
bio-mimetic-swarm · __main__.py
================================
CLI entry point: ``python -m swarm``

Supports named scenarios, parameter overrides, noise injection,
and multiple output formats. Run ``python -m swarm --help`` for details.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from . import __version__
from .config import build_config, list_scenarios
from .engine import SwarmEngine
from .renderer import render_animation, run_simulation

# ── CLI argument → SwarmConfig field mapping ──────────────────────────────
# Keys are argparse dest names, values are SwarmConfig field names.
# This single dict replaces 12 if-statements.

_ARG_TO_CONFIG: dict[str, str] = {
    "agents": "num_agents",
    "epsilon": "epsilon",
    "sigma": "sigma",
    "target_gain": "target_gain",
    "damping": "damping",
    "sensor_noise": "sensor_noise",
    "dropout": "dropout_rate",
    "dt": "dt",
    "steps": "steps",
    "format": "output_format",
    "output": "output_file",
    "fps": "anim_fps",
    "seed": "init_seed",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        prog="swarm",
        description="Bio-Mimetic Swarm - decentralised multi-agent simulation "
                    "using Lennard-Jones artificial potential fields.",
        epilog=(
            "Examples:\n"
            "  python -m swarm                              # default drifting-target scenario\n"
            "  python -m swarm --scenario tight-crystal      # named preset\n"
            "  python -m swarm --agents 100 --epsilon 40     # custom parameters\n"
            "  python -m swarm --sensor-noise 2 --dropout .1 # noisy conditions\n"
            "  python -m swarm --format mp4 -o demo.mp4      # MP4 output\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument(
        "--list-scenarios", action="store_true",
        help="list available scenario presets and exit",
    )

    g_scenario = p.add_argument_group("scenario")
    g_scenario.add_argument(
        "--scenario", "-s", type=str, default=None,
        help="named scenario preset (e.g. drifting-target, tight-crystal)",
    )

    g_physics = p.add_argument_group("physics")
    g_physics.add_argument("--agents", type=int, default=None, help="number of agents")
    g_physics.add_argument("--epsilon", type=float, default=None, help="LJ well depth ε")
    g_physics.add_argument("--sigma", type=float, default=None, help="LJ equilibrium base σ [m]")
    g_physics.add_argument("--target-gain", type=float, default=None, help="navigation gain k_att")
    g_physics.add_argument("--damping", type=float, default=None, help="damping coefficient k_d")

    g_realism = p.add_argument_group("realism")
    g_realism.add_argument(
        "--sensor-noise", type=float, default=None,
        help="Gaussian noise std-dev on distance measurements [m]",
    )
    g_realism.add_argument(
        "--dropout", type=float, default=None,
        help="fraction of pairwise links randomly dropped per step",
    )

    g_numerics = p.add_argument_group("numerics")
    g_numerics.add_argument("--dt", type=float, default=None, help="timestep [s]")
    g_numerics.add_argument("--steps", type=int, default=None, help="total simulation steps")
    g_numerics.add_argument("--seed", type=int, default=None, help="random seed")

    g_output = p.add_argument_group("output")
    g_output.add_argument(
        "--format", type=str, default=None, choices=["gif", "mp4"],
        help="output format (default: gif)",
    )
    g_output.add_argument("--output", "-o", type=str, default=None, help="output filename")
    g_output.add_argument("--fps", type=int, default=None, help="animation frames per second")
    g_output.add_argument("--quiet", "-q", action="store_true", help="suppress progress output")

    return p.parse_args(argv)


def _collect_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Extract non-None CLI arguments and map them to SwarmConfig field names."""
    overrides: dict[str, Any] = {}
    for arg_name, config_name in _ARG_TO_CONFIG.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[config_name] = value
    return overrides


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    # List scenarios and exit
    if args.list_scenarios:
        scenarios = list_scenarios()
        if scenarios:
            print("Available scenarios:")
            for s in scenarios:
                print(f"  {s}")
        else:
            print("No scenario files found in scenarios/ directory.")
            print("Install PyYAML and add .yaml files to scenarios/")
        return

    # Build config
    try:
        cfg = build_config(scenario=args.scenario, **_collect_overrides(args))
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    verbose = not args.quiet

    if verbose:
        print("╔" + "═" * 59 + "╗")
        print(f"║  bio-mimetic-swarm v{__version__}  ·  "
              f"APF Simulation Engine{' ' * 14}║")
        print(f"║  {cfg.summary()[:57]:<57}║")
        print("╚" + "═" * 59 + "╝")
        print()

    # Simulate
    engine = SwarmEngine(cfg)
    frames = run_simulation(engine, cfg, verbose=verbose)

    # Render
    output_path = render_animation(frames, cfg, verbose=verbose)

    if verbose:
        print()
        print("╔" + "═" * 59 + "╗")
        print(f"║  ✓ Complete - open {output_path:<40}║")
        print("╚" + "═" * 59 + "╝")


if __name__ == "__main__":
    main()
