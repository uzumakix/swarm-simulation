# Multi-agent spatial coverage with Lennard-Jones potential fields

Sixty agents released as a random cloud settle into a lattice whose nearest-neighbour spacing is
8.907 metres, against the 8.980 metres the Lennard-Jones potential predicts for this setting of
sigma. That is a difference of 0.81 per cent, and 95 per cent of agents sit between 8.858 and 8.980
metres. No agent is told where to go: each one computes its own acceleration from the distances to
its neighbours and from one attraction term pointing at the target.

With damping switched off, total energy changes by 0.067 per cent over 20,000 steps, with the largest
excursion along the way reaching 0.508 per cent. That is the test that matters for the integrator,
because an ordinary Euler step bleeds or gains energy steadily and this one does not.

![Sixty agents at rest in a lattice, beside a histogram of nearest-neighbour distances with the
Lennard-Jones minimum marked](results/figures/01_spacing_vs_theory.png)

## What the agents actually do

Each agent is a point mass. At every step it feels two things: a Lennard-Jones force from every other
agent within the cutoff, and a pull toward the current target position. Nothing else. There is no
shared map, no plan handed down, and no agent that knows what the others are doing beyond the
distances it can measure.

The Lennard-Jones force is the one used for molecules:

```
F(r) = 24 e [ 2 (s/r)^12 - (s/r)^6 ] / r
```

Close in it pushes hard, which keeps agents from colliding. Further out it pulls gently, which keeps
the group together. Between the two there is one distance where the force is zero, at `2^(1/6) s`,
and that is the spacing the swarm should settle at. Measuring whether it does is the first check
above, and it is the reason this potential was chosen over a plain spring: a spring has no hard core,
so agents pass through one another.

## The three checks

Every number below comes from `make_results.py` in this repository, run with seed 20261026.

**Spacing at rest.** 60 agents, sigma 8 metres, epsilon 150, 4000 steps at dt 0.05, attraction
switched off so only the inter-agent forces act. Median nearest-neighbour distance 8.907 metres
against a predicted 8.980. The small shortfall is expected: agents on the outside of the lattice have
fewer neighbours pushing back, so the group packs slightly tighter than the two-body minimum.

**Energy drift.** 40 agents, damping set to zero, 20,000 steps at dt 0.01, started from a relaxed
lattice rather than a random cloud. Starting from a cloud is meaningless here, because two agents
landing almost on top of each other send the `r^-12` term to an enormous value and swamp everything
else. Total energy, kinetic plus potential, moves by 0.067 per cent end to end.

![Total energy against time with damping off, staying within half a per cent of its starting
value](results/figures/02_energy_drift.png)

**Following a moving target.** 50 agents, sigma 12 metres, the target tracing a slow Lissajous path.
The centre of the swarm stays a median 0.65 metres from the target once it has settled. Spacing holds
at 12.37 metres while moving, below the 13.47 metre minimum for this sigma, because the pull toward
the target squeezes the group slightly. That compression is the honest result: formation and
navigation trade against each other, and the attraction gain sets where the balance lands.

![Swarm centre tracking a moving target, beside nearest-neighbour spacing held steady over
time](results/figures/03_target_following.png)

## Running it

```bash
pip install -r requirements.txt
PYTHONPATH=src python make_results.py
```

For the animation instead of the figures:

```bash
PYTHONPATH=src python -m swarm --scenario tight-crystal
```

`python -m swarm --list-scenarios` shows the presets in `configs/`.

## The C kernel

`src/swarm/forces_c.c` computes the pairwise forces in C, loaded through ctypes. It needs building
first:

```bash
gcc -O2 -shared -o src/swarm/forces_c.dll src/swarm/forces_c.c    # Windows
gcc -O2 -shared -fPIC -o src/swarm/forces_c.so src/swarm/forces_c.c
```

Without it the code falls back to a pure Python loop and warns that it has done so. The numbers above
were produced with the vectorised NumPy path in `engine.py`, which is what runs when you have no
compiler, so nothing here depends on the C build.

## What this does not do

- Agents feel each other through exact distances. Real hardware measures with error, and the
  `--sensor-noise` and `--dropout` flags exist to explore that, although no result above uses them.
- The boundary is a hard wall with a restitution of 0.5. Agents bounce off it rather than being
  steered away, which is cruder than anything a real system would do.
- The target path is a fixed curve rather than something that reacts to the swarm.
- What is measured is how the agents distribute themselves: the spacing they hold, whether that
  spacing matches theory, and whether it survives being dragged across the domain. Sweeping an area
  so that every cell gets visited is a related but separate objective, and it is not tested here.

## References

1. Lennard-Jones, J. E. (1931). Cohesion. *Proceedings of the Physical Society*, 43(5), 461-482.
2. Khatib, O. (1986). Real-time obstacle avoidance for manipulators and mobile robots.
   *The International Journal of Robotics Research*, 5(1), 90-98.
3. Hairer, E., Lubich, C. & Wanner, G. (2003). Geometric numerical integration illustrated by the
   Stormer-Verlet method. *Acta Numerica*, 12, 399-450.

## License

MIT License. See [LICENSE](LICENSE).
