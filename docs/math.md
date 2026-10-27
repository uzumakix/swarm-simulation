# Mathematical Foundation

This document provides the complete derivation of the control law used in the bio-mimetic swarm simulation. For the implementation, see `src/swarm/engine.py`.

## 1. Agent Dynamics

Each agent is modelled as a **double-integrator point mass** with linear viscous damping:

$$\ddot{\mathbf{x}}_i = \mathbf{u}_i - k_d \dot{\mathbf{x}}_i$$

where:
- $\mathbf{x}_i \in \mathbb{R}^2$ is the position of agent $i$
- $\dot{\mathbf{x}}_i$ is its velocity
- $k_d$ is the damping coefficient (models hydrodynamic drag)
- $\mathbf{u}_i$ is the control input

The control law is purely reactive. It is the **negative gradient of a scalar potential field**:

$$\mathbf{u}_i = -\nabla_{\mathbf{x}_i} U_{\text{total}}(\mathbf{x}_i)$$

This means the entire swarm behaviour is encoded in the shape of $U_{\text{total}}$, and the controller is just "roll downhill."

## 2. Lennard-Jones 12-6 Potential

### Origin

The Lennard-Jones potential was introduced by John Lennard-Jones in 1924 to model intermolecular interactions in noble gases. It captures two fundamental phenomena:

1. **Pauli repulsion** at short range (electron cloud overlap), modelled by $r^{-12}$
2. **Van der Waals attraction** at long range (induced dipole interaction), modelled by $r^{-6}$

### Definition

$$U_{LJ}(r) = 4\epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right]$$

Parameters:
- $\epsilon$: depth of the potential well (bond strength)
- $\sigma$: distance at which $U_{LJ} = 0$ (effective diameter)
- $r$: inter-agent Euclidean distance

### Equilibrium Distance

The potential minimum occurs where $\frac{dU}{dr} = 0$:

$$r_{eq} = 2^{1/6} \cdot \sigma \approx 1.122 \sigma$$

At $r = r_{eq}$, the attractive and repulsive forces balance exactly. This is the natural inter-agent spacing, analogous to a molecular bond length, or the preferred separation in a fish school.

### Force Derivation

The force on agent $i$ due to agent $j$ is $\mathbf{F}_{ij} = -\nabla_{x_i} U_{LJ}(r_{ij})$:

$$F_{LJ}(r) = -\frac{dU_{LJ}}{dr} = \frac{24\epsilon}{r} \left[ 2\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right]$$

In vector form:

$$\mathbf{F}_{ij} = F_{LJ}(r_{ij}) \cdot \hat{\mathbf{r}}_{ij}$$

where $\hat{\mathbf{r}}_{ij} = (\mathbf{x}_i - \mathbf{x}_j) / r_{ij}$ is the unit vector from $j$ to $i$.

**Sign convention:**
- $r < r_{eq}$: $F > 0$ → repulsion (push apart)
- $r > r_{eq}$: $F < 0$ → attraction (pull together)
- $r > r_{cut}$: $F = 0$ → interaction cutoff

### Interaction Cutoff

Beyond $r_{cut} = 50$ m, the LJ force is negligible and is set to zero. This is standard practice in molecular dynamics (the "cutoff radius") and keeps the computation $O(N^2)$ with a bounded constant.

## 3. Navigation Potential

A global quadratic attractive potential draws every agent toward the moving target:

$$U_{att}(\mathbf{x}_i) = \frac{1}{2} k_{att} \| \mathbf{x}_i - \mathbf{x}_{target}(t) \|^2$$

Its gradient yields a linear restoring force (a "spring" toward the target):

$$\mathbf{F}_{att,i} = -\nabla U_{att} = k_{att} (\mathbf{x}_{target}(t) - \mathbf{x}_i)$$

The target trajectory is a Lissajous curve:

$$x_t(t) = A_x \sin(\omega_x t), \quad y_t(t) = A_y \sin(\omega_y t + \varphi)$$

This gives the target smooth, time-varying motion that never repeats exactly.

## 4. Total Control Law

$$\mathbf{u}_i = \underbrace{-\sum_{j \neq i} \nabla U_{LJ}(r_{ij})}_{\text{LJ swarm forces}} + \underbrace{k_{att}(\mathbf{x}_{target} - \mathbf{x}_i)}_{\text{navigation}} - \underbrace{k_d \dot{\mathbf{x}}_i}_{\text{damping}}$$

**Key insight:** No agent communicates with any other. No agent knows the global swarm state. The collective behaviour is entirely a consequence of the potential field landscape.

## 5. Numerical Integration

### Semi-Implicit (Symplectic) Euler

$$\mathbf{v}_{n+1} = \mathbf{v}_n + \mathbf{a}_n \cdot \Delta t$$
$$\mathbf{x}_{n+1} = \mathbf{x}_n + \mathbf{v}_{n+1} \cdot \Delta t$$

The difference from explicit Euler is subtle but important: the position update uses the *already-updated* velocity $\mathbf{v}_{n+1}$. This preserves the **symplectic structure** of the Hamiltonian system, which means:

- **No artificial energy drift** over long simulations
- **Phase-space volume preservation** (Liouville's theorem)
- Qualitatively correct oscillatory dynamics at moderate timesteps

Explicit Euler, by contrast, introduces systematic energy gain that causes exponential blow-up in Hamiltonian systems.

### Stability Considerations

The LJ $r^{-12}$ term diverges rapidly as $r \to 0$. Two mechanisms prevent numerical instability:

1. **Force saturation:** $\|\mathbf{F}_i\| \leq F_{max}$ clips extreme forces
2. **Velocity saturation:** $\|\mathbf{v}_i\| \leq v_{max}$ prevents runaway agents

Together with the symplectic integrator, this allows stable simulation at relatively large timesteps ($\Delta t = 0.08$ s) without energy blow-up.

## 6. Sensor Noise Model

Real hardware has imperfect range sensors. The simulation optionally adds Gaussian noise to measured pairwise distances:

$$\tilde{r}_{ij} = r_{ij} + \mathcal{N}(0, \sigma_{noise}^2)$$

The noise is symmetrised ($\tilde{r}_{ij} = \tilde{r}_{ji}$) and clamped to non-negative values. This degrades the force computation but, remarkably, the swarm maintains cohesion for noise levels up to approximately $\sigma_{noise} \approx 3$ m with $\sigma = 12$ m.

## 7. Communication Dropout

Each pairwise interaction has a probability $p_{drop}$ of being disabled per timestep. This models intermittent link failures between agents. The dropout is also symmetrised. If agent $i$ cannot sense $j$, then $j$ also cannot sense $i$.

The decentralized control law degrades gracefully under dropout because each agent still has partial information from its remaining neighbours. This is a key advantage over centralized control, where a single link failure can compromise the entire system.
