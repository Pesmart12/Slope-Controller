# Ramp manipulation — task definition

A task definition for a repository that links **GRIP** and calls it. Nothing
here belongs in GRIP itself: the reward, the policy, the planner and the
training loop are all task definitions, and GRIP's rule is that a task
belongs to whoever is posing it.

The plan: build the whole stack against **penalty contact** (GRIP 1.0), take
measurements, then rerun it against the **NCP solve** (GRIP 2.0) and put the
two sets of numbers side by side. One task, no seed sweeps, no claim that any
of it generalizes — a build with measurements attached. What the comparison
shows is for the numbers to say.

---

## Why a ramp

A contact-rich task is not automatically a task where the *contact model*
matters. If contact is incidental, the formulation barely shows up in the
result. A ramp below the friction angle is the opposite case: it is the one
place penalty contact fails **structurally** rather than by a small margin,
and the size of that failure is a number you can write down in advance.

### The mechanism: friction creep

Penalty contact keeps the friction *force* constraint exact — `|β| ≤ μλ`
holds at every state by construction — and softens the *kinematic* one.
Sticking is not `s = 0`; it is

```
s = −β / b_slip
```

So a box resting on a slope must be **slipping** in order to generate the
friction that holds it up. It creeps, forever, at

```
creep = mg·sin α / (2·b_slip)
```

The 2 is the number of contact points. A box resting on a face touches at
both bottom corners, each carrying its own friction, so the pair together
supply `2·b_slip·s`. An earlier draft of this document omitted that factor
and predicted twice the real drift.

With GRIP's demo constants (`m = 1`, `b_slip = 200`) at a 20° ramp the
closed form gives 0.84 cm/s and simulation measures **0.95 cm/s**:

```
5 s of hold  →  4.7 cm of drift, about a sixth of a box width
```

Not an artifact you have to zoom in to see — the whole picture slides.

**The closed form is exact only while both corners stick.** It matches to
five figures from 5° to 18°, then departs sharply: 13% low at 20°, 48% low
at 26°. Friction acts at the contact points, 0.15 m below the centre of
mass, so it tilts the box by about a milliradian; the tilt redistributes
normal force between the corners by `k·w·δθ`, and the lightly loaded
uphill corner saturates on its own cone bound `μλ`. The downhill corner
then makes up the shortfall and creeps faster than an even split would.
Both corners saturating *is* the friction angle, where the box slides
freely and none of this applies.

So: exact below ~19°, a lower bound above it. Measured in
`experiments/drift.py`, whose right-hand panel is that crossover.

Under an NCP solve the same box sticks exactly. Zero drift, indefinitely.

### Why the ramp angle is the design parameter

At `μ = 0.5` the friction angle is `atan(0.5) = 26.6°`. **The ramp must sit
below it.** Below the friction angle, rigid physics says a released box holds
its position for all time — a closed-form fact, not a modelling opinion.
Above it, both formulations slide and the task measures nothing.

Nominal `α = 20°`. Everything else here is built on top of that
inequality holding.

### Why not penetration

Penalty's other headline artifact — bodies sinking by `mg/2k` — is
**0.49 mm** at `k = 1e4`. That is a fine visual for a static side-by-side,
but it is far too small for a policy to build a strategy around. Creep is
roughly a hundred times larger in effect.

Build the RL story on creep and impact softness. Keep penetration for the
static physics comparison.

---

## Scene

Three contact sets are live: box–ramp, pusher–ramp, and pusher–box. GRIP's
`penalty_forces_system` already sweeps all of it — per-body plane contacts
plus every `i < j` pair — so no new physics is required.

### Ramp

A tilted `HalfPlane`, not a body. Uphill is `+x`:

```
n = (−sin α, cos α)        o = 0        free space is  n·p ≥ 0
```

At `α = 0` this reduces to `(0, 1)` and recovers the ground plane, which is
a cheap sanity check. Gravity `(0, −g)` projected onto the uphill direction
`(cos α, sin α)` gives `−g·sin α`, i.e. downhill. Correct by construction.

### Box — the manipulated object

| | |
|---|---|
| shape | square, side `s = 0.3 m` |
| vertices (CCW, body frame) | `(−0.15,−0.15), (0.15,−0.15), (0.15,0.15), (−0.15,0.15)` |
| mass | `1.0` |
| inertia | `m·s²/6 = 0.015` |

### Pusher — the actuated body

| | |
|---|---|
| shape | rectangle, `0.15 × 0.3 m` |
| vertices (CCW, body frame) | `(−0.075,−0.15), (0.075,−0.15), (0.075,0.15), (−0.075,0.15)` |
| mass | `1.0` |
| inertia | `m(w²+h²)/12 = 0.009375` |

**Not gravity-compensated.** The pusher rests on the ramp like the box does,
so it has its own friction contact and its own creep. That is deliberate —
two creeping contacts, not one.

Vertices must wind counterclockwise; GRIP's SAT-and-clip path requires it.

### Initial placement

Both bodies rest flush on the ramp, so `θ_body = α` and the COM sits at
perpendicular distance `half-height` from the surface along `n`:

```
q_body = (point on ramp) + 0.15·n,   θ_body = α
```

Under penalty they will settle a further `mg·cos α / 2k ≈ 0.46 mm` into the
surface. Either start them there or give the episode a brief settle window.

---

## Actuation

The policy commands a **2D force on the pusher**, `(fx, fy)`, in world frame.

A fixed internal PD holds the pusher's orientation at `θ = α`, supplying the
`τ` component of the wrench. This is a stand-in for a wrist and it lives in
this repository, not in GRIP. Without it, a free-floating pusher tumbles and
early training is miserable.

Giving the policy all three components is a legitimate alternative — it just
costs training time and buys nothing the comparison needs.

---

## Objective

Let `ξ = p·(cos α, sin α)` be position measured **along** the ramp, and `ξ*`
the target.

```
r_t = −w_pos·(ξ_box − ξ*)²  −  w_ctrl·‖u_t‖²
```

Both terms matter. The control cost is **not** boilerplate: without it,
"keep pressing forever" costs nothing and is a perfectly good strategy under
either contact model. With it, holding position has a price, and how a policy
chooses to pay that price is the thing worth watching.

Under penalty the box is always sliding, so holding it needs continuous force.
Under a solve it sticks, and force after arrival buys nothing. Same reward
function, two different bills.

The gradient GRIP needs is a quadratic in position chained through
`∂ξ/∂q_box = (cos α, sin α, 0)` — exactly the seed shape `adjoint_batch`
already accepts. No new GRIP surface is required.

---

## Episode

**Control runs at 100 Hz in both simulators.** This is the detail that keeps
the comparison fair:

| | sim `dt` | substeps per action |
|---|---|---|
| penalty (1.0) | `5e-4` | 20 |
| NCP (2.0) | `1e-2` | 1 |

Without decoupling control rate from integration rate you are comparing two
different control problems and the result is meaningless.

Episode length **4 s = 400 control steps**: roughly 1.5 s to reach, 2.5 s to
hold. The hold window is where the formulations diverge, so it should be the
majority of the episode.

### Randomization

- initial `ξ_box`
- target `ξ*`
- ramp angle `α ∈ [15°, 22°]` — all strictly below the 26.6° friction angle

Enough to stop the policy memorizing a single trajectory, not enough to turn
this into a robustness study.

### Physics constants

Take GRIP's demo values; they are already exercised in a running program.

```
k = 1e4      b = 50      b_slip = 200      μ = 0.5      g = 9.81
```

For 2.0: `dt = 1e-2`, `μ = 0.5`, restitution 0, and whatever central-path
relaxation `κ` the solver exposes.

---

## Measurements

Every policy is scored in **NCP**, wherever it trained — not because a solve
is a perfect reference, but because it is the more accurate of the two, and
scoring in the sloppier one inverts the reading. A poor score there looks
like the accurate simulator produced a worse policy.

| trained in | scored in | |
|---|---|---|
| penalty | penalty | sanity check — did training work at all |
| NCP | NCP | the same, on the other side |
| penalty | NCP | the interesting cell |

Per cell, recorded: final distance to target, how that distance evolves
across the hold window, total control effort, and whether the box stays on
the ramp.

Training and evaluating in the same simulator is an advantage, so the two
matched cells are partly self-fulfilling. Worth saying alongside the numbers
rather than waiting to be asked.

### What to instrument for

Two mechanisms are plausible enough to measure deliberately, without assuming
either turns up:

**A feedback law tuned against constant drag.** Under penalty, letting go
always means losing ground. The same law in NCP, where the box sticks, has no
reason to stop pushing on arrival — so log position across the hold window,
and whether the box leaves the top of the ramp.

**Approach speed.** Penalty absorbs impacts over many timesteps, so a fast
approach is cheap. A solve resolves them at once. Log contact velocity at
first touch, and whether the box tips.

Instrument both. Report whichever shows up.

---

## Learning algorithms

**PPO is skipped.** It never touches GRIP's gradients, which are the thing
GRIP uniquely provides, and the sample budget is brutal — at `dt = 5e-4`,
one simulated second is 2000 integration steps. Not a judgement about PPO;
it is just not what this project is for.

### MPPI / CEM — the zeroth-order control

Sampling-based MPC. A planner, not a learner: no training loop, no sample
budget, a couple hundred lines. It answers *"is this task solvable and what
does good behaviour look like"* immediately, and it runs identically on both
simulators.

Its job is to remove ambiguity. If SHAC struggles, this is what tells you
whether the problem is the gradients or the task.

### SHAC — the one that uses the gradients

Short-horizon actor-critic with a learned value function, consuming analytic
gradients through the dynamics. It maps directly onto `adjoint_batch`:

- roll out a window of `W = 32` control steps
- seed `dl_dZ[t]` with `∂r_t/∂Z_t` at every step
- seed the terminal entry with `∂V/∂Z` from the critic
- read back `dJ_dU[t]` and step the policy

A result falls out of this for free: SHAC on penalty backpropagates through
`32 × 20 = 640` integration steps per window, against NCP's 32. **Twenty
times the adjoint cost and worse-conditioned gradients, from the same
training configuration.**

---

## What this produces

1. **The drift plot.** Box released on the ramp, no policy, no RL.
   Displacement against time: penalty drifts 4.7 cm in 5 s, NCP sits at
   zero, and Coulomb's law says zero. **Done** — `experiments/drift.py`.
2. **MPPI on both** — what good behaviour looks like, without a training
   loop in the way.
3. **SHAC on both** — the learned result.
4. **The cross-eval table.**
5. **A video** of whichever cell turns out to be the interesting one.

(1) is a measurement against a closed form and stands on its own whatever
the rest do. The rest are the build, and they are allowed to be modest.

---

## GRIP interface

GRIP 1.0 shipped, so this is the installed Python surface rather than a
sketch of one. State is `(environments, bodies, 6)`, controls are
`(steps, environments, bodies, 3)`.

```python
trajectory = grip.rollout_batch(scenes, initial, controls, substeps)
dJ_dZ0, dJ_dU = grip.adjoint_batch(scenes, trajectory, controls, substeps, dl_dZ, dl_dU)
```

GRIP never sees the reward. This repository computes `∂r/∂Z` and `∂r/∂U`,
hands them over as seeds, and gets total derivatives back.

`substeps` is what decouples control rate from integration rate, so one call
advances a whole control step and the same 100 Hz policy runs on both
formulations.

Every environment in a batch carries its own `Scene`, so ramp angle, masses
and contact parameters randomize across a batch for free — only the body
count has to match. That is what makes the randomization above cheap.

2.0 is expected to change the contact model behind these calls rather than
the calls themselves.

---

## Known hazard: parallel faces sit on a degeneracy

GRIP's own `pair_detection.md` flags this, and this task walks straight into
it.

When two faces are parallel and overlapping, both bodies' SAT queries report
*exactly* the same penetration, so which body owns the contact normal is a
tie. Tilting either way hands ownership to the same body, which makes exact
parallel an **isolated point whose tie-break disagrees with its entire
neighbourhood**. The analytic Jacobian there is not the limit from either
side.

A flat pusher pressed squarely against a flat box is precisely that
configuration. It is measure-zero in floating point and the forward
simulation is fine, but **SHAC gradients taken near it are suspect**.

Two options, and this is a judgment call worth making deliberately:

- **Accept it and note it.** Randomization and floating point mean the exact
  tie is essentially never hit, and the surrounding neighbourhood is
  well-behaved.
- **Angle the pusher's contact face by 2–3°**, so it meets the box at a
  vertex rather than flush. Cheap insurance, slightly less clean-looking,
  and it removes the degeneracy from the critical path entirely.

Recommended: angle it. The cost is cosmetic and the hazard sits exactly where
the policy spends most of its time.

---

## Build order

1. **Done.** Place a box on a 20° tilted `HalfPlane`, roll out five seconds
   of zero controls, plot `ξ` against time. It drifts 4.7 cm —
   `experiments/drift.py`.
2. Minimal task: box alone on the ramp, wrench applied directly to the box,
   no pusher. De-risks the reward, the observation space and the training
   loop with one object and one contact set.
3. Add the pusher. Body-body contact and friction join the critical path,
   and it starts looking like manipulation rather than a physics test.
4. MPPI on penalty. Confirms solvability.
5. SHAC on penalty. Completes the 1.0 column.
6. Wait for GRIP 2.0, rerun the whole column, fill in the cross-eval matrix.

Steps 1–5 need nothing from GRIP that does not already exist.
