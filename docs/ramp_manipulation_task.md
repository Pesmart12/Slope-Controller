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
| shape | trapezoid, nominally `0.30 × 0.15 m`, box-facing edge tilted 3° |
| vertices (CCW, body frame) | `(−0.148043,−0.075332), (0.144095,−0.075332), (0.151957, 0.074668), (−0.148043, 0.074668)` |
| mass | `2.0` |
| inertia | `0.018364` about the true centroid |

**Wide and short, not tall and narrow.** An earlier draft specified
`0.15 × 0.3 m` at `mass = 1.0`, standing on its narrow base. That shape
**tips at 4.55 N** — below the 7.35 N it needs to do its job — so it falls
over before it can hold the box, which is why that draft also needed an
orientation PD. Flipped to `0.30 × 0.15` it tips at 18.2 N per kg, or
36.4 N at `mass = 2.0`, far beyond anything the policy can command. **No PD
is required, and none should be added** — see Actuation for why that
matters more than it looks.

**The mass is 2.0 because the force budget demands it.** The pusher has to
hold *itself and* arrest the box's creep, without ever exceeding the normal
force pinning it to the ramp:

| pusher mass | hold self + box | lift-off ceiling | headroom |
|---|---|---|---|
| 0.5 | 5.51 N | 4.55 N | **impossible** |
| 1.0 | 7.35 N | 9.10 N | 1.24× |
| 2.0 | 11.02 N | 18.19 N | 1.65× |

at the steepest sampled slope. Below about 0.68 kg there is no valid force
limit at all. `mass = 1.0` leaves a 1.75 N window to work in; `2.0` leaves
seven, which is what makes a limit choosable with margin on both sides.

**The box-facing edge is tilted 3°, so the pusher meets the box at a
vertex** rather than face-to-face — the tie-break degeneracy at the end of
this document, removed from the critical path rather than managed on it.
The tilt leans the edge *outward*, putting the contact at the **top**
vertex; leaning it the other way would put contact at the bottom vertex,
down where the ramp contact already is.

One property worth naming, because it is structural rather than lucky: the
tilt only displaces vertices in `x`, so the contact vertex always sits a
full body-height above the ramp. At **half the box's side** that is exactly
the box's COM height, so the push generates **no tipping moment on the box
at all**, for any face angle.

**Not gravity-compensated.** The pusher rests on the ramp like the box does,
so it has its own friction contact and its own creep. That is deliberate —
two creeping contacts, not one. At `mass = 2.0` it creeps twice as fast as
the box.

Vertices must wind counterclockwise; GRIP's SAT-and-clip path requires it.
They are also given **relative to the true centroid**, which the 3° trim
moves 1.96 mm off the rectangle's centre — GRIP's `BodyShape` takes vertices
in a frame centred on the COM, so an un-recentred list puts a standing
torque offset into every contact. The trapezoid's inertia is 2.1% below
what the rectangle formula gives; use the computed value. A check should
assert the centroid is at the origin when this is built.

### Initial placement

Both bodies rest flush on the ramp, so `θ_body = α` and the COM sits at
perpendicular distance `half-height` from the surface along `n`:

```
q_body = (point on ramp) + 0.15·n,   θ_body = α
```

Under penalty they settle a further `mg·cos α / 2k ≈ 0.46 mm` into the
surface — but that is the **mean** of the two bottom corners, not a uniform
sink. Friction's moment arm tilts the box, so the corners come to rest at
different depths: 0.63 mm and 0.29 mm at 20°, widening with slope. No
uniform offset reaches that state, so "start them there" is not actually
available.

Start flush and give the episode a **settle window** instead. The contact
spring is underdamped — `ζ = 0.35` at these constants — so it overshoots by
about a quarter and rings at ~50 ms per cycle; **0.2 s** covers it across
the sampled slope range. The window is also the formulation-agnostic
choice, because it runs under zero control: an NCP solve settles in it
immediately and pays nothing, where starting at the penalty equilibrium
would hand the two columns different initial conditions and quietly spoil
the comparison they exist for.

Measured in `tests/check_task.py`, which asserts the window is still long
enough — that is what catches a later change to `k`, `b` or the box.

---

## Actuation

The policy commands a **2D force on the pusher**, in the ramp frame as at
step 2, rotated into GRIP's world wrench on the way in. The torque row stays
zero.

### Why there is no orientation PD

An earlier draft held the pusher's orientation with a fixed internal PD,
supplying the `τ` row. That was a workaround for a shape that tipped over,
and the shape is fixed now — the wide pusher cannot tip under any force the
policy can command, so the PD has nothing left to do.

Removing it matters for a reason beyond tidiness. **A PD is state feedback
inside the episode**, and `adjoint_batch` treats controls as exogenous: it
answers "how does the score change if I perturb `U_t` and let the physics
respond." If `τ_t` is a function of `Z_t`, that is no longer the derivative
anyone wants, because perturbing anything upstream moves `Z_t`, which moves
`τ_t`, which moves everything after it.

Keeping all feedback in the policy means there is exactly one place where
state reaches the controls, instead of two that have to be chained
correctly and independently.

This is **not** something to ask GRIP for. A state-dependent control is a
policy, and policies belong on this side of the split. GRIP's contract that
controls are exogenous is right; the consequence is ours to handle.

### The force limit

**The limit is a box, not a magnitude.** The two components of the action
are bounded by entirely different physics, and a single magnitude cap
conflates them:

| | closed form | at 22° | limit |
|---|---|---|---|
| tangential must **exceed** | `mg(sin α + μ·cos α)` | 8.22 N | 12 N |
| normal must stay **below** | `mg·cos α` | 9.10 N | 6 N |

The tangential bound is the force that breaks the box free of friction.
The normal bound is the load pinning it to the ramp — push outward harder
and the contact unloads and the body leaves the surface.

This was got wrong first. An earlier draft sized a single 5 N magnitude cap
against `hold_force = mg·sin α = 3.7 N` alone, never against the force
needed to *move* anything, and 5 N is below the break-free threshold at
every sampled slope. **The task was literally unsolvable**, and stayed that
way through a passing gradient check, because a correct gradient on an
impossible objective is still correct. `tests/check_trajopt.py` is what
caught it.

Tipping is **not** a constraint, though the same draft claimed it was. A
wrench acts at the centre of mass, so it exerts no moment there; friction's
couple at the base is balanced by the centre of pressure shifting `μ·h` =
7.5 cm, inside the 15 cm half-width, for any force whatever. Measured: 2
mrad of tilt under 25 N.

The purpose of the normal bound is worth restating: a body that can leave
the ramp makes this task de-risk nothing about contact, which is the only
reason it exists.

### Coulomb friction has no gentle regime

Worth stating separately, because it shapes what a solution can look like.
Measured at 15°, 20° and 22°, sweeping a constant uphill force:

```
20 deg    7.0 N ->   4.6 cm      stuck; this is arrested creep, not motion
          8.0 N ->  44.2 cm      breaking free
          9.0 N -> 840.7 cm      gone
```

There is no force that produces slow, controlled sliding. Below the
threshold nothing moves; a newton above it the box accelerates away
without bound. Any solution is therefore some form of shove-and-brake, and
the precision has to come from somewhere other than modulating the push.

### Step 2: the wrench applied directly

The minimal task has no pusher, so the force lands on the box itself. Two
decisions there, recorded so they don't get rediscovered later.

**The action is in the ramp frame**, `(tangential, normal)`, rotated into
GRIP's world wrench on the way in. The slope randomizes per episode, so a
world-frame action would mean something different in every environment; a
ramp-frame one means the same thing everywhere, and a positive tangential
component always pushes toward increasing `ξ`. The torque row stays zero,
mirroring the pusher, which is shaped so it needs no orientation control.

**The action saturates componentwise at `(12, 6)` N**, per the force-limit
section above. `tests/check_task.py` asserts both inequalities and their
physical counterparts: a full second of maximum outward push leaves the box
still in contact, and a full episode of maximum uphill push moves it far
enough to reach any target and brake.

### What the baseline solution looks like

`tests/check_trajopt.py` optimizes the raw 400-step control sequence with
Adam and reaches the target to **1.4 cm** at every sampled slope, from a
reward of −1195 down to −94. So the reward is solvable and its gradients
are navigable across 8000 integration steps — which is the thing that had
to be true before SHAC was worth writing.

The *shape* of the solution is the part worth recording:

```
        peak push   above break-free   last above    moved
 15 deg   10.61 N               6.5%      0.25 s    0.51 m
 20 deg   11.88 N               7.0%      0.27 s    0.51 m
 22 deg   12.00 N               7.8%      0.30 s    0.51 m
```

A shove of about **0.3 s** does essentially all the travel, friction brakes
it, and then the force settles into the band between `hold_force` and
`break_free_force` — too little to slide the box, more than enough to make
it creep *uphill* rather than down, at `(f − mg·sin α)/2b_slip`. Measured
at 0.67 cm/s across all three slopes, closing the last centimetre or two
over the remaining 3.7 s.

**That second phase is penalty contact only.** Below the friction bound a
rigid box does not move at all, so under an NCP solve this fine-positioning
mechanism does not exist and the last centimetre has to be closed some
other way. It is the first concrete instance of the thing this project
exists to measure, and it turned up in the baseline before any policy was
trained. What a learned controller does with it is for the measurements to
say.

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

### Trajectory optimization — the baseline

**MPPI is dropped.** It was here to remove an ambiguity — if SHAC
struggles, is it the gradients or the task? — but being zeroth order it
never calls the adjoint, so it could only ever answer half of that.
`tests/check_task.py` answered the other half by finite difference, and
direct trajectory optimization answers what was left more cheaply: Adam on
the raw control sequence, one real hyperparameter, no policy and no
training loop.

It establishes two things MPPI could not:

- **The reward is solvable.** 1.4 cm to target at every sampled slope.
- **The gradients are navigable**, not merely correct. A finite-difference
  check proves correctness at a point; this crosses 8000 integration steps
  of stiff contact and ends at its best iterate. SHAC's 32-step windows are
  a far easier gradient problem than the full episode solved here.

It is also where the force limit turned out to be unsolvable, which no
amount of gradient checking would have surfaced — a correct gradient on an
impossible objective is still correct.

MPPI keeps one job in reserve, and only one: if trajectory optimization
ever *fails*, being gradient-free is exactly what would separate "the
reward is wrong" from "the gradients are right but unusable." That is a
contingency, not a milestone.

### SHAC — the one that uses the gradients

Short-horizon actor-critic with a learned value function, consuming analytic
gradients through the dynamics.

**It does not map onto one `adjoint_batch` call.** An earlier draft here
prescribed exactly that — roll out `W = 32` steps, seed `dl_dZ[t]` at every
step and the terminal entry from the critic, read back `dJ_dU[t]`, step the
policy — and that recipe is **wrong for a policy**, measured:

```
one-parameter feedback a_t = -K(xi - xi*), 200 steps, dJ/dK
  K = 25    truth  5.129    one call   5.706   ( 11.3% off)
  K = 30    truth 28.790    one call  63.187   (119.5% off)
```

`adjoint_batch` returns `∂J/∂U_t` holding the other controls fixed. For a
control *sequence* that is exactly the gradient, which is why
`tests/check_trajopt.py` optimizes cleanly. For a *policy* it is not:
`U_t = π(Z_t)`, and `Z_t` depends on every earlier control, so perturbing a
parameter moves `U_0`, which moves `Z_1`, which moves `a_1` again through
the policy. Contracting `dJ_dU` with the direct `∂π/∂θ` picks up only the
first path.

Note the error is not a constant factor — 11% at one gain, 119% at another.
It cannot be absorbed into a learning rate.

**What works** is a per-step backward sweep. Each step's adjoint call
returns both pieces: `dJ_dU_t` to contract against `∂π/∂θ`, and `dJ_dZ0` to
carry the adjoint back one step. Between calls, add the path a single call
cannot see, `Z_t → a_t → Z_{t+1}`. Exact to 2e-6 relative, in
`tests/check_closed_loop.py`.

The cost is Python round trips, not simulation: `W` calls of `substeps`
each rather than one call of `W·substeps`, so the total adjoint work is
unchanged.

This is **not** something to ask GRIP for. A state-dependent control is a
policy, and policies belong on this side of the split — GRIP's contract
that controls are exogenous is correct.

A result still falls out for free: SHAC on penalty backpropagates through
`32 × 20 = 640` integration steps per window, against NCP's 32. **Twenty
times the adjoint cost and worse-conditioned gradients, from the same
training configuration.**

---

## What this produces

1. **The drift plot.** Box released on the ramp, no policy, no RL.
   Displacement against time: penalty drifts 4.7 cm in 5 s, NCP sits at
   zero, and Coulomb's law says zero. **Done** — `experiments/drift.py`.
2. **SHAC on both** — the learned result.
3. **The cross-eval table.**
4. **A video** of whichever cell turns out to be the interesting one.

(1) is a measurement against a closed form and stands on its own whatever
the rest do. The rest are the build, and they are allowed to be modest.

The trajectory-optimization baseline is deliberately **not** on this list.
It is machinery — it produces no number that means anything without an NCP
column to set it against, and unlike the drift plot it has no closed form
standing behind it. It lives in `tests/`, not `experiments/`.

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

**Decided, and already in the geometry above:** the pusher's box-facing
edge is tilted 3°, so it meets the box at a vertex and the tie never
arises. The alternative was to accept it and note it — randomization and
floating point mean the exact tie is essentially never hit, and the
neighbourhood is well-behaved — but the hazard sits exactly where the
policy spends most of its time, and the cost of removing it turned out to
be a 7.86 mm trim.

Two consequences of the trim are easy to miss, and both are recorded with
the vertex list: the centroid moves 1.96 mm, so the vertices have to be
re-centred before GRIP sees them, and the inertia is 2.1% below the
rectangle formula.

---

## Build order

1. **Done.** Place a box on a 20° tilted `HalfPlane`, roll out five seconds
   of zero controls, plot `ξ` against time. It drifts 4.7 cm —
   `experiments/drift.py`.
2. **Done.** Minimal task: box alone on the ramp, wrench applied directly,
   no pusher — `slope_control/task.py`, `tests/check_task.py`. De-risked
   the reward and the gradient path. It did **not** de-risk the observation
   space, which nothing consumes before SHAC.
3. **Done.** Trajectory-optimization baseline —
   `tests/check_trajopt.py`. Confirms the reward is solvable and its
   gradients navigable, and it is what caught the unsolvable force limit.
   Replaces the MPPI step.
4. **Next.** Add the pushers. Body-body contact and friction join the
   critical path, and it starts looking like manipulation rather than a
   physics test. **Two** pushers, not one — see below.
5. SHAC on penalty. Completes the 1.0 column.
6. Wait for GRIP 2.0, rerun the whole column, fill in the cross-eval matrix.

Steps 1–5 need nothing from GRIP that does not already exist.

### Why two pushers and not one

A convex pusher can only push, and which direction it pushes is fixed by
which side of the box it starts on — moving it does not change that. It
cannot get to the other side either: the box is twice its height, going
over means leaving the ramp, and `BodyShape` is convex-only so there is no
hook that could pull.

So with one pusher the box is drivable in one direction only, and
**overshoot is unrecoverable**. Not merely expensive:

```
overshoot recovery, if creep is the only restoring mechanism
  15°  0.63 cm/s -> 7.9 s to undo 5 cm   |  NCP: never
  22°  0.92 cm/s -> 5.4 s                |  NCP: never
```

against a 4 s episode. An earlier draft here claimed penalty "partially
forgives" overshoot where a solve does not — it does not, at the timescale
the task actually has. Neither forgives it, so the asymmetry is not a
measurement worth having, just brittleness in both columns.

Two pushers, one either side, make the box bidirectionally drivable. Cost
is three bodies at 98 episodes/s, and GRIP needs nothing new — it already
sweeps every `i < j` pair.

One consequence to size before building: the driving pusher can carry
itself and the box comfortably (1.65× headroom at `mass = 2.0`), but it
**cannot shove a passive stack** at 22° — that needs 18.37 N against an
18.19 N ceiling, and more pusher mass raises both sides of the inequality
rather than fixing it. So the trailing pusher has to actively yield. That
is real bilateral manipulation rather than bulldozing, but it means a naive
policy is physically impossible at the steep end of the slope range, which
is worth knowing before blaming the learner.
