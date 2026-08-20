# Ramp manipulation — a contact-formulation comparison task

A task definition for a repository that links **GRIP** and calls it. Nothing
here belongs in GRIP itself: the reward, the policy, the planner and the
training loop are all task definitions, and GRIP's rule is that a task
belongs to whoever is posing it.

The purpose of this task is narrow and should stay narrow. It exists to make
the difference between **penalty contact** (GRIP 1.0) and an **NCP contact
solve with IFT gradients** (GRIP 2.0) *load-bearing* for a learned policy —
so that the comparison produces a signal instead of two indistinguishable
reward curves.

It is deliberately **not** experimentally rigorous. It is a demonstration
with an honest caveat attached, not a study.

---

## Why this task, and not a generic pusher

A contact-rich task is not automatically a task where the *contact model*
matters. If contact is incidental, both formulations produce the same policy
and the comparison shows nothing. This task is built around the one place
penalty contact fails **structurally** rather than by a small margin.

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
creep = mg·sin α / b_slip
```

With GRIP's demo constants (`m = 1`, `b_slip = 200`) at a 20° ramp:

```
creep = 9.81 · sin(20°) / 200 ≈ 1.68 cm/s
```

Over a 2.5-second hold that is **4.2 cm** — about 14% of a box width. Over
five seconds, 8.4 cm. This is not an artifact you have to zoom in to see;
it is the whole picture sliding.

Under an NCP solve the same box sticks exactly. Zero drift, indefinitely.

### Why the ramp angle is the design parameter

At `μ = 0.5` the friction angle is `atan(0.5) = 26.6°`. **The ramp must sit
below it.** Below the friction angle, rigid physics says a released box holds
its position for all time — a closed-form fact, not a modelling opinion.
Above it, both formulations slide and the task measures nothing.

Nominal `α = 20°`. That inequality is the entire experiment.

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

Both terms matter. The control cost is **not** boilerplate:

> The control cost is what makes the two optimal strategies genuinely
> different. Without it, "keep pressing forever" is free, and both
> formulations converge on it.

The gradient GRIP needs is a quadratic in position chained through
`∂ξ/∂q_box = (cos α, sin α, 0)` — exactly the seed shape `adjoint_system`
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

## The crux: two different optimal strategies

This is the reason the task exists, and it is worth stating explicitly.

**Under penalty**, the box is always sliding downhill. The only way to hold
position is to keep the pusher pressed against it indefinitely. The optimal
policy is *press and never let go*, and it pays the control cost forever
because there is no alternative.

**Under NCP**, static friction holds. The optimal policy is *place the box,
then retreat* — pushing after arrival is pure control cost with no benefit.

These are not two parameter settings of one strategy. They are different
strategies, and the control cost term is what separates them.

---

## Evaluation protocol

**NCP is ground truth. Every policy is scored in NCP, regardless of where it
trained.**

This direction is not optional. Scoring a policy in the *less* accurate
simulator inverts the intuitive reading — a failure there looks like the
accurate simulator produced a worse policy, and undoing that impression
costs an explanation you should not have to give.

| trained in | scored in | expected |
|---|---|---|
| NCP | NCP | works |
| penalty | NCP | **degrades** |

### The predicted failure

The penalty-trained policy has learned a feedback law tuned against a
constant phantom drag — in its world, letting go always means losing ground.
Dropped into NCP, it reaches the target, the box *sticks*, and the policy
keeps pushing because it has never once observed a box that stays put. The
box walks past the target and off the top of the ramp.

Integrator windup against a disturbance that was an artifact. One sentence
to explain, and it points the right way.

### Secondary mechanism

Penalty absorbs impacts over many timesteps, so aggressive approach speeds
are free. NCP resolves them at once, so the same approach tips or scatters
the box. Expect this to show up as the penalty-trained policy being too
violent on contact.

### The honest caveat, stated before anyone asks

Training and evaluating in the same simulator is inherently an advantage, so
NCP-on-NCP winning is partly self-fulfilling. This is not treated as cheating
in the sim-to-real literature, because the question is not whether NCP is
self-consistent — it is **how much you lose by training in the cheap
approximation**. Say this yourself rather than being asked.

---

## Learning algorithms

**PPO is deliberately excluded.** It never touches GRIP's gradients, it is
the arm most likely to look identical across formulations, and the sample
budget is brutal — at `dt = 5e-4`, one simulated second is 2000 integration
steps.

### MPPI / CEM — the zeroth-order control

Sampling-based MPC. A planner, not a learner: no training loop, no sample
budget, a couple hundred lines. It answers *"is this task solvable and what
does good behaviour look like"* immediately, and it runs identically on both
simulators.

Its job is to remove ambiguity. If SHAC struggles, this is what tells you
whether the problem is the gradients or the task.

### SHAC — the headline

Short-horizon actor-critic with a learned value function, consuming analytic
gradients through the dynamics. It maps directly onto GRIP's existing
`adjoint_system`:

- roll out a window of `W = 32` control steps
- seed `dl_dZ[t]` with `∂r_t/∂Z_t` at every step
- seed the terminal entry with `∂V/∂Z` from the critic
- read back `dJ_dU[t]` and step the policy

A result falls out of this for free: SHAC on penalty backpropagates through
`32 × 20 = 640` integration steps per window, against NCP's 32. **Twenty
times the adjoint cost and worse-conditioned gradients, from the same
training configuration.**

---

## Artifacts this produces

Ordered by how much interpretation each one needs. The first needs none.

1. **The physics plot.** Box released on the ramp, no policy, no RL.
   Displacement against time: penalty drifts 8.4 cm in 5 s, NCP sits at
   zero, and Coulomb's law says zero. Three lines, one of them analytic.
   *This carries the correctness claim on its own.*
2. **MPPI on both** — the task is solvable; here is what good looks like.
3. **SHAC on both** — the headline learning result.
4. **The cross-eval matrix**, every cell scored in NCP.
5. **The video** — penalty-trained policy overshooting off the ramp in
   accurate physics.

Keeping (1) separate from (3) is the point. The claim *"2.0 is more
physically correct"* rests on closed-form ground truth and needs no
statistics. The RL results then answer a different and complementary
question — *"does that error change what a policy learns?"* — and are free
to be modest without undermining the headline.

---

## GRIP interface

The current C++ surface is pre-API and **will change** at GRIP's step 11
(public API and Python bindings), which is also where a batched
multi-scene entry point should land. Treat the calls below as the shape, not
the contract.

```cpp
// Forward: H controls in, H+1 system states out.
trajectory = rollout_system(initial, params, shapes, plane, penalty, controls, dt);

// Backward: caller supplies ∂ℓ/∂Z and ∂ℓ/∂U seeds, receives total derivatives.
gradients = adjoint_system(trajectory, params, shapes, plane, penalty, dl_dZ, dl_dU, dt);
// gradients.dJ_dZ0     6B
// gradients.dJ_dU[t]   3B per step
```

GRIP never sees the reward. This repository computes `∂r/∂Z` and `∂r/∂U`,
hands them over as seeds, and gets total derivatives back.

The only GRIP-side convenience worth requesting is a tilted-plane
constructor; everything else this task needs already exists.

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

1. **Now, against GRIP 1.0 as it stands.** Place a box on a 20° tilted
   `HalfPlane`, `rollout_system` for five seconds with zero controls, plot
   `ξ` against time. If it drifts 8.4 cm, half of artifact (1) is done —
   before joints, before the API, before any of it.
2. Minimal task: box alone on the ramp, wrench applied directly to the box,
   no pusher. De-risks the reward, the observation space and the training
   loop with two objects and one contact set.
3. Add the pusher. Body-body contact and friction join the critical path,
   and it starts looking like manipulation rather than a physics test.
4. MPPI on penalty. Confirms solvability.
5. SHAC on penalty. Completes the 1.0 column.
6. Wait for GRIP 2.0, rerun the whole column, fill in the cross-eval matrix.

Steps 1–5 need nothing from GRIP that does not already exist.
