# CLAUDE.md — slope-control

## What this project is

Control and reinforcement learning built on top of
[GRIP](https://github.com/Pesmart12/GRIP), a 2D differentiable rigid-body
contact simulator.

GRIP simulates and differentiates, and deliberately contains no policies,
rewards, or training loops. This is the other half of that split — the
repository that poses tasks and solves them. The rule cuts both ways: if
something requires knowing how the physics works, it belongs in GRIP; if a
consumer repository could reasonably implement it, it belongs here. Rewards,
observation spaces, planners, training loops and evaluation scripts are all
on this side of the line, without exception.

The plan in one sentence: **build an RL control stack on GRIP 1.0's penalty
contact, measure what it does, rebuild it on 2.0's NCP solve, measure again,
and put the two sets of numbers side by side.**

## This is not a research project

Recorded here because it was written the other way first, and rewritten.

An earlier draft of the README and the task doc posed a research question —
*does the choice of contact formulation change what a policy learns?* — with
a predicted failure, an evaluation protocol built to produce it, and a
pre-emptive caveats section. That framing is gone on purpose. It cannot
answer the question it poses (one task, one seed, no sweeps), and it decided
the result before taking the measurement.

What replaces it is smaller and true. Penalty contact creeps at a rate you
can write down in closed form; a rigid solve does not. That is a measurement
against Coulomb's law, and it needs no interpretation. Everything after it is
a build, and its results get reported as what they are. If the
penalty-trained policy misbehaves in NCP, show it. If it doesn't, say that.

**No hypotheses, no predicted results written as expectations, no defending a
claim.** GRIP's `CLAUDE.md` carries the same rule for the same reason.

The eventual comparison is also loose by design — these numbers will sit
next to other projects' numbers someday, informally. That is not a reason to
build experimental rigor nobody asked for. It is a reason to keep the
measurements honest and easy to read.

## How we work

Claude writes code — scenes, rewards, planners, training loops, plotting.
Move fast.

Pedro reviews, redirects, and makes the calls. If a decision has real
tradeoffs, surface them and ask rather than picking silently.

Two things Claude owes on anything non-trivial:

- **Explain what it wrote** — why this formulation and not the alternative,
  in a short paragraph in the response rather than a lecture.
- **Flag the judgment calls**, so they don't silently harden into
  assumptions nobody remembers making.

This file is **guidelines, not scripture**. It records decisions that were
made deliberately so they aren't re-litigated by accident. If following it
would produce something worse, say so.

## What exists

Four source files. The repository is small and should stay legible.

| | |
|---|---|
| `slope_control/ramp.py` | the ramp scene and its geometry conventions, shared by everything |
| `experiments/drift.py` | artifact 1 — the drift measurement and its figure |
| `figures/drift.png` | committed output, so results are visible without running anything |
| `docs/ramp_manipulation_task.md` | the task definition; the authority on scene numbers, reward and episode structure |

The one result so far, reproducible by `python experiments/drift.py`:

```
20° ramp, zero controls, 5 s   ->   4.75 cm of drift, where rigid physics says 0
closed form mg·sin α / (2·b_slip) exact to five figures up to 18°
departs at 19°, 48% low at 26°
```

Nothing else is built. No policy, no planner, no training loop, no tests, no
packaging.

## Build order

`docs/ramp_manipulation_task.md` is the authority; this is the short version
so a session knows where it is.

1. **Done.** Box on a 20° tilted `HalfPlane`, five seconds of zero controls,
   plot `ξ` against time.
2. **Next.** Minimal task — box alone, wrench applied directly to the box, no
   pusher. De-risks the reward, the observation space and the training loop
   against one contact set.
3. Add the pusher. Body-body contact and friction join the critical path.
4. MPPI on penalty. Confirms the task is solvable.
5. SHAC on penalty. Completes the 1.0 column.
6. Wait for GRIP 2.0, rerun the column, fill in the cross-eval table.

Steps 2–5 need nothing from GRIP that does not already exist. **Do not build
toward 2.0 while the 1.0 column is unfinished** — there is no NCP solve to
build against, and shaping code around one that doesn't exist is how the
column stays unfinished.

## The GRIP interface

GRIP 1.0 is installed and callable. The whole Python surface is five types
and three functions:

```python
grip.Scene(params=[...], shapes=[...], plane=..., penalty=..., dt=..., gravity=...)
grip.RigidBodyParams, grip.BodyShape, grip.HalfPlane, grip.PenaltyParams

trajectory = grip.rollout_batch(scenes, initial, controls, substeps)
dJ_dZ0, dJ_dU = grip.adjoint_batch(scenes, trajectory, controls, substeps, dl_dZ, dl_dU)
grip.step_batch(...)
```

State is `(environments, bodies, 6)` packed `(x, y, θ, vx, vy, ω)`; controls
are `(steps, environments, bodies, 3)` as a wrench at the centre of mass.
`rollout_batch` returns `(steps + 1, ...)` — one state per control step plus
the initial one.

Three things follow, and they shape most of the code here:

- **GRIP never sees the reward.** This repository computes `∂r/∂Z` and
  `∂r/∂U` and hands them over as seeds, and gets total derivatives back. A
  cost is a task definition, and tasks belong to whoever poses them.
- **`substeps` decouples control rate from integration rate.** Control runs
  at 100 Hz on both formulations; penalty integrates at `dt = 5e-4`, so
  `substeps = 20`. Without that decoupling the 1.0 and 2.0 columns are two
  different control problems and the comparison means nothing.
- **Every environment carries its own `Scene`.** Ramp angle, masses and
  contact parameters randomize across a batch for free; only the body count
  has to match. That is what makes the task's per-episode ramp angle cheap.

2.0 is expected to change the contact model behind these calls rather than
the calls themselves.

## Conventions

Fixed in `slope_control/ramp.py` so nothing downstream rederives them:

```
uphill = ( cos α,  sin α)     normal = (-sin α,  cos α)     offset = 0
```

At `α = 0` the normal is `(0, 1)` and the scene reduces to flat ground. That
is the cheap check that the signs are right; keep it working.

- **`α` is the ramp angle, `ξ` is position along the ramp, `μ` is friction.**
  In code these are `ramp_angle`, `xi` and the penalty dict's `friction`.
- **The ramp angle stays below the friction angle**, `atan(μ) = 26.6°` at
  `μ = 0.5`. Above it both formulations slide and the scene measures nothing.
  `experiments/drift.py` refuses such an angle outright — keep that guard in
  anything that takes an angle from a user.
- **Physics constants come from GRIP's demo values**, so they describe a
  configuration already exercised by a running program: `k = 1e4`, `b = 50`,
  `b_slip = 200`, `μ = 0.5`, `g = 9.81`.
- **Polygon vertices wind counterclockwise.** GRIP's SAT-and-clip path
  requires it and will not tell you politely.

`docs/derivations/notation.md` in GRIP is the canonical symbol table for
anything shared. Don't reuse one of its symbols for something else here.

## Code style

Pedro's preferences, the same ones GRIP uses where they carry over to Python.

- **Function parameters go on one line**, however long the line gets. Never
  wrap a parameter list one-per-line, in definitions or calls.
- **Two blank lines between top-level definitions.**
- **No type annotations.** None of the existing code has them; don't add them
  to code you touch.
- **Docstrings carry the reasoning, not just the signature.** `creep_rate` is
  the model: the formula, why the factor of 2 is there, and exactly where the
  closed form stops being true. A docstring that restates the function name
  is not worth the lines.
- **ASCII in code, Unicode in Markdown.** Python files use `--` and `alpha`;
  `.md` files use — and α. Don't mix them.
- **Comments explain why.** The what is already on the line above.
- No commented-out code, no TODO placeholders in reviewed paths.

## Known snags

Real, and each one will bite in a specific place:

- **`resting_state` hardcodes `(1, 1, 6)`** — one environment, one body. It
  is the first thing that has to generalize at step 2, since training wants a
  batch and the task randomizes ramp angle per environment.
- **No `pyproject.toml`.** `experiments/drift.py` reaches the package via
  `sys.path.insert`. Fine for one script; worth fixing before there are five.
- **An editable install of GRIP does not rebuild on C++ changes.** Reinstall
  after touching its `src/`. This has already cost time once.
- **The toolchain lives behind `vcvars`.** The import incantation is in the
  README's Setup section; a `pip install -e ../GRIP` without it fails
  confusingly.
- **Parallel faces sit on a tie-break degeneracy.** A flat pusher pressed
  squarely against a flat box is exactly the configuration GRIP's
  `pair_detection.md` flags: both bodies report identical penetration, so
  contact-normal ownership is a tie whose neighbourhood disagrees with it.
  Forward simulation is fine; **SHAC gradients taken near it are suspect.**
  The decision already made is to **angle the pusher's contact face by 2–3°**
  so it meets the box at a vertex. Do that when the pusher lands at step 3,
  rather than discovering it during training.

## Experiments and figures

`experiments/` holds one script per artifact, each runnable on its own with
no arguments. `figures/` holds their committed output.

- An experiment prints the numbers it measures, not just the file it wrote.
  The console output is what gets pasted into a commit message.
- Figures are committed. They are the reason the results are visible on the
  repository page.
- Matplotlib uses the `Agg` backend, set before `pyplot` is imported.
- **If a doc cites a number, it must come from something that still runs
  here.** The 4.75 cm, the 18° crossover and the 48% figure all trace to
  `drift.py`. If a change moves them, the README and the task doc move too.

If code and `docs/ramp_manipulation_task.md` disagree, flag it. Don't
silently pick one.

## Things to do proactively

- Say when something belongs in GRIP rather than here — and when a request
  is asking GRIP for something it should not have to know.
- Flag when a measurement is about to be reported as more than a measurement.
- Suggest the sanity check that would catch a sign error early; the `α = 0`
  reduction to flat ground is the model for that.
- Keep the reward's gradient seeds and the reward itself in one place, so
  they cannot drift apart.

## Things not to do

- **Reintroduce research framing.** No hypotheses, no predicted outcomes, no
  positioning, no defending a result. See the second section.
- Put physics, derivatives or contact code here. That is GRIP's half.
- Build toward GRIP 2.0 while the 1.0 column is unfinished.
- Add abstraction layers or generality no current experiment needs.
- Implement PPO. It never touches GRIP's gradients, which are the thing GRIP
  uniquely provides, and the sample budget at `dt = 5e-4` is brutal.
- Weaken a check to make something pass. Investigate instead.

## Standing reminder

The drift measurement is done and stands on its own. **Step 2 is next** —
box alone, wrench applied directly, no pusher — and the honest first task in
it is generalizing `resting_state` past its hardcoded single environment.

Steps 2 through 5 need nothing that does not already exist. The 2.0 column
is written down so it isn't re-litigated and so nothing here forecloses it,
**not** so it gets built early.

Keep this file current as steps land. GRIP's went stale once by claiming a
step was next while four more shipped, which is the kind of drift that makes
a session start by trusting the wrong thing.
