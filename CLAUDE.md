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

Eight source files. The repository is small and should stay legible.

| | |
|---|---|
| `slope_control/ramp.py` | the ramp scene and its geometry conventions, shared by everything |
| `slope_control/policy.py` | the actor and critic, and the seam where torch's autograd meets GRIP's adjoint |
| `slope_control/task.py` | the step-2 task — action limit and frame, reward and its gradient seeds, episode and settle window, `Batch` and its two builders |
| `experiments/drift.py` | artifact 1 — the drift measurement and its figure |
| `figures/drift.png` | committed output, so results are visible without running anything |
| `tests/check_task.py` | the checks step 2 rests on, chiefly the finite-difference of `dJ_dU` through GRIP |
| `tests/check_trajopt.py` | the baseline — Adam on the raw control sequence, which is what says the reward is solvable at all |
| `tests/check_closed_loop.py` | why SHAC needs a per-step adjoint sweep and not one call per window |
| `tests/check_policy_gradient.py` | the same statement for a real network — `dJ/d(theta)` against central differences |
| `docs/ramp_manipulation_task.md` | the task definition; the authority on scene numbers, reward and episode structure |
| `pyproject.toml` | the editable install, so nothing reaches the package by patching `sys.path` |

`slope_control/task.py` carries two variants, `BOX_ONLY` and `TWO_PUSHERS`,
which differ in body count, which body is scored, which are actuated, and
the force limits. Everything else — reward, seeds, observation frame,
settle — is shared, and every check runs both.

**`BOX_ONLY` is scheduled for deletion, and this is the decision record so
it does not become permanent by nobody remembering.** It is physically
fictional by its own docstring — nothing reaches into a box and pushes
from its centre — and it is the sole reason `Variant` exists at all.
Remove it and `bodies`, `box`, `actuated`, `limit`, `scale` and `weights`
all become module constants, `bodies_for` disappears, and `pushing_bodies`
becomes `[0, 2]`.

It stays **through SHAC bring-up only**. Its remaining value is as a
strictly easier version of the same problem — same reward, same
observation frame, same gradient path, no body-body contact — so if SHAC
does not learn on two pushers, "does it learn on box-only?" separates a
policy bug from a contact bug at no cost. **Delete it once step 5 lands.**

Two things need rework when it goes, and both would move recorded numbers:
`tests/check_closed_loop.py` is entirely box-only, so the 119% figure cited
here and in the task doc would have to be re-derived against pushers; and
`check_trajopt.report_creep_band` reads the settled force straight off the
action array, which only works when the action *is* the force.

Nothing takes a default variant. Every task function requires it
explicitly, so a caller that forgets cannot silently get the fictional
task — which is what the defaults used to do, `BOX_ONLY` everywhere
except `placement`.

The one **result** so far, reproducible by `python experiments/drift.py`:

```
20° ramp, zero controls, 5 s   ->   4.75 cm of drift, where rigid physics says 0
closed form mg·sin α / (2·b_slip) exact to five figures up to 18°
departs at 19°, 48% low at 26°
```

`tests/check_task.py` is not a result and produces no figure — it is what
makes the step-2 machinery trustworthy, and `python tests/check_task.py`
should stay at 9/9.

Still not built: no policy, no planner, no training loop, no packaging.

## Build order

`docs/ramp_manipulation_task.md` is the authority; this is the short version
so a session knows where it is.

1. **Done.** Box on a 20° tilted `HalfPlane`, five seconds of zero controls,
   plot `ξ` against time.
2. **Substrate done.** Minimal task — box alone, wrench applied directly to
   the box, no pusher. `slope_control/task.py` plus `tests/check_task.py`.
   The reward and the gradient path are de-risked: `dJ_dU` matches central
   differences to 3e-7 relative through GRIP. The **observation space is
   not** — nothing consumes an observation until SHAC, so `observe` is
   fixed but unvalidated, and step 2's original claim to de-risk it was
   never achievable before step 5.
3. **Done.** Trajectory-optimization baseline — `tests/check_trajopt.py`.
   Adam on the raw control sequence reaches the target to 1.4 cm at every
   sampled slope, so the reward is solvable and its gradients are
   navigable across 8000 integration steps, not merely correct at a point.
   **This replaces the MPPI step**, which was dropped: being zeroth order
   it never calls the adjoint, so it could only ever answer half the
   question it was there for. It keeps one job in reserve — if trajopt
   ever fails, gradient-free is what separates "bad reward" from "correct
   but unusable gradients."
4. **Done.** Two pushers, box between them, box unactuated. Trajopt
   solves it to 0.2–1.2 cm — so the manipulation task is worth training
   on. Every check now runs both variants; the adjoint holds through
   body-body contact at 2e-5 relative.
5. **Next.** SHAC on penalty. Completes the 1.0 column. Use the per-step
   adjoint sweep from `check_closed_loop.py`, not one call per window.
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

Four things follow, and they shape most of the code here:

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
- **Batching across environments is not parallelism, in 1.0.**
  `src/api/simulate.cpp` loops over environments serially — no OpenMP, no
  threads — so a batch buys one crossing of the binding instead of N, and
  the cost stays linear in N. The torch side genuinely does vectorize: an
  `(N, features)` observation is one GEMM per layer. So the batch size is
  chosen for **gradient quality**, not throughput — N randomized slopes,
  starts and targets are what force the policy to be a function of the
  observation rather than one memorized action sequence. Budget simulation
  cost accordingly. **GRIP 2.0 is planned to parallelize this on CUDA**,
  which is the right place for it (every environment is independent, and
  how the physics executes is GRIP's half of the split). Revisit batch
  sizing then; do not shape anything around it now.
- **`rollout_batch` hands back a view of the simulator's own buffer**, not a
  copy — its own docstring says so. One rollout at a time is fine, which is
  why `drift.py` never noticed. A training loop that keeps trajectories
  across iterations has to copy. `step_batch` returns a fresh array instead.

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

Two of its symbols are ours to fill in — it says so: **`ℓ` and `J` are the
consumer's, not GRIP's.** `slope_control/task.py`'s module docstring is
where they are defined, and the definitions are:

- **`ℓ` is one step's reward** (`l` in ASCII code). GRIP calls that slot the
  stage cost.
- **`J` is the whole objective**: `ℓ` summed over a window, plus the critic's
  estimate of what follows.
- **`dl_dZ` / `dl_dU` are partials** of one step and go *into* `adjoint_batch`.
  **`dJ_dZ0` / `dJ_dU` are totals** and come *out* of it. Turning the first
  into the second is the entire job of the adjoint.
- **There is no `r`.** An earlier draft used `r` in prose for the same
  quantity the code called `l`, which is how this got confusing enough to
  need writing down.
- **`ℓ` here holds a reward, so everything maximizes** — the opposite of the
  usual stage-cost convention. Stated in `task.py` rather than left to be
  discovered. If anything ever becomes a genuine cost, flip it everywhere
  at once.

One collision is **unresolved and belongs to GRIP**: its table uses `J` for
the caller's objective *and* `Jᵢ`/`J_A` for contact Jacobians. Bare against
subscripted is a thin separation, and it is the same class of thing
`notation.md` already registers for `U` and `F` — but it is not registered.
Raise it there; don't work around it here.

## Code style

Pedro's preferences, the same ones GRIP uses where they carry over to Python.

- **Function parameters go on one line**, however long the line gets. Never
  wrap a parameter list one-per-line, in definitions or calls.
- **Two blank lines between top-level definitions.**
- **No type annotations.** None of the existing code has them; don't add them
  to code you touch.
- **Docstrings: one plain sentence first, then only what a caller needs.**
  In this order — what the function does, then its arguments and return if
  the names do not already say, then the reasoning required to use it
  correctly. `creep_rate` is the model for the reasoning part: the formula,
  why the factor of 2 is there, where the closed form stops being true.
  A docstring that restates the function name is not worth the lines.
- **Three ways docstrings here have gone wrong. Do not repeat them.**
  - *History.* What the code used to be, how many arguments it had before,
    which file it moved out of, what the plan originally said. That is
    commit-message material and a reader trying to use the function gains
    nothing from it.
  - *Loose terms.* Naming a variable the code does not actually pass, or a
    vague verb where the operation is ordinary. Say "multiply by the
    observation Jacobian", not "project through it". Name the real
    variable.
  - *Essay register.* Long clause-heavy sentences and em-dash pileups.
    Write short sentences in plain language. Rationale comes after the
    reader knows what the thing does, never before.
- **ASCII in code, Unicode in Markdown.** Python files use `--` and `alpha`;
  `.md` files use — and α. Don't mix them.
- **Comments explain why in ordinary code, and *what* in dense code.** The
  old rule here was "comments explain why, the what is already on the line
  above." That is true of code that reads plainly and false of the code
  this repository is mostly made of. An einsum, an axis that has to be at
  −3, a sign that flips on which side of the box a body sits, a
  three-term chain-rule accumulation — none of those say what they do, and
  the opening docstring does not help someone reading one specific line.
  Comment those at the line. Name what each term of a multi-term
  expression contributes.
- **A docstring is not a substitute for commenting the body.** Both,
  where the body is hard. Do not comment lines that are already obvious;
  over-commenting simple code is the opposite failure and just as bad.
- No commented-out code, no TODO placeholders in reviewed paths.

## Known snags

Real, and each one will bite in a specific place:

- **Angles and scenes travel separately** at the `ramp.py` level.
  `resting_state` and `along_ramp` take one angle per environment, and
  nothing there binds an angle array to the scenes built from it — a
  mismatch projects a batch onto the wrong slopes with every shape still
  lining up. `ramp.scene_angle` reads the angle back out of a scene so
  that failure is checkable. Above that level it is solved: **`task.Batch`
  carries the scenes, the angles, the state, the targets, the substeps,
  the observation Jacobian and the variant as one value**, built by
  `task.fixed_batch` or `task.sample_batch`. Do not take a batch apart and
  pass the pieces on individually; that is the failure mode reintroducing
  itself.
- **`along_ramp` no longer accepts a squeezed array.** It needs the
  environment axis at −3. With one angle per environment there is no way to
  infer which axis is which, so passing `trajectory[:, 0, 0, :]` raises
  instead of quietly projecting onto the wrong thing.
- **The package is an editable install**, `pip install -e . --no-deps`
  into the conda environment. It replaced five copies of a
  `sys.path.insert` preamble. A fresh clone that skips it gets an
  `ImportError` from every script; the README's Setup section has the
  command.
- **An editable install of GRIP does not rebuild on C++ changes.** Reinstall
  after touching its `src/`. This has already cost time once.
- **The toolchain lives behind `vcvars`.** The import incantation is in the
  README's Setup section; a `pip install -e ../GRIP` without it fails
  confusingly.
- **The conda environment must be activated, not addressed by path.**
  Calling `envs/slope-control/python.exe` directly runs GRIP, numpy and
  torch perfectly and then kills the process on `savefig` with
  `0xC06D007F` and **no traceback** — a delay-load failure for a DLL that
  only activation puts on `PATH`. Every symptom points at matplotlib and
  none of them point at the environment. Use `conda run -n slope-control
  python ...` in scripts.
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

The drift measurement is done and stands on its own. Step 2's substrate is
done too, and checked — but it is **machinery, not a result**, and it was
deliberately landed without a figure. Don't go looking for the artifact it
didn't produce.

The baseline is done too, and it earned its place immediately: it caught a
force limit that made the task **unsolvable**, sized against the force to
*hold* the box and never against the force to *move* it. A passing gradient
check did not catch that and could not have — a correct gradient on an
impossible objective is still correct. When something is checked and still
doesn't work, suspect the task before the machinery.

**Step 5, SHAC, is next.** Everything it needs exists: a solvable task, a
verified closed-loop gradient, and a baseline to be measured against.

One finding from step 4, now fixed rather than merely flagged:
**gradients cannot discover a contact that does not exist.** With the task
objective alone the two-pusher case does not converge slowly — the driving
pusher's action stays at exactly 0.00 N forever, because with no contact
`d(box position)/d(pusher action)` is identically zero.

The fix is **reward shaping**, `task.approach_penalty`, named as shaping
and off by default: `reward(..., shaping=True)` for training,
`shaping=False` for anything reported. It is one-sided so it vanishes on
contact, reuses `w_pos` so it adds no tunable, and works because the
pushers are *directly* actuated. An earlier plan to fix this with a
holding-force initialization was dropped — it leans on penalty creep to
close the gap, so it would have stopped working at GRIP 2.0, where nothing
creeps and the flat region is total.

The two mistakes around it are not symmetric. Training unshaped fails
loudly, with the driving pusher pinned at 0.00 N. Reporting shaped is
silent and makes numbers incomparable across columns.

Three things left open on purpose, so they aren't mistaken for oversights:

- `observe` finally has a consumer: `policy.Actor` reads it, and
  `check_policy_gradient` differentiates through it. It is exercised for
  *shape and gradient*, not for whether it contains the right channels —
  that only training can say.
- `clip_action` is a hard clip with zero gradient once saturated, and the
  converged baseline sits on its limit **2.1%** of steps on box-only and
  **5.9%** on two pushers. The actor sidesteps it: it emits
  `limit * tanh(...)`, so every action is feasible by construction and the
  derivative survives everywhere. `clip_action` still runs downstream,
  where it is now a no-op that documents the guarantee. **Never widen the
  limit instead** — it is what keeps the bodies on the ramp.
- **SHAC needs a per-step adjoint sweep, not one call per window.**
  Measured in `tests/check_closed_loop.py`: one call gives the open-loop
  gradient, right for the baseline's fixed control sequence and **119%
  wrong** for a policy at one gain, 11% at another — state-dependent, so
  not something a learning rate absorbs. The sweep now lives in
  `policy.policy_gradient`, **not** `task.py` as originally planned: it
  needs torch, and keeping `task.py` torch-free is worth more, since
  `drift.py` and the numpy-side checks depend on it. It takes a
  `task.Batch` and a `policy.Window` — `policy_gradient(batch, window,
  actor, critic, shaping)` — rather than the thirteen loose arguments it
  started with, six of which `rollout` also took. Windows chain with
  `batch = batch._replace(state=window.states[-1])`.

Steps 4 and 5 need nothing that does not already exist. The 2.0 column
is written down so it isn't re-litigated and so nothing here forecloses it,
**not** so it gets built early.

Keep this file current as steps land. GRIP's went stale once by claiming a
step was next while four more shipped, which is the kind of drift that makes
a session start by trusting the wrong thing.
