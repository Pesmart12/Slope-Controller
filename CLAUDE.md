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

### The point of this repo is not penalty vs NCP

**Say it in full: the point is to build and train an RL control stack on
GRIP.** The contact comparison is a thing that falls out at the end, and
it is loose by design.

Written down twice over, because a session drifted past the rule above
once by treating trajopt as a bar to clear, then drifted again an hour
later by calling the contact comparison "the whole point of this repo".
The pull toward framing this as a comparison study is strong. Two
specific corrections, both from Pedro, both in one session:

> "My focus was not 'is RL better at control than a trad control?' that's
> a bad question for a simple task like ramp."

> "again the whole point of this repo is not ncp vs penalty."

**If a sentence you are about to write positions this as a study of
anything, delete it.** The deliverables are: did it learn the task, was
it efficient, a demo, and a loose look at how behaviour differs between
1.0 and 2.0.

**"Is RL better at control than classical control?" is not the question,
and it is a bad question for a task this simple.** The point is to build
and train an RL stack on GRIP. What is worth reporting:

- **Did it learn the task?**
- **Was it efficient?** — trajectory optimization is informative here, and
  only here.
- **A demo.** Show the trained policy doing the thing.
- **A loose comparison of behaviour under 1.0 and 2.0.** Loose is the
  design, not a shortfall. See the paragraph above about rigor nobody
  asked for.

`tests/check_trajopt.py` is a **check**, not a baseline. Its job is to say
the reward is solvable and its gradients navigable, and 800 iterations
answers that. **Its numbers are not converged**, and the check's own
"ended at its best: yes" line means *still improving* — the opposite of
the reassurance it reads as. So do not read −82.37 as the optimum, and do
not gate anything on beating it. Running it longer keeps improving the
reward and walks the parked position further downhill of the target; that
was measured on the box-only variant before it was removed, so there is no
table of it here any more.

**Both optimizers park downhill of the target**, and that is worth
keeping. Trajopt lands at +0.76 / −0.09 / −0.32 cm across the sampled
slopes, and SHAC at −3.14 cm uphill and −1.44 cm downhill. The reward
prices penalty contact correctly: closing the last centimetres needs force
above `break_free_force`, which costs more per step than the position
error it saves, and inside that band nothing slides. The standing reminder
has the full measurement.

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

The repository is small and should stay legible.

| | |
|---|---|
| `slope_control/ramp.py` | the ramp scene and its geometry conventions, shared by everything |
| `slope_control/policy.py` | the actor and critic, and the tensor plumbing they need |
| `slope_control/sweep.py` | the closed-loop gradient — `rollout` forward, `accumulate` at the torch/GRIP seam, `policy_gradient` back |
| `slope_control/task.py` | what the other three task modules share — body layout, action limit and frame, episode length |
| `slope_control/objective.py` | the reward, its shaping term, and their gradient seeds; defines `ℓ` and `J` |
| `slope_control/observation.py` | what the policy sees, and its constant Jacobian |
| `slope_control/batches.py` | `Batch` and its two builders — scenes, placement, settle window, targets |
| `slope_control/shac.py` | the training loop — windowed rollout, actor step, critic fit, target update |
| `slope_control/render.py` | draws a scene and animates an episode to a GIF; pure presentation, computes nothing |
| `experiments/drift.py` | artifact 1 — the drift measurement and its figure |
| `experiments/train_shac.py` | the experiment that trains — three arms, file logging, two checkpoints per arm |
| `experiments/demo.py` | the trained policy rendered — two GIFs, the creep figure, and the JSON the interactive page embeds |
| `figures/` | committed output, so results are visible without running anything |
| `runs/` | training logs, evaluation histories and checkpoints, one directory per run |
| `tests/check_task.py` | the checks the task rests on, chiefly the finite-difference of `dJ_dU` through GRIP |
| `tests/check_trajopt.py` | Adam on the raw control sequence, which is what says the reward is solvable at all |
| `tests/check_closed_loop.py` | why SHAC needs a per-step adjoint sweep and not one call per window |
| `tests/check_policy_gradient.py` | the same statement for a real network — `dJ/d(theta)` against central differences |
| `docs/ramp_manipulation_task.md` | the task definition; the authority on scene numbers, reward and episode structure |
| `pyproject.toml` | the editable install, so nothing reaches the package by patching `sys.path` |

**`policy.py` and `sweep.py` split on what a policy IS versus what running
and differentiating one takes.** The boundary is checkable rather than
aesthetic: `tests/check_policy_gradient.py` validates the whole gradient
path importing `policy` and `sweep` and never `shac`, so verifying a
derivative does not drag in the training loop. `Window` moving with them is
the tell that the seam is real — only those three functions touch it.

**The task is four modules split by job.** `objective`, `observation` and
the action conversions in `task` share constants and never call each
other; `batches` calls into `observation` and `task`, and nothing calls
into it. Constants with one user live with that user, so `WEIGHTS` is in
`objective` and `RESTING_OFFSETS` in `observation`. `TOUCHING` stays in
`task` because `objective.separations` and `batches.placement` both read
it, which is what keeps those two exact inverses. The module names avoid
`reward` and `episode`, which are already local names in callers.

Two things sit on the wrong side of that line and are left there on
purpose, being small: `ascend` is a generic optimizer step but lives in
`shac.py` because only `train` calls it and its default is a SHAC-tuned
constant, and `evaluate` and `calibration` would serve any algorithm.

**`BOX_ONLY` is gone, and this is the record so it is not reintroduced.**
It was a second variant of the task with one directly actuated box —
physically fictional, since nothing reaches into a box and pushes from its
centre — kept while SHAC was brought up so that "does it learn on
box-only?" could separate a policy bug from a contact bug. SHAC learns on
two pushers, so that job is finished. With it went `Variant`, `bodies_for`
and the `variant` argument every task function threaded; body count, the
scored body, the actuated set and the limit are module constants in
`task.py` now, and the scale and weights in `objective.py`.

One consequence worth knowing: **`tests/check_closed_loop.py` builds its
own one-box scene**, with a local wrench builder, reward and seeds. Its
subject is adjoint bookkeeping rather than contact, and a one-parameter
feedback law on the box's own position is not expressible against an
unactuated box. The fictional scene is a fixture there, not a task the
package offers. Keeping it that way is what preserved the 119% figure
exactly.

The two **results**, both reproducible:

```
python experiments/drift.py
  20° ramp, zero controls, 5 s   ->   4.75 cm of drift, where rigid physics says 0
  closed form mg·sin α / (2·b_slip) exact to five figures up to 18°
  departs at 19°, 48% low at 26°

python experiments/train_shac.py  then  python experiments/demo.py
  74.92 cm to within a box side in 0.30 s, and exactly on target downhill
  then creep takes it back off, which is the standing reminder's subject
```

`tests/check_task.py` is not a result and produces no figure — it is what
makes the task machinery trustworthy, and `python tests/check_task.py`
should stay at 11/11.

Still not built: no planner, and no packaging.

## Build order

`docs/ramp_manipulation_task.md` is the authority; this is the short version
so a session knows where it is.

1. **Done.** Box on a 20° tilted `HalfPlane`, five seconds of zero controls,
   plot `ξ` against time.
2. **Substrate done.** The task — `task.py`, `objective.py`,
   `observation.py` and `batches.py`, plus `tests/check_task.py`. The reward and the gradient path are de-risked:
   `dJ_dU` matches central differences to 1.9e-5 relative through GRIP,
   across a body-body contact. Built first against a single directly
   actuated box, which has since been removed.
3. **Done.** Trajectory optimization — `tests/check_trajopt.py`. Adam on
   the raw control sequence reaches the target to within a centimetre at
   every sampled slope, so the reward is solvable and its gradients are
   navigable across 8000 integration steps, not merely correct at a point.
   **This replaces the MPPI step**, which was dropped: being zeroth order
   it never calls the adjoint, so it could only ever answer half the
   question it was there for. It keeps one job in reserve — if trajopt
   ever fails, gradient-free is what separates "bad reward" from "correct
   but unusable gradients."
4. **Done.** Two pushers, box between them, box unactuated. Trajopt solves
   it to 0.76 / −0.09 / −0.32 cm, so the manipulation task is worth
   training on.
5. **Substantially done, and the open problem turned out not to be one.**
   SHAC on penalty — `slope_control/shac.py`, built on the per-step
   adjoint sweep in `sweep.policy_gradient`, run by
   `experiments/train_shac.py`. **The policy learns the task**: 74.92 cm
   to within a box side in 0.30 s, and exactly on target when the target
   is downhill. What no policy can do is *stay* there, because penalty
   creep will not allow it. Clipping and the critic's slope were both
   investigated and neither is the cause; see the standing reminder.
   **What is left is presentation**, not more training —
   `experiments/demo.py` is that, and the interactive page it feeds.
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
consumer's, not GRIP's.** `slope_control/objective.py`'s module docstring is
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
  usual stage-cost convention. Stated in `objective.py` rather than left to be
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
- **A comment says what the next line or block does. Rationale comes after
  that, or not at all.** This is the rule Pedro has corrected three times,
  so it is written out rather than left implied. Open in the imperative --
  "Concatenate the per-step observations into one array of rows" -- and put
  the shape change on its own line where there is one. A comment whose
  first sentence explains a *choice* is the failure mode: the reader wants
  to know what the line does before they can care why it does it. Two
  corollaries. A comment belongs immediately above the lines it describes
  and nowhere else; one that drifts to a neighbouring statement is worse
  than none. And a constant is the exception, since what it is IS its
  value, so its comment is allowed to be all rationale.
- **`batch` means `batches.Batch`.** Not a window, not the environment count,
  not the flattened rows a network is fitted on. Say "window",
  "environments", "rows".
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
  that failure is checkable. Above that level it is solved: **`batches.Batch`
  carries the scenes, the angles, the state, the targets, the substeps
  and the observation Jacobian as one value**, built by
  `batches.fixed_batch` or `batches.sample_batch`. Do not take a batch apart and
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
- **`cmake` and `ninja` come from conda-forge**, not from the system, so
  the build needs nothing installed system-wide on either platform. This
  does not replace `vcvars64.bat` on Windows — that is still needed, for
  the MSVC compiler itself.
- **The project builds on Windows and Linux, and each has one snag the
  other does not.** Both are live; neither is legacy.
- **Windows: the toolchain lives behind `vcvars`,** and the conda
  environment must be **activated, not addressed by path**. Calling
  `envs/slope-control/python.exe` directly runs GRIP, numpy and torch
  perfectly and then kills the process on `savefig` with `0xC06D007F` and
  **no traceback** — a delay-load failure for a DLL that only activation
  puts on `PATH`. Every symptom points at matplotlib and none point at the
  environment.
- **Linux: GRIP's `grip_core` needs `POSITION_INDEPENDENT_CODE`.** It is a
  static library linked into a shared Python module, and ELF linkers reject
  non-PIC objects in a shared object outright — `recompile with -fPIC`, at
  link time. Set in GRIP's `src/CMakeLists.txt`. Windows draws no such
  distinction, which is why the bindings built there for months and failed
  on the first Linux build.
- **`conda run -n slope-control python ...` is the safer form in scripts**
  on either platform.
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

Trajectory optimization is done too, and it earned its place immediately: it caught a
force limit that made the task **unsolvable**, sized against the force to
*hold* the box and never against the force to *move* it. A passing gradient
check did not catch that and could not have — a correct gradient on an
impossible objective is still correct. When something is checked and still
doesn't work, suspect the task before the machinery.

**Step 5, SHAC, trains and the policy learns the task.** For a long time
the open problem was recorded as "no run has yet held its best result",
read as a training pathology. That framing was wrong; the section below
replaces it.

The runs that produced it still stand as measurements. Two pushers, 8000
iterations, 64 environments, seed 0, same budget in all three:

```
                                        best      at it.   end of run
undiscounted, buffers copied            2.48 cm     500      8.95 cm
undiscounted, buffers blended           5.29 cm     500     11.26 cm
gamma = 0.99, no terminal bootstrap     2.37 cm     500     13.12 cm at it. 3500
```

Two things from that era are worth carrying forward. **The discount is not
a remedy and was never added as one** — it earned its place because at
gamma = 1 the Bellman operator is not a contraction, and the critic did
settle at −53.87 against episode returns near −180. And **the exploration
noise decays on its own**, sigma 0.368 to 0.116 by iteration 3500, so late
training is nearly deterministic and the reported error is not a noise
floor.

One run **crashed at iteration 3500 with exit code 4 and no traceback**,
six hours in, while running six times slower per iteration than its
predecessors. Never explained. `experiments/train_shac.py` logs to a file
and checkpoints at every evaluation because of it.

### WHAT THE LIMIT ACTUALLY IS: penalty creep, not training

The limit is penalty creep, and it bites in two opposite ways depending on
which side of the box the target sits. Measured on the trained policy, one
fixed episode per row:

```
   target              best cm   final cm   drift cm/s   hold N   break-free
   50 cm uphill          -3.14      -3.14       +0.27     20.76      23.89
   50 cm downhill        -0.00      -1.44       -0.49     15.13      23.89
```

**Uphill, creep is too slow to finish.** Closest approach equals final
error, so the error never stops improving — the policy pushes at or just
under break-free and creeps *toward* the target at 0.27–0.50 cm/s, and the
episode ends first. Parking short is running out of time, not being
dragged.

**Downhill, it arrives and then loses it.** The box reaches −0.00 cm, dead
on target, and slides to −1.44 cm while the policy holds at ~15 N, below
break-free. Driving downhill is about nine times cheaper than uphill, so
getting there is easy and staying there is what costs.

Both are the same artifact with opposite signs, and **every episode ends
downhill of the target** either way.

**Do not average `|error|` across target directions.** An earlier pass
here did, over eight sampled episodes, and produced an arrive-then-drift
curve that describes neither case — the mean is dominated by the downhill
minority. That error is what this section originally recorded.

**This is why the ablations barely moved the endpoint.** Two runs with
completely different critic arrangements ended at 6.69 and 6.72 cm on the
sampled batch, because the number is set by physics against a fixed
episode length, not by the optimizer.

**Did it learn the task? Yes** — it reaches the target within a box side
in 0.30 s, and hits it exactly when the target is downhill. What no policy
can do under penalty contact is *stay* there.

Recorded as measurement, not interpretation. One inference is **not**
measured: that the pusher's force goes mostly into holding itself, its mass
being twice the box's and so creeping twice as fast. Checkable, not
checked.

This is the thread already open above about trajopt parking downhill, and
it turned out to be the dominant term in the 1.0 column rather than a
footnote about the last millimetres. **Do not chase it under 1.0.** It has
a closed form behind it and it is what the 2.0 column exists to sit next
to. Do not write up what 2.0 will show either; measure it when there is a
solver.

#### Answered, so they are not re-run

**Clipping does not hold the policy, and it is worse.** Two arms, two
pushers, 2000 iterations, 64 environments, seed 0:

```
                best      at it.   end      gave up   clipped
  unclipped     1.39 cm     400    6.69 cm   5.30 cm    0/2000
  clipped       3.48 cm     300    9.19 cm   5.71 cm  610/2000
```

`clipped = 610` is the intended regime — neither inert nor binding every
iteration — so the comparison is real. Worse peak, worse end.

**The critic's slope is unreliable, and it is not the cause.** It was the
one unexamined suspect and it is genuinely bad: `dV/dξ` swings between
−184 and +129 and changes sign eight times across thirteen evaluations,
while the true slope drifts smoothly 8 to 48 and V itself holds at 0.97
correlation with bias near 1. But it is wrong at the policy's *best*
iteration too, not only late, and an ablation with `actor_bootstrap=False`
— the critic fitted and measured but never reaching the actor — still
ended at 6.72 cm. So the decay does not need it.

What the slope does cost is reward without position: that arm's reward
*improves* to the end, −186.92, where the bootstrapped arm's degrades to
−191.07 at the same error. **`observe` carrying no clock is the candidate
explanation** — the critic must fit return-to-go from an observation that
cannot tell early from late, which can leave V accurate on average and its
derivative incoherent. Not worth fixing under 1.0: it cannot move a
creep-bound endpoint, and it would change the observation space every
recorded number was measured against. Revisit when 2.0 removes creep and
the horizon becomes the binding constraint.

**`slope_calibration`'s delta was wrong and is fixed.** It was sized at
5 mm for physical plausibility — inside `POSITION_TOLERANCE`, above the
settle drift — when the binding criterion is where the difference quotient
converges. That is below about 0.1 mm here, and the two criteria disagreed
by 50×. Every slope number taken before the fix is void. `SLOPE_DELTA`
records the convergence table.

One finding from step 4, now fixed rather than merely flagged:
**gradients cannot discover a contact that does not exist.** With the task
objective alone the two-pusher case does not converge slowly — the driving
pusher's action stays at exactly 0.00 N forever, because with no contact
`d(box position)/d(pusher action)` is identically zero.

The fix is **reward shaping**, `objective.approach_penalty`, named as shaping
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

- **`observe`'s channels are still unvalidated as a set.** Training says
  they are sufficient — the policy learns the task from them — but nothing
  says any one of them earns its place, and no channel has been ablated.
  Its shape and its gradient are checked; its content is not.
- `clip_action` is a hard clip with zero gradient once saturated, and the
  converged trajopt solution sits on its limit **5.9%** of steps. The
  actor sidesteps it: it emits
  `limit * tanh(...)`, so every action is feasible by construction and the
  derivative survives everywhere. `clip_action` still runs downstream,
  where it is now a no-op that documents the guarantee. **Never widen the
  limit instead** — it is what keeps the bodies on the ramp.
- **SHAC needs a per-step adjoint sweep, not one call per window.**
  Measured in `tests/check_closed_loop.py`: one call gives the open-loop
  gradient, right for a fixed control sequence and **119%
  wrong** for a policy at one gain, 11% at another — state-dependent, so
  not something a learning rate absorbs. The sweep lives in
  `sweep.policy_gradient` rather than `objective.py`, because it needs torch
  and keeping the four task modules torch-free is worth more — `drift.py`
  and the numpy-side checks depend on that. It takes a `batches.Batch` and a
  `policy.Window`, and windows chain with
  `batch = batch._replace(state=window.states[-1])`.

The 2.0 column is written down so it isn't re-litigated and so nothing
here forecloses it, **not** so it gets built early.

Keep this file current as steps land. GRIP's went stale once by claiming a
step was next while four more shipped, which is the kind of drift that makes
a session start by trusting the wrong thing.
