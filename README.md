# slope-control

Control and reinforcement learning on top of [GRIP](https://github.com/Pesmart12/GRIP),
a 2D differentiable rigid-body contact simulator.

GRIP simulates and differentiates; it deliberately contains no policies,
rewards, or training loops. This is the other half of that split — the
repository that poses tasks and solves them.

## What it is for

Build an RL control stack on GRIP 1.0's penalty contact and measure what
it does. Then rebuild it on 2.0's NCP solve and measure again. Put the two
sets of numbers side by side and describe what changed.

The task is a box pushed up a ramp and held there, on a slope below the
friction angle — where rigid physics says a released box never moves.

GRIP 1.0 models contact as a penalty force — a stiff spring-damper with a
Coulomb friction cone. That formulation keeps the force constraint exact
and softens the kinematic one, so a box resting on a slope must be
*slipping* in order to generate the friction that holds it up. It creeps,
forever, at `mg·sinα / (2·b_slip)` — **4.7 cm in five seconds** on a 20°
ramp, about a sixth of a box width, on a slope where rigid physics says
it must not move at all.

GRIP 2.0 will replace that with a velocity-level NCP solve, where the
same box sticks exactly.

![measured drift on slopes below the friction angle](figures/drift.png)

Measured, in [`experiments/drift.py`](experiments/drift.py). The right
panel is a second finding that fell out of checking the first: the closed
form is exact to five figures below about 19°, and a lower bound above it,
because friction's moment arm tilts the box enough to redistribute normal
force between the corners until the lightly loaded one saturates on its
cone.

That is the physics half, and it needed no policy to produce. The other
half is what a learned controller runs into: the same creep is what stops
a trained policy holding the box once it gets there. See Status.

Full task definition, scene numbers, reward and what gets measured:
[`docs/ramp_manipulation_task.md`](docs/ramp_manipulation_task.md).

## Approach

| | |
|---|---|
| Task | push a box up a ramp to a target and hold it |
| Check | Trajectory optimization — Adam on the raw control sequence. Confirms the task is solvable and the gradients navigable, with no policy in the way. Not a baseline: it produces no number that means anything without an NCP column to set it against. |
| First-order | SHAC, consuming GRIP's analytic gradients through `adjoint_batch` |
| Scoring | NCP, whichever simulator a policy trained in — the more accurate of the two. |

PPO is skipped: it never touches the simulator's gradients, which are the
thing GRIP uniquely provides, and the sample budget at `dt = 5e-4` is
brutal.

The drift measurement stands on its own — a box on a slope below the
friction angle must not move, and that is Coulomb's law rather than a
modelling opinion. Everything after it is a build, and the comparison at
the end reports whatever the numbers turn out to say.

## Status

| | |
|---|---|
| Drift measurement | **done** — `experiments/drift.py`, the plot above |
| Ramp task, reward and gradient path | **done** — `slope_control/task.py`, `objective.py`, `observation.py`, `batches.py`, `tests/check_task.py` |
| Trajectory optimization | **done** — `tests/check_trajopt.py`, within a centimetre at every sampled slope |
| Two-pusher manipulation task | **done** — the box is unactuated, so every newton crossing it crosses a contact |
| SHAC | **done** — `experiments/train_shac.py`, 1365 s for 2000 iterations at 64 environments |
| NCP half of every comparison | waiting on GRIP 2.0 |

The first milestone needed no policy and no training, which is why it came
first: place a box on a tilted half-plane, roll out five seconds of zero
controls, and measure. The other half of that plot is the same figure with
a solve in place of a spring.

**What the policy does, and what it cannot do.** It drives the box 74.92 cm
to within a box width in 0.30 s, and lands exactly on target when the
target is downhill. Then it loses it. Penalty creep pulls the box off the
target for the rest of the episode, and arresting that needs force above
break-free, which the reward prices as not worth the centimetres it buys.
Every episode ends downhill of its target.

That is not a training failure — two runs with completely different critic
arrangements ended within 0.03 cm of each other, because the number is set
by the contact model against a fixed episode length. Under a rigid solve
none of it happens: the box sticks, and holding after arrival is free.
Which is what the 2.0 column is for.

## Setup

The project has been built on both Windows and Linux. The environment is
the same on either; only the step that compiles GRIP differs.

A conda environment, because PyTorch arrives that way and the CPU build
from conda-forge matches the Python the rest of the stack already uses.
`cmake` and `ninja` come from conda-forge as well, so the build needs
nothing installed system-wide. The two forms differ only in the
line-continuation character — PowerShell:

```powershell
conda create -n slope-control -c conda-forge --override-channels `
    python=3.13 numpy matplotlib "pytorch=*=cpu*" cmake ninja
```

and bash:

```bash
conda create -n slope-control -c conda-forge --override-channels \
    python=3.13 numpy matplotlib "pytorch=*=cpu*" cmake ninja
```

### Building GRIP — Windows

The MSVC toolchain has to be on `PATH` first, which `vcvars64.bat`
arranges. conda-forge supplies `cmake` and `ninja`, so only the compiler
itself comes from Visual Studio:

```powershell
$vs = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" && set" |
  Where-Object { $_ -match '^(PATH|INCLUDE|LIB|LIBPATH)=' } |
  ForEach-Object { $n, $v = $_ -split '=', 2; Set-Item -Path "env:$n" -Value $v }

conda activate slope-control
pip install -e ..\GRIP
pip install -e . --no-deps
```

**Activate the environment; do not call its `python.exe` by path.** Conda
puts its own DLLs on `PATH` at activation, and matplotlib's PNG writer
delay-loads one of them. Without activation `savefig` dies with
`0xC06D007F` and *no Python traceback at all* — every other part of the
stack, GRIP included, keeps working, so it reads as a matplotlib bug
rather than a missing path.

### Building GRIP — Linux

The system compiler is enough, and nothing needs root:

```bash
conda activate slope-control
pip install -e ../GRIP
pip install -e . --no-deps
```

**GRIP's `grip_core` must be compiled with `POSITION_INDEPENDENT_CODE`.**
It is a static library linked into a shared Python module, and ELF linkers
reject non-PIC objects inside a shared object outright, so the bindings
fail to link with a `recompile with -fPIC` error. Set in GRIP's
`src/CMakeLists.txt`. Windows draws no such distinction, which is why this
only appeared on the first Linux build.

### Both platforms

`pip install -e . --no-deps` installs this package so `slope_control`
imports from anywhere rather than only from the repository root.
`--no-deps` because numpy, matplotlib and torch already came from conda
and pip should not pull wheels over them.

An editable install of GRIP does **not** rebuild on C++ changes — reinstall
after touching its `src/`.

`conda run -n slope-control python ...` is the safer form in scripts, and
is what `conda activate` replaces interactively.

Verify against numbers that are already recorded, rather than against
"it imported":

```
python tests/check_task.py        # 11/11
python experiments/drift.py       # 4.75 cm at 20 degrees
python tests/check_trajopt.py     # reward -82.37, within a centimetre
```

If those three disagree with what is written here, the environment is
wrong and nothing measured in it is worth reading.

## Layout

```
slope_control/   the scene, the task, the policy and the training loop
experiments/     one script per artifact, each runnable on its own
figures/         their output, committed so the results are visible here
runs/            training logs, evaluation histories and checkpoints
tests/           the checks the build order rests on
docs/            task definition and experiment design
```

None of it goes into GRIP — that repository holds physics and derivatives,
and this one holds everything that decides what to do with them.
