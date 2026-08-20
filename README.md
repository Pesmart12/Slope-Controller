# slope-control

Control and reinforcement learning on top of [GRIP](https://github.com/Pesmart12/GRIP),
a 2D differentiable rigid-body contact simulator.

GRIP simulates and differentiates; it deliberately contains no policies,
rewards, or training loops. This is the other half of that split — the
repository that poses tasks and solves them.

## What it is for

One experiment, built to be legible rather than rigorous: **does the
choice of contact formulation change what a policy learns?**

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

So the task is built around a slope shallower than the friction angle,
where rigid physics says a released box must **never move**. Penalty
violates that by a visible margin; a solve satisfies it exactly. A policy
trained against the sloppy version learns to fight a disturbance that is
an artefact — and then walks its target off the ramp when the disturbance
disappears.

Full task definition, scene numbers, reward and evaluation protocol:
[`docs/ramp_manipulation_task.md`](docs/ramp_manipulation_task.md).

## Approach

| | |
|---|---|
| Task | push a box up a ramp to a target and hold it |
| Zeroth-order | MPPI / CEM — a planner, not a learner. Confirms the task is solvable and shows what good looks like. |
| First-order | SHAC, consuming GRIP's analytic gradients through `adjoint_batch` |
| Ground truth | NCP. **Every policy is scored there**, whichever simulator it trained in. |

PPO is deliberately excluded: it never touches the simulator's gradients,
which is the thing GRIP uniquely provides, and it is the arm most likely
to look identical across formulations.

The headline claim — that a solve is more physically correct — rests on a
closed-form comparison rather than on any learning curve. A box on a slope
below the friction angle must not move; that is Coulomb's law, not a
modelling opinion. The RL results answer a separate and more interesting
question, which is whether the error changes learned behaviour.

## Status

| | |
|---|---|
| Drift measurement | **done** — `experiments/drift.py`, the plot above |
| MPPI / CEM planner | not started |
| SHAC | not started |
| NCP half of every comparison | waiting on GRIP 2.0 |

The first milestone needed no policy and no training, which is why it came
first: place a box on a tilted half-plane, roll out five seconds of zero
controls, and measure. Half the headline result now exists; the other half
is the same plot with a solve in place of a spring.

## Setup

Needs GRIP installed. On a machine where the C++ toolchain is not on
`PATH`, import the build environment first:

```powershell
$vs = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" && set" |
  Where-Object { $_ -match '^(PATH|INCLUDE|LIB|LIBPATH)=' } |
  ForEach-Object { $n, $v = $_ -split '=', 2; Set-Item -Path "env:$n" -Value $v }
$env:PATH = "$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;" +
            "$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;$env:PATH"

pip install -e ..\GRIP
```

An editable install of GRIP does **not** rebuild on C++ changes — reinstall
after touching its `src/`.

## Layout

```
slope_control/   the ramp scene, shared by every experiment
experiments/     one script per artifact, each runnable on its own
figures/         their output, committed so the results are visible here
docs/            task definition and experiment design
```

Policies, planners and training loops will join `slope_control/`. None of
it goes into GRIP — that repository holds physics and derivatives, and
this one holds everything that decides what to do with them.
