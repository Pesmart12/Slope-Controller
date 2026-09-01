"""The policy gradient is not one adjoint_batch call.  Run: python tests/check_closed_loop.py

`adjoint_batch` returns dJ/dU_t holding the other controls fixed. For a
control sequence that is exactly the gradient, which is why
`check_trajopt.py` optimizes cleanly and why the baseline validated
against finite differences without trouble.

A policy is different. U_t is pi(Z_t), and Z_t depends on every earlier
control, so perturbing the parameters moves U_0, which moves Z_1, which
moves a_1 *again* through the policy. Contracting dJ_dU with the direct
dpi/dparam picks up only the first of those paths.

The task doc described SHAC as one call per window seeded at every step,
reading back dJ_dU. Measured here against a one-parameter feedback law,
that recipe comes out **more than double** the true gradient -- not a
small bias, and there is no reason its sign is reliable either.

What works is a per-step backward sweep. Each step's adjoint call returns
both pieces needed: dJ_dU_t, which contracts against dpi/dparam, and
dJ_dZ0, which carries the adjoint back one step. Between calls the sweep
adds the path the single call cannot see, Z_t -> a_t -> Z_{t+1}. Total
adjoint work is unchanged -- W calls of `substeps` each, rather than one
call of W*substeps -- so this costs Python round trips, not simulation.

None of this is something to ask GRIP for. A state-dependent control is a
policy, and policies belong on this side of the split; GRIP's contract
that controls are exogenous is right. The sweep is assembled here, and it
should move into `slope_control/task.py` when SHAC becomes its second
consumer.
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import grip  # noqa: E402
from slope_control import ramp, task  # noqa: E402

STEPS = 200
TARGET_OFFSET = 0.3


def rollout_closed_loop(scenes, state, angles, targets, gain, substeps, steps=STEPS):
    """Step one control step at a time, recomputing the action from the state.

    `step_batch` rather than `rollout_batch` because the control is not
    known in advance, and because it returns a fresh array per step rather
    than a view of the simulator's buffer.
    """
    states, controls, errors = [state], [], []
    for _ in range(steps):
        error = ramp.along_ramp(state, angles)[:, 0] - targets
        wrench = task.to_wrench(np.stack([-gain * error, np.zeros_like(error)], axis=-1), angles)

        errors.append(error)
        controls.append(wrench)
        state = grip.step_batch(scenes, state, wrench[None], substeps=substeps)
        states.append(state)

    return np.array(states), np.array(controls), np.array(errors)


def one_call_gradient(scenes, trajectory, controls, angles, dl_dZ, dl_dU, dpi_dgain, substeps):
    """The task doc's recipe: one sweep, contracted with the direct dpi/dK."""
    _, dJ_dU = grip.adjoint_batch(scenes, trajectory, controls, substeps, dl_dZ, dl_dU)
    return (task.to_action_gradient(dJ_dU, angles)[..., 0] * dpi_dgain).sum()


def per_step_gradient(scenes, trajectory, controls, angles, dl_dZ, dl_dU, dpi_dgain, gain, substeps):
    """The same sweep, one control step at a time, carrying dpi/dZ between steps."""
    adjoint = dl_dZ[-1].copy()
    total = 0.0
    for t in range(len(controls) - 1, -1, -1):
        seed = np.zeros((2,) + trajectory.shape[1:])
        seed[1] = adjoint
        dJ_dZ0, dJ_dU = grip.adjoint_batch(scenes, trajectory[t:t + 2], controls[t:t + 1], substeps, seed, dl_dU[t:t + 1])

        action_gradient = task.to_action_gradient(dJ_dU, angles)[0, :, 0]
        total += (action_gradient * dpi_dgain[t]).sum()

        # The path a single call cannot see: the state feeds the policy,
        # which feeds the next state. dpi/dZ is -K * d(xi)/d(x, y).
        through_policy = np.zeros_like(adjoint)
        through_policy[:, 0, 0:2] = (-gain * action_gradient)[:, None] * ramp.uphill(angles)
        adjoint = dJ_dZ0 + through_policy + dl_dZ[t]

    return total


def check(gain):
    angles = np.radians([18.0, 21.0])
    scenes = ramp.make_scenes(angles)
    substeps = task.substeps_for(scenes[0])
    state = task.settle(scenes, ramp.resting_state(0.0, angles), substeps)
    targets = ramp.along_ramp(state, angles)[:, 0] + TARGET_OFFSET

    trajectory, controls, errors = rollout_closed_loop(scenes, state, angles, targets, gain, substeps)

    peak = np.abs(gain * errors).max()
    assert peak < task.FORCE_LIMIT[0], f"the feedback law saturates at K = {gain}, so the clip is what is under test"

    # If the recorded controls do not replay to the same trajectory, the
    # adjoint is being handed a different problem than the one measured.
    replay = np.array(grip.rollout_batch(scenes, state, controls, substeps=substeps))
    assert np.abs(replay - trajectory).max() < 1e-12, "closed-loop and open-loop rollouts disagree"

    dl_dZ, dl_dU = task.reward_seeds(trajectory, controls, angles, targets)
    dpi_dgain = -errors  # d/dK of -K*(xi - xi*), at fixed state

    one_call = one_call_gradient(scenes, trajectory, controls, angles, dl_dZ, dl_dU, dpi_dgain, substeps)
    per_step = per_step_gradient(scenes, trajectory, controls, angles, dl_dZ, dl_dU, dpi_dgain, gain, substeps)

    def objective(k):
        rolled, held, _ = rollout_closed_loop(scenes, state, angles, targets, k, substeps)
        return task.reward(rolled, held, angles, targets).sum()

    h = 1e-3
    truth = (objective(gain + h) - objective(gain - h)) / (2.0 * h)

    print(f"  K = {gain:>5.1f}   truth {truth:>11.5f}   one call {one_call:>11.5f} ({100 * abs(one_call / truth - 1):>6.1f}% off)"
          f"   per step {per_step:>11.5f} ({100 * abs(per_step / truth - 1):.4f}% off)")
    return abs(one_call / truth - 1.0), abs(per_step / truth - 1.0)


def main():
    print(__doc__.strip().splitlines()[0])
    print(f"\n  one-parameter feedback a_t = -K (xi_t - xi*), {STEPS} control steps, 2 environments\n")

    results = [check(gain) for gain in [25.0, 30.0]]
    one_call_error = max(r[0] for r in results)
    per_step_error = max(r[1] for r in results)

    failures = []
    if per_step_error > 1e-4:
        failures.append(f"the per-step sweep is {100 * per_step_error:.3f}% off the finite difference; it is supposed to be exact")
    if one_call_error < 0.1:
        failures.append(f"one call per window is only {100 * one_call_error:.1f}% off -- the recorded finding no longer reproduces, so re-derive before trusting either")

    for failure in failures:
        print(f"\n  FAIL: {failure}")
    if not failures:
        print(f"\n  per-step sweep exact to {100 * per_step_error:.4f}%; the single-call recipe is {100 * one_call_error:.0f}% wrong and stays that way")
    print(f"\n{'PASS' if not failures else 'FAIL'} -- SHAC needs the per-step sweep, not one call per window")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
