"""Why a policy needs one adjoint call per step.  Run: python tests/check_closed_loop.py

`adjoint_batch` computes dJ/d(control) holding the other controls fixed.
For a fixed control sequence that IS the gradient, which is why
`check_trajopt.py` optimizes cleanly with one call per episode.

A policy is different. Its action is computed from the state, and the
state depends on every earlier action, so nudging the weights moves the
first action, which moves the next state, which moves the next action
again through the policy. One call sees only the first of those paths.

Measured here against a one-parameter feedback law, whose gradient can be
worked out by hand and checked against finite differences: one call per
window is 11% off at one gain and 119% off at another. The error depends
on the state, so a learning rate cannot absorb it.

The per-step sweep is exact to 0.0002%. It costs the same simulation --
W calls of `substeps` each instead of one call of W*substeps -- so the
price is Python round trips, not physics.

None of this is something to ask GRIP for. A state-dependent control is a
policy, and policies belong on this side of the split.

The sweep itself lives in `sweep.policy_gradient`. What is here is a
second implementation of it against a hand-differentiated feedback law,
which is a cross-check but also a copy that can drift.
`check_policy_gradient.py` is what tests the shipped one.
"""

import numpy as np

import grip
from slope_control import batches, objective, ramp, task

STEPS = 200
TARGET_OFFSET = 0.3

# This check builds its own scene: one box, driven by a wrench at its centre
# of mass. Physically fictional, and deliberately so. The subject here is
# adjoint bookkeeping rather than contact, and one directly actuated body is
# the smallest fixture that exposes the Z -> a -> Z path a single call cannot
# see. The task's scene has three bodies and an unactuated box, where a
# one-parameter feedback law on the box's own position is not expressible.
#
# The reward, weights and scale below mirror what the package used to carry
# for this scene, so the percentages this file reports stay comparable to the
# ones recorded against it.
BODIES = [ramp.BOX]
SCALE = 12.0
LIMIT = 12.0
WEIGHTS = dict(position=1.0, control=objective.control_weight(SCALE, [ramp.hold_force(ramp.DEFAULT_RAMP_ANGLE)]))


def fixture(angles, offset=TARGET_OFFSET):
    """A settled one-box batch per angle. Returns a `batches.Batch`.

    `jacobian` is None: nothing here observes, so there is no observation to
    differentiate.
    """
    scenes = ramp.make_scenes(angles, bodies=BODIES)
    substeps = task.substeps_for(scenes[0])
    state = batches.settle(scenes, ramp.resting_state(np.zeros(len(angles)), angles, bodies=BODIES), substeps)
    targets = ramp.along_ramp(state, angles)[:, 0] + offset
    return batches.Batch(scenes, angles, state, targets, substeps, None)


def to_wrench(force, angles):
    """One tangential force per environment -> (environments, 1, 3) wrench."""
    wrench = np.zeros((len(angles), 1, 3))
    wrench[:, 0, 0:2] = np.clip(force, -LIMIT, LIMIT)[:, None] * ramp.uphill(angles)
    return wrench


def to_action_gradient(dJ_dU, angles):
    """dJ/d(wrench) -> dJ/d(tangential force), the dot with uphill."""
    return np.einsum("...ni,ni->...n", dJ_dU[..., 0, 0:2], ramp.uphill(angles))


def reward(states, controls, angles, targets):
    """Position error and control effort, the same pair the task scores."""
    xi = ramp.along_ramp(states, angles)[1:, :, 0]
    error = (xi - targets) / ramp.BOX_SIDE
    effort = (controls[..., 0, 0:2] ** 2).sum(axis=-1)
    return -WEIGHTS["position"] * error ** 2 - WEIGHTS["control"] * effort / SCALE ** 2


def reward_seeds(states, controls, angles, targets):
    """Differentiate `reward`, giving the two seed arrays `adjoint_batch` wants."""
    xi = ramp.along_ramp(states, angles)[..., 0]
    coefficient = -2.0 * WEIGHTS["position"] * (xi - targets) / ramp.BOX_SIDE ** 2

    dl_dZ = np.zeros(states.shape)
    dl_dZ[1:, :, 0, 0:2] = coefficient[1:, :, None] * ramp.uphill(angles)[None, :, :]

    dl_dU = np.zeros(controls.shape)
    dl_dU[..., 0, 0:2] = -2.0 * WEIGHTS["control"] * controls[..., 0, 0:2] / SCALE ** 2
    return dl_dZ, dl_dU


def rollout_closed_loop(batch, gain, steps=STEPS):
    """Step one control step at a time, recomputing the action from the state.

    `step_batch` rather than `rollout_batch` because the control is not
    known in advance, and because it returns a fresh array per step rather
    than a view of the simulator's buffer.
    """
    state = batch.state
    states, controls, errors = [state], [], []
    for _ in range(steps):
        error = ramp.along_ramp(state, batch.angles)[:, 0] - batch.targets
        wrench = to_wrench(-gain * error, batch.angles)

        errors.append(error)
        controls.append(wrench)
        state = grip.step_batch(batch.scenes, state, wrench[None], substeps=batch.substeps)
        states.append(state)

    return np.array(states), np.array(controls), np.array(errors)


def one_call_gradient(batch, trajectory, controls, dl_dZ, dl_dU, dpi_dgain):
    """The task doc's recipe: one sweep, contracted with the direct dpi/dK."""
    _, dJ_dU = grip.adjoint_batch(batch.scenes, trajectory, controls, batch.substeps, dl_dZ, dl_dU)
    return (to_action_gradient(dJ_dU, batch.angles) * dpi_dgain).sum()


def per_step_gradient(batch, trajectory, controls, dl_dZ, dl_dU, dpi_dgain, gain):
    """The same sweep, one control step at a time, carrying dpi/dZ between steps."""
    adjoint = dl_dZ[-1].copy()
    total = 0.0
    for t in range(len(controls) - 1, -1, -1):
        seed = np.zeros((2,) + trajectory.shape[1:])
        seed[1] = adjoint
        dJ_dZ0, dJ_dU = grip.adjoint_batch(batch.scenes, trajectory[t:t + 2], controls[t:t + 1], batch.substeps, seed, dl_dU[t:t + 1])

        action_gradient = to_action_gradient(dJ_dU, batch.angles)[0]
        total += (action_gradient * dpi_dgain[t]).sum()

        # Build the term a single call cannot see, where the state feeds the
        # policy which feeds the next state. dpi/dZ is -K * d(xi)/d(x, y),
        # written onto the position columns of body 0.
        through_policy = np.zeros_like(adjoint)
        through_policy[:, 0, 0:2] = (-gain * action_gradient)[:, None] * ramp.uphill(batch.angles)
        adjoint = dJ_dZ0 + through_policy + dl_dZ[t]

    return total


def check(gain):
    batch = fixture(np.radians([18.0, 21.0]))

    trajectory, controls, errors = rollout_closed_loop(batch, gain)

    peak = np.abs(gain * errors).max()
    assert peak < LIMIT, f"the feedback law saturates at K = {gain}, so the clip is what is under test"

    # Replay the recorded controls open-loop and check they reproduce the
    # closed-loop trajectory. If they do not, the adjoint is being handed a
    # different problem than the one measured.
    replay = np.array(grip.rollout_batch(batch.scenes, batch.state, controls, substeps=batch.substeps))
    assert np.abs(replay - trajectory).max() < 1e-12, "closed-loop and open-loop rollouts disagree"

    dl_dZ, dl_dU = reward_seeds(trajectory, controls, batch.angles, batch.targets)
    dpi_dgain = -errors  # d/dK of -K*(xi - xi*), at fixed state

    one_call = one_call_gradient(batch, trajectory, controls, dl_dZ, dl_dU, dpi_dgain)
    per_step = per_step_gradient(batch, trajectory, controls, dl_dZ, dl_dU, dpi_dgain, gain)

    def objective(k):
        rolled, held, _ = rollout_closed_loop(batch, k)
        return reward(rolled, held, batch.angles, batch.targets).sum()

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
