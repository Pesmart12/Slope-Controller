"""Baseline: can gradient descent actually solve the task?  Run: python tests/check_trajopt.py

This replaces the MPPI step. MPPI's job in the plan was to remove an
ambiguity -- if SHAC struggles, is it the gradients or the task? -- but
being zeroth order it never calls the adjoint, so it can only ever answer
half of that. `check_task.py` already answered the other half by finite
difference. What is left open is narrower, and this closes it:

  * Does the reward produce sensible *behaviour*? Correct gradients say
    nothing about whether the thing they optimize is worth optimizing.
  * Are the gradients *navigable*? A finite-difference check proves
    correctness at a point. It says nothing about whether 8000 integration
    steps of stiff contact leave a landscape a descent method can cross.

Direct trajectory optimization answers both with no policy, no critic and
no training loop -- Adam on the raw control sequence, which has one real
hyperparameter. If this converges, SHAC's 32-step windows are a far
easier gradient problem than the full 400-step episode solved here.

Not an artifact and not an experiment: no figure, and it produces no
number that means anything without GRIP 2.0 to compare against. The
closed-form reference it *does* have is ramp.hold_force -- once the box is
parked, penalty contact gives it no friction for free, so the optimizer
has to discover that holding station costs exactly mg*sin(a).
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import grip  # noqa: E402
from slope_control import ramp, task  # noqa: E402

ITERATIONS = 300
LEARNING_RATE = 0.05
TARGET_OFFSET = 0.5


def total_reward(scenes, state, actions, angles, targets, substeps):
    """Score one control sequence. Returns the reward and the trajectory."""
    wrenches = task.to_wrench(actions, angles)
    trajectory = np.array(grip.rollout_batch(scenes, state, wrenches, substeps=substeps))
    return task.reward(trajectory, wrenches, angles, targets).sum(axis=0), trajectory


def optimize(scenes, state, angles, targets, steps, substeps, iterations=ITERATIONS, learning_rate=LEARNING_RATE):
    """Adam on the raw control sequence, projected back onto the force limit.

    Projected rather than penalized: `to_wrench` clips anyway, and a clipped
    action has zero derivative, so an unprojected iterate can wander far
    outside the limit while reporting no gradient at all. Clipping the
    iterate itself keeps every action feasible and the clip a no-op.

    Adam rather than plain ascent because the position and control terms
    differ in scale by orders of magnitude across a trajectory -- early
    steps move the box, late steps only hold it -- and a single global step
    size serves one or the other, not both.
    """
    actions = np.zeros((steps, len(scenes), 2))
    first_moment, second_moment = np.zeros_like(actions), np.zeros_like(actions)
    beta_1, beta_2, epsilon = 0.9, 0.999, 1e-8
    history = []

    for iteration in range(1, iterations + 1):
        wrenches = task.to_wrench(actions, angles)
        trajectory = np.array(grip.rollout_batch(scenes, state, wrenches, substeps=substeps))
        history.append(task.reward(trajectory, wrenches, angles, targets).sum(axis=0))

        dl_dZ, dl_dU = task.reward_seeds(trajectory, wrenches, angles, targets)
        _, dJ_dU = grip.adjoint_batch(scenes, trajectory, wrenches, substeps, dl_dZ, dl_dU)
        gradient = task.to_action_gradient(dJ_dU, angles)

        first_moment = beta_1 * first_moment + (1.0 - beta_1) * gradient
        second_moment = beta_2 * second_moment + (1.0 - beta_2) * gradient * gradient
        corrected_first = first_moment / (1.0 - beta_1 ** iteration)
        corrected_second = second_moment / (1.0 - beta_2 ** iteration)

        actions += learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)
        actions = task.clip_action(actions)

    return actions, np.array(history)


def main():
    degrees = np.array([15.0, 20.0, 22.0])
    angles = np.radians(degrees)
    scenes = ramp.make_scenes(angles)
    substeps = task.substeps_for(scenes[0])
    steps = task.EPISODE_STEPS

    state = task.settle(scenes, ramp.resting_state(0.0, angles), substeps)
    start = ramp.along_ramp(state, angles)[:, 0]
    targets = start + TARGET_OFFSET

    idle, idle_trajectory = total_reward(scenes, state, np.zeros((steps, len(scenes), 2)), angles, targets, substeps)
    idle_final = ramp.along_ramp(idle_trajectory, angles)[-1, :, 0]
    print(f"target is {100 * TARGET_OFFSET:.0f} cm uphill, episode {task.EPISODE_SECONDS:.0f} s, {steps} control steps")
    print(f"zero control: box ends {100 * (idle_final - targets).mean():.1f} cm from target, reward {idle.mean():.2f}\n")

    print(f"optimizing {steps * len(scenes) * 2} control variables, {ITERATIONS} Adam iterations")
    actions, history = optimize(scenes, state, angles, targets, steps, substeps)
    for iteration in [0, 9, 49, 99, 199, ITERATIONS - 1]:
        print(f"  iter {iteration + 1:>4}   reward {history[iteration].mean():>10.3f}")

    final, trajectory = total_reward(scenes, state, actions, angles, targets, substeps)
    xi = ramp.along_ramp(trajectory, angles)[:, :, 0]
    error = xi[-1] - targets

    # The settled window, minus the tail: with no future left to protect,
    # the optimizer always lets the force decay over the last few steps.
    settled = slice(int(2.0 * task.CONTROL_HZ), int(3.5 * task.CONTROL_HZ))
    held = actions[settled, :, 0].mean(axis=0)
    hold_force = ramp.hold_force(angles)
    break_free = ramp.break_free_force(angles)
    saturated = np.abs(np.abs(actions) - task.FORCE_LIMIT).min(axis=-1) < 1e-9

    print(f"\n  {'slope':>6} {'final err':>10} {'peak err':>9} {'mg sin':>8} {'settled':>9} {'break free':>11} {'saturated':>10}")
    for i, d in enumerate(degrees):
        peak = np.abs(xi[:, i] - targets[i]).max()
        print(f"  {d:>5.0f}d {100 * error[i]:>8.2f} cm {100 * peak:>7.1f} cm {hold_force[i]:>6.2f} N {held[i]:>7.2f} N {break_free[i]:>9.2f} N {100 * saturated[:, i].mean():>9.1f}%")

    # The solution has two phases and the second is the interesting one.
    # Covering 48 cm at creep rates would take a minute, so the optimizer
    # must exceed break-free early, brake, and only then creep in.
    print(f"\n  {'slope':>6} {'peak push':>10} {'above break-free':>17} {'last above':>11} {'moved':>8}")
    for i, d in enumerate(degrees):
        above = actions[:, i, 0] > break_free[i]
        last = np.nonzero(above)[0].max() / task.CONTROL_HZ if above.any() else float("nan")
        print(f"  {d:>5.0f}d {actions[:, i, 0].max():>8.2f} N {100 * above.mean():>15.1f}% {last:>9.2f} s {xi[-1, i] - xi[0, i]:>6.2f} m")

    print(f"\n  reward improved {history[0].mean():.2f} -> {final.mean():.2f}, ended at its best: "
          f"{'yes' if int(np.argmax(history.mean(axis=1))) == ITERATIONS - 1 else 'no'}")

    # The settled force lands between holding and breaking free, every
    # time. That band is the penalty creep regime: too little force to
    # slide the box, more than enough to make it creep uphill instead of
    # down, at (f - mg sin a) / 2*b_slip. The optimizer is using contact
    # softening as a fine-positioning mechanism, because the alternative --
    # exceed break-free and brake -- has no gentle setting.
    #
    # Worth stating plainly: this strategy does not exist under an NCP
    # solve. Below the friction bound a rigid box does not move at all.
    creep_band = (hold_force < held) & (held < break_free)
    print(f"  settled force sits in the creep band (hold < f < break free): {'all slopes' if creep_band.all() else 'NOT all slopes'}")
    print(f"  implied creep {100 * (held - hold_force) / (2 * ramp.DEFAULT_PENALTY['slip_damping'])} cm/s uphill")

    failures = []
    if np.abs(error).max() > task.POSITION_TOLERANCE:
        failures.append(f"worst final error {100 * np.abs(error).max():.2f} cm exceeds the {100 * task.POSITION_TOLERANCE:.0f} cm tolerance")
    if final.mean() <= idle.mean():
        failures.append("optimization did not beat doing nothing")
    if not creep_band.all():
        failures.append("settled force is outside the creep band, so the solution is not the one described above")

    for failure in failures:
        print(f"  FAIL: {failure}")
    print(f"\n{'PASS' if not failures else 'FAIL'} -- the reward is solvable and its gradients are navigable")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
