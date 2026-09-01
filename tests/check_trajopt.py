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

Run for both variants. The box-only case is the reward's own de-risking;
the two-pusher case is the manipulation task, where the box is unactuated
and every newton reaching it has to cross a body-body contact. Solving
that here is what says the task is worth training on at all.

The one thing this turned up that nothing else would have: **gradients
cannot discover a contact that does not exist.** From a zero
initialization the two-pusher case does not converge slowly, it does not
move at all -- the driving pusher's action sits at exactly 0.00 N for
every iteration, because with no contact d(box position)/d(pusher action)
is identically zero. See `warm_start`, and expect SHAC to meet the same
flat region.

Not an artifact and not an experiment: no figure, and it produces no
number that means anything without GRIP 2.0 to compare against.

Takes about seven minutes -- 800 Adam iterations over two variants, each
a full 400-step episode across three slopes. Lower ITERATIONS if you only
want the shape of the answer; the errors roughly triple at 300.
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import grip  # noqa: E402
from slope_control import ramp, task  # noqa: E402

ITERATIONS = 800
# Adam steps are about `learning_rate` per iteration whatever the gradient
# scale, so the step has to be a fraction of the force range or the two
# variants get very different numbers of effective iterations.
STEP_FRACTION = 0.05 / 12.0
TARGET_OFFSET = 0.5

# Zero, because a gap cannot be closed from a zero initialization. Under
# no control the 2 kg pusher creeps downhill at 1.68 cm/s and the 1 kg box
# at 0.84, so the gap between them OPENS, contact never forms, and
# d(box position)/d(pusher action) is identically zero -- the driving
# pusher's action stays at 0.00 N for all 300 iterations. Gradients cannot
# discover a contact that does not exist.
APPROACH_GAP = 0.0


def warm_start(variant, angles):
    """Start every actuated body at its own holding force, not at zero.

    Zero is a stationary point here, and not a useful one. Holding is the
    "do nothing but do not slide away" control, it encodes none of the
    solution, and it is enough to bring the bodies into contact: with the
    pushers holding station the box still creeps downhill into the lower
    one, so a gradient exists from the first iteration.

    This matters beyond the baseline. A SHAC policy initialized near zero
    output faces exactly the same flat region, so its initialization or
    its exploration noise has to solve the same bootstrap problem.
    """
    bodies = task.bodies_for(variant)
    actions = np.zeros((task.EPISODE_STEPS, len(angles), len(variant.actuated), 2))
    for slot, body in enumerate(variant.actuated):
        actions[:, :, slot, 0] = ramp.hold_force(angles, mass=bodies[body]["mass"])
    return actions


def setup(variant, degrees):
    """Scenes, a settled state and a target, for one variant."""
    angles = np.radians(degrees)
    bodies = task.bodies_for(variant)
    scenes = ramp.make_scenes(angles, bodies=bodies)
    substeps = task.substeps_for(scenes[0])

    start = np.zeros(len(degrees))
    positions = start[:, None] if variant.bodies == 1 else task.placement(start, np.full((len(degrees), len(variant.actuated)), APPROACH_GAP), variant)
    state = task.settle(scenes, ramp.resting_state(positions, angles, bodies=bodies), substeps)
    return scenes, angles, substeps, state, ramp.along_ramp(state, angles)[:, variant.box] + TARGET_OFFSET


def total_reward(variant, scenes, state, actions, angles, targets, substeps):
    """Score one control sequence. Returns the per-environment reward and trajectory."""
    wrenches = task.to_wrench(actions, angles, variant)
    trajectory = np.array(grip.rollout_batch(scenes, state, wrenches, substeps=substeps))
    return task.reward(trajectory, wrenches, angles, targets, variant).sum(axis=0), trajectory


def optimize(variant, scenes, state, angles, targets, substeps, iterations=ITERATIONS):
    """Adam on the raw control sequence, projected back onto the force limit.

    Projected rather than penalized: `to_wrench` clips anyway, and a clipped
    action has zero derivative, so an unprojected iterate can wander far
    outside the limit while reporting no gradient at all. Clipping the
    iterate itself keeps every action feasible and the clip a no-op.

    Adam rather than plain ascent because the position and control terms
    differ in scale by orders of magnitude across a trajectory -- early
    steps move the load, late steps only hold it -- and a single global
    step size serves one or the other, not both.
    """
    actions = warm_start(variant, angles)
    first_moment, second_moment = np.zeros_like(actions), np.zeros_like(actions)
    beta_1, beta_2, epsilon = 0.9, 0.999, 1e-8
    learning_rate = STEP_FRACTION * variant.scale
    history = []

    for iteration in range(1, iterations + 1):
        wrenches = task.to_wrench(actions, angles, variant)
        trajectory = np.array(grip.rollout_batch(scenes, state, wrenches, substeps=substeps))
        history.append(task.reward(trajectory, wrenches, angles, targets, variant).sum(axis=0))

        dl_dZ, dl_dU = task.reward_seeds(trajectory, wrenches, angles, targets, variant)
        _, dJ_dU = grip.adjoint_batch(scenes, trajectory, wrenches, substeps, dl_dZ, dl_dU)
        gradient = task.to_action_gradient(dJ_dU, angles, variant)

        first_moment = beta_1 * first_moment + (1.0 - beta_1) * gradient
        second_moment = beta_2 * second_moment + (1.0 - beta_2) * gradient * gradient
        corrected_first = first_moment / (1.0 - beta_1 ** iteration)
        corrected_second = second_moment / (1.0 - beta_2 ** iteration)

        actions += learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)
        actions = task.clip_action(actions, variant)

    return actions, np.array(history)


def solve(name, variant, degrees):
    """Optimize one variant and report what the solution looks like."""
    scenes, angles, substeps, state, targets = setup(variant, degrees)
    idle, _ = total_reward(variant, scenes, state, warm_start(variant, angles), angles, targets, substeps)

    actions, history = optimize(variant, scenes, state, angles, targets, substeps)
    final, trajectory = total_reward(variant, scenes, state, actions, angles, targets, substeps)

    xi = ramp.along_ramp(trajectory, angles)
    error = xi[-1, :, variant.box] - targets
    saturated = (np.abs(np.abs(actions) - variant.limit).min(axis=-1) < 1e-9).mean()

    print(f"\n{name}: {variant.bodies} bodies, {len(variant.actuated)} actuated, target {100 * TARGET_OFFSET:.0f} cm uphill")
    print(f"  reward {idle.mean():.2f} (zero control) -> {final.mean():.2f}, ended at its best: "
          f"{'yes' if int(np.argmax(history.mean(axis=1))) == ITERATIONS - 1 else 'no'}, saturated {100 * saturated:.1f}% of the time")
    print(f"  {'slope':>6} {'final err':>10} {'box moved':>10}" + "".join(f" {'push ' + str(i):>10}" for i in range(len(variant.actuated))))
    for i, d in enumerate(degrees):
        peaks = "".join(f" {actions[:, i, slot, 0].max():>8.2f} N" for slot in range(len(variant.actuated)))
        print(f"  {d:>5.0f}d {100 * error[i]:>8.2f} cm {xi[-1, i, variant.box] - xi[0, i, variant.box]:>8.2f} m{peaks}")
    return error, final, idle, actions, angles


def report_creep_band(actions, angles, degrees):
    """The box-only solution's second phase, which is a penalty artifact.

    The settled force lands between holding and breaking free every time.
    That band is the creep regime: too little force to slide the box, more
    than enough to make it creep uphill instead of down, at
    (f - mg sin a) / 2*b_slip. The optimizer is using contact softening as
    a fine-positioning mechanism, because the alternative -- exceed
    break-free and brake -- has no gentle setting.

    Worth stating plainly: this strategy does not exist under an NCP solve.
    Below the friction bound a rigid box does not move at all.
    """
    settled = slice(int(2.0 * task.CONTROL_HZ), int(3.5 * task.CONTROL_HZ))
    held = actions[settled, :, 0, 0].mean(axis=0)
    hold_force, break_free = ramp.hold_force(angles), ramp.break_free_force(angles)

    in_band = (hold_force < held) & (held < break_free)
    creep = 100 * (held - hold_force) / (2 * ramp.DEFAULT_PENALTY["slip_damping"])
    print(f"  settled force in the creep band at every slope: {'yes' if in_band.all() else 'NO'}"
          f"   implied uphill creep {np.array2string(creep, precision=2)} cm/s")
    return in_band.all()


def main():
    degrees = np.array([15.0, 20.0, 22.0])
    box_error, box_final, box_idle, actions, angles = solve("box only", task.BOX_ONLY, degrees)
    in_band = report_creep_band(actions, angles, degrees)
    pusher_error, pusher_final, pusher_idle, _, _ = solve("two pushers", task.TWO_PUSHERS, degrees)

    failures = []
    if np.abs(box_error).max() > task.POSITION_TOLERANCE:
        failures.append(f"box only: worst error {100 * np.abs(box_error).max():.2f} cm exceeds the {100 * task.POSITION_TOLERANCE:.0f} cm tolerance")
    if box_final.mean() <= box_idle.mean():
        failures.append("box only: optimization did not beat doing nothing")
    if not in_band:
        failures.append("box only: settled force left the creep band, so the recorded solution shape no longer holds")
    if pusher_final.mean() <= pusher_idle.mean():
        failures.append("two pushers: optimization did not beat doing nothing")
    if np.abs(pusher_error).max() > 10.0 * task.POSITION_TOLERANCE:
        failures.append(f"two pushers: worst error {100 * np.abs(pusher_error).max():.2f} cm is more than 10x the tolerance; the task may not be solvable as posed")

    print()
    for failure in failures:
        print(f"  FAIL: {failure}")
    print(f"\n{'PASS' if not failures else 'FAIL'} -- the reward is solvable and its gradients are navigable")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
