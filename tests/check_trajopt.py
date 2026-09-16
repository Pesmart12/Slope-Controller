"""Can gradient descent actually solve this task?  Run: python tests/check_trajopt.py

Adam on the raw control sequence, with no policy, no critic and no
training loop. Two questions that a gradient check cannot answer:

  Does the reward produce sensible behaviour? A correct gradient says
  nothing about whether the thing it optimizes is worth optimizing.

  Are the gradients navigable? A finite-difference check proves
  correctness at one point. It says nothing about whether 8000 integration
  steps of stiff contact leave a landscape a descent method can cross.

If this converges, SHAC's 32-step windows are a far easier gradient
problem than the full 400-step episode solved here.

The box is unactuated, so every newton reaching it crosses a body-body
contact. Solving that is what says the task is worth training on.

This is also where the shaping term is held honest. Optimization runs with
`shaping=True`, but every number printed is the reward WITHOUT shaping. A
shaping term that only looked good on its own objective would show up here
as a good shaped score and a bad task score.

Takes a few minutes: 800 Adam iterations across three slopes. Lower
ITERATIONS for a rough answer; errors roughly triple at 300.
"""

import numpy as np

import grip
from slope_control import batches, objective, ramp, task

ITERATIONS = 800
# Adam steps are about `learning_rate` per iteration whatever the gradient
# scale, so the step is a fraction of the force range rather than an
# absolute number -- a change to the force limit then does not silently
# change how far 800 iterations get.
STEP_FRACTION = 0.05 / 12.0
TARGET_OFFSET = 0.5

# The top of the sampled range, and deliberately the hard case: at 5 cm
# neither creep nor a holding-force initialization brings the pushers to
# the box within an episode. The shaping term has to do it, so this is
# where it earns its place.
APPROACH_GAP = 0.05


def total_reward(batch, actions):
    """Score one control sequence. Returns the per-environment reward, the
    trajectory and the wrenches.

    The wrenches come back because `optimize` needs them to seed the
    adjoint, and recomputing them would mean this rollout and that one
    could drift apart.

    Unshaped, always. `optimize` maximizes the shaped objective but every
    number reported comes from here, which is what keeps the shaping term
    honest.
    """
    wrenches = task.to_wrench(actions, batch.angles)
    trajectory = np.array(grip.rollout_batch(batch.scenes, batch.state, wrenches, substeps=batch.substeps))
    return objective.reward(trajectory, wrenches, batch.angles, batch.targets).sum(axis=0), trajectory, wrenches


def optimize(batch, iterations=ITERATIONS):
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
    # Zero, deliberately. The shaping term is what has to lift the
    # optimizer off this point; an initialization that did it instead
    # would hide whether the term works.
    actions = np.zeros((task.EPISODE_STEPS, len(batch.scenes), len(task.ACTUATED), 2))
    first_moment, second_moment = np.zeros_like(actions), np.zeros_like(actions)
    beta_1, beta_2, epsilon = 0.9, 0.999, 1e-8
    learning_rate = STEP_FRACTION * objective.SCALE
    history = []

    for iteration in range(1, iterations + 1):
        # Roll the current iterate and record its score.
        reward, trajectory, wrenches = total_reward(batch, actions)
        history.append(reward)

        # Optimize the SHAPED objective but record the unshaped one. A shaping
        # term that only looked good on its own objective would show up as a
        # good shaped score and a bad task score, which is the point of the split.
        dl_dZ, dl_dU = objective.reward_seeds(trajectory, wrenches, batch.angles, batch.targets, shaping=True)

        # One call for the whole episode is CORRECT here, unlike for a policy:
        # the control sequence is a fixed set of variables, not a function of
        # the state, so there is no Z -> a -> Z path for it to miss.
        _, dJ_dU = grip.adjoint_batch(batch.scenes, trajectory, wrenches, batch.substeps, dl_dZ, dl_dU)
        gradient = task.to_action_gradient(dJ_dU, batch.angles)

        # Adam, by hand so this file stays numpy-only. First moment is a
        # running mean of the gradient, second an uncentred running variance.
        first_moment = beta_1 * first_moment + (1.0 - beta_1) * gradient
        second_moment = beta_2 * second_moment + (1.0 - beta_2) * gradient * gradient

        # Divide out the initialization bias. Both averages start at zero
        # and so read low for the first iterations; (1 - beta^t) undoes
        # exactly that, and decays to nothing as beta^t does.
        corrected_first = first_moment / (1.0 - beta_1 ** iteration)
        corrected_second = second_moment / (1.0 - beta_2 ** iteration)

        # Step by the corrected first moment over the root of the corrected
        # second, scaled by `learning_rate`. That division is what holds the
        # step near `learning_rate` whatever the gradient scale, which is why
        # this handles a trajectory whose early steps move the load and whose
        # late steps only hold it. `+` rather than `-`: reward, not loss.
        actions += learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)

        # Project back onto the feasible box every iteration. Without this the
        # iterate can wander far outside the limit while `to_wrench` silently
        # clips it, reporting zero gradient from a point it can never leave.
        actions = task.clip_action(actions)

    return actions, np.array(history)


def solve(degrees):
    """Optimize the control sequence and report what the solution looks like."""
    batch = batches.fixed_batch(np.radians(degrees), offset=TARGET_OFFSET, gaps=APPROACH_GAP)
    idle, _, _ = total_reward(batch, np.zeros((task.EPISODE_STEPS, len(degrees), len(task.ACTUATED), 2)))

    actions, history = optimize(batch)
    final, trajectory, _ = total_reward(batch, actions)

    xi = ramp.along_ramp(trajectory, batch.angles)
    error = xi[-1, :, task.BOX] - batch.targets
    saturated = (np.abs(np.abs(actions) - task.LIMIT).min(axis=-1) < 1e-9).mean()

    print(f"\ntwo pushers: {task.BODY_COUNT} bodies, {len(task.ACTUATED)} actuated, target {100 * TARGET_OFFSET:.0f} cm uphill")
    print(f"  reward {idle.mean():.2f} (zero control) -> {final.mean():.2f}, ended at its best: "
          f"{'yes' if int(np.argmax(history.mean(axis=1))) == ITERATIONS - 1 else 'no'}, saturated {100 * saturated:.1f}% of the time")
    print(f"  {'slope':>6} {'final err':>10} {'box moved':>10}" + "".join(f" {'push ' + str(i):>10}" for i in range(len(task.ACTUATED))))
    for i, d in enumerate(degrees):
        peaks = "".join(f" {actions[:, i, slot, 0].max():>8.2f} N" for slot in range(len(task.ACTUATED)))
        print(f"  {d:>5.0f}d {100 * error[i]:>8.2f} cm {xi[-1, i, task.BOX] - xi[0, i, task.BOX]:>8.2f} m{peaks}")
    return error, final, idle, actions, batch.angles


def report_creep_band(actions, angles):
    """Where the driving pusher's settled force sits between holding and moving.

    Once parked, the force settles near the balance point and stays there:
    what it never does is approach break-free again. The endgame runs
    entirely inside the creep regime, where nothing can slide and can only
    ooze.

    That regime is a penalty artifact with no counterpart under an NCP
    solve, where a body below the friction bound does not move at all and
    the last millimetre has to be closed some other way.

    Both bounds are for the pusher AND the box, since slot 0 drives both.
    No creep RATE is reported: the box and the pusher each rest on two
    corners, so the four contacts do not reduce to the single-body closed
    form, and a number here would be invented rather than derived.
    """
    settled = slice(int(2.0 * task.CONTROL_HZ), int(3.5 * task.CONTROL_HZ))
    driven = ramp.PUSHER_MASS + ramp.BOX_MASS
    held = actions[settled, :, 0, 0].mean(axis=0)
    hold_force = ramp.hold_force(angles, mass=driven)
    break_free = ramp.break_free_force(angles, mass=driven)

    # Locate the settled force between holding and breaking free. Deep
    # inside that band is what says the converged solution is oozing rather
    # than anywhere near sliding.
    margin = np.abs(held - hold_force) / (break_free - hold_force)
    print(f"  settled force {np.array2string(held, precision=2)} N against mg sin(a) {np.array2string(hold_force, precision=2)} N"
          f"   -> {np.array2string(100 * margin, precision=0)}% of the way to break-free")
    return (margin < 0.5).all()


def main():
    degrees = np.array([15.0, 20.0, 22.0])
    error, final, idle, actions, angles = solve(degrees)
    in_creep_regime = report_creep_band(actions, angles)

    failures = []
    if final.mean() <= idle.mean():
        failures.append("optimization did not beat doing nothing")
    if np.abs(error).max() > 10.0 * objective.POSITION_TOLERANCE:
        failures.append(f"worst error {100 * np.abs(error).max():.2f} cm is more than 10x the tolerance; the task may not be solvable as posed")
    if not in_creep_regime:
        failures.append("the settled force is no longer deep in the creep regime, so the recorded solution shape has changed")

    print()
    for failure in failures:
        print(f"  FAIL: {failure}")
    print(f"\n{'PASS' if not failures else 'FAIL'} -- the reward is solvable and its gradients are navigable")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
