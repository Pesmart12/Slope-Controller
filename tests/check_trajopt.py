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
cannot discover a contact that does not exist.** With the task objective
alone the two-pusher case does not converge slowly, it does not move at
all -- the driving pusher's action sits at exactly 0.00 N for every
iteration, because with no contact d(box position)/d(pusher action) is
identically zero.

The fix is the shaping term in `task.approach_penalty`, and this check is
what holds it honest: optimization runs with `shaping=True`, and every
number reported is the UNSHAPED task reward. A shaping term that only
looked good on its own objective would show up here as a good shaped
score and a bad task score.

Not an artifact and not an experiment: no figure, and it produces no
number that means anything without GRIP 2.0 to compare against.

Takes about seven minutes -- 800 Adam iterations over two variants, each
a full 400-step episode across three slopes. Lower ITERATIONS if you only
want the shape of the answer; the errors roughly triple at 300.
"""

import numpy as np

import grip
from slope_control import ramp, task

ITERATIONS = 800
# Adam steps are about `learning_rate` per iteration whatever the gradient
# scale, so the step has to be a fraction of the force range or the two
# variants get very different numbers of effective iterations.
STEP_FRACTION = 0.05 / 12.0
TARGET_OFFSET = 0.5

# The top of the sampled range, and deliberately the hard case: at 5 cm
# neither creep nor a holding-force initialization brings the pushers to
# the box within an episode. The shaping term has to do it, so this is
# where it earns its place.
APPROACH_GAP = 0.05


def total_reward(batch, actions):
    """Score one control sequence. Returns the per-environment reward and trajectory."""
    wrenches = task.to_wrench(actions, batch.angles, batch.variant)
    trajectory = np.array(grip.rollout_batch(batch.scenes, batch.state, wrenches, substeps=batch.substeps))
    return task.reward(trajectory, wrenches, batch.angles, batch.targets, batch.variant).sum(axis=0), trajectory


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
    variant = batch.variant
    actions = np.zeros((task.EPISODE_STEPS, len(batch.scenes), len(variant.actuated), 2))
    first_moment, second_moment = np.zeros_like(actions), np.zeros_like(actions)
    beta_1, beta_2, epsilon = 0.9, 0.999, 1e-8
    learning_rate = STEP_FRACTION * variant.scale
    history = []

    for iteration in range(1, iterations + 1):
        wrenches = task.to_wrench(actions, batch.angles, variant)
        trajectory = np.array(grip.rollout_batch(batch.scenes, batch.state, wrenches, substeps=batch.substeps))
        history.append(task.reward(trajectory, wrenches, batch.angles, batch.targets, variant).sum(axis=0))  # unshaped, for reporting

        # Optimize the SHAPED objective but record the unshaped one. A shaping
        # term that only looked good on its own objective would show up as a
        # good shaped score and a bad task score, which is the point of the split.
        dl_dZ, dl_dU = task.reward_seeds(trajectory, wrenches, batch.angles, batch.targets, variant, shaping=True)

        # One call for the whole episode is CORRECT here, unlike for a policy:
        # the control sequence is a fixed set of variables, not a function of
        # the state, so there is no Z -> a -> Z path for it to miss.
        _, dJ_dU = grip.adjoint_batch(batch.scenes, trajectory, wrenches, batch.substeps, dl_dZ, dl_dU)
        gradient = task.to_action_gradient(dJ_dU, batch.angles, variant)

        # Adam, by hand so this file stays numpy-only. First moment is a
        # running mean of the gradient, second an uncentred running variance.
        first_moment = beta_1 * first_moment + (1.0 - beta_1) * gradient
        second_moment = beta_2 * second_moment + (1.0 - beta_2) * gradient * gradient

        # Both averages start at zero and so are biased low for the first
        # iterations. Dividing by (1 - beta^t) undoes exactly that, and the
        # correction decays to nothing as beta^t does.
        corrected_first = first_moment / (1.0 - beta_1 ** iteration)
        corrected_second = second_moment / (1.0 - beta_2 ** iteration)

        # Dividing by the root second moment is what makes the step size
        # roughly `learning_rate` regardless of gradient scale -- which is why
        # this handles a trajectory whose early steps move the load and whose
        # late steps only hold it. `+` rather than `-`: reward, not loss.
        actions += learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)

        # Project back onto the feasible box every iteration. Without this the
        # iterate can wander far outside the limit while `to_wrench` silently
        # clips it, reporting zero gradient from a point it can never leave.
        actions = task.clip_action(actions, variant)

    return actions, np.array(history)


def solve(name, variant, degrees):
    """Optimize one variant and report what the solution looks like."""
    batch = task.fixed_batch(np.radians(degrees), variant, offset=TARGET_OFFSET, gaps=APPROACH_GAP)
    idle, _ = total_reward(batch, np.zeros((task.EPISODE_STEPS, len(degrees), len(variant.actuated), 2)))

    actions, history = optimize(batch)
    final, trajectory = total_reward(batch, actions)

    xi = ramp.along_ramp(trajectory, batch.angles)
    error = xi[-1, :, variant.box] - batch.targets
    saturated = (np.abs(np.abs(actions) - variant.limit).min(axis=-1) < 1e-9).mean()

    print(f"\n{name}: {variant.bodies} bodies, {len(variant.actuated)} actuated, target {100 * TARGET_OFFSET:.0f} cm uphill")
    print(f"  reward {idle.mean():.2f} (zero control) -> {final.mean():.2f}, ended at its best: "
          f"{'yes' if int(np.argmax(history.mean(axis=1))) == ITERATIONS - 1 else 'no'}, saturated {100 * saturated:.1f}% of the time")
    print(f"  {'slope':>6} {'final err':>10} {'box moved':>10}" + "".join(f" {'push ' + str(i):>10}" for i in range(len(variant.actuated))))
    for i, d in enumerate(degrees):
        peaks = "".join(f" {actions[:, i, slot, 0].max():>8.2f} N" for slot in range(len(variant.actuated)))
        print(f"  {d:>5.0f}d {100 * error[i]:>8.2f} cm {xi[-1, i, variant.box] - xi[0, i, variant.box]:>8.2f} m{peaks}")
    return error, final, idle, actions, batch.angles


def report_creep_band(actions, angles, degrees):
    """The box-only solution's second phase, which is a penalty artifact.

    Once parked, the force settles NEAR mg*sin(a) -- the balance point --
    and the residual, either sign, trims the last few millimetres by
    creeping at (f - mg sin a) / 2*b_slip. What it never does is approach
    break-free again: the endgame is entirely inside the creep regime,
    where the box cannot slide and can only ooze.

    That regime is a penalty artifact and has no counterpart under an NCP
    solve, where a box below the friction bound does not move at all and
    the last millimetre has to be closed some other way. It is the sharpest
    thing the baseline says about the comparison.

    A correction worth recording: an earlier run reported the settled force
    sitting strictly between hold and break-free, creeping uphill at
    0.67 cm/s at every slope. That was an under-converged optimizer still
    travelling the last centimetre. Converged, it parks at the balance
    point instead and the residual creep drops to +/-0.3 cm/s with either
    sign. The mechanism was right; the number was measuring how far the
    solution still had to go.
    """
    settled = slice(int(2.0 * task.CONTROL_HZ), int(3.5 * task.CONTROL_HZ))
    held = actions[settled, :, 0, 0].mean(axis=0)
    hold_force, break_free = ramp.hold_force(angles), ramp.break_free_force(angles)

    # Well inside the creep regime, rather than anywhere near sliding.
    margin = np.abs(held - hold_force) / (break_free - hold_force)
    creep = 100 * (held - hold_force) / (2 * ramp.DEFAULT_PENALTY["slip_damping"])
    print(f"  settled force {np.array2string(held, precision=2)} N against mg sin(a) {np.array2string(hold_force, precision=2)} N"
          f"   -> {np.array2string(100 * margin, precision=0)}% of the way to break-free")
    print(f"  residual creep {np.array2string(creep, precision=2)} cm/s, trimming rather than travelling")
    return (margin < 0.5).all()


def main():
    degrees = np.array([15.0, 20.0, 22.0])
    box_error, box_final, box_idle, actions, angles = solve("box only", task.BOX_ONLY, degrees)
    in_creep_regime = report_creep_band(actions, angles, degrees)
    pusher_error, pusher_final, pusher_idle, _, _ = solve("two pushers", task.TWO_PUSHERS, degrees)

    failures = []
    if np.abs(box_error).max() > task.POSITION_TOLERANCE:
        failures.append(f"box only: worst error {100 * np.abs(box_error).max():.2f} cm exceeds the {100 * task.POSITION_TOLERANCE:.0f} cm tolerance")
    if box_final.mean() <= box_idle.mean():
        failures.append("box only: optimization did not beat doing nothing")
    if not in_creep_regime:
        failures.append("box only: the settled force is no longer deep in the creep regime, so the recorded solution shape has changed")
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
