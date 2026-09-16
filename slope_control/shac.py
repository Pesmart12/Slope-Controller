"""SHAC: train the actor by differentiating short windows of simulation.

The loop, one iteration:

    1. Roll the policy forward W control steps from wherever the last
       window ended.
    2. Differentiate that window's reward back to the actor's weights,
       through the physics, and take an Adam step.
    3. Fit the critic to the returns the window actually produced.
    4. Nudge the target critic toward the critic.
    5. Carry the state forward. Resample once an episode is used up.

What makes this SHAC rather than a policy-gradient method is step 2. The
gradient is computed, not estimated: GRIP differentiates the contact and
`sweep.policy_gradient` carries that through the network. No sampling, no
score function, no advantage estimate.

The window is short because differentiating a whole 400-step episode
through stiff contact is badly conditioned. The critic is what makes a
short window sound -- it estimates the reward after the window ends, so
the actor is not optimizing 0.32 s in isolation.

Discounted at gamma = 0.99, which is a training device and not a claim
about what the task wants. Everything reported is the plain undiscounted
reward -- `evaluate` sees no gamma at all.

An earlier version ran undiscounted, on the reasoning that the episode is
finite so nothing diverges and the whole 4 s should count. Two things are
wrong with that:

    At gamma = 1 the Bellman operator is not a contraction, so the
    bootstrap has no unique fixed point and no reason to converge. It
    drifted to one: against episode returns near -180 the critic settled
    at -53.87.

    The value of a state under gamma = 1 also depends on how many steps
    remain, and `observe` carries no clock. So the critic was being asked
    to fit "-180 early in the episode" and "-0.5 late in it" with one
    number and no way to tell them apart. Discounting makes the far future
    irrelevant, which is most of why it is standard for episodic tasks.

Both of those stand on their own, which is why the discount stays. It is
not a fix for anything else, and was never added as one.

The actor's gradient is norm-clipped by default, at MAX_GRADIENT_NORM.
Measured against an unclipped arm at the same seed it is WORSE on both
counts -- a worse peak, 3.48 cm against 1.39, and a worse endpoint, 9.19
against 6.69 -- with the ceiling binding on 610 of 2000 iterations, which
is the regime where the comparison means something. Kept because it is
cheap insurance against a genuine gradient spike, not because it helped.

**The reported error rising after an early minimum is not a training
failure.** It is penalty creep taking the box off the target, and it
happens with the critic cut out of the actor's objective entirely.
CLAUDE.md's standing reminder has the measurement and the two opposite
ways it bites. Do not go looking for the optimizer bug; there isn't one.

The critic is fitted in NORMALIZED units. Returns are still a badly scaled
target for a network starting near zero, so `policy.Critic` keeps a
running mean and standard deviation, fits in those units, and converts
back on the way out. Callers and the actor's gradient see real reward.

The reward is MAXIMIZED, following the sign convention `task.py`
documents. Adam minimizes, so the gradient is negated before every step.
"""

import copy

import numpy as np
import torch

from . import batches, objective, observation, policy, ramp, sweep, task

WINDOW = 32
N_ENVS = 64

# Effective horizon 1/(1 - gamma) = 100 control steps, one second, against
# a four-second episode. Long enough to cover the approach and the arrival,
# short enough that the bootstrap is a contraction. Raise toward 0.997 if a
# longer horizon is ever wanted; it still contracts.
GAMMA = 0.99

ACTOR_LR = 1e-3
CRITIC_LR = 1e-3

# How far the target critic moves toward the critic each iteration. Small,
# because its whole job is to be a slower-moving thing for the critic to
# aim at. 0.005 is the usual starting point.
POLYAK = 0.005

# Gradient passes over the window's data per iteration. More than one
# because a single pass barely moves the critic; many would overfit a
# window of 32 steps.
CRITIC_EPOCHS = 4

# Where along the episode the critic's slope is read. 3 s in, inside the
# hold window, which is where the policy spends most of an episode and
# where the reported error is decided.
SLOPE_AT_STEP = 300

# How far the box is displaced to central-difference the true return.
#
# Sized by where the difference quotient CONVERGES, not by what counts as a
# physically sensible nudge. Measured on a trained checkpoint 3 s into an
# episode, the quotient settles near 48 reward/m below about 0.1 mm and
# then walks away as the step grows: 48.6 at 0.1 mm, 43.3 at 1 mm, 35.8 at
# 2 mm, 23.4 at 5 mm. Past that it is reading the return's curvature over a
# centimetre rather than its slope at a point.
#
# The first version of this used 5 mm, chosen for sitting inside
# POSITION_TOLERANCE and above the millimetre the settle window drifts.
# Both true, neither relevant, and the two criteria disagree by a factor of
# 50. Penalty contact is stiff and Coulomb friction has no gentle regime,
# so the return is only linear over a very short distance.
SLOPE_DELTA = 1.0e-4

# Ceiling on the actor's gradient norm. Applied after `ascend` divides by
# the environment count, so one value holds however many environments run.
#
# Sized off a measured run. Over 2000 iterations at 64 environments the
# post-division norm had a median near 1.2 and a tail at 4.4, 7.7, 13.9 and
# 22.5. 2.0 sits above the bulk and cuts that tail.
#
# Adam does not absorb those on its own: a spike enters the numerator at
# once and the second-moment denominator only over the following
# iterations, so the outsized step lands before the correction does.
MAX_GRADIENT_NORM = 2.0


def window_returns(batch, window, target_critic, shaping, gamma=GAMMA, bootstrap=True):
    """The return from each state in the window: discounted reward to the end
    of the window, plus the target critic's estimate of everything after it.

    Returns (steps + 1, environments), lined up with `window.states`.

    `bootstrap=False` for the window that ends an episode. There is no
    "after" there, so the correct terminal value is zero. Bootstrapping
    anyway teaches the critic that reward keeps arriving past step 400,
    which is a bias with nothing to correct it.

    The bootstrap comes from the TARGET critic rather than the critic being
    trained, so the regression target does not move every time the critic
    does.
    """
    rewards = objective.reward(window.states, window.wrenches, batch.angles, batch.targets, shaping=shaping)
    steps, n_envs = rewards.shape

    returns = torch.zeros((steps + 1, n_envs), dtype=torch.float64)
    if bootstrap:
        # Observe the window's final state, score it with the target critic,
        # and write that into the last row. Under no_grad because this is a
        # regression target, and a target has to be a constant -- otherwise
        # fitting the critic would push gradients into the target network,
        # which is the one thing it exists to avoid.
        with torch.no_grad():
            terminal = policy.as_tensor(observation.observe(window.states[-1], batch.angles, batch.targets))
            returns[-1] = target_critic(terminal)

    # Walk backwards: the value of being at step t is this step's reward plus
    # the discounted value of where it lands. rewards[t] is the reward for
    # the transition out of state t, which is why the indices line up.
    rewards = policy.as_tensor(rewards)
    for t in range(steps - 1, -1, -1):
        returns[t] = rewards[t] + gamma * returns[t + 1]
    return returns


def fit_critic(critic, optimizer, observations, returns, epochs=CRITIC_EPOCHS):
    """Regress the critic onto the window's returns. Returns the final loss,
    or nan if `epochs` is zero.

    `observations` is a list of (environments, features), one per control
    step; `returns` is (steps + 1, environments). Both are flattened to one
    row per (step, environment) pair and fitted as independent samples --
    the critic reads an observation and nothing else, so the order the rows
    were produced in carries no information.

    Fitted in normalized units. `update_statistics` runs first so the
    predictions and the targets are on the same scale.
    """
    critic.update_statistics(returns)

    # Concatenate the per-step observations into one array of rows, cutting
    # each off the actor's autograd graph on the way:
    #   steps x (environments, features) -> (steps * environments, features)
    # Without the detach, loss.backward() would continue through the physics
    # and into the actor.
    features = torch.cat([observation.detach() for observation in observations], dim=0)

    # Drop the last return, flatten to one value per row of `features`, and
    # convert to the units the network predicts in:
    #   (steps + 1, environments) -> (steps * environments,)
    # The dropped row holds the bootstrap, which has no observation to pair
    # with.
    targets = critic.normalize(returns[:-1].reshape(-1)).detach()

    final_loss = float("nan")
    for _ in range(epochs):
        # Clear the gradients the previous epoch left on the parameters.
        optimizer.zero_grad()

        # Mean squared error between the network's raw output and the
        # normalized returns. `normalized` rather than `forward`, which
        # would un-normalize and put the two sides on different scales.
        loss = torch.nn.functional.mse_loss(critic.normalized(features), targets)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
    return final_loss


def soft_update(target, source, tau=POLYAK):
    """Move the target critic a fraction `tau` of the way toward the critic.

    Weights AND buffers move on the same schedule, which is the point. The
    buffers hold the normalization statistics, and `normalized` predicts in
    whatever units the statistics defined when it was fitted. A lagged
    network un-normalized with current statistics would mix vintages:
    `old_net(obs) * new_std + new_mean` is not an estimate of anything.
    Blending both keeps them matched, so the target stays a coherent older
    version of the critic rather than a spliced one.

    An earlier version copied the buffers outright and had exactly that
    mismatch. It is self-correcting once `count` is large, since the
    statistics stop moving, but it is worst in the first few hundred
    iterations when they move fastest.

    `count` gets blended too. It is meaningless on a target -- nothing
    calls `update_statistics` on one -- so it is inert either way, and
    special-casing it by name would cost more than it saves.
    """
    with torch.no_grad():
        for target_parameter, parameter in zip(target.parameters(), source.parameters()):
            target_parameter.mul_(1.0 - tau).add_(parameter, alpha=tau)
        for target_buffer, buffer in zip(target.buffers(), source.buffers()):
            target_buffer.mul_(1.0 - tau).add_(buffer, alpha=tau)


def ascend(module, optimizer, n_envs, max_norm=MAX_GRADIENT_NORM):
    """Take one Adam step UPHILL on the accumulated gradient, norm-clipped.

    Returns the gradient norm measured before clipping, so a caller can
    count how often the ceiling actually binds. Compare it against
    `max_norm`, not against the norm `train` logs, which is measured before
    the division below and so is larger by `n_envs`.

    `policy_gradient` accumulates dJ/d(weights) where J is a reward to be
    maximized, and torch optimizers minimize, so every gradient is negated
    first. This is the sign convention `task.py` documents, and this is the
    one place it has to be acted on.

    Also divides by the environment count, so the step size means the same
    thing whether the batch holds 8 environments or 64. Clipping happens
    after that division, which is why `MAX_GRADIENT_NORM` does not have to
    be resized when the environment count is.

    Pass `max_norm=None` to skip clipping entirely, which is what reproduces
    the runs taken before it existed.
    """
    for parameter in module.parameters():
        if parameter.grad is not None:
            parameter.grad.neg_().div_(n_envs)

    # Measure the length of the flattened gradient, which is the length the
    # step would have without a ceiling. Raises here if anything went NaN or
    # infinite. Same quantity `train` logs as |grad|, differing only by the
    # division above.
    total = policy.gradient_norm(module)

    if max_norm is not None:
        # Multiply every gradient by the ratio max_norm / total, whenever
        # total is the larger. The ratio is then below 1, so the flattened
        # gradient comes out with length exactly max_norm and its direction
        # unchanged.
        torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm)

    optimizer.step()
    return total


def rolled_episode(actor, batch=None, n_envs=8, seed=1000):
    """Build a batch if one is not given, and roll one deterministic episode
    on it. Returns (batch, window).

    `evaluate` and `calibration` both need exactly this, and at their shared
    default seed they would each compute the same trajectory. Rolling it
    once here and passing the pair to both is what stops that.
    """
    if batch is None:
        batch = batches.sample_batch(np.random.default_rng(seed), n_envs)

    # Roll with the noise off and no autograd graph. `rollout` normally
    # keeps 400 live graphs so the sweep can walk back through them;
    # nothing that consumes this differentiates, so building them would be
    # 400 steps of bookkeeping thrown away.
    with torch.no_grad():
        return batch, sweep.rollout(batch, actor, task.EPISODE_STEPS, deterministic=True)


def evaluate(actor, batch=None, n_envs=8, seed=1000, shaping=False, episode=None):
    """Run full episodes with the noise switched off and score them.

    Returns the per-environment unshaped reward and the final position
    error, signed.

    Deterministic, and by default on a seed the training loop never saw,
    so this measures the policy rather than the exploration it happened to
    do. Pass a `batch` to score a specific configuration instead --
    `check_trajopt`'s, for one, which is the only way its numbers and
    these are answers to the same question.

    `episode` takes an already-rolled (batch, window) pair from
    `rolled_episode`, which is how `train` scores the policy and the critic
    on one rollout rather than two. Both halves travel together because
    scoring a window against a different batch's targets is not a thing
    that should be expressible.
    """
    batch, rolled = rolled_episode(actor, batch=batch, n_envs=n_envs, seed=seed) if episode is None else episode

    reward = objective.reward(rolled.states, rolled.wrenches, batch.angles, batch.targets, shaping=shaping).sum(axis=0)

    # Score the same rollout again with the shaping term switched on.
    # Training maximizes this one; every reported number is the other.
    shaped = objective.reward(rolled.states, rolled.wrenches, batch.angles, batch.targets, shaping=True).sum(axis=0)

    final = ramp.along_ramp(rolled.states[-1], batch.angles)[:, task.BOX] - batch.targets
    return reward, final, shaped


def calibration(actor, critic, n_envs=8, seed=1000, gamma=GAMMA, shaping=True, episode=None):
    """Measure the critic against the discounted return it is meant to predict.

    Returns a dict: `bias` (mean V - mean true return), `rmse`, `correlation`,
    and `initial` / `initial_true`, the two numbers for the episode's first
    state.

    The critic's own training loss cannot answer this. With a 32-step window
    at gamma = 0.99 the regression target is 32 real rewards plus
    gamma^32 = 0.725 times the target critic's output, so roughly 72% of what
    the critic is fitted to is a lagged copy of itself. A loss near 1e-4 says
    those two agree, not that either is right. This runs whole episodes to
    termination instead, where the discounted return is a fact and needs no
    bootstrap.

    Scored on SHAPED reward by default, because that is what the critic was
    trained on. Passing shaping=False measures it against a return it was
    never asked to predict.

    `episode` takes an already-rolled (batch, window) pair, so this and
    `evaluate` can share one rollout; see `rolled_episode`.
    """
    batch, rolled = rolled_episode(actor, n_envs=n_envs, seed=seed) if episode is None else episode

    rewards = objective.reward(rolled.states, rolled.wrenches, batch.angles, batch.targets, shaping=shaping)
    steps = len(rewards)

    # Accumulate the discounted return to go, walking backwards from a
    # terminal value of zero. Zero because the episode really ends here, so
    # there is no "after" to estimate -- this is the quantity the training
    # bootstrap stands in for.
    true = np.zeros((steps + 1, n_envs))
    for t in range(steps - 1, -1, -1):
        true[t] = rewards[t] + gamma * true[t + 1]

    observations = observation.observe(rolled.states, batch.angles, batch.targets)
    with torch.no_grad():
        # (steps + 1, environments, features) -> one row per (step,
        # environment) for the forward pass, then back to the original shape.
        flat = policy.as_tensor(observations.reshape(-1, observations.shape[-1]))
        predicted = critic(flat).numpy().reshape(steps + 1, n_envs)

    error = predicted - true
    return dict(bias=float(error.mean()), rmse=float(np.sqrt((error ** 2).mean())),
                correlation=float(np.corrcoef(predicted.reshape(-1), true.reshape(-1))[0, 1]),
                initial=float(predicted[0].mean()), initial_true=float(true[0].mean()))


def slope_calibration(actor, critic, n_envs=8, seed=1000, gamma=GAMMA, shaping=True, delta=SLOPE_DELTA, start_step=0, episode=None):
    """Measure the critic's SLOPE against the slope of the true return.

    Returns a dict: `critic` and `true`, the two mean slopes in reward per
    metre; `bias`, `rmse` and `correlation` between them; and `agreement`,
    the fraction of environments where the two carry the same sign.

    `calibration` asks whether V is the right number. This asks whether it
    points the right way, which is a different question and the one the
    actor actually depends on. What enters the actor's objective is
    dV/d(obs) and never V -- `sweep.policy_gradient` seeds the window's
    final adjoint with it and with nothing else the critic produces. A
    network can fit values to 0.97 correlation and still have noisy local
    derivatives, and nothing here has ever checked.

    The perturbation moves the box along the ramp. That direction keeps its
    height above the surface, so contact geometry is unchanged and the two
    rollouts differ in position error and nothing else. `SLOPE_DELTA` sets
    how far, and it has to be small enough that the return is still linear
    across it -- see the constant, which records where the quotient
    converges and what happens well above it.

    `start_step` rolls the policy forward that many steps before measuring,
    so the slope can be read at a state the policy actually reaches rather
    than only at the start. The degradation this exists to diagnose happens
    during the hold, which is where `start_step` needs to land to see it.

    The true slope is a central difference of the return under the SAME
    deterministic policy, so it differentiates the quantity the critic is
    fitted to predict rather than a nearby one.

    `episode` takes an already-rolled (batch, window) pair, and `start_step`
    then indexes into it rather than re-simulating those steps; see
    `rolled_episode`.
    """
    if episode is None:
        batch = batches.sample_batch(np.random.default_rng(seed), n_envs)

        # Advance to the phase being measured, so the slope is read where
        # the policy spends its time rather than only at the settled start.
        if start_step:
            with torch.no_grad():
                batch = batch._replace(state=sweep.rollout(batch, actor, start_step, deterministic=True).states[-1])
    else:
        # Step `start_step` of the episode `evaluate` already scored. Same
        # batch, same deterministic policy, so it is the same state the
        # branch above would re-simulate to reach.
        batch, rolled = episode
        batch = batch._replace(state=rolled.states[start_step])

    remaining = task.EPISODE_STEPS - start_step

    up = ramp.uphill(batch.angles)

    # Displace the box `delta` along the ramp, each way, and roll the rest of
    # the episode from both. Along-ramp is the one direction that leaves the
    # contact geometry alone, so the difference isolates position error.
    returns = []
    for sign in (1.0, -1.0):
        state = batch.state.copy()
        state[:, task.BOX, 0:2] += sign * delta * up
        shifted = batch._replace(state=state)
        with torch.no_grad():
            rolled = sweep.rollout(shifted, actor, remaining, deterministic=True)

        # `bootstrap=False` sets the terminal value to zero, which is the
        # real one here -- the episode ends at the end of the roll. It also
        # leaves the target critic unused, so None is safe to pass. Row 0
        # is the return from the perturbed state, which is what the
        # difference below needs.
        returns.append(window_returns(shifted, rolled, None, shaping, gamma=gamma, bootstrap=False)[0].numpy())
    true = (returns[0] - returns[1]) / (2.0 * delta)

    # The critic's own slope at the unperturbed state.
    obs = policy.as_tensor(observation.observe(batch.state, batch.angles, batch.targets), grad=True)
    (dV_dobs,) = torch.autograd.grad(critic(obs).sum(), [obs])

    # Multiply by the observation Jacobian to get dV/dZ, then project the
    # box's two position entries onto uphill -- the same line the
    # perturbation moved along, so both numbers are slopes along one
    # direction and are directly comparable.
    dV_dZ = observation.state_gradient(dV_dobs.detach().numpy(), batch.jacobian)
    predicted = np.einsum("ni,ni->n", dV_dZ[:, task.BOX, 0:2], up)

    error = predicted - true
    return dict(critic=float(predicted.mean()), true=float(true.mean()), bias=float(error.mean()),
                rmse=float(np.sqrt((error ** 2).mean())), correlation=float(np.corrcoef(predicted, true)[0, 1]),
                agreement=float(np.mean(np.sign(predicted) == np.sign(true))))


def train(iterations, n_envs=N_ENVS, window=WINDOW, gamma=GAMMA, max_norm=MAX_GRADIENT_NORM, seed=0, shaping=True, eval_every=100, eval_envs=8, log=print, slope_at=SLOPE_AT_STEP, checkpoint=None, actor_bootstrap=True):
    """Run SHAC. Returns the actor, the critic, and the evaluation log.

    Episodes advance in lockstep: every environment starts together, runs
    `task.EPISODE_STEPS` control steps in windows, and the whole batch is
    resampled at once. The final window of an episode is shortened rather
    than dropped, so the last steps of the hold are still trained on.

    `shaping` applies to the actor's objective and to the critic's targets
    together. They have to agree -- the critic stands in for future reward,
    so it must estimate the same reward the actor is maximizing. Every
    number reported is the UNSHAPED task reward regardless.

    `actor_bootstrap=False` keeps the critic's estimate out of the actor's
    objective, so `dV/d(obs)` never reaches the actor's weights. The critic
    is still fitted, still bootstraps its own targets, and is still
    measured by `calibration` and `slope_calibration` -- it becomes a
    spectator rather than leaving the loop.

    That makes the actor myopic, optimizing one 32-step window with no
    estimate of the 3.7 s after it, which is the critic's whole job. So a
    worse policy is expected on those grounds alone and the two outcomes
    are not symmetric: a run that still decays says the decay does not need
    the critic's slope, while a run that holds cannot separate "a harmful
    gradient was removed" from "a myopic objective is steadier".

    Progress is measured by running full deterministic episodes every
    `eval_every` iterations, NOT by the reward of the training window. A
    window's reward depends on where in the episode it happens to fall:
    the first window of an episode starts with the box up to a metre from
    target and scores about -200, while a window in the middle of the hold
    scores about -1. That makes the training reward useless as a curve --
    it moves with episode phase far more than with policy quality.

    `slope_at` is the episode phase where the critic's slope is read; see
    `slope_calibration`. `checkpoint`, if given, is called at every
    evaluation with (iteration, actor, critic, history) -- enough to save a
    run that might not finish, without deciding here where files go.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    batch = batches.sample_batch(rng, n_envs)
    observation_size = observation.observe(batch.state, batch.angles, batch.targets).shape[-1]

    actor = policy.Actor(observation_size, len(task.ACTUATED), task.LIMIT)
    critic = policy.Critic(observation_size)
    target_critic = copy.deepcopy(critic)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)

    history, steps_done = [], 0
    clipped = 0
    for iteration in range(1, iterations + 1):
        # Shorten rather than overrun, so an episode ends exactly at
        # EPISODE_STEPS. 400 is not a multiple of 32.
        steps = min(window, task.EPISODE_STEPS - steps_done)

        # Flag whether this window runs to the end of the episode. The last
        # one has no future to estimate, so both the actor's objective and
        # the critic's target stop at its end.
        ends_episode = steps_done + steps >= task.EPISODE_STEPS

        policy.zero_gradients(actor)
        rolled = sweep.rollout(batch, actor, steps)
        # Withhold the critic on the last window of an episode, where there
        # is no future to estimate, and on every window when
        # `actor_bootstrap` is off.
        sweep.policy_gradient(batch, rolled, actor, critic=None if ends_episode or not actor_bootstrap else target_critic,
                               shaping=shaping, gamma=gamma)
        gradient_norm = policy.gradient_norm(actor)

        # Step the actor, and count the iterations where the ceiling bound.
        # A count of zero means the clip is inert and a comparison run says
        # nothing; a count near every iteration means it is a learning-rate
        # change wearing a disguise.
        stepped = ascend(actor, actor_optimizer, n_envs, max_norm=max_norm)
        if max_norm is not None and stepped > max_norm:
            clipped += 1

        returns = window_returns(batch, rolled, target_critic, shaping, gamma=gamma, bootstrap=not ends_episode)
        critic_loss = fit_critic(critic, critic_optimizer, rolled.observations, returns)
        soft_update(target_critic, critic)

        steps_done += steps
        if steps_done >= task.EPISODE_STEPS:
            batch, steps_done = batches.sample_batch(rng, n_envs), 0
        else:
            batch = batch._replace(state=rolled.states[-1])

        if eval_every and (iteration % eval_every == 0 or iteration == 1):
            # Roll the scoring episode once and score it twice -- the
            # policy's error and the critic's calibration are two readings
            # of the same trajectory.
            episode = rolled_episode(actor, n_envs=eval_envs)
            reward, error, shaped = evaluate(actor, episode=episode)
            calibrated = calibration(actor, critic, gamma=gamma, episode=episode)

            # Prefixed, because `slope_calibration` reports its own bias,
            # rmse and correlation and those names are already taken by the
            # value calibration above.
            slope = slope_calibration(actor, critic, gamma=gamma, start_step=slope_at, episode=episode)

            record = dict(iteration=iteration, reward=float(reward.mean()), shaped=float(shaped.mean()),
                          error=float(np.abs(error).mean()), worst=float(np.abs(error).max()),
                          critic_loss=critic_loss, gradient_norm=gradient_norm,
                          noise=float(torch.exp(actor.log_std.detach()).mean()), clipped=clipped, **calibrated,
                          **{f"slope_{name}": value for name, value in slope.items()})
            history.append(record)

            # Print two lines per evaluation: the policy, then the critic.
            # The critic line compares V(s_0) against the return the episode
            # actually paid, which is what `critic_loss` cannot report.
            log(f"  {iteration:>6}  reward {record['reward']:>9.2f}  shaped {record['shaped']:>9.2f}"
                f"  err {100 * record['error']:>7.2f} cm  worst {100 * record['worst']:>7.2f} cm"
                f"  |grad| {gradient_norm:>8.1f}  sigma {record['noise']:.3f}")
            log(f"          critic loss {critic_loss:>8.4f}  V(s0) {record['initial']:>9.2f}"
                f"  true {record['initial_true']:>9.2f}  bias {record['bias']:>8.2f}"
                f"  rmse {record['rmse']:>8.2f}  corr {record['correlation']:>6.3f}"
                f"  clipped {record['clipped']:>4}")

            # The critic's slope, which is what the actor's objective
            # actually reads. `dV/dxi` against the true return's slope at
            # the same state, both in reward per metre.
            log(f"          dV/dxi {record['slope_critic']:>9.2f}  true {record['slope_true']:>9.2f}"
                f"  bias {record['slope_bias']:>8.2f}  rmse {record['slope_rmse']:>8.2f}"
                f"  corr {record['slope_correlation']:>6.3f}  sign {record['slope_agreement']:>5.2f}")
            clipped = 0

            if checkpoint is not None:
                checkpoint(iteration, actor, critic, history)

    return actor, critic, history
