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
reward -- `evaluate` sees no gamma at all. The discount is there for two
reasons:

    At gamma = 1 the Bellman operator is not a contraction, so the
    bootstrap has no unique fixed point to converge to. Undiscounted, the
    critic settled at -53.87 against episode returns near -180.

    Under gamma = 1 the value of a state depends on how many steps remain,
    and `observe` carries no clock. The critic would have to fit "-180
    early in the episode" and "-0.5 late in it" with no way to tell the
    two apart.

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
# Do not size it for physical plausibility. A step inside POSITION_TOLERANCE
# sounds reasonable and is 50 times too large: penalty contact is stiff and
# Coulomb friction has no gentle regime, so the return is only linear over a
# very short distance.
SLOPE_DELTA = 1.0e-4

# The evaluation batch: a seed the training loop never draws from, and few
# enough environments that scoring every 100 iterations stays cheap.
EVAL_SEED = 1000
EVAL_ENVS = 8


def discounted_returns(rewards, gamma, terminal=0.0):
    """The discounted return from every step to the end of `rewards`.

    (steps, environments) -> (steps + 1, environments), numpy. The last row
    is `terminal`, the value of whatever follows the final step: zero where
    the episode really ends, a critic's estimate where it does not.
    """
    returns = np.zeros((len(rewards) + 1,) + np.shape(rewards)[1:])
    returns[-1] = terminal

    # Walk backwards: the value of being at step t is this step's reward plus
    # the discounted value of where it lands. rewards[t] is the reward for
    # the transition out of state t, which is why the indices line up.
    for t in range(len(rewards) - 1, -1, -1):
        returns[t] = rewards[t] + gamma * returns[t + 1]
    return returns


def window_returns(batch, window, target_critic, shaping, gamma=GAMMA, bootstrap=True):
    """The return from each state in the window: discounted reward to the end
    of the window, plus the target critic's estimate of everything after it.

    Returns a (steps + 1, environments) tensor, lined up with
    `window.states`.

    `bootstrap=False` for the window that ends an episode. There is no
    "after" there, so the correct terminal value is zero. Bootstrapping
    anyway teaches the critic that reward keeps arriving past step 400,
    which is a bias with nothing to correct it.

    The bootstrap comes from the TARGET critic rather than the critic being
    trained, so the regression target does not move every time the critic
    does.
    """
    rewards = objective.reward(window.states, window.wrenches, batch.angles, batch.targets, shaping=shaping)

    terminal = 0.0
    if bootstrap:
        # Observe the window's final state and score it with the target
        # critic. Under no_grad because this is a regression target, and a
        # target has to be a constant -- otherwise fitting the critic would
        # push gradients into the target network, which is the one thing it
        # exists to avoid.
        with torch.no_grad():
            observed = policy.as_tensor(observation.observe(window.states[-1], batch.angles, batch.targets))
            terminal = target_critic(observed).numpy()

    return policy.as_tensor(discounted_returns(rewards, gamma, terminal))


def fit_critic(critic, optimizer, observations, returns, epochs=CRITIC_EPOCHS):
    """Regress the critic onto the window's returns. Returns the final loss.

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
    """
    with torch.no_grad():
        for target_parameter, parameter in zip(target.parameters(), source.parameters()):
            target_parameter.mul_(1.0 - tau).add_(parameter, alpha=tau)
        for target_buffer, buffer in zip(target.buffers(), source.buffers()):
            target_buffer.mul_(1.0 - tau).add_(buffer, alpha=tau)


def ascend(module, optimizer, n_envs):
    """Take one Adam step UPHILL on the accumulated gradient.

    `policy_gradient` accumulates dJ/d(weights) where J is a reward to be
    maximized, and torch optimizers minimize, so every gradient is negated
    first. This is the sign convention `task.py` documents, and this is the
    one place it has to be acted on.

    Also divides by the environment count, so the step size means the same
    thing whether the batch holds 8 environments or 64.
    """
    for parameter in module.parameters():
        if parameter.grad is not None:
            parameter.grad.neg_().div_(n_envs)
    optimizer.step()


def rolled_episode(actor, batch=None, n_envs=EVAL_ENVS, seed=EVAL_SEED):
    """Roll one deterministic episode. Returns (batch, window), the
    `episode` that `evaluate`, `calibration` and `slope_calibration` take.

    Builds the evaluation batch from `seed` unless a `batch` is given. The
    two halves travel together because scoring a window against a
    different batch's targets is not a thing that should be expressible.
    """
    if batch is None:
        batch = batches.sample_batch(np.random.default_rng(seed), n_envs)

    # Roll with the noise off and no autograd graph. `rollout` normally
    # keeps 400 live graphs so the sweep can walk back through them;
    # nothing that consumes this differentiates, so building them would be
    # 400 steps of bookkeeping thrown away.
    with torch.no_grad():
        return batch, sweep.rollout(batch, actor, task.EPISODE_STEPS, deterministic=True)


def evaluate(episode):
    """Score a deterministic episode. Returns the per-environment reward,
    the same reward with the shaping term, and the final position error,
    signed.

    The unshaped reward is the one every reported number uses. The shaped
    one is what training maximizes.
    """
    batch, rolled = episode
    reward = objective.reward(rolled.states, rolled.wrenches, batch.angles, batch.targets).sum(axis=0)
    shaped = objective.reward(rolled.states, rolled.wrenches, batch.angles, batch.targets, shaping=True).sum(axis=0)
    final = ramp.along_ramp(rolled.states[-1], batch.angles)[:, task.BOX] - batch.targets
    return reward, shaped, final


def calibration(critic, episode, gamma=GAMMA, shaping=True):
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

    """
    batch, rolled = episode

    # The discounted return to go, with a terminal value of zero because the
    # episode really ends here. This is what the training bootstrap stands
    # in for.
    rewards = objective.reward(rolled.states, rolled.wrenches, batch.angles, batch.targets, shaping=shaping)
    true = discounted_returns(rewards, gamma)
    steps, n_envs = rewards.shape

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


def slope_calibration(actor, critic, episode, gamma=GAMMA, shaping=True, delta=SLOPE_DELTA, start_step=0):
    """Measure the critic's SLOPE against the slope of the true return.

    Returns a dict: `critic` and `true`, the two mean slopes in reward per
    metre; `bias`, `rmse` and `correlation` between them; and `agreement`,
    the fraction of environments where the two carry the same sign.

    Pass the target critic: it is the network `sweep.policy_gradient`
    differentiates, so it is the slope the actor actually receives.

    The slope is read at step `start_step` of `episode`, a state the policy
    actually reaches. The true slope is a central difference of the return
    from there, under the same deterministic policy.

    `calibration` asks whether V is the right number; this asks whether it
    points the right way, which is what the actor depends on. What enters
    the actor's objective is dV/d(obs) and never V, and a network can fit
    values to 0.97 correlation and still have noisy local derivatives.

    The box is displaced along the ramp, which keeps its height above the
    surface, so the two rollouts differ in position error and nothing else.
    `SLOPE_DELTA` sets how far, and records why it is that small.
    """
    # Restart the episode's batch from its step `start_step`.
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

        # Keep row 0, the return from the perturbed state. The terminal value
        # is zero because the roll ends where the episode does.
        rewards = objective.reward(rolled.states, rolled.wrenches, shifted.angles, shifted.targets, shaping=shaping)
        returns.append(discounted_returns(rewards, gamma)[0])
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


def evaluation(actor, critic, target_critic, gamma=GAMMA, slope_at=SLOPE_AT_STEP, n_envs=EVAL_ENVS):
    """Score the policy and the critic on one deterministic episode.

    The value is measured on `critic`, which is what the fit produces. The
    slope is measured on `target_critic`, which is what the actor's
    bootstrap reads.

    Returns a flat dict of floats: the policy's reward, shaped reward, mean
    and worst final error and exploration noise; `calibration`'s entries;
    and `slope_calibration`'s entries prefixed `slope_`, since the two
    share the names bias, rmse and correlation.
    """
    episode = rolled_episode(actor, n_envs=n_envs)
    reward, shaped, error = evaluate(episode)
    calibrated = calibration(critic, episode, gamma=gamma)
    slope = slope_calibration(actor, target_critic, episode, gamma=gamma, start_step=slope_at)

    return dict(reward=float(reward.mean()), shaped=float(shaped.mean()), error=float(np.abs(error).mean()),
                worst=float(np.abs(error).max()), noise=float(torch.exp(actor.log_std.detach()).mean()),
                **calibrated, **{f"slope_{name}": value for name, value in slope.items()})


def log_evaluation(record, log):
    """Print one evaluation as three lines: the policy, the critic's value,
    and the critic's slope.

    The value line compares V(s_0) against the return the episode actually
    paid, which `critic_loss` cannot report. The slope line compares dV/dxi
    against the true return's slope at the same state, both in reward per
    metre.
    """
    log(f"  {record['iteration']:>6}  reward {record['reward']:>9.2f}  shaped {record['shaped']:>9.2f}"
        f"  err {100 * record['error']:>7.2f} cm  worst {100 * record['worst']:>7.2f} cm"
        f"  |grad| {record['gradient_norm']:>8.1f}  sigma {record['noise']:.3f}")
    log(f"          critic loss {record['critic_loss']:>8.4f}  V(s0) {record['initial']:>9.2f}"
        f"  true {record['initial_true']:>9.2f}  bias {record['bias']:>8.2f}"
        f"  rmse {record['rmse']:>8.2f}  corr {record['correlation']:>6.3f}")
    log(f"          dV/dxi {record['slope_critic']:>9.2f}  true {record['slope_true']:>9.2f}"
        f"  bias {record['slope_bias']:>8.2f}  rmse {record['slope_rmse']:>8.2f}"
        f"  corr {record['slope_correlation']:>6.3f}  sign {record['slope_agreement']:>5.2f}")


def train(iterations, n_envs=N_ENVS, window=WINDOW, gamma=GAMMA, seed=0, shaping=True, eval_every=100, eval_envs=EVAL_ENVS, log=print, slope_at=SLOPE_AT_STEP, checkpoint=None, trace=None):
    """Run SHAC. Returns the actor, the critic, and the evaluation log.

    Episodes advance in lockstep: every environment starts together, runs
    `task.EPISODE_STEPS` control steps in windows, and the whole batch is
    resampled at once. The final window of an episode is shortened rather
    than dropped, so the last steps of the hold are still trained on.

    `shaping` applies to the actor's objective and to the critic's targets
    together. They have to agree -- the critic stands in for future reward,
    so it must estimate the same reward the actor is maximizing. Every
    number reported is the UNSHAPED task reward regardless.

    Progress is measured by running full deterministic episodes every
    `eval_every` iterations, NOT by the reward of the training window. A
    window's reward depends on where in the episode it happens to fall:
    the first window of an episode starts with the box up to a metre from
    target and scores about -200, while a window in the middle of the hold
    scores about -1. That makes the training reward useless as a curve --
    it moves with episode phase far more than with policy quality.

    `slope_at` is the episode phase where the critic's slope is read; see
    `slope_calibration`. `checkpoint`, if given, is called at every
    evaluation with (iteration, actor, critic, target_critic, history) --
    enough to save a run that might not finish, without deciding here where
    files go. The target critic is included because it, not the critic, is
    what the actor's bootstrap reads.

    `trace`, if given, is called every iteration with (iteration, window
    index within the episode, window objective, gradient norm). The window
    objective is the mean over environments of J at the window's first
    state: shaped, discounted, and bootstrapped from the target critic,
    which is the quantity the actor step climbs.
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
        # is no future to estimate.
        sweep.policy_gradient(batch, rolled, actor, critic=None if ends_episode else target_critic, shaping=shaping, gamma=gamma)
        gradient_norm = policy.gradient_norm(actor)

        ascend(actor, actor_optimizer, n_envs)

        returns = window_returns(batch, rolled, target_critic, shaping, gamma=gamma, bootstrap=not ends_episode)
        critic_loss = fit_critic(critic, critic_optimizer, rolled.observations, returns)
        soft_update(target_critic, critic)

        if trace is not None:
            # Report the window's objective, averaged over environments.
            # returns[0] was computed on the same rollout the actor's
            # gradient came from, and before the soft update, so it is the
            # objective the actor step just climbed.
            trace(iteration, steps_done // window, float(returns[0].mean()), gradient_norm)

        steps_done += steps
        if steps_done >= task.EPISODE_STEPS:
            batch, steps_done = batches.sample_batch(rng, n_envs), 0
        else:
            batch = batch._replace(state=rolled.states[-1])

        if eval_every and (iteration % eval_every == 0 or iteration == 1):
            record = dict(iteration=iteration, critic_loss=critic_loss, gradient_norm=gradient_norm,
                          **evaluation(actor, critic, target_critic, gamma=gamma, slope_at=slope_at, n_envs=eval_envs))
            history.append(record)
            log_evaluation(record, log)

            if checkpoint is not None:
                checkpoint(iteration, actor, critic, target_critic, history)

    return actor, critic, history
