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
`policy.policy_gradient` carries that through the network. No sampling, no
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

Both of those stand on their own, which is why the discount stays. What it
did NOT do is fix training. Two pushers peaked at 2.37 cm on iteration 500
and then degraded monotonically to 13.12 cm by 3500 -- earlier and more
steadily than the undiscounted runs it replaced, which peaked at the same
iteration 500 and ended near 9-11 cm. Every run so far, both variants,
peaks early and decays. That is open and the cause is not identified, so
do not read the discount as the remedy for it.

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

from . import policy, ramp, task

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
    rewards = task.reward(window.states, window.wrenches, batch.angles, batch.targets, batch.variant, shaping=shaping)
    steps, n_envs = rewards.shape

    returns = torch.zeros((steps + 1, n_envs), dtype=torch.float64)
    if bootstrap:
        # No graph. This is a regression target, and a target has to be a
        # constant -- otherwise fitting the critic would push gradients into
        # the target network, which is the one thing it exists to avoid.
        with torch.no_grad():
            terminal = policy.as_tensor(task.observe(window.states[-1], batch.angles, batch.targets, batch.variant))
            returns[-1] = target_critic(terminal)

    # Walk backwards: the value of being at step t is this step's reward plus
    # the discounted value of where it lands. rewards[t] is the reward for
    # the transition out of state t, which is why the indices line up.
    rewards = policy.as_tensor(rewards)
    for t in range(steps - 1, -1, -1):
        returns[t] = rewards[t] + gamma * returns[t + 1]
    return returns


def fit_critic(critic, optimizer, observations, returns, epochs=CRITIC_EPOCHS):
    """Regress the critic onto the window's returns. Returns the final loss.

    Fitted in normalized units, so the statistics are updated first and
    both sides of the loss are then in the same scale.
    """
    critic.update_statistics(returns)

    # One flat batch of (state, return) pairs. The graph is dropped -- the
    # critic is fitted to numbers, and letting the actor's graph reach into
    # the critic's loss would tie two independent updates together.
    features = torch.cat([observation.detach() for observation in observations], dim=0)
    targets = critic.normalize(returns[:-1].reshape(-1)).detach()

    # .item() rather than float(), which would warn about converting a
    # tensor that still carries a graph.
    final_loss = float("nan")
    for _ in range(epochs):
        optimizer.zero_grad()
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


def evaluate(actor, variant, batch=None, n_envs=8, seed=1000, shaping=False):
    """Run full episodes with the noise switched off and score them.

    Returns the per-environment unshaped reward and the final position
    error, signed.

    Deterministic, and by default on a seed the training loop never saw,
    so this measures the policy rather than the exploration it happened to
    do. Pass a `batch` to score a specific configuration instead --
    `check_trajopt`'s, for one, which is the only way its numbers and
    these are answers to the same question.
    """
    if batch is None:
        batch = task.sample_batch(np.random.default_rng(seed), n_envs, variant)

    # No graph. `rollout` normally keeps 400 live autograd graphs so the
    # sweep can walk back through them; nothing here differentiates, so
    # building them would be 400 steps of bookkeeping thrown away.
    with torch.no_grad():
        rolled = policy.rollout(batch, actor, task.EPISODE_STEPS, deterministic=True)

    reward = task.reward(rolled.states, rolled.wrenches, batch.angles, batch.targets, variant, shaping=shaping).sum(axis=0)

    final = ramp.along_ramp(rolled.states[-1], batch.angles)[:, variant.box] - batch.targets
    return reward, final


def train(variant, iterations, n_envs=N_ENVS, window=WINDOW, gamma=GAMMA, seed=0, shaping=True, eval_every=100, eval_envs=8, log=print):
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
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    batch = task.sample_batch(rng, n_envs, variant)
    observation_size = task.observe(batch.state, batch.angles, batch.targets, variant).shape[-1]

    actor = policy.Actor(observation_size, len(variant.actuated), variant.limit)
    critic = policy.Critic(observation_size)
    target_critic = copy.deepcopy(critic)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)

    history, steps_done = [], 0
    for iteration in range(1, iterations + 1):
        # Shorten rather than overrun, so an episode ends exactly at
        # EPISODE_STEPS. 400 is not a multiple of 32.
        steps = min(window, task.EPISODE_STEPS - steps_done)

        # The last window of an episode has no future to estimate, so both
        # the actor's objective and the critic's target stop at its end.
        ends_episode = steps_done + steps >= task.EPISODE_STEPS

        policy.zero_gradients(actor)
        rolled = policy.rollout(batch, actor, steps)
        policy.policy_gradient(batch, rolled, actor, critic=None if ends_episode else target_critic,
                               shaping=shaping, gamma=gamma)
        gradient_norm = float(policy.flat_gradients(actor).norm())
        ascend(actor, actor_optimizer, n_envs)

        returns = window_returns(batch, rolled, target_critic, shaping, gamma=gamma, bootstrap=not ends_episode)
        critic_loss = fit_critic(critic, critic_optimizer, rolled.observations, returns)
        soft_update(target_critic, critic)

        steps_done += steps
        if steps_done >= task.EPISODE_STEPS:
            batch, steps_done = task.sample_batch(rng, n_envs, variant), 0
        else:
            batch = batch._replace(state=rolled.states[-1])

        if eval_every and (iteration % eval_every == 0 or iteration == 1):
            reward, error = evaluate(actor, variant, n_envs=eval_envs)
            record = dict(iteration=iteration, reward=float(reward.mean()),
                          error=float(np.abs(error).mean()), worst=float(np.abs(error).max()),
                          critic_loss=critic_loss, gradient_norm=gradient_norm,
                          noise=float(torch.exp(actor.log_std.detach()).mean()))
            history.append(record)
            log(f"  {iteration:>6}  reward {record['reward']:>9.2f}  err {100 * record['error']:>7.2f} cm"
                f"  worst {100 * record['worst']:>7.2f} cm  critic {critic_loss:>8.4f}"
                f"  |grad| {gradient_norm:>8.1f}  sigma {record['noise']:.3f}")

    return actor, critic, history
