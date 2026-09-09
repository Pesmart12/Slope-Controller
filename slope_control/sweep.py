"""The closed-loop gradient: run the policy, then differentiate back through it.

Two autodiff systems have to meet. GRIP is numpy and differentiates the
physics; the networks are torch. They meet at the action:

    forward   observation -> [torch] -> action -> [GRIP] -> next state
    backward  dJ/d(state) <- [torch] <- dJ/d(action) <- [GRIP]

GRIP computes dJ/d(action) and torch continues from there. `rollout` runs
the forward direction, `accumulate` is where the handover happens, and
`policy_gradient` runs the backward one. `Window` is what the first hands
the third.

The sweep is PER STEP, not one call per window. One call per window gives
the open-loop gradient, which is right for a fixed control sequence and
119% wrong for a policy -- `tests/check_closed_loop.py` measures that and
says why.

Split out of `policy.py`, which now holds only what a policy IS. This file
holds what running and differentiating one takes, and it is what
`tests/check_policy_gradient.py` exercises.
"""

from collections import namedtuple

import numpy as np
import torch

import grip

from . import policy, task


# What one closed-loop rollout leaves behind, which is exactly what the
# backward sweep consumes. `observations` and `actions` are live torch
# tensors with their graph intact, NOT logs -- they are the only reason
# the sweep can push an adjoint back through the policy rather than
# treating each action as an independent variable.
#
# states        (steps + 1, environments, bodies, 6), one per control step plus the start
# wrenches      (steps, environments, bodies, 3), what GRIP was actually given
# observations  per step, requires_grad, the leaf the input gradient comes back on
# actions       per step, the network's output, seeded by GRIP on the way back
Window = namedtuple("Window", "states wrenches observations actions")

def accumulate(module, outputs, seed, inputs):
    """Back-propagate one step through the network, starting from GRIP's seed.

        module   the network to differentiate
        outputs  what it produced -- the action tensor, graph attached
        seed     dJ/d(outputs), which GRIP computed
        inputs   the observation tensor it was given

    Returns dJ/d(inputs). Adds dJ/d(parameters) into each parameter's
    `.grad` as a side effect, so the two results leave by different doors.

    Usually you call `loss.backward()` and torch seeds the pass with 1.0 on
    a scalar. Here the objective is on the far side of the simulator, so
    the seed has to be supplied by hand. `outputs` says where in the graph
    to start; `seed` says what the derivative is there.

    Both results come from one pass over the graph because they are the
    same pass. They are then used differently:

      The parameter gradients are summed, not assigned. The same weights
      produced the action at every step in the window, so the window's
      total derivative is a sum over steps.

      The input gradient belongs to this step alone and is returned
      immediately. `policy_gradient` multiplies it by the observation
      Jacobian to get dJ/d(state), which is how the adjoint travels back
      through the policy to the previous step.
    """
    parameters = [p for p in module.parameters() if p.requires_grad]

    # Differentiate the actions with respect to every parameter and the
    # observation in one walk of the graph, returning the parameter gradients
    # followed by dJ/d(observation) as the last element.
    #
    # `grad_outputs` is the handoff between the two autodiff systems. Torch
    # normally seeds a backward pass with 1.0 on a scalar loss; here the
    # scalar lives on the far side of the simulator, so GRIP's dJ/d(action)
    # goes in as the seed instead. `retain_graph` because the sweep comes
    # back through this graph once per step; `allow_unused` because
    # `log_std` never enters a deterministic pass and comes back as None.
    gradients = torch.autograd.grad(outputs, parameters + [inputs], grad_outputs=policy.as_tensor(seed), retain_graph=True, allow_unused=True)

    # Sum rather than assign: the same parameters produced the action at every
    # step of the window, so the window's total derivative is the sum over
    # steps and the sweep visits each step exactly once.
    for parameter, gradient in zip(parameters, gradients[:-1]):
        if gradient is not None:
            parameter.grad = gradient if parameter.grad is None else parameter.grad + gradient

    # The trailing entry, dJ/d(observation) -- what carries the adjoint back
    # one step through the policy. Detached because GRIP takes it as numpy.
    return gradients[-1].detach().numpy()


def rollout(batch, actor, steps, deterministic=False):
    """Run the policy for `steps` control steps and return everything the
    backward sweep will need.

    Each step: look at the state, decide a force, apply it, repeat. All the
    environments in the batch advance together; only time is sequential.

    Returns a `Window`. `states` has one more entry than the rest, because
    it includes the state the rollout started from.

    Uses `step_batch`, not `rollout_batch`, for two reasons. The controls
    are not known ahead of time -- the action at step 5 depends on the
    state at step 5 -- which is what closed loop means. And `rollout_batch`
    returns a view of GRIP's internal buffer, which the next call would
    overwrite; `step_batch` returns a fresh array each time.

    To continue past the end of a window without resetting the episode:

        batch = batch._replace(state=window.states[-1])
    """
    state = batch.state

    # `states` starts with the initial state, so it ends up one longer than
    # the other three -- the same (steps + 1) convention `rollout_batch` uses.
    states, observations, actions, wrenches = [state], [], [], []

    for _ in range(steps):
        # grad=True marks the observation as a leaf, which is what lets
        # `accumulate` hand dJ/d(observation) back on the return trip.
        observation = policy.as_tensor(task.observe(state, batch.angles, batch.targets, batch.variant), grad=True)
        action = actor(observation, deterministic=deterministic)

        # Rotate the action into a world wrench as numpy, cutting it off the
        # graph on the way out. GRIP knows nothing about torch, so the link
        # is re-established by hand later, when `accumulate` seeds the
        # backward pass with what GRIP computed.
        wrench = task.to_wrench(action.detach().numpy(), batch.angles, batch.variant)

        # observations and actions are kept as LIVE tensors with their graph
        # attached. Storing numpy copies would make each action an independent
        # variable with no path back to theta, which is precisely the
        # open-loop gradient -- the one measured at 119% wrong.
        observations.append(observation)
        actions.append(action)
        wrenches.append(wrench)

        # wrench[None] adds the single-step axis `step_batch` expects.
        state = grip.step_batch(batch.scenes, state, wrench[None], substeps=batch.substeps)
        states.append(state)

    return Window(np.array(states), np.array(wrenches), observations, actions)


def policy_gradient(batch, window, actor, critic=None, shaping=True, gamma=1.0):
    """Differentiate a window's reward with respect to the actor's weights.

    Adds the result into each parameter's `.grad`, ready for an optimizer
    step. Returns dJ/d(state) at the start of the window, which callers can
    usually ignore.

    Walks backwards one CONTROL step at a time, 32 calls to `adjoint_batch`
    for a 32-step window, each covering that step's 20 integration
    substeps. One call for the whole window would cover the same 640
    substeps and cost the same simulation, so the price here is Python
    round trips, not physics.

    The reason for the round trips is what happens between the calls. A
    single call treats the controls as fixed inputs, which is correct for
    an optimizer tuning a control sequence and wrong for a policy that
    reacts to the state. Measured against finite differences, one call is
    11% off at one feedback gain and 119% off at another. Being
    state-dependent, that error is not something a learning rate absorbs.

    At each step the state can reach the objective three ways, and the code
    adds all three:

        through the physics      Z_t -> Z_{t+1}, force held fixed
        through the policy       Z_t -> obs_t -> a_t -> Z_{t+1}
        through the reward       r(Z_t) directly

    `adjoint_batch` supplies the first and the third. The middle one has to
    be built here, because GRIP's contract is that controls come from
    outside and it has no idea the action was computed from the state:

        1. `to_action_gradient` rotates GRIP's dJ/d(wrench) into
           dJ/d(action), the ramp-frame force the network emits.
        2. `accumulate` back-propagates that through the network, banking
           dJ/d(weights) and returning dJ/d(observation).
        3. `state_gradient` multiplies dJ/d(observation) by the observation
           Jacobian to get dJ/d(state).

    `critic`, when given, estimates the reward after the window ends and
    its derivative joins the seed at the window's final state. Pass None to
    make the objective the window's reward alone. Two callers need that: a
    finite-difference check, which has to differentiate exactly what it
    perturbs, and the last window of an episode, where there is genuinely
    no future to estimate.

    `gamma` discounts. The objective is

        sum over t of gamma^t * reward[t]  +  gamma^steps * V(final state)

    applied by scaling the seeds before the sweep rather than inside it.
    The backward recursion then carries the weights on its own, because
    each reward's contribution enters through its own seed. Note the two
    exponents differ by one: `dl_dU[t]` differentiates reward[t], but
    `dl_dZ[s]` differentiates reward[s-1], since control u_t is scored
    against the state it produces.
    """
    trajectory, wrenches = window.states, window.wrenches
    steps = len(wrenches)

    # Partials of one step's reward -- dl/dZ_t and dl/dU_t, holding
    # everything else fixed. `adjoint_batch` turns them into total
    # derivatives of the objective, which is the l-to-J step.
    dl_dZ, dl_dU = task.reward_seeds(trajectory, wrenches, batch.angles, batch.targets, batch.variant, shaping=shaping)

    if gamma != 1.0:
        # Multiply seed t by gamma^t, broadcasting the weight across the
        # environment, body and component axes. Discounting the seeds this
        # way discounts the objective and leaves the sweep below untouched.
        weights = gamma ** np.arange(steps)
        dl_dU = dl_dU * weights[:, None, None, None]

        # Apply the same weights to dl_dZ, shifted one row later, because
        # dl_dZ[s] differentiates reward[s-1] rather than reward[s]. Row 0 is
        # skipped and stays zero -- no control reaches Z_0.
        dl_dZ = dl_dZ.copy()
        dl_dZ[1:] *= weights[:, None, None, None]

    # `adjoint` is the running quantity, dJ/dZ at whichever step the sweep has
    # reached: "how much does the total objective move if the state is nudged
    # right here?" It starts at the far end of the window, where the only
    # contribution is the reward's own derivative at the final state.
    adjoint = dl_dZ[-1].copy()

    if critic is not None:
        # Observe the window's final state, score it with the critic, and
        # take dV/d(observation). This is the terminal bootstrap: everything
        # past the window's end is summarized by V(Z_W), so its derivative
        # joins the seed.
        terminal = policy.as_tensor(task.observe(trajectory[-1], batch.angles, batch.targets, batch.variant), grad=True)
        value = critic(terminal)
        (dV_dobs,) = torch.autograd.grad(value.sum(), [terminal])

        # Multiply the critic's observation gradient by the observation
        # Jacobian to get dV/dZ, weight it by gamma^steps, and add it to the
        # seed. gamma^steps because V estimates reward starting one step past
        # the window's last reward.
        adjoint = adjoint + gamma ** steps * task.state_gradient(dV_dobs.detach().numpy(), batch.jacobian)

    # Backwards, one control step at a time. One call per WINDOW would give
    # the open-loop gradient; the difference is the middle term below.
    for t in range(len(wrenches) - 1, -1, -1):
        # Two-state slice, Z_t -> Z_{t+1}. The seed is zero on the first state
        # and `adjoint` on the second, which says: only account for how the
        # objective arrives at the downstream state.
        seed = np.zeros((2,) + trajectory.shape[1:])
        seed[1] = adjoint
        dJ_dZ0, dJ_dU = grip.adjoint_batch(batch.scenes, trajectory[t:t + 2], wrenches[t:t + 1], batch.substeps, seed, dl_dU[t:t + 1])

        # Rotate the wrench gradient back into the ramp frame and drop the
        # single-step axis with [0]. GRIP differentiates world wrenches; the
        # network emits ramp-frame actions.
        action_gradient = task.to_action_gradient(dJ_dU, batch.angles, batch.variant)[0]

        # Push that through the network: banks dJ/d(theta) for this step and
        # returns dJ/d(observation) for the term below.
        dJ_dobs = accumulate(actor, window.actions[t], action_gradient, window.observations[t])

        # The three paths by which Z_t reaches the objective, summed:
        #   dJ_dZ0            Z_t -> Z_{t+1} through physics, force held fixed
        #   state_gradient(.) Z_t -> obs_t -> a_t -> Z_{t+1}, THROUGH THE POLICY
        #   dl_dZ[t]          this step's own reward, dl/dZ_t
        # The middle term is what a single whole-window call cannot see: it
        # treats the controls as fixed inputs, which is right for a trajectory
        # optimizer and wrong for anything that reacts to the state.
        adjoint = dJ_dZ0 + task.state_gradient(dJ_dobs, batch.jacobian) + dl_dZ[t]

    return adjoint
