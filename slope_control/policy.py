"""The actor and critic, and the code that joins torch's autograd to GRIP's.

Two autodiff systems have to meet. GRIP is numpy and differentiates the
physics; the networks are torch. They meet at the action:

    forward   observation -> [torch] -> action -> [GRIP] -> next state
    backward  dJ/d(state) <- [torch] <- dJ/d(action) <- [GRIP]

GRIP computes dJ/d(action) and torch continues from there. `accumulate` is
where the handover happens, `rollout` runs the forward direction and
`policy_gradient` runs the backward one.

Nothing here knows about ramps, rewards or contact. Those are in `task.py`.
"""

from collections import namedtuple

import numpy as np
import torch

import grip

from . import task

HIDDEN = (64, 64)

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

# Pre-squash noise. The action is limit*tanh(x + sigma*eps), so sigma is
# in the units of the pre-tanh activation rather than newtons; 0.37 puts
# meaningful exploration across the whole range without pinning the sample
# to either end of the squash.
INITIAL_LOG_STD = -1.0


def mlp(inputs, outputs, hidden=HIDDEN, output_gain=1.0):
    """Build a small fully connected network: Linear, ELU, Linear, ELU, Linear.

    `output_gain` bounds the last layer's initial weights, and its bias
    starts at zero, so an untrained network outputs close to nothing. That
    keeps the first few updates from being dominated by the random
    initialization.

    ELU rather than ReLU because ReLU has a kink, so its derivative jumps.
    The critic's gradient feeds into the actor's update here, meaning these
    networks get differentiated through rather than just backpropagated
    once, and a kinked function makes that gradient field discontinuous.
    """
    layers, width = [], inputs
    for size in hidden:
        layers += [torch.nn.Linear(width, size), torch.nn.ELU()]
        width = size
    final = torch.nn.Linear(width, outputs)
    torch.nn.init.uniform_(final.weight, -output_gain, output_gain)
    torch.nn.init.zeros_(final.bias)
    return torch.nn.Sequential(*layers, final)


class Actor(torch.nn.Module):
    """Turn an observation into a force for each actuated body.

        action = limit * tanh(network(obs) + sigma * noise)

    Forces are in the ramp frame, (tangential, normal), in newtons.

    The tanh is what enforces the force limit. A hard clip would work too,
    but its derivative is exactly zero once saturated, and the trajectory
    optimization baseline sits at its limit 2-6% of the time. A policy
    would then spend real time somewhere the gradient says nothing. tanh
    keeps every action inside the limit and every derivative nonzero.

    The noise is added before the tanh, and it is drawn outside the
    network. That means the sampled action is still a differentiable
    function of the network's weights, which is what lets SHAC
    differentiate straight through the sample instead of estimating the
    gradient statistically the way PPO does.
    """

    def __init__(self, observation_size, actuated, limit, hidden=HIDDEN):
        super().__init__()
        self.actuated = actuated
        self.register_buffer("limit", torch.as_tensor(np.asarray(limit), dtype=torch.float64))
        self.net = mlp(observation_size, 2 * actuated, hidden, output_gain=1e-3).double()
        self.log_std = torch.nn.Parameter(torch.full((actuated, 2), INITIAL_LOG_STD, dtype=torch.float64))

    def forward(self, observation, deterministic=False):
        """(environments, features) -> (environments, actuated, 2), in newtons.

        `deterministic=True` skips the noise. Use it whenever two runs have
        to be comparable, such as a finite-difference check.
        """
        pre = self.net(observation).reshape(observation.shape[0], self.actuated, 2)
        if not deterministic:
            pre = pre + torch.exp(self.log_std) * torch.randn_like(pre)
        return self.limit * torch.tanh(pre)


class Critic(torch.nn.Module):
    """Estimate the total reward still to come from a given observation.

    SHAC optimizes short windows -- 32 control steps, 0.32 s -- out of a
    4 s episode. This estimate stands in for everything after the window
    ends. Without it the policy would optimize a third of a second and
    know nothing about the 3.7 s hold that follows, which is exactly where
    penalty contact and a rigid solve differ.
    """

    def __init__(self, observation_size, hidden=HIDDEN):
        super().__init__()
        self.net = mlp(observation_size, 1, hidden, output_gain=1e-2).double()

    def forward(self, observation):
        return self.net(observation).squeeze(-1)


def as_tensor(array, grad=False):
    """numpy -> torch, in float64.

    `grad=True` marks the tensor as something to differentiate with respect
    to, which is how `accumulate` can return dJ/d(observation).

    float64 rather than the usual float32 because the finite-difference
    checks compare against a 1e-6 tolerance, and float32 rounding would
    swamp that.
    """
    tensor = torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float64)
    return tensor.requires_grad_(True) if grad else tensor


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

    # `grad_outputs` IS the handoff between the two autodiff systems. Normally
    # torch seeds a backward pass with 1.0 on a scalar loss; here the scalar
    # lives on the far side of the simulator, so GRIP's dJ/d(action) is handed
    # in as the seed instead. Asking for `parameters + [inputs]` walks the
    # graph once and returns both halves: the parameter gradients, and
    # dJ/d(observation) as the last element. `retain_graph` because the sweep
    # comes back through this graph once per step; `allow_unused` because
    # `log_std` never enters a deterministic pass and comes back as None.
    gradients = torch.autograd.grad(outputs, parameters + [inputs], grad_outputs=as_tensor(seed), retain_graph=True, allow_unused=True)

    # Sum rather than assign: the same parameters produced the action at every
    # step of the window, so the window's total derivative is the sum over
    # steps and the sweep visits each step exactly once.
    for parameter, gradient in zip(parameters, gradients[:-1]):
        if gradient is not None:
            parameter.grad = gradient if parameter.grad is None else parameter.grad + gradient

    # The trailing entry, dJ/d(observation) -- what carries the adjoint back
    # one step through the policy. Detached because GRIP takes it as numpy.
    return gradients[-1].detach().numpy()


def flat_parameters(module):
    """Every parameter flattened into one vector.

    A finite-difference check nudges parameter i and remeasures, so it
    needs the parameters as a single indexable vector rather than a list of
    differently shaped tensors.
    """
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()])


def set_flat_parameters(module, flat):
    """Inverse of `flat_parameters`, in place."""
    offset = 0
    with torch.no_grad():
        for parameter in module.parameters():
            size = parameter.numel()
            parameter.copy_(torch.as_tensor(flat[offset:offset + size]).reshape(parameter.shape))
            offset += size


def flat_gradients(module):
    """Every parameter's `.grad` flattened into one vector, to line up with
    `flat_parameters`.

    A parameter that never entered the computation has `.grad` of None.
    Those become zeros rather than being skipped, so both vectors have the
    same length and index i means the same parameter in each.
    """
    return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).reshape(-1) for p in module.parameters()])


def zero_gradients(module):
    for parameter in module.parameters():
        parameter.grad = None


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
        observation = as_tensor(task.observe(state, batch.angles, batch.targets, batch.variant), grad=True)
        action = actor(observation, deterministic=deterministic)

        # .detach() is the handoff out of torch: GRIP is numpy and knows
        # nothing about the graph. The link is re-established by hand later,
        # when `accumulate` seeds the backward pass with what GRIP computed.
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


def policy_gradient(batch, window, actor, critic=None, shaping=True):
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
    make the objective the window's reward alone. A finite-difference check
    needs that, because it perturbs the weights and remeasures the window's
    reward -- if this function included the critic's contribution and the
    check did not, the two would differ for a legitimate reason and the
    check would prove nothing. Training always passes a critic.
    """
    trajectory, wrenches = window.states, window.wrenches

    # Partials of one step's reward -- dl/dZ_t and dl/dU_t, holding
    # everything else fixed. `adjoint_batch` turns them into total
    # derivatives of the objective, which is the l-to-J step.
    dl_dZ, dl_dU = task.reward_seeds(trajectory, wrenches, batch.angles, batch.targets, batch.variant, shaping=shaping)

    # `adjoint` is the running quantity, dJ/dZ at whichever step the sweep has
    # reached: "how much does the total objective move if the state is nudged
    # right here?" It starts at the far end of the window, where the only
    # contribution is the reward's own derivative at the final state.
    adjoint = dl_dZ[-1].copy()

    if critic is not None:
        # The terminal bootstrap. Everything past the window's end is
        # summarized by V(Z_W), so its derivative joins the seed. The critic
        # reads observations, so dV/d(obs) has to be pulled back to dV/dZ
        # through the observation Jacobian before it can be added.
        terminal = as_tensor(task.observe(trajectory[-1], batch.angles, batch.targets, batch.variant), grad=True)
        value = critic(terminal)
        (dV_dobs,) = torch.autograd.grad(value.sum(), [terminal])
        adjoint = adjoint + task.state_gradient(dV_dobs.detach().numpy(), batch.jacobian)

    # Backwards, one control step at a time. One call per WINDOW would give
    # the open-loop gradient; the difference is the middle term below.
    for t in range(len(wrenches) - 1, -1, -1):
        # Two-state slice, Z_t -> Z_{t+1}. The seed is zero on the first state
        # and `adjoint` on the second, which says: only account for how the
        # objective arrives at the downstream state.
        seed = np.zeros((2,) + trajectory.shape[1:])
        seed[1] = adjoint
        dJ_dZ0, dJ_dU = grip.adjoint_batch(batch.scenes, trajectory[t:t + 2], wrenches[t:t + 1], batch.substeps, seed, dl_dU[t:t + 1])

        # GRIP differentiates world wrenches; the network emits ramp-frame
        # actions. Rotate back, then drop the single-step axis with [0].
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
