"""The actor and critic SHAC trains, and the seam between torch and GRIP.

GRIP is numpy and the networks are torch, so every step of a closed-loop
rollout crosses the boundary twice: the observation goes in as a tensor,
the action comes out as an array, and on the way back GRIP's dJ/d(action)
is handed to `torch.autograd` as the seed of a backward pass. That pass
returns both things the sweep needs at once -- the parameter gradients,
accumulated across the window, and the gradient with respect to the
observation, which carries the adjoint back one step through the policy.

Nothing here knows about ramps, rewards or contact. It is a network and a
boundary.
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
    """A small ELU stack. ELU rather than ReLU because a kinked value
    function is a poor thing to differentiate a policy through, and this
    whole project exists to take derivatives.

    The last layer is scaled down so an untrained policy starts near zero
    output. That used to be a trap -- a pusher not touching the box has no
    gradient at all -- but `task.approach_penalty` now supplies one, so
    starting quiet is safe again and is what keeps the first updates from
    being dominated by whatever the initialization happened to be.
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
    """Observation -> a bounded, stochastic ramp-frame force per actuated body.

        action = limit * tanh(mu(obs) + sigma * eps)

    The tanh IS the action limit, not a clip applied afterwards. A hard
    clip has exactly zero gradient once saturated, and the converged
    baseline sits on its limit 2-6% of the time, so a policy would very
    likely spend real time in a region where it cannot be improved.
    Squashing keeps every action feasible by construction and keeps the
    derivative alive everywhere. `task.clip_action` still runs downstream,
    where it is now a no-op that documents the guarantee.

    Reparameterized, so the sample is differentiable with respect to the
    parameters -- which is the whole point, since SHAC differentiates the
    pathway rather than estimating a score function. There is no
    log-probability anywhere in SHAC, so the tanh needs no change-of-
    variables correction; it is a squash, not a density.
    """

    def __init__(self, observation_size, actuated, limit, hidden=HIDDEN):
        super().__init__()
        self.actuated = actuated
        self.register_buffer("limit", torch.as_tensor(np.asarray(limit), dtype=torch.float64))
        self.net = mlp(observation_size, 2 * actuated, hidden, output_gain=1e-3).double()
        self.log_std = torch.nn.Parameter(torch.full((actuated, 2), INITIAL_LOG_STD, dtype=torch.float64))

    def forward(self, observation, deterministic=False):
        """(environments, features) -> (environments, actuated, 2), in newtons."""
        pre = self.net(observation).reshape(observation.shape[0], self.actuated, 2)
        if not deterministic:
            pre = pre + torch.exp(self.log_std) * torch.randn_like(pre)
        return self.limit * torch.tanh(pre)


class Critic(torch.nn.Module):
    """Observation -> the value of continuing from here.

    Its whole job is to stand in for the reward beyond the end of a short
    window. Without it a 32-step window optimizes 0.32 s of behaviour and
    nothing about the 3.7 s hold that follows, which is where the two
    contact formulations actually differ.
    """

    def __init__(self, observation_size, hidden=HIDDEN):
        super().__init__()
        self.net = mlp(observation_size, 1, hidden, output_gain=1e-2).double()

    def forward(self, observation):
        return self.net(observation).squeeze(-1)


def as_tensor(array, grad=False):
    """numpy -> torch, keeping float64 so the finite-difference checks mean something."""
    tensor = torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float64)
    return tensor.requires_grad_(True) if grad else tensor


def accumulate(module, outputs, seed, inputs):
    """One backward pass, seeded by GRIP, read from both ends.

    `outputs` is what the network produced, `seed` is dJ/d(outputs) as
    computed through the simulator, and `inputs` is the observation tensor.
    Returns dJ/d(inputs) and accumulates dJ/d(parameters) into `.grad`.

    Both come out of a single pass because they are the same pass: the
    parameter gradient is what the update needs, and the input gradient is
    what carries the adjoint back another step through the policy. Doing
    them separately would double the cost for nothing.

    Accumulating rather than assigning is exactly right -- a window's total
    derivative is the sum over its steps, and the sweep visits each once.
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
    """Every parameter as one vector, for finite-difference checking."""
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
    """Every parameter's accumulated gradient as one vector, zeros where absent."""
    return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).reshape(-1) for p in module.parameters()])


def zero_gradients(module):
    for parameter in module.parameters():
        parameter.grad = None


def rollout(batch, actor, steps, deterministic=False):
    """Step the policy forward from `batch.state`, keeping what the sweep needs.

    `step_batch` rather than `rollout_batch` because the control is not
    known in advance -- that is what closed loop means -- and because it
    returns a fresh array per step rather than a view of the simulator's
    buffer, which a trajectory kept across a whole window would otherwise
    have overwritten under it.

    Takes a whole `task.Batch` rather than the six loose pieces it needs.
    The scenes, the slopes and the variant have to describe the same world
    or the rollout is quietly simulating something else, and handing them
    over as one value is what makes that unexpressible.

    Windows chain by advancing the batch: `batch = batch._replace(state =
    window.states[-1])`, which is how SHAC carries a state across a window
    boundary without resetting the episode.
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
    """dJ/d(actor parameters) for a closed-loop window, accumulated into .grad.

    The per-step sweep, lifted out of `tests/check_closed_loop.py` now that
    SHAC is its second consumer. One `adjoint_batch` call per window gives
    the OPEN-loop gradient, which is right for a fixed control sequence and
    measured at 119% wrong for a policy at one gain and 11% at another --
    state-dependent, so not something a learning rate can absorb.

    Each step's call returns both pieces the sweep needs: dJ_dU_t to seed
    the network's backward pass, and dJ_dZ0 to carry the adjoint back one
    step. Between calls it adds the path a single call cannot see,
    Z_t -> a_t -> Z_{t+1}, which is the network's input gradient projected
    through the (constant) observation Jacobian.

    Deliberately NOT in `task.py`, though the plan said to put it there.
    It needs torch, and task.py is the only thing `drift.py` and the
    numpy-side checks depend on -- keeping it torch-free is worth more
    than filing this by its original address.

    `critic` supplies the terminal bootstrap, dV/dZ_W. Passing None makes
    the objective the windowed reward alone, which is what a
    finite-difference check needs, since the check has to differentiate
    exactly the quantity it perturbs.

    Takes the `task.Batch` and the `Window` whole. The two used to arrive
    as eleven loose positional arguments, six of which were also `rollout`
    arguments, and a swapped pair among them raises nothing -- it just
    differentiates a different problem.
    """
    trajectory, wrenches = window.states, window.wrenches

    # Partials of the stage cost only -- dr/dZ_t and dr/dU_t at each step,
    # holding everything else fixed. GRIP turns them into total derivatives.
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
        #   dl_dZ[t]          the reward's own dependence on Z_t
        # The middle term is what a single whole-window call cannot see: it
        # treats the controls as fixed inputs, which is right for a trajectory
        # optimizer and wrong for anything that reacts to the state.
        adjoint = dJ_dZ0 + task.state_gradient(dJ_dobs, batch.jacobian) + dl_dZ[t]

    return adjoint
