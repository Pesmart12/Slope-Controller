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

import numpy as np
import torch

import grip

from . import task

HIDDEN = (64, 64)

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
    gradients = torch.autograd.grad(outputs, parameters + [inputs], grad_outputs=as_tensor(seed), retain_graph=True, allow_unused=True)
    for parameter, gradient in zip(parameters, gradients[:-1]):
        if gradient is not None:
            parameter.grad = gradient if parameter.grad is None else parameter.grad + gradient
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


def rollout(scenes, state, ramp_angles, targets, actor, variant, substeps, steps, deterministic=False):
    """Step the policy forward, keeping what the backward sweep will need.

    `step_batch` rather than `rollout_batch` because the control is not
    known in advance -- that is what closed loop means -- and because it
    returns a fresh array per step rather than a view of the simulator's
    buffer, which a trajectory kept across a whole window would otherwise
    have overwritten under it.

    The observation tensors are kept with their graph intact. They are the
    only reason the sweep can push an adjoint back through the policy
    rather than treating each action as an independent variable.
    """
    states, observations, actions, wrenches = [state], [], [], []
    for _ in range(steps):
        observation = as_tensor(task.observe(state, ramp_angles, targets, variant), grad=True)
        action = actor(observation, deterministic=deterministic)
        wrench = task.to_wrench(action.detach().numpy(), ramp_angles, variant)

        observations.append(observation)
        actions.append(action)
        wrenches.append(wrench)
        state = grip.step_batch(scenes, state, wrench[None], substeps=substeps)
        states.append(state)

    return np.array(states), np.array(wrenches), observations, actions


def policy_gradient(scenes, trajectory, wrenches, observations, actions, ramp_angles, targets, actor, variant, substeps, jacobian, critic=None, shaping=True):
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
    """
    dl_dZ, dl_dU = task.reward_seeds(trajectory, wrenches, ramp_angles, targets, variant, shaping=shaping)

    adjoint = dl_dZ[-1].copy()
    if critic is not None:
        terminal = as_tensor(task.observe(trajectory[-1], ramp_angles, targets, variant), grad=True)
        value = critic(terminal)
        (dV_dobs,) = torch.autograd.grad(value.sum(), [terminal])
        adjoint = adjoint + task.state_gradient(dV_dobs.detach().numpy(), jacobian)

    for t in range(len(wrenches) - 1, -1, -1):
        seed = np.zeros((2,) + trajectory.shape[1:])
        seed[1] = adjoint
        dJ_dZ0, dJ_dU = grip.adjoint_batch(scenes, trajectory[t:t + 2], wrenches[t:t + 1], substeps, seed, dl_dU[t:t + 1])

        action_gradient = task.to_action_gradient(dJ_dU, ramp_angles, variant)[0]
        dJ_dobs = accumulate(actor, actions[t], action_gradient, observations[t])
        adjoint = dJ_dZ0 + task.state_gradient(dJ_dobs, jacobian) + dl_dZ[t]

    return adjoint
