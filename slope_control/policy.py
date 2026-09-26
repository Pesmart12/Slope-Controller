"""The actor and critic, and the tensor plumbing they need.

`Actor` turns an observation into a force, `Critic` estimates the reward
still to come, and the rest of this file converts between numpy and torch
and flattens parameters and gradients into indexable vectors.

Running a policy and differentiating back through it live in `sweep.py`.
Nothing here knows about ramps, rewards or contact; those are in `task.py`.
"""

import numpy as np
import torch

# Hidden layer widths, from reference SHAC's Ant and Cheetah configs.
ACTOR_HIDDEN = (128, 64, 32)
CRITIC_HIDDEN = (64, 64)

# Pre-squash noise. The action is limit*tanh(x + sigma*eps), so sigma is
# in the units of the pre-tanh activation rather than newtons; 0.37 puts
# meaningful exploration across the whole range without pinning the sample
# to either end of the squash.
INITIAL_LOG_STD = -1.0


def mlp(inputs, outputs, hidden, orthogonal=False):
    """Build a fully connected network: Linear, ELU and LayerNorm for each
    hidden layer, then a final Linear. The same layout reference SHAC uses.

    `orthogonal=True` initializes every Linear layer with orthogonal weights
    at gain sqrt(2) and zero biases, which is how the reference initializes
    its critic. Otherwise torch's default initialization applies, as it does
    for the reference actor.

    ELU rather than ReLU because ReLU has a kink, so its derivative jumps.
    The critic's gradient feeds into the actor's update here, meaning these
    networks get differentiated through rather than just backpropagated
    once, and a kinked function makes that gradient field discontinuous.
    """
    layers, width = [], inputs
    for size in hidden:
        layers += [torch.nn.Linear(width, size), torch.nn.ELU(), torch.nn.LayerNorm(size)]
        width = size
    layers.append(torch.nn.Linear(width, outputs))

    if orthogonal:
        for layer in layers:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
                torch.nn.init.zeros_(layer.bias)
    return torch.nn.Sequential(*layers)


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

    def __init__(self, observation_size, actuated, limit, hidden=ACTOR_HIDDEN):
        super().__init__()
        self.actuated = actuated
        self.register_buffer("limit", torch.as_tensor(np.asarray(limit), dtype=torch.float64))
        self.net = mlp(observation_size, 2 * actuated, hidden).double()
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
    know nothing about the 3.7 s hold that follows.

    Predicts raw returns, in reward units, as the reference does: no running
    normalization of the targets. The LayerNorm after each hidden layer is
    what keeps the fit well conditioned.
    """

    def __init__(self, observation_size, hidden=CRITIC_HIDDEN):
        super().__init__()
        self.net = mlp(observation_size, 1, hidden, orthogonal=True).double()

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


def gradients(module):
    """Every parameter's `.grad`, in parameter order, a missing one
    materialized as zeros.

    A parameter that never entered the computation has `.grad` of None --
    `log_std` on a deterministic pass, which reaches the action only through
    the noise term. Substituting zeros rather than dropping the entry is what
    keeps position meaningful: the list stays one entry per parameter, so
    nothing downstream has to know which were absent. Dropping them instead
    would shift every later index and misalign `flat_gradients` against
    `flat_parameters` silently.
    """
    return [torch.zeros_like(p) if p.grad is None else p.grad for p in module.parameters()]


def flat_gradients(module):
    """Every parameter's `.grad` flattened into one vector, to line up with
    `flat_parameters`.

    Index i means the same parameter in both vectors, which is what a
    finite-difference check needs: it nudges parameter i and compares against
    gradient i.
    """
    return torch.cat([gradient.reshape(-1) for gradient in gradients(module)])


def gradient_norm(module, error_if_nonfinite=True):
    """The L2 norm of every gradient taken as one vector.

    Raises by default if any gradient is infinite or NaN. That is worth a
    hard failure rather than a warning: a NaN entering Adam's second-moment
    estimate stays there, so every step after it is NaN too and the run is
    producing nothing. Failing at the first iteration beats discovering it
    hours in.

    Torch's `get_total_norm` does the arithmetic. It sums the squared
    per-tensor norms rather than squaring 5448 elements in one pass, which
    can differ from `flat_gradients(module).norm()` in the last few bits and
    never by more than about 1e-15 relative.
    """
    return float(torch.nn.utils.get_total_norm(gradients(module), error_if_nonfinite=error_if_nonfinite))


def zero_gradients(module):
    for parameter in module.parameters():
        parameter.grad = None
