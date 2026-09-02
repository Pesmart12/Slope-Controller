"""dJ/d(policy parameters), finite-differenced.  Run: python tests/check_policy_gradient.py

`check_closed_loop.py` established that a policy needs a per-step adjoint
sweep and measured what one call per window costs -- 119% at one gain.
That was a one-parameter feedback law, hand-differentiated. This is the
same statement for the thing SHAC actually trains: an MLP, with the chain
running

    dJ/dU  ->  dJ/d(action)  ->  dJ/d(theta)   and   dJ/d(obs)  ->  dJ/dZ

through torch on one side and GRIP on the other. Every one of those hops
is a place a sign or a transpose can hide, and none of them raise when
wrong -- they just train something that is not the objective.

Finite-differencing a network means perturbing parameters and re-rolling,
so the policy must be DETERMINISTIC here: with sampling on, the two
evaluations differ by noise far larger than the perturbation and the
comparison is meaningless. Sampling is a separate concern and is not what
this checks.

The critic is left out for the same reason. It would be a legitimate part
of the objective, but the check has to differentiate exactly the quantity
it perturbs, and the windowed reward is that quantity.
"""

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from slope_control import policy, ramp, task  # noqa: E402

WINDOW = 24
PROBES = 12


def setup(variant, n_envs=2, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    scenes, angles, state, targets = task.sample_batch(rng, n_envs, variant)
    substeps = task.substeps_for(scenes[0])
    observation_size = task.observe(state, angles, targets, variant).shape[-1]
    actor = policy.Actor(observation_size, len(variant.actuated), variant.limit)

    # Small but not tiny, so the network is genuinely nonlinear at the
    # operating point rather than sitting in tanh's linear region.
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.add_(0.25 * torch.randn_like(parameter))

    return scenes, angles, state, targets, substeps, actor, task.observation_jacobian(angles, variant)


def windowed_reward(scenes, state, angles, targets, actor, variant, substeps, shaping):
    """The scalar the sweep claims to differentiate."""
    trajectory, wrenches, _, _ = policy.rollout(scenes, state, angles, targets, actor, variant, substeps, WINDOW, deterministic=True)
    return task.reward(trajectory, wrenches, angles, targets, variant, shaping=shaping).sum()


def check(name, variant, shaping):
    scenes, angles, state, targets, substeps, actor, jacobian = setup(variant)

    policy.zero_gradients(actor)
    trajectory, wrenches, observations, actions = policy.rollout(scenes, state, angles, targets, actor, variant, substeps, WINDOW, deterministic=True)
    policy.policy_gradient(scenes, trajectory, wrenches, observations, actions, angles, targets, actor, variant, substeps, jacobian, critic=None, shaping=shaping)
    analytic = policy.flat_gradients(actor).numpy()

    baseline = policy.flat_parameters(actor).numpy().copy()
    rng = np.random.default_rng(1)
    indices = rng.choice(baseline.size, size=min(PROBES, baseline.size), replace=False)

    def perturbed_reward(index, delta):
        shifted = baseline.copy()
        shifted[index] += delta
        policy.set_flat_parameters(actor, shifted)
        return windowed_reward(scenes, state, angles, targets, actor, variant, substeps, shaping)

    worst, scale, h = 0.0, np.abs(analytic).max(), 1e-6
    for index in indices:
        finite = (perturbed_reward(index, h) - perturbed_reward(index, -h)) / (2.0 * h)
        policy.set_flat_parameters(actor, baseline)
        # Relative to the gradient's own scale, not to this component --
        # a component that happens to be near zero would otherwise
        # dominate the score for no reason.
        worst = max(worst, abs(finite - analytic[index]) / max(abs(finite), 1e-3 * scale))

    print(f"  {name}: {baseline.size} parameters, {len(indices)} probed through {WINDOW * substeps} integration steps"
          f"  ->  worst relative error {worst:.2e}")
    return worst


def main():
    print(__doc__.strip().splitlines()[0])
    print(f"\n  MLP actor, deterministic, {WINDOW}-step window, per-step adjoint sweep\n")

    worst = 0.0
    for name, variant, shaping in [("box only,  unshaped", task.BOX_ONLY, False),
                                   ("box only,  shaped  ", task.BOX_ONLY, True),
                                   ("2 pushers, shaped  ", task.TWO_PUSHERS, True)]:
        worst = max(worst, check(name, variant, shaping))

    failed = worst > 1e-4
    print(f"\n{'FAIL' if failed else 'PASS'} -- the policy gradient matches central differences to {worst:.2e}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
