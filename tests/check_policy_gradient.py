"""Check the policy gradient against finite differences.  Run: python tests/check_policy_gradient.py

`check_closed_loop.py` made this point for a one-parameter feedback law
differentiated by hand. This makes it for the thing SHAC actually trains:
a network, with the derivative crossing between GRIP and torch twice per
step.

    GRIP gives dJ/d(wrench)
      -> rotate to dJ/d(action)
      -> back-propagate to dJ/d(weights) and dJ/d(observation)
      -> multiply by the observation Jacobian to get dJ/d(state)
      -> hand back to GRIP as the next seed

Every one of those hops can hide a sign or a transpose, and none of them
raise when wrong. They just train something that is not the objective.

The policy must run deterministically here. Finite differencing means
perturbing a weight and re-rolling, and with sampling on the two rollouts
would differ by noise far larger than the perturbation.

The critic is left out for the same reason. It would be a legitimate part
of the objective, but the check has to differentiate exactly what it
perturbs, and what it perturbs is the window's reward.
"""

import numpy as np
import torch

from slope_control import policy, task

WINDOW = 24
PROBES = 12


def setup(variant, n_envs=2, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    batch = task.sample_batch(rng, n_envs, variant)
    observation_size = task.observe(batch.state, batch.angles, batch.targets, variant).shape[-1]
    actor = policy.Actor(observation_size, len(variant.actuated), variant.limit)

    # Small but not tiny, so the network is genuinely nonlinear at the
    # operating point rather than sitting in tanh's linear region.
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.add_(0.25 * torch.randn_like(parameter))

    return batch, actor


def windowed_reward(batch, actor, shaping):
    """The scalar the sweep claims to differentiate."""
    window = policy.rollout(batch, actor, WINDOW, deterministic=True)
    return task.reward(window.states, window.wrenches, batch.angles, batch.targets, batch.variant, shaping=shaping).sum()


def check(name, variant, shaping):
    batch, actor = setup(variant)

    policy.zero_gradients(actor)
    window = policy.rollout(batch, actor, WINDOW, deterministic=True)
    policy.policy_gradient(batch, window, actor, critic=None, shaping=shaping)
    analytic = policy.flat_gradients(actor).numpy()

    baseline = policy.flat_parameters(actor).numpy().copy()
    rng = np.random.default_rng(1)
    indices = rng.choice(baseline.size, size=min(PROBES, baseline.size), replace=False)

    def perturbed_reward(index, delta):
        shifted = baseline.copy()
        shifted[index] += delta
        policy.set_flat_parameters(actor, shifted)
        return windowed_reward(batch, actor, shaping)

    worst, scale, h = 0.0, np.abs(analytic).max(), 1e-6
    for index in indices:
        finite = (perturbed_reward(index, h) - perturbed_reward(index, -h)) / (2.0 * h)
        policy.set_flat_parameters(actor, baseline)
        # Relative to the gradient's own scale, not to this component --
        # a component that happens to be near zero would otherwise
        # dominate the score for no reason.
        worst = max(worst, abs(finite - analytic[index]) / max(abs(finite), 1e-3 * scale))

    print(f"  {name}: {baseline.size} parameters, {len(indices)} probed through {WINDOW * batch.substeps} integration steps"
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
