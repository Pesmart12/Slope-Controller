"""Check that the critic's TD(lambda) targets are the ones reference SHAC fits.

Three checks on random rewards and values, no simulation:

    lam = 0      every target is one reward plus the next state's value
    lam = 1      every target is the discounted reward to the window's end
                 plus the value there, the same as `discounted_returns`
    lam = 0.95   the targets match a line-for-line port of the reference's
                 own recursion, NVlabs/DiffRL `compute_target_values`

Run:  python tests/check_critic_targets.py
"""

import numpy as np

from slope_control import shac

STEPS, ENVS = 32, 64
GAMMA = shac.GAMMA
TOLERANCE = 1e-12


def reference_targets(rewards, values, gamma, lam):
    """The reference's TD(lambda) recursion, ported line for line.

    Its window has `done_mask` zero everywhere except the last step, which
    it sets to one so every window ends with a bootstrap from `values`.
    """
    steps, envs = rewards.shape
    done_mask = np.zeros((steps, envs))
    done_mask[-1] = 1.0

    targets = np.zeros((steps, envs))
    Ai, Bi, lam_i = np.zeros(envs), np.zeros(envs), np.ones(envs)
    for i in reversed(range(steps)):
        lam_i = lam_i * lam * (1.0 - done_mask[i]) + done_mask[i]
        Ai = (1.0 - done_mask[i]) * (lam * gamma * Ai + gamma * values[i] + (1.0 - lam_i) / (1.0 - lam) * rewards[i])
        Bi = gamma * (values[i] * done_mask[i] + Bi * (1.0 - done_mask[i])) + rewards[i]
        targets[i] = (1.0 - lam) * Ai + lam_i * Bi
    return targets


def report(name, ours, expected):
    error = float(np.max(np.abs(ours - expected)))
    passed = error <= TOLERANCE
    print(f"  {'pass' if passed else 'FAIL'}  {name:<48} max |difference| {error:.2e}")
    return passed


def main():
    rng = np.random.default_rng(0)
    rewards = rng.normal(-1.0, 0.5, size=(STEPS, ENVS))
    values = rng.normal(-20.0, 5.0, size=(STEPS, ENVS))

    one_step = rewards + GAMMA * values
    discounted = shac.discounted_returns(rewards, GAMMA, terminal=values[-1])[:-1]

    results = [
        report("lam = 0 is one reward plus the next value", shac.lambda_returns(rewards, values, GAMMA, lam=0.0), one_step),
        report("lam = 1 is the discounted return to the window end", shac.lambda_returns(rewards, values, GAMMA, lam=1.0), discounted),
        report("lam = 0.95 matches the reference recursion", shac.lambda_returns(rewards, values, GAMMA, lam=shac.LAMBDA), reference_targets(rewards, values, GAMMA, shac.LAMBDA)),
    ]
    print(f"\n{sum(results)}/{len(results)} checks passed")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
