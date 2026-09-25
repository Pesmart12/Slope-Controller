"""Find out why SHAC loses its best policy as training continues.

The run `train_shac.py` makes peaks at iteration 400 and ends worse on both reward and
error at iteration 2000. Physics is the same at every iteration, so the
late policy is a worse policy. This script scores every checkpoint of that
run against several objectives to see which of them training is actually
improving:

    reported         unshaped, undiscounted, deterministic, 4 s
    surrogate        shaped, gamma = 0.99, from every window start, with the
                     true return-to-go -- what SHAC maximizes if its critic
                     were perfect
    surrogate noisy  the same, rolled with the checkpoint's exploration noise
    fixed critic     the same, with everything past each window replaced by
                     one target critic held fixed across every checkpoint --
                     what the actor climbs, with the critic's own drift
                     taken out

Reading them:

    surrogate rises while reported falls       the objectives disagree
    fixed-critic rises while surrogate falls   the critic misleads the actor
    everything falls                           the optimization is unstable

The critic has to be held fixed. Scored with each checkpoint's own target
critic, the bootstrapped surrogate rises from -26.8 to -13.3 over that
run, but that is the critic's calibration catching up, not the
actor improving: against any single critic the late actors score no
better than iteration 400.

Two phases. The first reruns `train_shac.py`'s training with a
checkpoint at every evaluation and a per-iteration trace; that run keeps
only two checkpoints, and no target critic. The second rescores every
checkpoint on 64 environments drawn from a seed neither training nor
`shac.evaluate` uses. The recorded best was chosen over 21 evaluations of
the same 8 episodes, which flatters it.

Run:  python experiments/diagnose_shac.py
      python experiments/diagnose_shac.py --skip-train     rescore only
"""

import argparse
import json
import pathlib
import time

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

from slope_control import batches, objective, observation, policy, ramp, shac, sweep, task

ITERATIONS = 2000
N_ENVS = 64
SEED = 0
EVAL_EVERY = 100

# The rescoring batch. 2000 is neither the training seed nor the seed
# `shac.evaluate` defaults to, so no checkpoint was selected on it.
SCORE_SEED = 2000
SCORE_ENVS = 64

# Noise seeds per checkpoint for the noisy surrogate. Four is enough to see
# a trend across 21 checkpoints; it is not enough to quote a single value.
NOISE_SEEDS = 4

# The checkpoints whose target critics are held fixed and used to score
# every actor. Early, middle and last, so a conclusion does not hang on one
# critic. Any that a shortened run did not produce are skipped.
FIXED_CRITICS = (400, 1000, 2000)

# The first second of an episode. Before it the box is still being driven
# to the target; after it the policy is holding.
APPROACH_STEPS = 100

ROOT = pathlib.Path(__file__).resolve().parents[1]


def train_run(directory, iterations, n_envs, seed, eval_every):
    """Rerun `train_shac.py`'s training, saving a checkpoint at every evaluation.

    Writes `checkpoints/<iteration>.pt` with the actor, critic and target
    critic, `history.json` with the evaluation log, `trace.json` with one
    row per iteration, and `train.log`.
    """
    (directory / "checkpoints").mkdir(parents=True, exist_ok=True)
    handle = (directory / "train.log").open("w")
    rows = []

    def log(line):
        print(line)
        handle.write(line + "\n")
        handle.flush()

    def trace(iteration, phase, objective_value, gradient_norm):
        rows.append(dict(iteration=iteration, phase=phase, objective=objective_value, gradient_norm=gradient_norm))

    def checkpoint(iteration, actor, critic, target_critic, history):
        state = dict(iteration=iteration, actor=actor.state_dict(), critic=critic.state_dict(), target_critic=target_critic.state_dict())
        torch.save(state, directory / "checkpoints" / f"{iteration:05d}.pt")
        (directory / "history.json").write_text(json.dumps(history, indent=2))
        (directory / "trace.json").write_text(json.dumps(rows))

    log(f"### rerun: {iterations} iterations, {n_envs} environments, seed {seed}")
    started = time.time()
    shac.train(iterations, n_envs=n_envs, seed=seed, eval_every=eval_every, log=log, checkpoint=checkpoint, trace=trace)
    log(f"### {time.time() - started:.0f} s")
    handle.close()


def load(path, observation_size):
    """Rebuild the actor and target critic a checkpoint was saved from."""
    state = torch.load(path)
    actor = policy.Actor(observation_size, len(task.ACTUATED), task.LIMIT)
    actor.load_state_dict(state["actor"])
    target_critic = policy.Critic(observation_size)
    target_critic.load_state_dict(state["target_critic"])
    return state["iteration"], actor, target_critic


def window_starts():
    """The steps at which training windows begin: 0, 32, ..., 384."""
    return list(range(0, task.EPISODE_STEPS, shac.WINDOW))


def surrogate(rewards, gamma):
    """What SHAC maximizes with a perfect critic: the discounted return-to-go
    from each window start, averaged over windows and environments."""
    return float(shac.discounted_returns(rewards, gamma)[window_starts()].mean())


def critic_surrogate(rewards, states, batch, target_critic, gamma):
    """What the actor actually climbed: each window's own discounted reward,
    plus the target critic's value at the window's end.

    Mirrors `shac.train`. The last window is shortened to end the episode
    and has no bootstrap.
    """
    returns = shac.discounted_returns(rewards, gamma)
    values = []
    for start in window_starts():
        end = min(start + shac.WINDOW, task.EPISODE_STEPS)

        # Subtract the discounted tail to keep only this window's rewards.
        inside = returns[start] - gamma ** (end - start) * returns[end]

        tail = 0.0
        if end < task.EPISODE_STEPS:
            with torch.no_grad():
                terminal = policy.as_tensor(observation.observe(states[end], batch.angles, batch.targets))
                tail = gamma ** (end - start) * target_critic(terminal).numpy()
        values.append(inside + tail)
    return float(np.mean(values))


def score(actor, critics, batch, gamma=shac.GAMMA, noise_seeds=NOISE_SEEDS):
    """Score one checkpoint's actor on one batch. Returns a dict of plain
    floats.

    `critics` maps an iteration to the target critic saved there; each one
    adds a `fixed_<iteration>` entry, the bootstrapped surrogate scored with
    that critic.
    """
    _, rolled = shac.rolled_episode(actor, batch=batch)
    states, wrenches = rolled.states, rolled.wrenches

    # Score the rollout three ways, each (steps, environments):
    #   unshaped    the reported reward
    #   shaped      the reward training maximizes
    #   position    the reward with the controls zeroed, which leaves the
    #               position term alone; unshaped minus it is the control term
    unshaped = objective.reward(states, wrenches, batch.angles, batch.targets)
    shaped = objective.reward(states, wrenches, batch.angles, batch.targets, shaping=True)
    position = objective.reward(states, np.zeros_like(wrenches), batch.angles, batch.targets)

    # Final error, signed, and which side of the start each target sits on.
    final = ramp.along_ramp(states[-1], batch.angles)[:, task.BOX] - batch.targets
    uphill = batch.targets > ramp.along_ramp(batch.state, batch.angles)[:, task.BOX]

    # Roll again with the policy's own exploration noise, as training does,
    # and score what training would have seen.
    noisy = []
    for seed in range(noise_seeds):
        torch.manual_seed(seed)
        with torch.no_grad():
            window = sweep.rollout(batch, actor, task.EPISODE_STEPS)
        noisy.append(surrogate(objective.reward(window.states, window.wrenches, batch.angles, batch.targets, shaping=True), gamma))

    return dict(reported=float(unshaped.sum(axis=0).mean()),
                approach=float(unshaped[:APPROACH_STEPS].sum(axis=0).mean()),
                hold=float(unshaped[APPROACH_STEPS:].sum(axis=0).mean()),
                position=float(position.sum(axis=0).mean()),
                control=float((unshaped - position).sum(axis=0).mean()),
                shaping=float((shaped - unshaped).sum(axis=0).mean()),
                surrogate=surrogate(shaped, gamma),
                surrogate_unshaped=surrogate(unshaped, gamma),
                surrogate_undiscounted=surrogate(shaped, 1.0),
                surrogate_noisy=float(np.mean(noisy)),
                error_uphill=float(np.abs(final[uphill]).mean()),
                error_downhill=float(np.abs(final[~uphill]).mean()),
                signed_uphill=float(final[uphill].mean()),
                signed_downhill=float(final[~uphill].mean()),
                sigma=float(torch.exp(actor.log_std.detach()).mean()),
                **{f"fixed_{iteration}": critic_surrogate(shaped, states, batch, critic, gamma) for iteration, critic in critics.items()})


def rescore(directory):
    """Score every checkpoint in `directory`. Returns one dict per checkpoint.

    Also checks each checkpoint's reported reward on `shac.evaluate`'s own
    batch against the value the training run logged, so the rescoring is
    known to be scoring the same policies.
    """
    batch = batches.sample_batch(np.random.default_rng(SCORE_SEED), SCORE_ENVS)
    check_batch = batches.sample_batch(np.random.default_rng(shac.EVAL_SEED), shac.EVAL_ENVS)
    observation_size = observation.observe(batch.state, batch.angles, batch.targets).shape[-1]
    logged = {record["iteration"]: record["reward"] for record in json.loads((directory / "history.json").read_text())}

    # Load every checkpoint up front, since each actor is scored against
    # critics saved at other iterations.
    loaded = [load(path, observation_size) for path in sorted((directory / "checkpoints").glob("*.pt"))]
    critics = {iteration: critic for iteration, _, critic in loaded if iteration in FIXED_CRITICS}

    results = []
    for iteration, actor, _ in loaded:
        # Rescore on the batch training evaluated on, and require the logged
        # number back.
        reward, _, _ = shac.evaluate(shac.rolled_episode(actor, batch=check_batch))
        if not np.isclose(reward.mean(), logged[iteration], rtol=1e-9, atol=1e-9):
            raise RuntimeError(f"iteration {iteration}: rescored {reward.mean():.6f}, logged {logged[iteration]:.6f}")

        started = time.time()
        results.append(dict(iteration=iteration, **score(actor, critics, batch)))
        print(f"  scored {iteration:>5} in {time.time() - started:.0f} s")
    return results


def report(results, trace):
    """Print the rescoring as a table, and the trace summarized by phase."""
    fixed = [key for key in results[0] if key.startswith("fixed_")]
    print("\n  iter  reported  approach    hold  position  control |  surr  s.noisy  s.unshaped  s.undisc |"
          + "".join(f"  V@{key[6:]:>4}" for key in fixed) + " |  err up  err down (cm)  sigma")
    for r in results:
        print(f"  {r['iteration']:>4}  {r['reported']:>8.2f}  {r['approach']:>8.2f}  {r['hold']:>6.2f}  {r['position']:>8.2f}"
              f"  {r['control']:>7.2f} | {r['surrogate']:>6.2f}  {r['surrogate_noisy']:>7.2f}"
              f"  {r['surrogate_unshaped']:>10.2f}  {r['surrogate_undiscounted']:>8.2f} |"
              + "".join(f"  {r[key]:>6.2f}" for key in fixed)
              + f" |  {100 * r['error_uphill']:>6.2f}  {100 * r['error_downhill']:>8.2f}       {r['sigma']:.3f}")

    # Bin the trace by 200 iterations, split approach windows from hold
    # windows, and report the median gradient norm and mean objective of
    # each. Approach is the windows that start inside APPROACH_STEPS.
    approach_windows = APPROACH_STEPS // shac.WINDOW + 1
    print(f"\n  trace, windows 0-{approach_windows - 1} as approach, the rest as hold")
    print("   iters        |grad| approach   |grad| hold   J approach   J hold")
    for low in range(0, trace[-1]["iteration"], 200):
        rows = [row for row in trace if low < row["iteration"] <= low + 200]
        early = [row for row in rows if row["phase"] < approach_windows]
        late = [row for row in rows if row["phase"] >= approach_windows]
        print(f"  {low + 1:>5}-{low + 200:<5}  {np.median([row['gradient_norm'] for row in early]):>14.1f}"
              f"  {np.median([row['gradient_norm'] for row in late]):>12.1f}"
              f"  {np.mean([row['objective'] for row in early]):>11.2f}  {np.mean([row['objective'] for row in late]):>7.2f}")


def plot(results, trace, path):
    # Take the reference as the checkpoint with the best reported reward,
    # skipping iteration 1, whose scores are an order of magnitude off.
    trained = [r for r in results if r["iteration"] > 1]
    reference = max(trained, key=lambda r: r["reported"])

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))

    # Plot each score's change relative to its value at the reference
    # checkpoint. Rewards are negative and maximized, so up is better on
    # every curve.
    ax = axes[0, 0]
    series = [("reported", "reported", "#222222"), ("surrogate", "surrogate", "#4477aa"),
              ("surrogate_noisy", "surrogate, noisy", "#66ccee"), ("surrogate_unshaped", "surrogate, unshaped", "#228833"),
              ("surrogate_undiscounted", "surrogate, gamma = 1", "#aa3377")]
    reds = ["#f4a582", "#d6604d", "#b2182b"]
    series += [(key, f"critic from iter {key[6:]}, fixed", reds[i % len(reds)]) for i, key in enumerate(key for key in reference if key.startswith("fixed_"))]
    for key, label, colour in series:
        base = reference[key]
        ax.plot([r["iteration"] for r in trained], [100 * (r[key] - base) / abs(base) for r in trained], color=colour, linewidth=1.8, marker="o", markersize=3.0, label=label)
    ax.axhline(0.0, color="#888888", linewidth=1.0, linestyle=":")
    ax.axvline(reference["iteration"], color="#888888", linewidth=1.0, linestyle=":")
    ax.set_ylabel(f"% change from iteration {reference['iteration']}")
    ax.set_title("Which objective is training improving?\nup is better on every curve")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for key, label, colour in [("position", "position term", "#4477aa"), ("control", "control term", "#c1442e"), ("shaping", "shaping term", "#228833")]:
        ax.plot([r["iteration"] for r in trained], [r[key] for r in trained], color=colour, linewidth=1.8, marker="o", markersize=3.0, label=label)
    ax.set_ylabel("episode reward")
    ax.set_title("What the reported reward is made of\ndeterministic, 4 s")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.plot([r["iteration"] for r in trained], [100 * r["signed_uphill"] for r in trained], color="#c1442e", linewidth=1.8, marker="o", markersize=3.0, label="uphill targets")
    ax.plot([r["iteration"] for r in trained], [100 * r["signed_downhill"] for r in trained], color="#4477aa", linewidth=1.8, marker="o", markersize=3.0, label="downhill targets")
    ax.axhline(0.0, color="#888888", linewidth=1.0, linestyle=":")
    ax.set_xlabel("iteration")
    ax.set_ylabel("final error, signed (cm)\nnegative is downhill of target")
    ax.set_title("Where the box ends up, by target direction")
    ax.legend(frameon=False, fontsize=8)

    # Plot a running median of the gradient norm, one curve per episode
    # phase group, on a log axis because the spikes span three decades.
    ax = axes[1, 1]
    approach_windows = APPROACH_STEPS // shac.WINDOW + 1
    for label, keep, colour in [("approach windows", lambda p: p < approach_windows, "#c1442e"), ("hold windows", lambda p: p >= approach_windows, "#4477aa")]:
        rows = [row for row in trace if keep(row["phase"])]
        x = np.array([row["iteration"] for row in rows])
        y = np.array([row["gradient_norm"] for row in rows])
        medians = [np.median(y[max(0, i - 25):i + 25]) for i in range(len(y))]
        ax.plot(x, y, color=colour, alpha=0.15, linewidth=0.6)
        ax.plot(x, medians, color=colour, linewidth=1.8, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("actor gradient norm, before the step")
    ax.set_title("Which part of the episode drives the actor")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--envs", type=int, default=N_ENVS)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--skip-train", action="store_true", help="rescore the checkpoints already in --runs")
    parser.add_argument("--runs", type=pathlib.Path, default=ROOT / "runs" / "shac_diagnosis")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "figures" / "shac_diagnosis.png")
    args = parser.parse_args()

    if not args.skip_train:
        train_run(args.runs, args.iterations, args.envs, SEED, args.eval_every)

    results = rescore(args.runs)
    trace = json.loads((args.runs / "trace.json").read_text())
    (args.runs / "scores.json").write_text(json.dumps(results, indent=2))

    report(results, trace)
    plot(results, trace, args.out)
    print(f"\nwrote {args.out}")
    print(f"wrote {args.runs}")


if __name__ == "__main__":
    main()
