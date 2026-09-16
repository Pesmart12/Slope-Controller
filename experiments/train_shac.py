"""Train SHAC on penalty contact, and measure whether clipping holds the peak.

The experiment that trains, and the one every reported SHAC number comes
from.

Three arms, same seed and same environments, all run here rather than one
here and the rest quoted from an earlier machine:

    unclipped       max_norm None                 what every run so far did
    clipped         max_norm 2.0                  the ceiling `shac.MAX_GRADIENT_NORM` sizes
    no-bootstrap    actor_bootstrap False         the critic never reaches the actor

Read the `clipped` count before reading the first two arms' error. A count
of zero means the ceiling never bound and they are the same experiment run
twice; a count near every iteration means the ceiling is a learning-rate
change wearing a disguise. Neither says anything about clipping.

The third arm keeps `dV/d(obs)` out of the actor's objective while leaving
the critic fitted and measured. Its outcomes are not symmetric: a run that
still decays shows the decay does not need the critic's slope, while a run
that holds cannot separate that from the actor simply being myopic. See
`shac.train`.

The critic's slope is logged alongside, at every evaluation. What enters
the actor's objective is dV/d(obs) and never V, and no run has ever
checked it -- `shac.slope_calibration` measures it against a central
difference of the true return.

Writes a figure, a JSON history and two checkpoints per arm: the latest
evaluation, and the lowest-error one as `<arm>-best.pt`. The latest exists
because a previous long run died six hours in and left nothing. The best
one exists because on this task it lands early and is overwritten long
before the run ends, which is what a demo wants and what the creep
measurement needs to compare against.

Run:  python experiments/train_shac.py
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

from slope_control import shac

ITERATIONS = 2000
N_ENVS = 64
SEED = 0
EVAL_EVERY = 100

# name, the actor's gradient ceiling, whether the critic reaches the actor.
ARMS = [("unclipped", None, True), ("clipped", shac.MAX_GRADIENT_NORM, True), ("no-bootstrap", None, False)]


def run_arm(name, max_norm, actor_bootstrap, iterations, n_envs, seed, eval_every, directory):
    """Train one arm, logging to console and to a file as it goes.

    Returns the evaluation history. The actor, critic and history are
    written to `directory` at every evaluation, so a run that dies leaves
    everything up to its last evaluation behind.
    """
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / f"{name}.log"
    handle = log_path.open("w")

    def log(line):
        print(line)
        handle.write(line + "\n")

        # Flush every line: this is the file that has to survive a run
        # dying, and buffered output would lose the tail that says how far
        # it got.
        handle.flush()

    def checkpoint(iteration, actor, critic, history):
        state = dict(iteration=iteration, actor=actor.state_dict(), critic=critic.state_dict())
        torch.save(state, directory / f"{name}.pt")
        (directory / f"{name}.json").write_text(json.dumps(history, indent=2))

        # Keep the lowest-error evaluation as its own file. The latest
        # checkpoint overwrites itself every evaluation, and on this task
        # the best one lands early and is gone by the end of the run --
        # which is what left the 1.39 cm policy unrecoverable once.
        if history[-1]["error"] <= min(record["error"] for record in history):
            torch.save(state, directory / f"{name}-best.pt")

    ceiling = "none" if max_norm is None else f"{max_norm}"
    log(f"### {name}: max_norm {ceiling}, actor bootstrap {'on' if actor_bootstrap else 'off'}, "
        f"{iterations} iterations, {n_envs} environments, seed {seed}")

    started = time.time()
    _, _, history = shac.train(iterations, n_envs=n_envs, seed=seed, max_norm=max_norm, eval_every=eval_every, log=log, checkpoint=checkpoint, actor_bootstrap=actor_bootstrap)
    elapsed = time.time() - started

    log(f"### {name}: {elapsed:.0f} s, {elapsed / iterations:.2f} s/iteration")
    handle.close()
    return history


def report(name, history):
    """Print the two numbers that decide what an arm means: where it peaked,
    and where it ended."""
    errors = [record["error"] for record in history]
    best = history[int(np.argmin(errors))]
    end = history[-1]
    total_clipped = sum(record["clipped"] for record in history)

    print(f"\n  {name}")
    print(f"    best     iteration {best['iteration']:>5}   err {100 * best['error']:>6.2f} cm   reward {best['reward']:>9.2f}")
    print(f"    end      iteration {end['iteration']:>5}   err {100 * end['error']:>6.2f} cm   reward {end['reward']:>9.2f}")
    print(f"    gave up  {100 * (end['error'] - best['error']):>6.2f} cm between them")
    print(f"    clipped  {total_clipped} of {history[-1]['iteration']} iterations")
    print(f"    slope    corr {best['slope_correlation']:>6.3f} at best, {end['slope_correlation']:>6.3f} at end"
          f"   sign {best['slope_agreement']:>5.2f} -> {end['slope_agreement']:>5.2f}")
    return best, end


def plot(axes_error, axes_slope, results):
    colours = {"unclipped": "#333333", "clipped": "#c1442e"}
    for name, history in results.items():
        iterations = [record["iteration"] for record in history]
        colour = colours.get(name, "#4477aa")

        axes_error.plot(iterations, [100 * record["error"] for record in history], color=colour, linewidth=2.0, marker="o", markersize=3.0, label=name)
        axes_slope.plot(iterations, [record["slope_correlation"] for record in history], color=colour, linewidth=2.0, marker="o", markersize=3.0, label=name)

    axes_error.set_xlabel("iteration")
    axes_error.set_ylabel("final position error (cm)")
    axes_error.set_title("Does the policy hold its best?\ndeterministic episodes, unshaped reward")
    axes_error.legend(frameon=False)

    axes_slope.axhline(1.0, color="#888888", linewidth=1.0, linestyle=":")
    axes_slope.set_xlabel("iteration")
    axes_slope.set_ylabel(r"corr($dV/d\xi$, true slope)")
    axes_slope.set_title("What the actor actually reads\nthe critic's slope, not its value")
    axes_slope.legend(frameon=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--envs", type=int, default=N_ENVS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--arm", choices=[name for name, _, _ in ARMS], help="run one arm instead of all of them")
    parser.add_argument("--runs", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1] / "runs" / "shac_clipping")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1] / "figures" / "shac_clipping.png")
    args = parser.parse_args()

    arms = [arm for arm in ARMS if args.arm is None or arm[0] == args.arm]

    results = {}
    for name, max_norm, actor_bootstrap in arms:
        results[name] = run_arm(name, max_norm, actor_bootstrap, args.iterations, args.envs, args.seed, args.eval_every, args.runs)

    print("\n" + "=" * 78)
    for name, history in results.items():
        report(name, history)

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    plot(left, right, results)
    for axes in (left, right):
        axes.grid(alpha=0.25)
        axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")
    print(f"wrote {args.runs}")


if __name__ == "__main__":
    main()
