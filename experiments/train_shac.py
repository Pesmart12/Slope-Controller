"""Train SHAC on penalty contact.

The experiment that trains, and the one every reported SHAC number comes
from. One run: 2000 iterations, 64 environments, seed 0.

Logs to the console and to a file, and writes a JSON history, a figure,
and two checkpoints: the latest evaluation, and the lowest-error one as
`shac-best.pt`. The latest exists because a previous long run died six
hours in and left nothing. The best one exists because on this task it
lands early, around iteration 400, and is overwritten long before the run
ends; `experiments/diagnose_shac.py` measures why.

The critic's value and slope are logged at every evaluation next to the
policy's error. What enters the actor's objective is dV/d(obs) and never
V, so `shac.slope_calibration` measures the slope against a central
difference of the true return.

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
NAME = "shac"


def run(iterations, n_envs, seed, eval_every, directory):
    """Train, logging to console and to a file as it goes.

    Returns the evaluation history. The actor, critic and history are
    written to `directory` at every evaluation, so a run that dies leaves
    everything up to its last evaluation behind.
    """
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / f"{NAME}.log").open("w")

    def log(line):
        print(line)
        handle.write(line + "\n")

        # Flush every line: this is the file that has to survive a run
        # dying, and buffered output would lose the tail that says how far
        # it got.
        handle.flush()

    def checkpoint(iteration, actor, critic, target_critic, history):
        state = dict(iteration=iteration, actor=actor.state_dict(), critic=critic.state_dict())
        torch.save(state, directory / f"{NAME}.pt")
        (directory / f"{NAME}.json").write_text(json.dumps(history, indent=2))

        # Keep the lowest-error evaluation as its own file. The latest
        # checkpoint overwrites itself every evaluation, and on this task
        # the best one lands early and is gone by the end of the run.
        if history[-1]["error"] <= min(record["error"] for record in history):
            torch.save(state, directory / f"{NAME}-best.pt")

    log(f"### {NAME}: {iterations} iterations, {n_envs} environments, seed {seed}")

    started = time.time()
    _, _, history = shac.train(iterations, n_envs=n_envs, seed=seed, eval_every=eval_every, log=log, checkpoint=checkpoint)
    elapsed = time.time() - started

    log(f"### {NAME}: {elapsed:.0f} s, {elapsed / iterations:.2f} s/iteration")
    handle.close()
    return history


def report(history):
    """Print where the run peaked and where it ended."""
    errors = [record["error"] for record in history]
    best = history[int(np.argmin(errors))]
    end = history[-1]

    print(f"    best     iteration {best['iteration']:>5}   err {100 * best['error']:>6.2f} cm   reward {best['reward']:>9.2f}")
    print(f"    end      iteration {end['iteration']:>5}   err {100 * end['error']:>6.2f} cm   reward {end['reward']:>9.2f}")
    print(f"    gave up  {100 * (end['error'] - best['error']):>6.2f} cm between them")
    print(f"    slope    corr {best['slope_correlation']:>6.3f} at best, {end['slope_correlation']:>6.3f} at end"
          f"   sign {best['slope_agreement']:>5.2f} -> {end['slope_agreement']:>5.2f}")


def plot(axes_error, axes_slope, history):
    iterations = [record["iteration"] for record in history]

    axes_error.plot(iterations, [100 * record["error"] for record in history], color="#333333", linewidth=2.0, marker="o", markersize=3.0)
    axes_error.set_xlabel("iteration")
    axes_error.set_ylabel("final position error (cm)")
    axes_error.set_title("Does the policy hold its best?\ndeterministic episodes, unshaped reward")

    axes_slope.plot(iterations, [record["slope_correlation"] for record in history], color="#333333", linewidth=2.0, marker="o", markersize=3.0)
    axes_slope.axhline(1.0, color="#888888", linewidth=1.0, linestyle=":")
    axes_slope.set_xlabel("iteration")
    axes_slope.set_ylabel(r"corr($dV/d\xi$, true slope)")
    axes_slope.set_title("What the actor actually reads\nthe critic's slope, not its value")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--envs", type=int, default=N_ENVS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--runs", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1] / "runs" / "shac")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1] / "figures" / "shac_training.png")
    args = parser.parse_args()

    history = run(args.iterations, args.envs, args.seed, args.eval_every, args.runs)

    print("\n" + "=" * 78)
    report(history)

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    plot(left, right, history)
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
