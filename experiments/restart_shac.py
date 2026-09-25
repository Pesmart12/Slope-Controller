"""Restart training from its best checkpoint and change one thing at a time.

`diagnose_shac.py` shows the policy is at its best at iteration 400 and
loses the hold after it. Something that changes during training has to be
the cause, since everything fixed was fixed at iteration 400 too. This
restarts from that checkpoint and runs 400 more iterations in four arms,
each changing one thing:

    control         nothing -- has to reproduce the decline, or the restart
                    itself is confounded
    actor lr x0.1   smaller actor steps
    frozen critic   no critic fitting and no target update, so the actor
                    bootstraps from iteration 400's target critic throughout
    frozen noise    the actor's log_std held at its iteration-400 value

All four use the same seed, so they see the same batches and the same noise
draws. The iteration-400 checkpoint carries no optimizer state, so every
arm restarts Adam from scratch; that is why the control arm exists.

Every checkpoint is scored the way `diagnose_shac.py` scores, on the same
64 environments, next to the original run's curve.

Run:  python experiments/restart_shac.py
      python experiments/restart_shac.py --skip-train     rescore only
"""

import argparse
import json
import multiprocessing
import pathlib
import time

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

import diagnose_shac
from slope_control import batches, observation, shac

ITERATIONS = 400
EVAL_EVERY = 100
SEED = 0

# name, the `shac.train` arguments that make the arm, and its colour.
ARMS = [("control", {}, "#222222"),
        ("actor lr x0.1", dict(actor_lr=0.1 * shac.ACTOR_LR), "#4477aa"),
        ("frozen critic", dict(train_critic=False), "#c1442e"),
        ("frozen noise", dict(train_noise=False), "#228833")]

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run_arm(name, overrides, start, directory, iterations):
    """Train one arm from `start`, writing a log and a checkpoint at every
    evaluation to `directory / name`."""
    # One thread per process, since the four arms run side by side.
    torch.set_num_threads(1)

    arm = directory / name.replace(" ", "_")
    arm.mkdir(parents=True, exist_ok=True)
    handle = (arm / "train.log").open("w")

    def log(line):
        handle.write(line + "\n")
        handle.flush()

    def checkpoint(iteration, state, history):
        torch.save(state, arm / f"{iteration:05d}.pt")

    initial = torch.load(start)
    log(f"### {name}: from iteration {initial['iteration']}, {iterations} iterations, {overrides}")
    started = time.time()
    shac.train(iterations, seed=SEED, eval_every=EVAL_EVERY, log=log, checkpoint=checkpoint, initial=initial, **overrides)
    log(f"### {time.time() - started:.0f} s")
    handle.close()


def rescore(start, directory):
    """Score the starting checkpoint and every arm's checkpoints on the
    diagnosis batch. Returns {arm: [score dicts]}, each list starting at
    the shared starting checkpoint."""
    batch = batches.sample_batch(np.random.default_rng(diagnose_shac.SCORE_SEED), diagnose_shac.SCORE_ENVS)
    observation_size = observation.observe(batch.state, batch.angles, batch.targets).shape[-1]

    iteration, actor, _ = diagnose_shac.load(start, observation_size)
    first = dict(iteration=iteration, **diagnose_shac.score(actor, {}, batch))

    results = {}
    for name, _, _ in ARMS:
        results[name] = [first]
        for path in sorted((directory / name.replace(" ", "_")).glob("*.pt")):
            iteration, actor, _ = diagnose_shac.load(path, observation_size)
            results[name].append(dict(iteration=iteration, **diagnose_shac.score(actor, {}, batch)))
    return results


def report(results, original):
    """Print each arm's scores beside the original run's."""
    columns = "  iter  reported     hold   surrogate   err up  err down (cm, signed)  sigma"
    rows = [("original run", [r for r in original if r["iteration"] in {r["iteration"] for r in results["control"]}])]
    for name, scores in [*rows, *results.items()]:
        print(f"\n  {name}\n{columns}")
        for r in scores:
            print(f"  {r['iteration']:>4}  {r['reported']:>8.2f}  {r['hold']:>7.2f}  {r['surrogate']:>10.2f}"
                  f"  {100 * r['signed_uphill']:>7.2f}  {100 * r['signed_downhill']:>8.2f}               {r['sigma']:.3f}")


def plot(results, original, path):
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    span = {r["iteration"] for r in results["control"]}
    reference = [r for r in original if r["iteration"] in span]

    # Plot the hold reward on the left and the mean signed final error, in
    # cm, on the right: the original run as a dotted reference, then each arm.
    readings = [(left, lambda r: r["hold"]), (right, lambda r: 50.0 * (r["signed_uphill"] + r["signed_downhill"]))]
    for axes, value in readings:
        axes.plot([r["iteration"] for r in reference], [value(r) for r in reference], color="#888888", linewidth=1.5, linestyle=":", marker="o", markersize=3.0, label="original run")
        for name, _, colour in ARMS:
            scores = results[name]
            axes.plot([r["iteration"] for r in scores], [value(r) for r in scores], color=colour, linewidth=1.8, marker="o", markersize=3.0, label=name)

    left.set_ylabel("hold reward, 1-4 s")
    left.set_title("Does the hold survive past iteration 400?\nup is better")
    right.axhline(0.0, color="#888888", linewidth=1.0, linestyle=":")
    right.set_ylabel("final error, signed (cm)\nmean of uphill and downhill targets")
    right.set_title("Where the box ends up")
    for axes in (left, right):
        axes.set_xlabel("iteration")
        axes.legend(frameon=False, fontsize=8)
        axes.grid(alpha=0.25)
        axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--skip-train", action="store_true", help="rescore the checkpoints already in --runs")
    parser.add_argument("--start", type=pathlib.Path, default=ROOT / "runs" / "shac_diagnosis" / "checkpoints" / "00400.pt")
    parser.add_argument("--runs", type=pathlib.Path, default=ROOT / "runs" / "shac_restart")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "figures" / "shac_restart.png")
    args = parser.parse_args()

    if not args.skip_train:
        # Run the arms as separate processes. Spawn rather than fork, so no
        # process inherits another's GRIP or torch state.
        started = time.time()
        context = multiprocessing.get_context("spawn")
        workers = [context.Process(target=run_arm, args=(name, overrides, args.start, args.runs, args.iterations)) for name, overrides, _ in ARMS]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
            if worker.exitcode != 0:
                raise SystemExit(f"an arm exited with code {worker.exitcode}; see its train.log under {args.runs}")
        print(f"trained {len(ARMS)} arms in {time.time() - started:.0f} s")

    results = rescore(args.start, args.runs)
    original = json.loads((ROOT / "runs" / "shac_diagnosis" / "scores.json").read_text())
    (args.runs / "scores.json").write_text(json.dumps(results, indent=2))

    report(results, original)
    plot(results, original, args.out)
    print(f"\nwrote {args.out}")
    print(f"wrote {args.runs}")


if __name__ == "__main__":
    main()
