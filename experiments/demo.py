"""Watch a trained policy work, and watch penalty creep undo it.

Renders one episode per checkpoint as a GIF, and a companion figure of
position error and commanded force against time. The GIF is the demo; the
figure is what makes the creep legible, because 2.8 cm of drift is under a
tenth of a box width and is easy to miss on the animation alone.

Two checkpoints, because they show different things. The lowest-error one
holds near the target and is the policy at its best. The last one drifts,
and is where the creep regime shows up plainly. Both are the same training
run and neither is cherry-picked -- the second is simply what the first
becomes.

Scores the SAME fixed configuration for both, so the two GIFs differ only
by the policy. Unshaped throughout; the shaping term is a training device
and never appears in a reported number.

Also writes `demo.json`: every policy crossed with every scenario, as
poses rather than states, for the interactive page to play back. That page
can do the two things a GIF cannot -- scrub the hold window in slow motion,
and run the two policies side by side on one configuration.

Run:  python experiments/demo.py
"""

import argparse
import json
import pathlib

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

from slope_control import batches, observation, policy, ramp, render, sweep, task

# One representative slope from the middle of the sampled range, a target
# half a metre uphill and the 5 cm standoff the baseline uses. Fixed rather
# than sampled so the demo is the same episode every time it is rendered.
DEMO_ANGLE = 20.0
DEMO_OFFSET = 0.5
DEMO_GAP = 0.05

# Where the hold is measured from. The policy has arrived well before this
# on every checkpoint seen, so the window holds only the creep.
HOLD_FROM = 150


# The configurations the interactive page can switch between. Two slopes
# from the ends of the sampled range and one from the middle, and a target
# on each side of the start -- downhill is the case the second pusher
# exists for, and it is nine times cheaper to travel, so it should look
# different.
SCENARIOS = [("20 deg, 50 cm uphill", 20.0, 0.5), ("15 deg, 50 cm uphill", 15.0, 0.5),
             ("22 deg, 50 cm uphill", 22.0, 0.5), ("20 deg, 50 cm downhill", 20.0, -0.5)]


def episode(checkpoint, angle=DEMO_ANGLE, offset=DEMO_OFFSET):
    """Roll one deterministic episode of a fixed configuration.

    Returns the batch, the window, and the iteration the checkpoint came
    from.
    """
    batch = batches.fixed_batch(np.radians([angle]), offset=offset, gaps=DEMO_GAP)
    observation_size = observation.observe(batch.state, batch.angles, batch.targets).shape[-1]

    saved = torch.load(checkpoint, weights_only=True)
    actor = policy.Actor(observation_size, len(task.ACTUATED), task.LIMIT)
    actor.load_state_dict(saved["actor"])

    with torch.no_grad():
        rolled = sweep.rollout(batch, actor, task.EPISODE_STEPS, deterministic=True)
    return batch, rolled, saved["iteration"]


def measure(batch, rolled):
    """The numbers the demo is showing. Returns a dict, all in SI.

    `arrival` is the first step within one box side of the target, which is
    where the approach ends and the hold begins. `drift` is signed along
    the ramp over the hold window, so a negative value is downhill.
    """
    xi = ramp.along_ramp(rolled.states, batch.angles)[:, 0, task.BOX]
    signed = xi - batch.targets[0]

    # Sum the force magnitudes the actuated bodies were commanded, one
    # entry per control step.
    force = sum(np.linalg.norm(rolled.wrenches[:, 0, body, 0:2], axis=-1) for body in task.ACTUATED)

    # First step inside a box side of the target. `argmax` on a boolean
    # array gives the first True, and the `any` guards the case where the
    # policy never got there.
    within = np.abs(signed) < ramp.BOX_SIDE
    arrival = int(np.argmax(within)) if within.any() else len(signed) - 1

    seconds = (len(signed) - 1 - HOLD_FROM) / task.CONTROL_HZ
    return dict(signed=signed, force=force, arrival=arrival,
                arrival_error=float(signed[arrival]), final_error=float(signed[-1]),
                best_error=float(signed[np.abs(signed).argmin()]),
                drift=float((signed[-1] - signed[HOLD_FROM]) / seconds),
                hold_force=float(force[HOLD_FROM:].mean()))


def export(checkpoints, path):
    """Write every scenario and policy to one JSON file for the web demo.

    Poses rather than states: the page draws bodies, so it needs (x, y,
    theta) and none of the velocity columns. Rounded to a tenth of a
    millimetre, which is finer than the creep this is meant to show and
    keeps the file small enough to embed.
    """
    payload = dict(controlHz=task.CONTROL_HZ, boxSide=ramp.BOX_SIDE, boxIndex=task.BOX,
                   bodies=[dict(vertices=[[round(vx, 5), round(vy, 5)] for vx, vy in body["vertices"]]) for body in task.BODIES],
                   episodes=[])

    for label, checkpoint in checkpoints.items():
        for title, angle, offset in SCENARIOS:
            batch, rolled, iteration = episode(checkpoint, angle=angle, offset=offset)
            measured = measure(batch, rolled)

            # states is (steps + 1, environments, bodies, 6); take the one
            # environment and columns 0:3, which are x, y and theta.
            poses = rolled.states[:, 0, :, 0:3]
            payload["episodes"].append(dict(
                policy=label, scenario=title, iteration=iteration,
                rampAngle=float(batch.angles[0]), target=float(batch.targets[0]),
                holdForce=float(ramp.hold_force(batch.angles[0])),
                breakFree=float(ramp.break_free_force(batch.angles[0], mass=ramp.BOX_MASS + ramp.PUSHER_MASS)),
                creepRate=float(ramp.creep_rate(batch.angles[0])),
                arrival=measured["arrival"], drift=measured["drift"],
                poses=np.round(poses, 5).tolist(),
                error=np.round(measured["signed"], 5).tolist(),
                force=np.round(measured["force"], 3).tolist()))

    path.write_text(json.dumps(payload, separators=(",", ":")))
    return payload


def plot(axes_error, axes_force, runs):
    """Draw error and commanded force against time, one line per checkpoint."""
    colours = {"best": "#2b6cb0", "final": "#c1442e"}
    for label, (_, measured, iteration) in runs.items():
        colour = colours.get(label, "#4a5568")
        signed, force = measured["signed"], measured["force"]
        time = np.arange(len(signed)) / task.CONTROL_HZ

        axes_error.plot(time, 100 * signed, color=colour, linewidth=2.0, label=f"{label}, iteration {iteration}")
        axes_force.plot(time[:-1], force, color=colour, linewidth=2.0, label=f"{label}, iteration {iteration}")

    axes_error.axhline(0.0, color="#888888", linewidth=1.0, linestyle=":")
    axes_error.set_xlabel("time (s)")
    axes_error.set_ylabel("position error along ramp (cm)")
    axes_error.set_title("Arrives, then slides back\nnegative is downhill of target")
    axes_error.legend(frameon=False, fontsize=9)

    # Mark the two forces that bound the creep regime. Between them the box
    # can only ooze: above the lower one it is held, below the upper one it
    # cannot be driven.
    angle = np.radians(DEMO_ANGLE)
    axes_force.axhline(ramp.hold_force(angle), color="#888888", linewidth=1.0, linestyle=":")
    axes_force.axhline(ramp.break_free_force(angle, mass=ramp.BOX_MASS + ramp.PUSHER_MASS), color="#888888", linewidth=1.0, linestyle="--")
    axes_force.set_xlabel("time (s)")
    axes_force.set_ylabel("commanded force (N)")
    axes_force.set_title("Inside the creep band\ndotted: holds the box   dashed: breaks it free")
    axes_force.legend(frameon=False, fontsize=9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = pathlib.Path(__file__).resolve().parents[1]
    parser.add_argument("--runs", type=pathlib.Path, default=root / "runs" / "demo")
    parser.add_argument("--arm", default="unclipped")
    parser.add_argument("--figures", type=pathlib.Path, default=root / "figures")
    args = parser.parse_args()

    checkpoints = {"best": args.runs / f"{args.arm}-best.pt", "final": args.runs / f"{args.arm}.pt"}
    missing = [str(path) for path in checkpoints.values() if not path.exists()]
    if missing:
        raise SystemExit(f"no checkpoint at {', '.join(missing)} -- run experiments/train_shac.py first")

    args.figures.mkdir(parents=True, exist_ok=True)
    runs = {}
    for label, path in checkpoints.items():
        batch, rolled, iteration = episode(path)
        measured = measure(batch, rolled)
        runs[label] = (batch, measured, iteration)

        gif = args.figures / f"demo_{label}.gif"
        frames = render.animate(rolled.states[:, 0], float(batch.angles[0]), float(batch.targets[0]), task.BODIES, task.BOX,
                                gif, title=f"SHAC on penalty contact -- {label}, iteration {iteration}")

        print(f"\n{label}: iteration {iteration}")
        print(f"  arrives within a box side at {measured['arrival'] / task.CONTROL_HZ:.2f} s")
        print(f"  closest approach {100 * measured['best_error']:>6.2f} cm")
        print(f"  final error      {100 * measured['final_error']:>6.2f} cm")
        print(f"  hold-window drift {100 * measured['drift']:>5.2f} cm/s against a closed form of "
              f"{100 * ramp.creep_rate(batch.angles[0]):.2f} cm/s")
        print(f"  mean force while holding {measured['hold_force']:.2f} N, between "
              f"{ramp.hold_force(batch.angles[0]):.2f} N and "
              f"{ramp.break_free_force(batch.angles[0], mass=ramp.BOX_MASS + ramp.PUSHER_MASS):.2f} N")
        print(f"  wrote {gif} ({frames} frames)")

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    plot(left, right, runs)
    for axes in (left, right):
        axes.grid(alpha=0.25)
        axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()

    out = args.figures / "demo_creep.png"
    figure.savefig(out, dpi=150)
    print(f"\nwrote {out}")

    data = args.runs / "demo.json"
    payload = export(checkpoints, data)
    print(f"wrote {data} ({len(payload['episodes'])} episodes, {data.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    raise SystemExit(main())
