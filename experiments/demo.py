"""Build the data the interactive demo page replays.

Rolls one deterministic episode for every point of a grid -- slope,
target offset and training checkpoint -- and writes them compactly for
`demo/index.html`. The page cannot run GRIP, so its sliders pick among
these precomputed episodes.

Writes two files next to the page:

    demo/episodes.txt   int16, little-endian, base64: every pose, then every
                        force. Text because the artifact host serves no raw
                        binary files
    demo/meta.json      the grid axes, shapes, scale factors, body outlines,
                        targets and a camera window per scenario

Poses are (x, y, theta) per body per step, in units of POSE_SCALE. Forces
are the two pushers' ramp-frame actions, (uphill, outward normal),
in units of FORCE_SCALE. Unshaped, deterministic, the same policy
`diagnose_shac.py` scores.

Run:  python experiments/demo.py
"""

import argparse
import base64
import json
import pathlib

import numpy as np
import torch

from slope_control import batches, observation, policy, ramp, sweep, task

ROOT = pathlib.Path(__file__).resolve().parents[1]

SLOPES_DEG = [15, 16, 17, 18, 19, 20, 21, 22]

# Target offsets from the box's settled start, in metres. Inside the range
# training samples from, `batches.TARGET_OFFSET_RANGE`, on both sides.
OFFSETS = [-1.0, -0.85, -0.7, -0.55, -0.4, 0.4, 0.55, 0.7, 0.85, 1.0]

CHECKPOINTS = [1, 100, 200, 400, 700, 1000, 1500, 2000]

# The standoff each pusher starts at, as in training's typical approach.
GAP = 0.05

# Quantization steps: 0.2 mm for positions, 1e-4 rad for angles. Finer
# than anything the page can draw, and int16 then holds positions out to
# 6.5 m -- early checkpoints push bodies well past 3 m.
POSE_SCALE = [2e-4, 2e-4, 1e-4]
FORCE_SCALE = 1e-2

# How far the camera window reaches past the trained policy's path.
CAMERA_MARGIN = 0.25


def load_actor(path, observation_size):
    """Rebuild the actor a checkpoint was saved from."""
    state = torch.load(path)
    actor = policy.Actor(observation_size, len(task.ACTUATED), task.LIMIT)
    actor.load_state_dict(state["actor"])
    return actor


def scenario_batch():
    """One environment per (slope, offset) pair, slope-major."""
    angles = np.radians([slope for slope in SLOPES_DEG for _ in OFFSETS])
    offsets = np.array([offset for _ in SLOPES_DEG for offset in OFFSETS])
    return batches.fixed_batch(angles, start=0.0, offset=offsets, gaps=GAP)


def roll(actor, batch):
    """Roll one deterministic episode on every environment. Returns poses
    (steps + 1, environments, bodies, 3) and actions (steps, environments,
    pushers, 2)."""
    with torch.no_grad():
        window = sweep.rollout(batch, actor, task.EPISODE_STEPS, deterministic=True)
    actions = np.stack([action.numpy() for action in window.actions])
    return window.states[..., 0:3], actions


def quantize(values, scale, name):
    """Round to integer units of `scale` as int16, refusing to overflow."""
    units = np.round(values / np.asarray(scale))
    if np.abs(units).max() > np.iinfo(np.int16).max:
        raise ValueError(f"{name} overflows int16 at scale {scale}: max {np.abs(values).max():.3f}")
    return units.astype("<i2")


def camera_windows(poses, batch):
    """A world rectangle per scenario covering the trained policy's path and
    the target. Returns (environments, 4) as (xmin, ymin, xmax, ymax).

    `poses` is the final checkpoint's (steps + 1, environments, bodies, 3).
    Body extents come from the outlines' largest radius, which is enough
    for a camera and avoids rotating every outline at every step.
    """
    radius = max(np.linalg.norm(np.asarray(body["vertices"]), axis=1).max() for body in task.BODIES)

    # Stack every body centre over the episode with each target's position:
    #   (steps + 1, environments, bodies, 2) -> (environments, points, 2)
    centres = poses[..., 0:2].transpose(1, 0, 2, 3).reshape(poses.shape[1], -1, 2)
    targets = batch.targets[:, None] * ramp.uphill(batch.angles)
    points = np.concatenate([centres, targets[:, None, :]], axis=1)

    reach = radius + CAMERA_MARGIN
    return np.concatenate([points.min(axis=1) - reach, points.max(axis=1) + reach], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=pathlib.Path, default=ROOT / "runs" / "shac_tol1cm_seed0")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "demo")
    args = parser.parse_args()

    batch = scenario_batch()
    observation_size = observation.observe(batch.state, batch.angles, batch.targets).shape[-1]

    # Roll every checkpoint on the same scenarios, then order the axes as
    # (checkpoint, environment, step, ...) so each episode is contiguous.
    all_poses, all_actions = [], []
    for iteration in CHECKPOINTS:
        poses, actions = roll(load_actor(args.runs / "checkpoints" / f"{iteration:05d}.pt", observation_size), batch)
        all_poses.append(poses.transpose(1, 0, 2, 3))
        all_actions.append(actions.transpose(1, 0, 2, 3))
    poses, actions = np.stack(all_poses), np.stack(all_actions)

    pose_units = quantize(poses, POSE_SCALE, "poses")
    force_units = quantize(actions, FORCE_SCALE, "forces")

    args.out.mkdir(parents=True, exist_ok=True)
    data = args.out / "episodes.txt"
    data.write_text(base64.b64encode(pose_units.tobytes() + force_units.tobytes()).decode("ascii"))

    final = poses[-1].transpose(1, 0, 2, 3)
    meta = dict(
        slopesDeg=SLOPES_DEG, offsets=OFFSETS, checkpoints=CHECKPOINTS,
        controlHz=task.CONTROL_HZ, steps=task.EPISODE_STEPS,
        bodies=[dict(vertices=np.round(body["vertices"], 5).tolist()) for body in task.BODIES],
        boxIndex=task.BOX, pushers=list(task.ACTUATED), boxSide=ramp.BOX_SIDE,
        tolerance=0.01,
        poseShape=list(pose_units.shape), poseScale=POSE_SCALE,
        forceShape=list(force_units.shape), forceScale=FORCE_SCALE, forceLimit=task.LIMIT.tolist(),
        targets=np.round(batch.targets, 5).tolist(),
        cameras=np.round(camera_windows(final, batch), 4).tolist())
    (args.out / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))

    # Decode what was written and require the trajectories back to within
    # half a quantization step.
    raw = np.frombuffer(base64.b64decode(data.read_text()), dtype="<i2")
    decoded = raw[:pose_units.size].reshape(pose_units.shape) * np.asarray(POSE_SCALE)
    worst = np.abs(decoded - poses).max()
    assert worst <= 0.5 * max(POSE_SCALE) + 1e-12, f"round trip off by {worst}"

    # Print where the trained policy parks, split by target direction.
    # poses[-1][:, -1] is the last checkpoint's final state, (environments,
    # bodies, 3), with the environment axis at -3 as `along_ramp` needs.
    xi = ramp.along_ramp(poses[-1][:, -1], batch.angles)[:, task.BOX]
    error = xi - batch.targets
    uphill = np.array([offset for _ in SLOPES_DEG for offset in OFFSETS]) > 0
    print(f"grid: {len(SLOPES_DEG)} slopes x {len(OFFSETS)} targets x {len(CHECKPOINTS)} checkpoints = {poses.shape[0] * poses.shape[1]} episodes")
    print(f"iteration {CHECKPOINTS[-1]} parks {100 * error[uphill].mean():+.2f} cm from uphill targets, {100 * error[~uphill].mean():+.2f} cm from downhill")
    print(f"round trip exact to {worst * 1000:.3f} mm")
    print(f"wrote {data} ({data.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {args.out / 'meta.json'} ({(args.out / 'meta.json').stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
