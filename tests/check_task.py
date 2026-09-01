"""Checks for the step-2 task machinery.  Run:  python tests/check_task.py

Not an experiment and not an artifact -- nothing here produces a figure.
These are the failures that have to be ruled out before a training loop
is worth writing, because every one of them looks like "RL is hard" from
the outside rather than like a bug.

The two that matter most:

  * The gradient seeds are finite-differenced against the reward they
    claim to differentiate. A wrong seed raises nothing; it silently
    optimizes something else.
  * The total derivative from `adjoint_batch` is finite-differenced
    through GRIP end to end. MPPI at step 4 cannot cover this -- it is
    zeroth order and never calls the adjoint -- so without it a
    struggling SHAC leaves you unable to tell a bad gradient from a bad
    hyperparameter.
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import grip  # noqa: E402
from slope_control import ramp, task  # noqa: E402


def corner_depths(state, ramp_angles, side=ramp.BOX_SIDE):
    """Signed distance of both bottom corners, downhill first. Negative is into the ramp."""
    angles = np.atleast_1d(ramp_angles)
    centre, theta = state[:, 0, 0:2], state[:, 0, 2]
    cos, sin = np.cos(theta), np.sin(theta)
    h = 0.5 * side
    depths = []
    for vx, vy in [(-h, -h), (h, -h)]:
        world = np.stack([centre[:, 0] + cos * vx - sin * vy, centre[:, 1] + sin * vx + cos * vy], axis=1)
        depths.append(np.einsum("ni,ni->n", world, ramp.normal(angles)))
    return depths


def check_flat_ground():
    """At a = 0 everything must reduce to flat ground, signs included."""
    assert np.allclose(ramp.uphill(0.0), [1.0, 0.0])
    assert np.allclose(ramp.normal(0.0), [0.0, 1.0])
    assert math.isclose(ramp.scene_angle(ramp.make_scene(ramp_angle=0.0)), 0.0, abs_tol=1e-12)
    assert math.isclose(ramp.creep_rate(0.0), 0.0, abs_tol=1e-15)
    assert math.isclose(ramp.hold_force(0.0), 0.0, abs_tol=1e-15)

    # xi collapses to world x, and a ramp-frame action to a world force.
    state = ramp.resting_state([0.7], [0.0])
    assert np.allclose(ramp.along_ramp(state, [0.0]), [[0.7]])
    assert np.allclose(state[0, 0, 1], 0.5 * ramp.BOX_SIDE)
    assert np.allclose(task.to_wrench(np.array([[1.3, -0.4]]), [0.0])[0, 0], [1.3, -0.4, 0.0])

    # And the physical version: on flat ground there is nothing to creep towards.
    scene = ramp.make_scene(ramp_angle=0.0)
    substeps = task.substeps_for(scene)
    settled = task.settle([scene], state, substeps)
    trajectory = grip.rollout_batch([scene], settled, np.zeros((200, 1, 1, 3)), substeps=substeps)
    drift = abs(ramp.along_ramp(trajectory, [0.0])[-1, 0, 0] - ramp.along_ramp(settled, [0.0])[0, 0])
    print(f"  flat ground, 2 s of zero control: {1e9 * drift:.3f} nm of drift")
    assert drift < 1e-9, drift


def check_scene_angles():
    """Scenes and the angles that built them must not disagree."""
    angles = np.linspace(*task.RAMP_ANGLE_RANGE, 7)
    recovered = np.array([ramp.scene_angle(s) for s in ramp.make_scenes(angles)])
    worst = np.abs(recovered - angles).max()
    print(f"  7 scenes, angle recovered from the plane normal: worst error {math.degrees(worst):.2e} deg")
    assert worst < 1e-12, worst


def check_substeps():
    """Control must land at exactly 100 Hz, not near it."""
    scene = ramp.make_scene()
    substeps = task.substeps_for(scene)
    assert substeps == 20, substeps
    assert math.isclose(1.0 / (substeps * scene.dt), task.CONTROL_HZ)
    print(f"  dt = {scene.dt:.1e}, substeps = {substeps}, control rate {1.0 / (substeps * scene.dt):.1f} Hz")


def check_settle_window():
    """The settle window has to actually reach steady state.

    Guards SETTLE_SECONDS against a change in k, b or the box: if the
    contact spring gets softer or less damped, the window silently stops
    being long enough and the first control steps of every episode become
    a transient.
    """
    angles = np.radians([15.0, 20.0, 22.0])
    scenes = ramp.make_scenes(angles)
    substeps = task.substeps_for(scenes[0])

    settled = task.settle(scenes, ramp.resting_state(0.0, angles), substeps)
    later = np.array(grip.rollout_batch(scenes, settled, np.zeros((200, len(scenes), 1, 3)), substeps=substeps)[-1])

    v_settled = np.einsum("ni,ni->n", settled[:, 0, 3:5], ramp.uphill(angles))
    v_steady = np.einsum("ni,ni->n", later[:, 0, 3:5], ramp.uphill(angles))
    error = np.abs(v_settled - v_steady) / np.abs(v_steady)

    downhill, uphill_corner = corner_depths(settled, angles)
    for i, degrees in enumerate(np.degrees(angles)):
        mean = -0.5e3 * (downhill[i] + uphill_corner[i])
        closed = 1e3 * ramp.BOX_MASS * ramp.GRAVITY * math.cos(angles[i]) / (2.0 * ramp.DEFAULT_PENALTY["stiffness"])
        print(f"  {degrees:.0f} deg  corners {-1e3 * downhill[i]:.3f} / {-1e3 * uphill_corner[i]:.3f} mm  mean {mean:.3f} vs mg cos(a)/2k = {closed:.3f} mm   creep settled to {100 * error[i]:.2f}%")
    assert error.max() < 0.01, error


def check_reward_seeds():
    """The seeds must differentiate the reward that sits next to them."""
    rng = np.random.default_rng(0)
    n_envs, steps = 3, 5
    angles = rng.uniform(*task.RAMP_ANGLE_RANGE, size=n_envs)
    targets = rng.uniform(-0.5, 0.5, size=n_envs)
    states = rng.normal(size=(steps + 1, n_envs, 1, 6))
    controls = rng.normal(size=(steps, n_envs, 1, 3))

    dl_dZ, dl_dU = task.reward_seeds(states, controls, angles, targets)
    worst = 0.0
    for array, analytic in [(states, dl_dZ), (controls, dl_dU)]:
        for index in np.ndindex(array.shape):
            saved, h = array[index], 1e-6
            array[index] = saved + h
            plus = task.reward(states, controls, angles, targets).sum()
            array[index] = saved - h
            minus = task.reward(states, controls, angles, targets).sum()
            array[index] = saved
            worst = max(worst, abs((plus - minus) / (2.0 * h) - analytic[index]))

    scale = max(np.abs(dl_dZ).max(), np.abs(dl_dU).max())
    print(f"  {states.size + controls.size} partials finite-differenced: worst error {worst:.2e} against scale {scale:.2e}")
    assert worst < 1e-6 * scale, worst


def check_adjoint():
    """dJ_dU from GRIP, finite-differenced through the real dynamics.

    The seeds passing on their own does not mean the chain is right: the
    index offset between states and controls, and the direction of the
    projection, both live here rather than in the reward.
    """
    rng = np.random.default_rng(1)
    n_envs, steps, probes = 2, 12, 10
    angles = rng.uniform(*task.RAMP_ANGLE_RANGE, size=n_envs)
    scenes = ramp.make_scenes(angles)
    substeps = task.substeps_for(scenes[0])

    state = task.settle(scenes, ramp.resting_state([0.1, 0.3], angles), substeps)
    targets = ramp.along_ramp(state, angles)[:, 0] + np.array([0.5, -0.5])
    # Well inside MAX_FORCE, so the clip is not what is under test here.
    wrenches = task.to_wrench(rng.uniform(-1.5, 1.5, size=(steps, n_envs, 2)), angles)

    trajectory = np.array(grip.rollout_batch(scenes, state, wrenches, substeps=substeps))
    dl_dZ, dl_dU = task.reward_seeds(trajectory, wrenches, angles, targets)
    _, dJ_dU = grip.adjoint_batch(scenes, trajectory, wrenches, substeps, dl_dZ, dl_dU)

    def total(controls):
        rolled = grip.rollout_batch(scenes, state, controls, substeps=substeps)
        return task.reward(rolled, controls, angles, targets).sum()

    indices = [(rng.integers(steps), rng.integers(n_envs), 0, rng.integers(3)) for _ in range(probes)]
    worst = 0.0
    for index in indices:
        saved, h = wrenches[index], 1e-5
        wrenches[index] = saved + h
        plus = total(wrenches)
        wrenches[index] = saved - h
        minus = total(wrenches)
        wrenches[index] = saved
        finite = (plus - minus) / (2.0 * h)
        worst = max(worst, abs(finite - dJ_dU[index]) / max(abs(finite), 1e-9))

    print(f"  {probes} components of dJ_dU vs central differences through {steps * substeps} integration steps: worst relative error {worst:.2e}")
    assert worst < 1e-4, worst


def check_observation():
    """Shape, and the a = 0 reduction. This is all that exercises `observe`."""
    angles = np.array([0.0, math.radians(20.0)])
    state = ramp.resting_state([0.4, 0.4], angles)
    targets = np.array([1.0, 1.0])
    observation = task.observe(state, angles, targets)
    assert observation.shape == (2, 1, 7), observation.shape

    flat = observation[0, 0]
    assert np.allclose(flat[0], (0.4 - 1.0) / ramp.BOX_SIDE)  # error in box widths
    assert np.allclose(flat[2:], [0.0, 0.0, 0.0, 0.0, 0.0])   # flush, at rest, on the flat
    # Slope-invariance: the same placement on a tilted ramp reads identically
    # except for the gravity load in the last entry.
    assert np.allclose(observation[0, 0, :6], observation[1, 0, :6])
    assert np.allclose(observation[1, 0, 6], math.sin(angles[1]))
    print(f"  observation {observation.shape}, slope-invariant except sin(a) = {observation[1, 0, 6]:.4f}")


def check_action_limit():
    """MAX_FORCE has to be small enough that the box cannot leave the ramp."""
    lift_off = ramp.BOX_MASS * ramp.GRAVITY * math.cos(task.RAMP_ANGLE_RANGE[1])
    hold = ramp.hold_force(task.RAMP_ANGLE_RANGE[1])
    print(f"  MAX_FORCE {task.MAX_FORCE:.1f} N   lift-off needs {lift_off:.2f} N   holding at 22 deg needs {hold:.2f} N")
    assert task.MAX_FORCE < lift_off, "the box could break contact and fly to the target"
    assert task.MAX_FORCE > 1.2 * hold, "not enough authority left over to move the box"

    saturated = task.to_wrench(np.array([[30.0, 40.0]]), [0.0])
    assert math.isclose(np.linalg.norm(saturated[0, 0, 0:2]), task.MAX_FORCE)

    # The physical version: push straight into free space as hard as allowed
    # and the box must stay in contact.
    angles = np.array([math.radians(20.0)])
    scenes = ramp.make_scenes(angles)
    substeps = task.substeps_for(scenes[0])
    state = task.settle(scenes, ramp.resting_state(0.0, angles), substeps)
    lifting = task.to_wrench(np.tile([0.0, task.MAX_FORCE], (100, 1, 1)), angles)
    trajectory = grip.rollout_batch(scenes, state, lifting, substeps=substeps)
    gap = max(corner_depths(np.array(trajectory[-1]), angles))
    print(f"  1 s of full outward force: deepest corner still {-1e3 * gap[0]:.3f} mm inside the ramp")
    assert gap[0] < 0.0, "the box left the surface"


def check_batch():
    """A sampled batch has to be internally consistent."""
    rng = np.random.default_rng(7)
    scenes, angles, state, targets = task.sample_batch(rng, 6)
    assert len(scenes) == 6 and state.shape == (6, 1, 6) and targets.shape == (6,)
    assert np.allclose([ramp.scene_angle(s) for s in scenes], angles)
    assert (angles >= task.RAMP_ANGLE_RANGE[0]).all() and (angles <= task.RAMP_ANGLE_RANGE[1]).all()
    assert (angles < ramp.friction_angle(ramp.DEFAULT_PENALTY["friction"])).all(), "a sampled slope is above the friction angle"

    offsets = targets - ramp.along_ramp(state, angles)[:, 0]
    assert (np.abs(offsets) >= task.TARGET_OFFSET_RANGE[0] - 1e-9).all()
    print(f"  6 environments, slopes {np.degrees(angles).min():.1f}-{np.degrees(angles).max():.1f} deg, {(offsets < 0).sum()} of 6 targets downhill")


def main():
    checks = [check_flat_ground, check_scene_angles, check_substeps, check_settle_window, check_reward_seeds, check_adjoint, check_observation, check_action_limit, check_batch]
    failures = 0
    for check in checks:
        print(f"{check.__name__}  --  {check.__doc__.splitlines()[0]}")
        try:
            check()
        except AssertionError as error:
            print(f"  FAIL: {error}")
            failures += 1
        print()

    print(f"{len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
