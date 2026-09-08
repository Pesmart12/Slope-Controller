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

import numpy as np

import grip
from slope_control import ramp, task


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
    assert np.allclose(task.to_wrench(np.array([[[1.3, -0.4]]]), [0.0], task.BOX_ONLY)[0, 0], [1.3, -0.4, 0.0])

    # And the physical version: on flat ground there is nothing to creep towards.
    scene = ramp.make_scene(ramp_angle=0.0)
    substeps = task.substeps_for(scene)
    settled = task.settle([scene], state, substeps)
    trajectory = grip.rollout_batch([scene], settled, np.zeros((200, 1, 1, 3)), substeps=substeps)
    drift = abs(ramp.along_ramp(trajectory, [0.0])[-1, 0, 0] - ramp.along_ramp(settled, [0.0])[0, 0])
    print(f"  flat ground, 2 s of zero control: {1e9 * drift:.3f} nm of drift")
    assert drift < 1e-9, drift


def check_body_geometry():
    """Mass properties and resting offsets, against what they are supposed to be.

    The inertias are derived from the outlines rather than written down, so
    what needs asserting is that the derivation agrees with the closed form
    where one exists, and that the trapezoid's centroid really did end up
    at the origin -- GRIP takes vertices in a COM-centred frame, and an
    un-recentred list puts a standing torque into every contact.
    """
    closed_form = ramp.BOX_MASS * ramp.BOX_SIDE ** 2 / 6.0
    print(f"  box inertia: derived {ramp.BOX['inertia']:.9f} vs m*s^2/6 = {closed_form:.9f}")
    assert math.isclose(ramp.BOX["inertia"], closed_form, rel_tol=1e-12)

    for name, shape in [("box", ramp.BOX), ("pusher", ramp.PUSHER), ("mirrored", ramp.PUSHER_MIRRORED)]:
        area, centroid, _ = ramp.polygon_properties(shape["vertices"])
        assert area > 0.0, f"{name} winds clockwise; GRIP's SAT path requires counterclockwise"
        assert np.abs(centroid).max() < 1e-12, (name, centroid)

    # Reflection preserves the polar moment, so the two pushers must match.
    assert math.isclose(ramp.PUSHER["inertia"], ramp.PUSHER_MIRRORED["inertia"], rel_tol=1e-12)

    # The claim resting_state's docstring makes, in numbers.
    offsets = {name: ramp.resting_offset(s["vertices"]) for name, s in [("box", ramp.BOX), ("pusher", ramp.PUSHER)]}
    print(f"  resting offset: box {1e3 * offsets['box']:.3f} mm (= half side), pusher {1e3 * offsets['pusher']:.3f} mm (NOT half height)")
    assert math.isclose(offsets["box"], 0.5 * ramp.BOX_SIDE, rel_tol=1e-12)
    assert not math.isclose(offsets["pusher"], 0.5 * ramp.PUSHER_HEIGHT, rel_tol=1e-6), "the face trim no longer moves the centroid"

    # Both pushers must reach the box by the same amount, or the two sides
    # of a symmetric placement are not symmetric.
    low = ramp.contact_reach(ramp.PUSHER, toward_uphill=True)
    high = ramp.contact_reach(ramp.PUSHER_MIRRORED, toward_uphill=False)
    print(f"  contact reach: lower {1e3 * low:.3f} mm, upper {1e3 * high:.3f} mm")
    assert math.isclose(low, high, rel_tol=1e-12)

    # And the contact vertex sits at the box's centre of mass height, which
    # is what makes the push exert no tipping moment on the box.
    height = ramp.resting_offset(ramp.PUSHER["vertices"]) + max(vy for _, vy in ramp.PUSHER["vertices"])
    print(f"  contact vertex {1e3 * height:.3f} mm above the ramp, box COM at {1e3 * 0.5 * ramp.BOX_SIDE:.3f} mm")
    assert math.isclose(height, 0.5 * ramp.BOX_SIDE, rel_tol=1e-12)


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
    """The seeds must differentiate the reward that sits next to them.

    Run for both variants, because the two-pusher reward scores a
    different body than it actuates -- so a seed written against the wrong
    index is a mistake the box-only case structurally cannot make.
    """
    for name, variant in [("box only ", task.BOX_ONLY), ("2 pushers", task.TWO_PUSHERS)]:
        for shaping in [False, True]:
            rng = np.random.default_rng(0)
            n_envs, steps = 3, 5
            angles = rng.uniform(*task.RAMP_ANGLE_RANGE, size=n_envs)
            targets = rng.uniform(-0.5, 0.5, size=n_envs)
            states = rng.normal(size=(steps + 1, n_envs, variant.bodies, 6))
            controls = rng.normal(size=(steps, n_envs, variant.bodies, 3))

            dl_dZ, dl_dU = task.reward_seeds(states, controls, angles, targets, variant, shaping=shaping)
            worst = 0.0
            for array, analytic in [(states, dl_dZ), (controls, dl_dU)]:
                for index in np.ndindex(array.shape):
                    saved, h = array[index], 1e-6
                    array[index] = saved + h
                    plus = task.reward(states, controls, angles, targets, variant, shaping=shaping).sum()
                    array[index] = saved - h
                    minus = task.reward(states, controls, angles, targets, variant, shaping=shaping).sum()
                    array[index] = saved
                    worst = max(worst, abs((plus - minus) / (2.0 * h) - analytic[index]))

            scale = max(np.abs(dl_dZ).max(), np.abs(dl_dU).max())
            label = "shaped  " if shaping else "unshaped"
            print(f"  {name} {label}: {states.size + controls.size:>4} partials, worst error {worst:.2e} against scale {scale:.2e}")
            assert worst < 1e-6 * scale, (name, shaping, worst)


def check_shaping_vanishes():
    """The approach term must be exactly zero once the pushers touch.

    That is what keeps it from distorting behaviour in the regime the task
    objective actually cares about -- it buys a gradient where there was
    none and then gets out of the way.
    """
    variant = task.TWO_PUSHERS
    bodies = task.bodies_for(variant)
    angles = np.radians([15.0, 20.0, 22.0])

    # Overlapping, not merely adjacent: a zero placement sits exactly on
    # the knife edge, where max(0, 1e-17)^2 is small but not bitwise zero.
    # Real contact means the separation is negative, and there the term has
    # to be off entirely rather than just nearly off.
    for gap, expect_zero in [(-0.001, True), (0.03, False)]:
        positions = task.placement(np.zeros(len(angles)), np.full((len(angles), 2), gap), variant)
        state = ramp.resting_state(positions, angles, bodies=bodies)
        penalty = task.approach_penalty(state, angles, variant)
        gaps = np.array(task.separations(state, angles, variant)).ravel()
        print(f"  {100 * gap:>5.1f} cm placement: separations {np.round(1e3 * gaps, 2)} mm, penalty {penalty.max():.4f}")
        assert (penalty.max() == 0.0) == expect_zero, (gap, penalty.max())

    # And it contributes nothing at all to the box-only variant.
    assert task.pushing_bodies(task.BOX_ONLY) == []
    state = ramp.resting_state([0.0], [0.35])
    assert task.approach_penalty(state, [0.35], task.BOX_ONLY).max() == 0.0
    print("  box only: no pushing bodies, so the term is structurally absent")


def check_adjoint():
    """dJ_dU from GRIP, finite-differenced through the real dynamics.

    The seeds passing on their own does not mean the chain is right: the
    index offset between states and controls, and the direction of the
    projection, both live here rather than in the reward.

    Run for both variants. With two pushers the box is unactuated, so the
    whole path from a control to the scored position runs through a
    body-body contact -- a longer chain than the box-only case exercises,
    and the one the manipulation task actually depends on.
    """
    for name, variant in [("box only ", task.BOX_ONLY), ("2 pushers", task.TWO_PUSHERS)]:
        worst = probe_adjoint(variant)
        print(f"  {name}: {variant.bodies} bodies, worst relative error {worst:.2e}")
        assert worst < 1e-4, (name, worst)


def probe_adjoint(variant):
    """Finite-difference a handful of dJ_dU entries for one variant."""
    rng = np.random.default_rng(1)
    n_envs, steps, probes = 2, 12, 10
    angles = rng.uniform(*task.RAMP_ANGLE_RANGE, size=n_envs)
    bodies = task.bodies_for(variant)
    scenes = ramp.make_scenes(angles, bodies=bodies)
    substeps = task.substeps_for(scenes[0])

    start = np.array([0.1, 0.3])
    positions = task.placement(start, 0.0, variant)
    state = task.settle(scenes, ramp.resting_state(positions, angles, bodies=bodies), substeps)
    targets = ramp.along_ramp(state, angles)[:, variant.box] + np.array([0.5, -0.5])
    # Well inside the limit, so the clip is not what is under test here.
    wrenches = task.to_wrench(rng.uniform(-1.5, 1.5, size=(steps, n_envs, len(variant.actuated), 2)), angles, variant)

    trajectory = np.array(grip.rollout_batch(scenes, state, wrenches, substeps=substeps))
    dl_dZ, dl_dU = task.reward_seeds(trajectory, wrenches, angles, targets, variant)
    _, dJ_dU = grip.adjoint_batch(scenes, trajectory, wrenches, substeps, dl_dZ, dl_dU)

    def total(controls):
        rolled = grip.rollout_batch(scenes, state, controls, substeps=substeps)
        return task.reward(rolled, controls, angles, targets, variant).sum()

    indices = [(rng.integers(steps), rng.integers(n_envs), rng.integers(variant.bodies), rng.integers(3)) for _ in range(probes)]
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

    return worst


def check_observation():
    """Shape, and the a = 0 reduction. This is all that exercises `observe`."""
    angles = np.array([0.0, math.radians(20.0)])
    state = ramp.resting_state([0.4, 0.4], angles)
    targets = np.array([1.0, 1.0])
    observation = task.observe(state, angles, targets, task.BOX_ONLY)
    assert observation.shape == (2, 7), observation.shape

    flat = observation[0]
    assert np.allclose(flat[0], (0.4 - 1.0) / ramp.BOX_SIDE)  # error in box widths
    assert np.allclose(flat[2:], [0.0, 0.0, 0.0, 0.0, 0.0])   # flush, at rest, on the flat
    # Slope-invariance: the same placement on a tilted ramp reads identically
    # except for the gravity load in the last entry.
    assert np.allclose(observation[0, :6], observation[1, :6])
    assert np.allclose(observation[1, 6], math.sin(angles[1]))

    # Two pushers: the box's seven, then four per pusher, relative to it.
    pushers = task.TWO_PUSHERS
    bodies = task.bodies_for(pushers)
    placed = task.placement(np.full(2, 0.4), np.zeros((2, 2)), pushers)
    wide = task.observe(ramp.resting_state(placed, angles, bodies=bodies), angles, targets, pushers)
    assert wide.shape == (2, 15), wide.shape
    assert np.allclose(wide[:, 0:7], observation)  # the box block is unchanged
    # Pushers sit either side, touching, so their relative xi is symmetric.
    assert np.allclose(wide[:, 7], -wide[:, 11])
    assert np.allclose(wide[:, [9, 10, 13, 14]], 0.0)  # flush, level, at rest

    print(f"  box only {observation.shape}, two pushers {wide.shape}; slope-invariant except sin(a) = {observation[1, 6]:.4f}")
    print(f"  pushers read at {wide[0, 7]:+.4f} and {wide[0, 11]:+.4f} box widths from the box")


def check_action_limit():
    """The limit must let the box move, and still not let it leave the ramp.

    Two bounds from different physics, which is why the limit is a box
    and not a magnitude. Sizing one alone is how an earlier 5 N limit came
    out below the force needed to move the box at all, leaving the task
    unsolvable until `check_trajopt.py` reported it.
    """
    steep = task.RAMP_ANGLE_RANGE[1]
    tangential, perpendicular = task.BOX_ONLY.limit
    break_free = ramp.break_free_force(steep)
    load = ramp.normal_load(steep)

    print(f"  at {math.degrees(steep):.0f} deg: break free needs {break_free:.2f} N, tangential limit {tangential:.1f} N ({tangential / break_free:.2f}x)")
    print(f"  {' ':>15}lift-off needs {load:.2f} N, normal limit {perpendicular:.1f} N (leaves lambda >= {load - perpendicular:.2f} N)")
    assert tangential > break_free, "the box cannot be moved at all"
    assert perpendicular < load, "an outward push could peel the box off the ramp"

    assert np.allclose(task.to_wrench(np.array([[[99.0, -99.0]]]), [0.0], task.BOX_ONLY)[0, 0], [tangential, -perpendicular, 0.0])

    angles = np.array([math.radians(20.0)])
    scenes = ramp.make_scenes(angles)
    substeps = task.substeps_for(scenes[0])
    state = task.settle(scenes, ramp.resting_state(0.0, angles), substeps)

    # Push straight out as hard as allowed: the box must stay in contact.
    lifting = task.to_wrench(np.tile([0.0, perpendicular], (100, 1, 1, 1)), angles, task.BOX_ONLY)
    gap = max(corner_depths(np.array(grip.rollout_batch(scenes, state, lifting, substeps=substeps)[-1]), angles))
    assert gap[0] < 0.0, "the box left the surface"

    # Push along the ramp as hard as allowed: the box must actually go
    # somewhere, and must not tip while doing it.
    driving = task.to_wrench(np.tile([tangential, 0.0], (task.EPISODE_STEPS, 1, 1, 1)), angles, task.BOX_ONLY)
    trajectory = np.array(grip.rollout_batch(scenes, state, driving, substeps=substeps))
    moved = ramp.along_ramp(trajectory, angles)[-1, 0, 0] - ramp.along_ramp(state, angles)[0, 0]
    tilt = np.abs(trajectory[:, 0, 0, 2] - angles[0]).max()
    print(f"  full outward push: {-1e3 * gap[0]:.3f} mm still inside the ramp   full uphill push: {moved:.2f} m in an episode, {1e3 * tilt:.2f} mrad of tilt")
    assert moved > 4.0 * task.TARGET_OFFSET_RANGE[1], "not enough authority to reach a far target and brake"
    assert tilt < 0.05, "the box tipped"


def check_batch():
    """A sampled batch has to be internally consistent."""
    rng = np.random.default_rng(7)
    scenes, angles, state, targets = task.sample_batch(rng, 6, task.BOX_ONLY)
    assert len(scenes) == 6 and state.shape == (6, 1, 6) and targets.shape == (6,)
    assert np.allclose([ramp.scene_angle(s) for s in scenes], angles)
    assert (angles >= task.RAMP_ANGLE_RANGE[0]).all() and (angles <= task.RAMP_ANGLE_RANGE[1]).all()
    assert (angles < ramp.friction_angle(ramp.DEFAULT_PENALTY["friction"])).all(), "a sampled slope is above the friction angle"

    offsets = targets - ramp.along_ramp(state, angles)[:, 0]
    assert (np.abs(offsets) >= task.TARGET_OFFSET_RANGE[0] - 1e-9).all()
    print(f"  6 environments, slopes {np.degrees(angles).min():.1f}-{np.degrees(angles).max():.1f} deg, {(offsets < 0).sum()} of 6 targets downhill")


def main():
    checks = [check_flat_ground, check_body_geometry, check_scene_angles, check_substeps, check_settle_window, check_reward_seeds, check_shaping_vanishes, check_adjoint, check_observation, check_action_limit, check_batch]
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
