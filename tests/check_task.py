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
from slope_control import batches, objective, observation, ramp, task


def corner_depths(state, ramp_angles, body=0, vertices=None):
    """Signed distance of one body's vertices from the ramp, in vertex order.

    Negative is into the surface. The box's list winds from the bottom-left,
    so its first two entries are the bottom corners, downhill first.
    """
    vertices = ramp.BOX["vertices"] if vertices is None else vertices
    angles = ramp.as_angles(ramp_angles)
    centre, theta = state[:, body, 0:2], state[:, body, 2]
    cos, sin = np.cos(theta), np.sin(theta)
    depths = []
    for vx, vy in vertices:
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
    wrench = task.to_wrench(np.array([[[1.3, -0.4], [0.5, 0.2]]]), [0.0])[0]
    assert np.allclose(wrench[task.ACTUATED[0]], [1.3, -0.4, 0.0])
    assert np.allclose(wrench[task.ACTUATED[1]], [0.5, 0.2, 0.0])
    assert np.allclose(wrench[task.BOX], 0.0), "the box is unactuated and must never be written"

    # And the physical version: on flat ground there is nothing to creep towards.
    scene = ramp.make_scene(ramp_angle=0.0)
    substeps = task.substeps_for(scene)
    settled = batches.settle([scene], state, substeps)
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

    # Check the resting offsets against what `resting_state`'s docstring
    # claims: half a side for the box, and NOT half its height for the
    # pusher, whose centroid the face trim moved.
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
    angles = np.linspace(*batches.RAMP_ANGLE_RANGE, 7)
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

    settled = batches.settle(scenes, ramp.resting_state(0.0, angles), substeps)
    later = np.array(grip.rollout_batch(scenes, settled, np.zeros((200, len(scenes), 1, 3)), substeps=substeps)[-1])

    v_settled = np.einsum("ni,ni->n", settled[:, 0, 3:5], ramp.uphill(angles))
    v_steady = np.einsum("ni,ni->n", later[:, 0, 3:5], ramp.uphill(angles))
    error = np.abs(v_settled - v_steady) / np.abs(v_steady)

    downhill, uphill_corner = corner_depths(settled, angles)[:2]
    for i, degrees in enumerate(np.degrees(angles)):
        mean = -0.5e3 * (downhill[i] + uphill_corner[i])
        closed = 1e3 * ramp.BOX_MASS * ramp.GRAVITY * math.cos(angles[i]) / (2.0 * ramp.DEFAULT_PENALTY["stiffness"])
        print(f"  {degrees:.0f} deg  corners {-1e3 * downhill[i]:.3f} / {-1e3 * uphill_corner[i]:.3f} mm  mean {mean:.3f} vs mg cos(a)/2k = {closed:.3f} mm   creep settled to {100 * error[i]:.2f}%")
    assert error.max() < 0.01, error


def check_reward_seeds():
    """The seeds must differentiate the reward that sits next to them.

    Shaped and unshaped separately: the shaping term is the only thing that
    puts the pushers into `dl_dZ` at all, and it writes two entries with
    opposite signs where the position term writes one.
    """
    for shaping in [False, True]:
        rng = np.random.default_rng(0)
        n_envs, steps = 3, 5
        angles = rng.uniform(*batches.RAMP_ANGLE_RANGE, size=n_envs)
        targets = rng.uniform(-0.5, 0.5, size=n_envs)
        states = rng.normal(size=(steps + 1, n_envs, task.BODY_COUNT, 6))
        controls = rng.normal(size=(steps, n_envs, task.BODY_COUNT, 3))

        dl_dZ, dl_dU = objective.reward_seeds(states, controls, angles, targets, shaping=shaping)
        worst = 0.0
        for array, analytic in [(states, dl_dZ), (controls, dl_dU)]:
            for index in np.ndindex(array.shape):
                saved, h = array[index], 1e-6
                array[index] = saved + h
                plus = objective.reward(states, controls, angles, targets, shaping=shaping).sum()
                array[index] = saved - h
                minus = objective.reward(states, controls, angles, targets, shaping=shaping).sum()
                array[index] = saved
                worst = max(worst, abs((plus - minus) / (2.0 * h) - analytic[index]))

        scale = max(np.abs(dl_dZ).max(), np.abs(dl_dU).max())
        label = "shaped  " if shaping else "unshaped"
        print(f"  {label}: {states.size + controls.size:>4} partials, worst error {worst:.2e} against scale {scale:.2e}")
        assert worst < 1e-6 * scale, (shaping, worst)


def check_shaping_vanishes():
    """The approach term must be exactly zero once the pushers touch.

    That is what keeps it from distorting behaviour in the regime the task
    objective actually cares about -- it buys a gradient where there was
    none and then gets out of the way.
    """
    angles = np.radians([15.0, 20.0, 22.0])

    # Overlapping, not merely adjacent: a zero placement sits exactly on
    # the knife edge, where max(0, 1e-17)^2 is small but not bitwise zero.
    # Real contact means the separation is negative, and there the term has
    # to be off entirely rather than just nearly off.
    for gap, expect_zero in [(-0.001, True), (0.03, False)]:
        positions = batches.placement(np.zeros(len(angles)), np.full((len(angles), 2), gap))
        state = ramp.resting_state(positions, angles, bodies=task.BODIES)
        penalty = objective.approach_penalty(state, angles)
        gaps = np.array(objective.separations(state, angles)).ravel()
        print(f"  {100 * gap:>5.1f} cm placement: separations {np.round(1e3 * gaps, 2)} mm, penalty {penalty.max():.4f}")
        assert (penalty.max() == 0.0) == expect_zero, (gap, penalty.max())


def check_adjoint():
    """dJ_dU from GRIP, finite-differenced through the real dynamics.

    The seeds passing on their own does not mean the chain is right: the
    index offset between states and controls, and the direction of the
    projection, both live here rather than in the reward.

    The box is unactuated, so the whole path from a control to the scored
    position runs through a body-body contact. That is the chain the
    manipulation task depends on, and it is the long one.
    """
    worst = probe_adjoint()
    print(f"  {task.BODY_COUNT} bodies, worst relative error {worst:.2e}")
    assert worst < 1e-4, worst


def probe_adjoint():
    """Finite-difference a handful of dJ_dU entries."""
    rng = np.random.default_rng(1)
    n_envs, steps, probes = 2, 12, 10
    angles = rng.uniform(*batches.RAMP_ANGLE_RANGE, size=n_envs)
    batch = batches.fixed_batch(angles, start=np.array([0.1, 0.3]), offset=np.array([0.5, -0.5]))

    # Draw random actions in [-1.5, 1.5] N and rotate them into wrenches.
    # Well inside the limit, so the clip is not what is under test here.
    wrenches = task.to_wrench(rng.uniform(-1.5, 1.5, size=(steps, n_envs, len(task.ACTUATED), 2)), batch.angles)

    trajectory = np.array(grip.rollout_batch(batch.scenes, batch.state, wrenches, substeps=batch.substeps))
    dl_dZ, dl_dU = objective.reward_seeds(trajectory, wrenches, batch.angles, batch.targets)
    _, dJ_dU = grip.adjoint_batch(batch.scenes, trajectory, wrenches, batch.substeps, dl_dZ, dl_dU)

    def total(controls):
        rolled = grip.rollout_batch(batch.scenes, batch.state, controls, substeps=batch.substeps)
        return objective.reward(rolled, controls, batch.angles, batch.targets).sum()

    indices = [(rng.integers(steps), rng.integers(n_envs), rng.integers(task.BODY_COUNT), rng.integers(3)) for _ in range(probes)]
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
    targets = np.array([1.0, 1.0])

    # Everything touching, so the pushers' relative channels are symmetric.
    placed = batches.placement(np.full(2, 0.4), np.zeros((2, 2)))
    obs = observation.observe(ramp.resting_state(placed, angles, bodies=task.BODIES), angles, targets)

    # Seven channels for the box, then four for each other body.
    assert obs.shape == (2, 7 + 4 * (task.BODY_COUNT - 1)), obs.shape

    flat = obs[0]
    assert np.allclose(flat[0], (0.4 - 1.0) / ramp.BOX_SIDE)  # error in box widths
    assert np.allclose(flat[2:7], [0.0, 0.0, 0.0, 0.0, 0.0])  # flush, at rest, on the flat

    # Slope-invariance: the same placement on a tilted ramp reads identically
    # except for the gravity load in the last box entry.
    assert np.allclose(obs[0, :6], obs[1, :6])
    assert np.allclose(obs[1, 6], math.sin(angles[1]))

    # The pushers sit either side, touching, so their relative xi is
    # symmetric and their gap, tilt and speed are all zero.
    assert np.allclose(obs[:, 7], -obs[:, 11])
    assert np.allclose(obs[:, [9, 10, 13, 14]], 0.0)

    print(f"  {obs.shape}; slope-invariant except sin(a) = {obs[1, 6]:.4f}")
    print(f"  pushers read at {obs[0, 7]:+.4f} and {obs[0, 11]:+.4f} box widths from the box")


def check_action_limit():
    """The limit must let the box move, and still not let it leave the ramp.

    Two bounds from different physics, which is why the limit is a box
    and not a magnitude. Sizing one alone is how an earlier 5 N limit came
    out below the force needed to move the box at all, leaving the task
    unsolvable until `check_trajopt.py` reported it.
    """
    steep = batches.RAMP_ANGLE_RANGE[1]
    tangential, perpendicular = task.LIMIT

    # The driving pusher has to break itself AND the box free of friction,
    # against a ceiling set by its own normal load -- the two bounds sit on
    # different bodies, which is the other half of why the limit is a box.
    break_free = ramp.break_free_force(steep, mass=ramp.PUSHER_MASS + ramp.BOX_MASS)
    load = ramp.normal_load(steep, mass=ramp.PUSHER_MASS)

    print(f"  at {math.degrees(steep):.0f} deg: break free needs {break_free:.2f} N, tangential limit {tangential:.1f} N ({tangential / break_free:.2f}x)")
    print(f"  {' ':>15}lift-off needs {load:.2f} N, normal limit {perpendicular:.1f} N (leaves lambda >= {load - perpendicular:.2f} N)")
    assert tangential > break_free, "the pusher cannot drive the box at all"
    assert perpendicular < load, "an outward push could peel the pusher off the ramp"

    # `to_wrench` does not clip, so the clamp is asserted on its own and then
    # carried through the conversion.
    over = np.array([[[99.0, -99.0], [99.0, -99.0]]])
    assert np.allclose(task.clip_action(over)[0, 0], [tangential, -perpendicular])
    saturated = task.to_wrench(task.clip_action(over), [0.0])[0]
    assert np.allclose(saturated[task.ACTUATED[0]], [tangential, -perpendicular, 0.0])

    angles = np.array([math.radians(20.0)])
    batch = batches.fixed_batch(angles, gaps=0.0)
    scenes, state = batch.scenes, batch.state

    # Push both pushers straight out as hard as allowed: they must stay in
    # contact. The pusher, not the box, is what an outward push can peel off.
    lifting = task.to_wrench(np.tile([0.0, perpendicular], (100, 1, 2, 1)), angles)
    lifted = np.array(grip.rollout_batch(scenes, state, lifting, substeps=batch.substeps)[-1])

    # The pusher's list winds from the bottom-left, so [:2] is its two bottom
    # corners. Taking the higher of them asserts both are still in the
    # surface, rather than the body tipping up onto one edge.
    corners = corner_depths(lifted, angles, body=task.ACTUATED[0], vertices=ramp.PUSHER["vertices"])[:2]
    gap = max(depth.max() for depth in corners)
    assert gap < 0.0, "the pusher left the surface"

    # Push both along the ramp as hard as allowed: the box must actually go
    # somewhere, and must not tip while being driven.
    driving = task.to_wrench(np.tile([tangential, 0.0], (task.EPISODE_STEPS, 1, 2, 1)), angles)
    trajectory = np.array(grip.rollout_batch(scenes, state, driving, substeps=batch.substeps))
    xi = ramp.along_ramp(trajectory, angles)[:, 0, task.BOX]
    moved = xi[-1] - xi[0]
    tilt = np.abs(trajectory[:, 0, task.BOX, 2] - angles[0]).max()
    print(f"  full outward push: {-1e3 * gap:.3f} mm still inside the ramp   full uphill push: {moved:.2f} m in an episode, {1e3 * tilt:.2f} mrad of tilt")
    assert moved > 4.0 * batches.TARGET_OFFSET_RANGE[1], "not enough authority to reach a far target and brake"
    assert tilt < 0.05, "the box tipped"


def check_batch():
    """A sampled batch has to be internally consistent."""
    rng = np.random.default_rng(7)
    batch = batches.sample_batch(rng, 6)
    scenes, angles, state, targets = batch.scenes, batch.angles, batch.state, batch.targets
    assert len(scenes) == 6 and state.shape == (6, task.BODY_COUNT, 6) and targets.shape == (6,)
    assert np.allclose([ramp.scene_angle(s) for s in scenes], angles)
    assert (angles >= batches.RAMP_ANGLE_RANGE[0]).all() and (angles <= batches.RAMP_ANGLE_RANGE[1]).all()
    assert (angles < ramp.friction_angle(ramp.DEFAULT_PENALTY["friction"])).all(), "a sampled slope is above the friction angle"

    # The batch carries its own derived pieces, so nothing downstream can
    # rebuild one of them against a different slope.
    assert batch.substeps == task.substeps_for(scenes[0])
    assert batch.jacobian.shape == (6, observation.observe(state, angles, targets).shape[-1], task.BODY_COUNT, 6)

    offsets = targets - ramp.along_ramp(state, angles)[:, task.BOX]
    assert (np.abs(offsets) >= batches.TARGET_OFFSET_RANGE[0] - 1e-9).all()
    print(f"  6 environments, slopes {np.degrees(angles).min():.1f}-{np.degrees(angles).max():.1f} deg, {(offsets < 0).sum()} of 6 targets downhill")
    print(f"  batch carries its own substeps ({batch.substeps}) and observation jacobian {batch.jacobian.shape}")


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
