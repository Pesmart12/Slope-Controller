"""Build a `Batch`: the scenes, the placed and settled bodies, and the targets."""

import math
from collections import namedtuple

import numpy as np

import grip

from . import observation, ramp, task

# Long enough for the contact spring to ring down before scoring starts.
# Measured across the sampled range: 20 and 22 degrees are within 1% of
# steady state by about 105 ms, but 15 degrees needs nearer 150 ms -- not
# because it settles more slowly but because its creep is the smallest, so
# a relative bar is tightest there. 0.2 s leaves margin at the shallow end
# instead of sitting on the threshold. `check_task.py` asserts it still
# holds, which is what catches a change to k, b or the box.
SETTLE_SECONDS = 0.2
SETTLE_STEPS = int(round(SETTLE_SECONDS * task.CONTROL_HZ))

RAMP_ANGLE_RANGE = (math.radians(15.0), math.radians(22.0))
START_RANGE = (0.0, 0.4)
TARGET_OFFSET_RANGE = (0.4, 1.0)
APPROACH_GAP_RANGE = (0.0, 0.05)

# Everything one rollout needs, built together so no two pieces of it can
# disagree. `scenes` and `angles` are the pairing CLAUDE.md lists as a
# snag -- an angle array and the scenes built from it travel separately
# and a mismatch projects onto the wrong slopes with every shape still
# lining up.
#
# scenes     one grip.Scene per environment, each with its own slope
# angles     the slope those scenes were built from
# state      where this rollout starts, already settled
# targets    where the box should end up, in ramp coordinates
# substeps   integration steps per control step, derived from the scenes
# jacobian   d(observation)/d(state), constant because `observation.observe` is affine
Batch = namedtuple("Batch", "scenes angles state targets substeps jacobian")


def settle(scenes, state, substeps):
    """Let the bodies bounce to rest before the episode starts. Returns the
    settled state, as a copy.

    A body placed flush on the ramp sits 0.46 mm above where the contact
    spring wants it. The spring is underdamped, so it overshoots by about a
    quarter, rings with a period near 50 ms, and needs roughly 100 ms to
    get within 1% of steady state. Scoring an episode through that would
    make its first tenth of a second a disturbance the policy has to learn
    around for no reason.

    Runs with zero control, which is what keeps it fair to both contact
    models. Under a rigid solve the same window settles instantly and costs
    nothing, so both columns start the same way.

    The alternative -- placing bodies at the spring's equilibrium directly
    -- was rejected. That position is a fact about penalty contact, so it
    would hand the two columns different starting conditions. It is also
    not reachable by shifting every body the same distance, because
    friction tilts the box and loads its two corners unequally.

    The copy matters: `rollout_batch` returns a view of GRIP's internal
    buffer, and this state has to survive the next rollout.
    """
    controls = np.zeros((SETTLE_STEPS, len(scenes), state.shape[1], 3))
    return np.array(grip.rollout_batch(scenes, state, controls, substeps=substeps)[-1])


def placement(box_xi, gaps):
    """Place every body along the ramp, given where the box goes and how far
    back each pusher should start.

    A gap of zero starts a pusher touching the box. A few centimetres buys
    an approach and an impact, which a standing start cannot show.

    `gaps` broadcasts, so one number puts every pusher at the same standoff
    on every slope.
    """
    box_xi = np.asarray(box_xi, dtype=float)

    # Accepts a scalar, one gap per pusher, or one per environment per pusher.
    # Broadcasting here means the callers do not each build a full array.
    gaps = np.broadcast_to(np.asarray(gaps, dtype=float), box_xi.shape + (len(task.PUSHING_BODIES),))

    positions = np.zeros(box_xi.shape + (task.BODY_COUNT,))
    positions[..., task.BOX] = box_xi
    for slot, (body, (sign, touching)) in enumerate(zip(task.PUSHING_BODIES, task.TOUCHING)):
        # Step out from the box by the touching distance plus the standoff
        # asked for, on whichever side this pusher belongs. Exactly what
        # `objective.separations` undoes, from the same two constants.
        positions[..., body] = box_xi + sign * (touching + gaps[..., slot])
    return positions


def fixed_batch(ramp_angles, start=0.0, offset=0.0, gaps=0.0):
    """Assemble a `Batch` from stated numbers: build the scenes, place the
    bodies, settle them, and work out the targets.

    The deterministic counterpart to `sample_batch`, which differs only in
    where its numbers come from and then calls this. The two therefore
    cannot drift apart on the mechanics -- body count, whether the state was
    settled, or where the target offset is measured from.

    `offset` is measured from where the box ends up AFTER settling, not
    where it was placed. Everything creeps a millimetre or two while
    settling.
    """
    angles = ramp.as_angles(ramp_angles)

    scenes = ramp.make_scenes(angles, bodies=task.BODIES)
    substeps = task.substeps_for(scenes[0])

    # `start` broadcasts to one box position per environment, so a scalar puts
    # the box at the same point on every slope.
    positions = placement(np.broadcast_to(np.asarray(start, dtype=float), angles.shape), gaps)

    # Flush placement, then let the contact spring ring down. Scoring through
    # that transient would make the opening of every episode a disturbance.
    state = settle(scenes, ramp.resting_state(positions, angles, bodies=task.BODIES), substeps)

    # Measured from the SETTLED position, not the placed one -- everything
    # creeps a millimetre or two during the settle window.
    targets = ramp.along_ramp(state, angles)[:, task.BOX] + offset

    # Build the observation Jacobian once and hand it to the Batch. It is
    # constant because `observation.observe` is affine, so no backward sweep
    # recomputes it.
    return Batch(scenes, angles, state, targets, substeps, observation.observation_jacobian(angles))


def sample_batch(rng, n_envs):
    """Build a batch with a random slope, start, target and approach gap for
    each environment.

    Targets land on either side of the start, not always uphill. If they
    were always uphill a policy could score well by learning "push hard
    uphill" without representing the target at all.
    """
    angles = rng.uniform(*RAMP_ANGLE_RANGE, size=n_envs)
    start = rng.uniform(*START_RANGE, size=n_envs)

    # Magnitude and direction drawn separately, so the target is never closer
    # than TARGET_OFFSET_RANGE[0] and lands on either side with equal odds.
    offset = rng.uniform(*TARGET_OFFSET_RANGE, size=n_envs) * rng.choice([-1.0, 1.0], size=n_envs)

    gaps = rng.uniform(*APPROACH_GAP_RANGE, size=(n_envs, len(task.PUSHING_BODIES)))

    return fixed_batch(angles, start, offset, gaps)
