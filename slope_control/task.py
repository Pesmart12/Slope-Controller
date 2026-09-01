"""The minimal ramp task: one box, a wrench applied straight to it, no pusher.

Everything here is a task definition and none of it belongs in GRIP. The
reward, its gradient seeds, the action limit, the observation frame and
the episode structure are all decisions about what to ask for, not about
how bodies move.

The action is a two-component force in the *ramp* frame -- tangential
first, then normal -- rotated into GRIP's world wrench on the way in. The
slope randomizes per episode, so a world-frame action would mean
something different in every environment; a ramp-frame one means the same
thing everywhere, and a positive first component always pushes toward
increasing `xi`.

The torque component is never written. That mirrors the pusher planned
for step 3, which gets a 2D force and a fixed PD holding its orientation.
"""

import math

import numpy as np

import grip

from . import ramp

CONTROL_HZ = 100.0
EPISODE_SECONDS = 4.0
EPISODE_STEPS = int(round(EPISODE_SECONDS * CONTROL_HZ))

# Long enough for the contact spring to ring down before scoring starts.
# Measured across the sampled range: 20 and 22 degrees are within 1% of
# steady state by about 105 ms, but 15 degrees needs nearer 150 ms -- not
# because it settles more slowly but because its creep is the smallest, so
# a relative bar is tightest there. 0.2 s leaves margin at the shallow end
# instead of sitting on the threshold. `check_task.py` asserts it still
# holds, which is what catches a change to k, b or the box.
SETTLE_SECONDS = 0.2
SETTLE_STEPS = int(round(SETTLE_SECONDS * CONTROL_HZ))

# Three constraints meet here, and 5 N clears all of them.
#   Cannot break contact: lift-off needs mg*cos(a) = 9.2 N of normal
#     force, so under this limit the box physically cannot leave the ramp
#     and contact stays live for the whole episode by construction. That
#     matters more than it sounds -- a box that can fly to the target
#     makes this task de-risk nothing about contact, which is its job.
#   Cannot tip: overturning about the downhill corner also needs roughly
#     mg*cos(a), since the centre of mass is 0.15 m up and the base half
#     width is 0.15 m too.
#   Enough authority: holding at the steepest sampled slope costs
#     mg*sin(22 deg) = 3.7 N, leaving 1.3 N of net uphill acceleration.
MAX_FORCE = 5.0

RAMP_ANGLE_RANGE = (math.radians(15.0), math.radians(22.0))
START_RANGE = (0.0, 0.4)
TARGET_OFFSET_RANGE = (0.4, 1.0)

# How far off target the policy may sit before it is worth spending the
# steady holding force to close the gap. Rather under the 4.75 cm that
# drift.py measures for doing nothing at all, so the controller is being
# asked for something a released box does not already achieve.
POSITION_TOLERANCE = 0.02


def control_weight(tolerance=POSITION_TOLERANCE, ramp_angle=ramp.DEFAULT_RAMP_ANGLE):
    """Fix w_ctrl from a stated tolerance instead of picking a number.

    Both reward terms are normalized -- error by the box side, force by
    MAX_FORCE -- so the weights are directly comparable and the choice
    reduces to one question: at what position error does holding stop
    being worth the force it costs? Setting the two terms equal there,

        w_ctrl * (mg*sin(a) / F_max)^2  =  (tolerance / side)^2

    which at the nominal slope comes out near 0.01. Stating the tolerance
    rather than the weight means the number stays meaningful if MAX_FORCE
    or the box changes.
    """
    return (tolerance / ramp.BOX_SIDE) ** 2 / (ramp.hold_force(ramp_angle) / MAX_FORCE) ** 2


WEIGHTS = dict(position=1.0, control=control_weight())


def substeps_for(scene):
    """Integration steps per control step, so control runs at CONTROL_HZ.

    20 at penalty's dt = 5e-4. This is the decoupling that keeps the 1.0
    and 2.0 columns the same control problem rather than two different
    ones, so it is derived from the scene rather than written down.
    """
    return int(round(1.0 / (CONTROL_HZ * scene.dt)))


def clip_action(actions):
    """Saturate the action magnitude at MAX_FORCE.

    Applied inside `to_wrench`, which is the only place an action becomes
    physics, so the limit cannot be bypassed by a caller that forgets it.

    A hard clip has zero gradient once saturated. That is fine for a
    planner and acceptable for SHAC as long as the policy is not pinned
    to the limit; if it turns out to be, a smooth squash is the fix, not
    a larger limit.
    """
    actions = np.asarray(actions, dtype=float)
    magnitude = np.linalg.norm(actions, axis=-1, keepdims=True)
    return actions * np.minimum(1.0, MAX_FORCE / np.maximum(magnitude, 1e-12))


def to_wrench(actions, ramp_angles):
    """Ramp-frame actions (..., N, 2) -> GRIP wrenches (..., N, 1, 3).

    The columns of the map are `uphill` and `normal`, which is the
    rotation by the ramp angle. The torque row stays zero.
    """
    actions = clip_action(actions)
    angles = np.atleast_1d(np.asarray(ramp_angles, dtype=float))
    force = actions[..., 0:1] * ramp.uphill(angles) + actions[..., 1:2] * ramp.normal(angles)
    wrench = np.zeros(actions.shape[:-1] + (1, 3))
    wrench[..., 0, 0:2] = force
    return wrench


def to_action_gradient(dJ_dU, ramp_angles):
    """Pull GRIP's wrench derivatives back to the ramp-frame action.

    The action-to-wrench map is orthogonal, so its pullback is the
    transpose: project the force gradient onto uphill and normal. The
    torque row is dropped because the action never wrote it.

    Does not account for saturation -- a clipped action has zero
    derivative and the caller has to mask it. Kept out of here so the
    rotation stays one obvious thing.
    """
    angles = np.atleast_1d(np.asarray(ramp_angles, dtype=float))
    force = np.asarray(dJ_dU)[..., 0, 0:2]
    tangential = np.einsum("...ni,ni->...n", force, ramp.uphill(angles))
    perpendicular = np.einsum("...ni,ni->...n", force, ramp.normal(angles))
    return np.stack([tangential, perpendicular], axis=-1)


def reward(states, controls, ramp_angles, targets, weights=None):
    """Per-step reward, shaped (steps, environments).

        r_t = -w_pos * ((xi - xi*)/side)^2  -  w_ctrl * (|f|/F_max)^2

    Both terms are normalized, which is what makes the weights order one
    and comparable. In raw units the ratio is about 1e-5 -- metres
    squared against newtons squared -- a number that tells a reader
    nothing and invites someone to "fix" it later.

    The control term is not boilerplate. Under penalty contact a held box
    gets no friction for free, so holding costs mg*sin(a) forever
    (`ramp.hold_force`); under a solve it costs nothing after arrival.
    Same reward, two different bills, and that is the thing worth
    watching in the comparison.

    Index convention: control u_t is scored against the state it
    produces, Z_{t+1}. Z_0 is fixed by the initial condition and no
    control reaches it, so it contributes a constant and is left out.
    """
    weights = WEIGHTS if weights is None else weights
    xi = ramp.along_ramp(states, ramp_angles)[1:, :, 0]
    error = (xi - np.asarray(targets)) / ramp.BOX_SIDE
    force = np.linalg.norm(np.asarray(controls)[..., 0, 0:2], axis=-1) / MAX_FORCE
    return -weights["position"] * error ** 2 - weights["control"] * force ** 2


def reward_seeds(states, controls, ramp_angles, targets, weights=None):
    """dl_dZ and dl_dU for `adjoint_batch`, matching `reward` term for term.

    Deliberately adjacent to the reward it differentiates. The two have
    to agree, and the cheapest guarantee of that is a change to one being
    visibly next to the other -- a wrong seed raises nothing, it just
    quietly trains for something else.

    The state enters only through xi = p . uphill, so

        dr/d(x, y) = -2*w_pos*(xi - xi*)/side^2 * uphill

    and every other component of the 6-vector is zero: the reward does
    not see theta, and it does not see velocity. dl_dZ[0] is zero for the
    index convention `reward` describes. For the control,

        dr/df = -2*w_ctrl*f / F_max^2

    with the torque row zero. Both are partials of the stage cost only --
    GRIP supplies everything that makes them total derivatives.
    """
    weights = WEIGHTS if weights is None else weights
    states = np.asarray(states)
    controls = np.asarray(controls)
    angles = np.atleast_1d(np.asarray(ramp_angles, dtype=float))

    xi = ramp.along_ramp(states, angles)[..., 0]
    coefficient = -2.0 * weights["position"] * (xi - np.asarray(targets)) / ramp.BOX_SIDE ** 2

    dl_dZ = np.zeros(states.shape)
    dl_dZ[1:, :, 0, 0:2] = coefficient[1:, :, None] * ramp.uphill(angles)[None, :, :]

    dl_dU = np.zeros(controls.shape)
    dl_dU[..., 0, 0:2] = -2.0 * weights["control"] * controls[..., 0, 0:2] / MAX_FORCE ** 2
    return dl_dZ, dl_dU


def observe(states, ramp_angles, targets):
    """What a policy sees, in the ramp frame so it reads the same at every slope.

        [ (xi - xi*)/side, v_uphill, gap/side, v_normal, theta - a, omega, sin a ]

    Everything but the last entry is slope-invariant. sin(a) is the one
    thing that leaks through from the world frame, because it sets the
    gravity load the controller has to fight, and without it the policy
    would have to infer the slope from how fast the box is losing ground.
    The frame matches the action's, so a positive first action always
    pushes toward a larger first observation.

    Nothing consumes this yet: MPPI replans from full state, and there is
    no policy before SHAC. It is fixed here only to pin the frame down
    alongside the action mapping that shares it, and it is exercised only
    for shape and for the a = 0 reduction. Treat it as unvalidated.
    """
    states = np.asarray(states)
    angles = np.atleast_1d(np.asarray(ramp_angles, dtype=float))
    targets = np.asarray(targets)
    up, out = ramp.uphill(angles), ramp.normal(angles)

    position, velocity = states[..., 0:2], states[..., 3:5]
    xi = np.einsum("...nbi,ni->...nb", position, up)
    eta = np.einsum("...nbi,ni->...nb", position, out)
    return np.stack([
        (xi - targets[:, None]) / ramp.BOX_SIDE,
        np.einsum("...nbi,ni->...nb", velocity, up),
        (eta - 0.5 * ramp.BOX_SIDE) / ramp.BOX_SIDE,
        np.einsum("...nbi,ni->...nb", velocity, out),
        states[..., 2] - angles[:, None],
        states[..., 5],
        np.broadcast_to(np.sin(angles)[:, None], xi.shape),
    ], axis=-1)


def settle(scenes, state, substeps):
    """Ring the contact spring down before the episode is scored.

    A flush start sits 0.46 mm above the penalty equilibrium and the
    contact spring is underdamped -- zeta = 0.35 at GRIP's demo constants
    -- so it overshoots by about a quarter, rings at roughly 50 ms per
    cycle, and needs about 100 ms to come within 1% of steady state.
    Scoring through that makes the opening of every episode a disturbance
    the policy has to learn around for no reason.

    Zero control throughout, which is what keeps it formulation-agnostic:
    under an NCP solve the same window settles at once and costs nothing.
    That is the whole argument for settling rather than starting at the
    penalty equilibrium, which is a 1.0-specific state and, being tilted,
    not reachable by a uniform offset anyway.

    Returns a copy, because `rollout_batch` hands back a view of the
    simulator's own buffer and this state has to outlive the next
    rollout.
    """
    controls = np.zeros((SETTLE_STEPS, len(scenes), 1, 3))
    return np.array(grip.rollout_batch(scenes, state, controls, substeps=substeps)[-1])


def sample_batch(rng, n_envs):
    """One randomized episode setup per environment, already settled.

    Returns scenes, angles, initial state and targets together rather
    than separately, so the angles and the scenes built from them cannot
    drift apart in a caller.

    Targets sit a fixed distance to either side of the start, not always
    uphill. Uphill-only would let a policy score well by learning "push
    hard uphill" with no representation of the target at all, and hide
    that until the pusher arrives at step 3.

    Offsets are measured from where the box actually ends up after
    settling, not from where it was placed, since it creeps a millimetre
    or two during the window.
    """
    angles = rng.uniform(*RAMP_ANGLE_RANGE, size=n_envs)
    start = rng.uniform(*START_RANGE, size=n_envs)
    offset = rng.uniform(*TARGET_OFFSET_RANGE, size=n_envs) * rng.choice([-1.0, 1.0], size=n_envs)

    scenes = ramp.make_scenes(angles)
    state = settle(scenes, ramp.resting_state(start, angles), substeps_for(scenes[0]))
    return scenes, angles, state, ramp.along_ramp(state, angles)[:, 0] + offset
