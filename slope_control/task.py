"""The ramp task, in two variants that share a reward.

Everything here is a task definition and none of it belongs in GRIP. The
reward, its gradient seeds, the action limits, the observation frame and
the episode structure are all decisions about what to ask for, not about
how bodies move.

Actions are two-component forces in the *ramp* frame -- tangential first,
then normal -- rotated into GRIP's world wrench on the way in, one per
actuated body. The slope randomizes per episode, so a world-frame action
would mean something different in every environment; a ramp-frame one
means the same thing everywhere, and a positive tangential component
always pushes toward increasing `xi`. The torque row is never written.

Two variants, because the force limits are set by the bodies being driven
and there is no single number that serves both:

    BOX_ONLY      one box, wrench applied straight to it. Physically
                  fictional -- nothing reaches into a box and pushes from
                  its centre -- but it de-risks the reward against a
                  single contact set.
    TWO_PUSHERS   pusher, box, pusher. The box is unactuated and
                  everything it does arrives through a contact. Two of
                  them, because a convex pusher only pushes and its
                  direction is fixed by the side it starts on, so one
                  pusher leaves overshoot unrecoverable.
"""

import math
from collections import namedtuple

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

RAMP_ANGLE_RANGE = (math.radians(15.0), math.radians(22.0))
START_RANGE = (0.0, 0.4)
TARGET_OFFSET_RANGE = (0.4, 1.0)
APPROACH_GAP_RANGE = (0.0, 0.05)

# How far off target the policy may sit before it is worth spending the
# steady holding force to close the gap. Rather under the 4.75 cm that
# drift.py measures for doing nothing at all, so the controller is being
# asked for something a released box does not already achieve.
POSITION_TOLERANCE = 0.02

# bodies    how many the scene carries
# box       which one the reward scores
# actuated  which ones take an action, in action order
# limit     per-component force clamp, (tangential, normal)
# scale     the isotropic force scale the control cost divides by
# weights   position and control, derived from POSITION_TOLERANCE
Variant = namedtuple("Variant", "bodies box actuated limit scale weights")


def control_weight(scale, hold_forces, tolerance=POSITION_TOLERANCE):
    """Fix w_ctrl from a stated tolerance instead of picking a number.

    Both reward terms are normalized -- error by the box side, force by
    `scale` -- so the weights are comparable and the choice reduces to one
    question: at what position error does holding stop being worth the
    force it costs? Setting the two terms equal there,

        w_ctrl * sum_i (hold_i / scale)^2  =  (tolerance / side)^2

    where the sum runs over actuated bodies, because under penalty contact
    every one of them has to be pushed continuously just to stay put.
    Stating the tolerance rather than the weight keeps the number
    meaningful when the force scale or the bodies change.
    """
    return (tolerance / ramp.BOX_SIDE) ** 2 / sum((force / scale) ** 2 for force in hold_forces)


def _variant(bodies, box, actuated, limit, scale, hold_forces):
    weights = dict(position=1.0, control=control_weight(scale, hold_forces))
    return Variant(bodies, box, tuple(actuated), np.asarray(limit, dtype=float), scale, weights)


# A box, not a ball. The two components are bounded by different physics
# and a single magnitude cap conflates them -- sizing one leaves the other
# wrong, which is how the first version came out unable to move anything.
#
#   Tangential must EXCEED the force that breaks the driven bodies free of
#     friction. Below it nothing moves; a newton above it they accelerate
#     away, because Coulomb friction has no gentle regime.
#   Normal must stay BELOW the actuated body's own normal load, or an
#     outward push unloads the contact and it leaves the ramp. A body that
#     can fly to its target makes this task de-risk nothing about contact,
#     which is its job.
#
# Tipping is NOT a constraint for either variant, though an earlier draft
# claimed it was. A wrench acts at the centre of mass so exerts no moment
# there, and friction's couple is balanced by the centre of pressure
# shifting within the base. Measured: 2 mrad on the box at 25 N, 4 mrad on
# the pusher at 60 N.

# Box alone: break free 8.22 N against a 9.10 N normal load, at 22 degrees.
BOX_ONLY = _variant(bodies=1, box=0, actuated=[0], limit=[12.0, 6.0], scale=12.0, hold_forces=[ramp.hold_force(ramp.DEFAULT_RAMP_ANGLE)])

# Two pushers: the driving one must move itself and the box, 24.67 N at the
# steepest slope, against its own 18.19 N normal load. 30 N is deliberately
# short of the 41.11 N that would shove a passive partner along, so the
# trailing pusher has to be driven out of the way rather than bulldozed --
# which is the reason for having two of them. Coordinated, the numbers are
# comfortable: 24.67 N to drive the box, 16.45 N for the trailing pusher to
# carry itself, both inside 30.
TWO_PUSHERS = _variant(
    bodies=3, box=1, actuated=[0, 2], limit=[30.0, 12.0], scale=30.0,
    hold_forces=[ramp.hold_force(ramp.DEFAULT_RAMP_ANGLE, mass=ramp.PUSHER_MASS + ramp.BOX_MASS), ramp.hold_force(ramp.DEFAULT_RAMP_ANGLE, mass=ramp.PUSHER_MASS)],
)


def bodies_for(variant):
    """The shape list a variant's scenes are built from."""
    return ramp.BOX_ONLY_BODIES if variant.bodies == 1 else ramp.TWO_PUSHER_BODIES


def substeps_for(scene):
    """Integration steps per control step, so control runs at CONTROL_HZ.

    20 at penalty's dt = 5e-4. This is the decoupling that keeps the 1.0
    and 2.0 columns the same control problem rather than two different
    ones, so it is derived from the scene rather than written down.
    """
    return int(round(1.0 / (CONTROL_HZ * scene.dt)))


def clip_action(actions, variant=BOX_ONLY):
    """Clamp each component to its own entry in the variant's limit.

    Applied inside `to_wrench`, which is the only place an action becomes
    physics, so the limit cannot be bypassed by a caller that forgets it.
    Beware when measuring anything against force: a sweep that routes
    through `to_wrench` silently makes every value above the limit the
    same experiment.

    A hard clip has zero gradient once saturated. That is fine for a
    planner and acceptable for SHAC as long as the policy is not pinned to
    the limit; if it turns out to be, a smooth squash is the fix, not a
    larger limit.
    """
    return np.clip(np.asarray(actions, dtype=float), -variant.limit, variant.limit)


def to_wrench(actions, ramp_angles, variant=BOX_ONLY):
    """Ramp-frame actions (..., N, actuated, 2) -> GRIP wrenches (..., N, bodies, 3).

    The columns of the map are `uphill` and `normal`, which is the rotation
    by the ramp angle. Unactuated bodies get zero rows, and the torque row
    is zero for everything.
    """
    actions = clip_action(actions, variant)
    angles = np.atleast_1d(np.asarray(ramp_angles, dtype=float))
    up, out = ramp.uphill(angles)[:, None, :], ramp.normal(angles)[:, None, :]
    force = actions[..., 0:1] * up + actions[..., 1:2] * out

    wrench = np.zeros(actions.shape[:-2] + (variant.bodies, 3))
    for slot, body in enumerate(variant.actuated):
        wrench[..., body, 0:2] = force[..., slot, :]
    return wrench


def to_action_gradient(dJ_dU, ramp_angles, variant=BOX_ONLY):
    """Pull GRIP's wrench derivatives back to the ramp-frame actions.

    The action-to-wrench map is orthogonal, so its pullback is the
    transpose: project each actuated body's force gradient onto uphill and
    normal. Torque rows are dropped because no action wrote them.

    Does not account for saturation -- a clipped action has zero derivative
    and the caller has to mask it. Kept out of here so the rotation stays
    one obvious thing.
    """
    angles = np.atleast_1d(np.asarray(ramp_angles, dtype=float))
    up, out = ramp.uphill(angles), ramp.normal(angles)
    dJ_dU = np.asarray(dJ_dU)

    gradients = []
    for body in variant.actuated:
        force = dJ_dU[..., body, 0:2]
        gradients.append(np.stack([np.einsum("...ni,ni->...n", force, up), np.einsum("...ni,ni->...n", force, out)], axis=-1))
    return np.stack(gradients, axis=-2)


def pushing_bodies(variant):
    """Actuated bodies that are not the scored one -- the ones that approach.

    Empty for BOX_ONLY, where the actuated body *is* the box and there is
    nothing to approach.
    """
    return [body for body in variant.actuated if body != variant.box]


def separations(states, ramp_angles, variant):
    """Each pushing body's gap to the box face it meets. Positive is a gap.

    The reach is read off the vertex list rather than written down, since
    the 3 degree face trim moves it, and from the correct side -- the
    uphill pusher is mirrored, so its contact vertex is at -x.
    """
    bodies = bodies_for(variant)
    xi = ramp.along_ramp(states, ramp_angles)
    gaps = []
    for body in pushing_bodies(variant):
        uphill_of_box = body > variant.box
        reach = 0.5 * ramp.BOX_SIDE + ramp.contact_reach(bodies[body], toward_uphill=not uphill_of_box)
        sign = 1.0 if uphill_of_box else -1.0
        gaps.append(sign * (xi[..., body] - xi[..., variant.box]) - reach)
    return gaps


def reward(states, controls, ramp_angles, targets, variant=BOX_ONLY, weights=None, shaping=False):
    """Per-step reward, shaped (steps, environments).

        r_t = -w_pos * ((xi_box - xi*)/side)^2  -  w_ctrl * sum_i |f_i|^2 / scale^2

    Both terms are normalized, which is what makes the weights order one
    and comparable. In raw units the ratio is about 1e-5 -- metres squared
    against newtons squared -- a number that tells a reader nothing and
    invites someone to "fix" it later.

    Only the box's position is scored. Where the pushers end up is their
    own business, and pricing it would be deciding for the policy how to
    use its second body.

    The control term is not boilerplate. Under penalty contact a held body
    gets no friction for free, so holding costs mg*sin(a) forever
    (`ramp.hold_force`) for every actuated body; under a solve it costs
    nothing after arrival. Same reward, two different bills, and that is
    the thing worth watching in the comparison.

    Index convention: control u_t is scored against the state it produces,
    Z_{t+1}. Z_0 is fixed by the initial condition and no control reaches
    it, so it contributes a constant and is left out.

    `shaping=True` adds the approach term documented in
    `approach_penalty`. It is OFF by default, so a reported number is the
    task objective unless someone asked for otherwise. Train with it on,
    score with it off; the two mistakes are not symmetric, since training
    without it fails loudly -- the driving pusher's action sits at exactly
    0.00 N -- while reporting with it on is silent and makes numbers
    incomparable across columns.
    """
    weights = variant.weights if weights is None else weights
    controls = np.asarray(controls)

    xi = ramp.along_ramp(states, ramp_angles)[1:, :, variant.box]
    error = (xi - np.asarray(targets)) / ramp.BOX_SIDE
    effort = sum((controls[..., body, 0:2] ** 2).sum(axis=-1) for body in variant.actuated)
    total = -weights["position"] * error ** 2 - weights["control"] * effort / variant.scale ** 2

    if shaping:
        total = total - approach_penalty(states, ramp_angles, variant, weights)[1:]
    return total


def approach_penalty(states, ramp_angles, variant, weights=None):
    """REWARD SHAPING: the cost of a pusher not being at the box yet.

    Named as such on purpose. This is not part of the task objective; it
    is a term added to the *training* objective to remove a region where
    the task objective has no gradient at all.

    THIS IS REWARD SHAPING and is named as such on purpose. It is not part
    of the task objective; it is a term added to the *training* objective
    to remove a region where the task objective has no gradient at all.

    The region: a pusher not touching the box contributes nothing to the
    box's position, so d(box position)/d(pusher action) is identically
    zero. Measured -- from a zero initialization at a 5 cm approach gap,
    trajectory optimization leaves the driving pusher at exactly 0.00 N
    for every iteration and the box 52 cm short. Gradients cannot discover
    a contact that does not exist.

    The term is

        -w_pos * sum_i max(0, separation_i / side)^2

    over the pushing bodies. Three properties, each deliberate:

      One-sided, so it is EXACTLY zero once contact is made and cannot
      distort behaviour in the regime where the task objective takes over.
      max(0, x)^2 is C1, so the gradient stays continuous at the kink.

      No new weight. It reuses w_pos and the same normalization by the box
      side, which says: penalize a pusher being away from the box exactly
      as much as we penalize the box being away from its target, but only
      while it actually is away.

      Live from the first iteration, because the pushers are DIRECTLY
      actuated -- d(pusher position)/d(pusher action) is never zero,
      contact or no contact. That is the whole mechanism.

    Measured effect, same setup as the failure above: the driving pusher
    reaches its 30 N limit, both contacts form, and the box lands 1.8-2.4
    cm from target. The *unshaped* reward improves from -1193 to -93, so
    the term is not buying its result by moving the goalposts.

    It belongs in both columns identically. It is formulation-agnostic --
    approaching is the same problem under penalty and under a solve --
    and under an NCP solve it is not merely convenient but necessary:
    penalty creep happens to close one of the two gaps on its own, and a
    rigid solve closes neither, so the flat region there is total.

    Worth re-checking once a policy optimizes against it rather than an
    open-loop sequence. A policy has more freedom to find a degenerate way
    to satisfy a shaped term.

    Returned over all steps including the initial state, so callers slice
    it the same way they slice the position term.
    """
    weights = variant.weights if weights is None else weights
    gaps = separations(states, ramp_angles, variant)
    if not gaps:
        return np.zeros(np.asarray(states).shape[:-2])
    return weights["position"] * sum(np.maximum(0.0, gap / ramp.BOX_SIDE) ** 2 for gap in gaps)


def reward_seeds(states, controls, ramp_angles, targets, variant=BOX_ONLY, weights=None, shaping=False):
    """dl_dZ and dl_dU for `adjoint_batch`, matching `reward` term for term.

    Deliberately adjacent to the reward it differentiates. The two have to
    agree, and the cheapest guarantee of that is a change to one being
    visibly next to the other -- a wrong seed raises nothing, it just
    quietly trains for something else.

    The state enters only through the box's xi = p . uphill, so

        dr/d(x, y)_box = -2*w_pos*(xi - xi*)/side^2 * uphill

    and every other entry is zero: the reward does not see orientation, it
    does not see velocity, and it does not see where the pushers are. For
    each actuated body,

        dr/df_i = -2*w_ctrl*f_i / scale^2

    with the torque row zero. Both are partials of the stage cost only --
    GRIP supplies everything that makes them total derivatives.

    For a *policy* these seeds are not the whole story. One `adjoint_batch`
    call over a window gives the open-loop gradient, which is right for a
    fixed control sequence and wrong for state feedback; see
    `tests/check_closed_loop.py`.
    """
    weights = variant.weights if weights is None else weights
    states, controls = np.asarray(states), np.asarray(controls)
    angles = np.atleast_1d(np.asarray(ramp_angles, dtype=float))

    xi = ramp.along_ramp(states, angles)[..., variant.box]
    coefficient = -2.0 * weights["position"] * (xi - np.asarray(targets)) / ramp.BOX_SIDE ** 2

    dl_dZ = np.zeros(states.shape)
    dl_dZ[1:, :, variant.box, 0:2] = coefficient[1:, :, None] * ramp.uphill(angles)[None, :, :]

    dl_dU = np.zeros(controls.shape)
    for body in variant.actuated:
        dl_dU[..., body, 0:2] = -2.0 * weights["control"] * controls[..., body, 0:2] / variant.scale ** 2

    if shaping:
        # d/d(sep) of -w*max(0, sep/side)^2, chained through
        # sep = sign*(xi_body - xi_box) - reach and xi = p . uphill.
        up = ramp.uphill(angles)[None, :, :]
        for body, gap in zip(pushing_bodies(variant), separations(states, angles, variant)):
            sign = 1.0 if body > variant.box else -1.0
            coefficient = -2.0 * weights["position"] * np.maximum(0.0, gap) / ramp.BOX_SIDE ** 2
            dl_dZ[1:, :, body, 0:2] += (sign * coefficient[1:])[:, :, None] * up
            dl_dZ[1:, :, variant.box, 0:2] += (-sign * coefficient[1:])[:, :, None] * up
    return dl_dZ, dl_dU


def observe(states, ramp_angles, targets, variant=BOX_ONLY):
    """What a policy sees, in the ramp frame so it reads the same at every slope.

    Seven numbers for the box,

        [ (xi - xi*)/side, v_uphill, gap/side, v_normal, theta - a, omega, sin a ]

    then four more for every other body, relative to the box,

        [ (xi_body - xi_box)/side, v_uphill, gap/side, theta - a ]

    Everything but sin(a) is slope-invariant. That one entry is what leaks
    through from the world frame, because it sets the gravity load the
    controller has to fight, and without it the policy would have to infer
    the slope from how fast things are losing ground. The frame matches the
    action's, so a positive first action always pushes toward a larger
    first observation.

    Nothing consumes this yet -- there is no policy before SHAC -- so it is
    fixed here only to pin the frame down alongside the action mapping that
    shares it, and it is exercised only for shape and for the a = 0
    reduction. Treat it as unvalidated.
    """
    states = np.asarray(states)
    angles = np.atleast_1d(np.asarray(ramp_angles, dtype=float))
    targets = np.asarray(targets)
    up, out = ramp.uphill(angles), ramp.normal(angles)

    position, velocity = states[..., 0:2], states[..., 3:5]
    xi = np.einsum("...nbi,ni->...nb", position, up)
    eta = np.einsum("...nbi,ni->...nb", position, out)
    v_up = np.einsum("...nbi,ni->...nb", velocity, up)
    v_out = np.einsum("...nbi,ni->...nb", velocity, out)
    tilt = states[..., 2] - angles[:, None]

    offsets = np.array([ramp.resting_offset(body["vertices"]) for body in bodies_for(variant)])
    gap = (eta - offsets) / ramp.BOX_SIDE

    box = variant.box
    channels = [
        (xi[..., box] - targets) / ramp.BOX_SIDE,
        v_up[..., box],
        gap[..., box],
        v_out[..., box],
        tilt[..., box],
        states[..., box, 5],
        np.broadcast_to(np.sin(angles), xi[..., box].shape),
    ]
    for body in range(variant.bodies):
        if body != box:
            channels += [(xi[..., body] - xi[..., box]) / ramp.BOX_SIDE, v_up[..., body], gap[..., body], tilt[..., body]]
    return np.stack(channels, axis=-1)


def observation_jacobian(ramp_angles, variant=BOX_ONLY):
    """d(observation)/d(state), shaped (environments, features, bodies, 6).

    `observe` is affine in the state -- every channel is a projection onto
    uphill or normal, a difference of two such, or a component passed
    straight through, and the only nonlinear thing in it, sin(a), is a
    constant with respect to Z. So this is a CONSTANT matrix per
    environment: build it once when the batch is sampled, never again.

    That matters because the closed-loop policy gradient needs
    d(action)/d(state) = (d action/d obs)(d obs/d Z) at every step of the
    backward sweep. The first factor is the network's input gradient and
    has to be recomputed; the second is this, and does not.

    Built by evaluating `observe` on unit states rather than by finite
    differences -- for an affine map that is exact, not an approximation.
    `check_task.py` asserts the affinity that makes it legitimate.
    """
    angles = np.atleast_1d(np.asarray(ramp_angles, dtype=float))
    shape = (angles.size, variant.bodies, 6)
    targets = np.zeros(angles.size)  # only shifts the constant term

    base = observe(np.zeros(shape), angles, targets, variant)
    jacobian = np.zeros(base.shape + (variant.bodies, 6))
    for body in range(variant.bodies):
        for component in range(6):
            probe = np.zeros(shape)
            probe[:, body, component] = 1.0
            jacobian[..., body, component] = observe(probe, angles, targets, variant) - base
    return jacobian


def state_gradient(dJ_dobs, jacobian):
    """Pull an adjoint on the observation back to an adjoint on the state."""
    return np.einsum("...nf,nfbc->...nbc", np.asarray(dJ_dobs), jacobian)


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
    simulator's own buffer and this state has to outlive the next rollout.
    """
    controls = np.zeros((SETTLE_STEPS, len(scenes), state.shape[1], 3))
    return np.array(grip.rollout_batch(scenes, state, controls, substeps=substeps)[-1])


def placement(box_xi, gaps, variant=TWO_PUSHERS):
    """Where each body starts along the ramp, from the box and the approach gaps.

    Each pusher is set back so its contact vertex sits `gap` short of the
    box face it will meet. Zero starts them touching; a few centimetres
    buys an approach and an impact, which is the second thing the task doc
    wants instrumented and the one thing a standing start cannot show.

    The reach is read off the vertex list rather than written down, since
    the 3 degree trim moves it, and it is read from the correct side --
    the uphill pusher is mirrored, so its contact vertex is at -x.
    """
    bodies = bodies_for(variant)
    gaps, box_xi = np.asarray(gaps, dtype=float), np.asarray(box_xi, dtype=float)

    positions = np.zeros(box_xi.shape + (variant.bodies,))
    positions[..., variant.box] = box_xi
    for slot, body in enumerate(variant.actuated):
        uphill_of_box = body > variant.box
        reach = ramp.contact_reach(bodies[body], toward_uphill=not uphill_of_box)
        offset = 0.5 * ramp.BOX_SIDE + reach + gaps[..., slot]
        positions[..., body] = box_xi + offset if uphill_of_box else box_xi - offset
    return positions


def sample_batch(rng, n_envs, variant=BOX_ONLY):
    """One randomized episode setup per environment, already settled.

    Returns scenes, angles, initial state and targets together rather than
    separately, so the angles and the scenes built from them cannot drift
    apart in a caller.

    Targets sit a fixed distance to either side of the start, not always
    uphill. Uphill-only would let a policy score well by learning "push
    hard uphill" with no representation of the target at all.

    Offsets are measured from where the box actually ends up after
    settling, not from where it was placed, since everything creeps a
    millimetre or two during the window.
    """
    angles = rng.uniform(*RAMP_ANGLE_RANGE, size=n_envs)
    start = rng.uniform(*START_RANGE, size=n_envs)
    offset = rng.uniform(*TARGET_OFFSET_RANGE, size=n_envs) * rng.choice([-1.0, 1.0], size=n_envs)

    if variant.bodies == 1:
        positions = start[:, None]
    else:
        positions = placement(start, rng.uniform(*APPROACH_GAP_RANGE, size=(n_envs, len(variant.actuated))), variant)

    scenes = ramp.make_scenes(angles, bodies=bodies_for(variant))
    state = settle(scenes, ramp.resting_state(positions, angles, bodies=bodies_for(variant)), substeps_for(scenes[0]))
    return scenes, angles, state, ramp.along_ramp(state, angles)[:, variant.box] + offset
