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

    TWO_PUSHERS   pusher, box, pusher. The box is unactuated and
                  everything it does arrives through a contact. Two of
                  them, because a convex pusher only pushes and its
                  direction is fixed by the side it starts on, so one
                  pusher leaves overshoot unrecoverable. This is the task.
    BOX_ONLY      one box, wrench applied straight to it. Physically
                  fictional -- nothing reaches into a box and pushes from
                  its centre -- and kept only through SHAC bring-up, as a
                  version of the same problem with the contact taken out.
                  If a policy fails on pushers and works here, the fault
                  is in the contact rather than the gradient path.
                  Scheduled for deletion once step 5 lands; CLAUDE.md
                  carries the decision.

No function here defaults its `variant`. One of the two is fictional, so a
caller that forgets the argument must not silently get it.

The reward has two terms, position and control effort, and that pair is
the task objective -- it is what gets reported, and it is identical in
both the penalty and NCP columns. There is a third term, `approach_penalty`,
which is REWARD SHAPING: off by default, added only to the training
objective, and documented at length where it is defined. Anything scored
for the record is scored without it.
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

# Everything one rollout needs, built together so no two pieces of it can
# disagree. `scenes` and `angles` are the pairing CLAUDE.md lists as a
# snag -- an angle array and the scenes built from it travel separately
# and a mismatch projects onto the wrong slopes with every shape still
# lining up. `variant` is here for the same reason: the scenes carry its
# body count, so passing a different one to `observe` is not a thing that
# should be expressible.
#
# scenes     one grip.Scene per environment, each with its own slope
# angles     the slope those scenes were built from
# state      where this rollout starts, already settled
# targets    where the box should end up, in ramp coordinates
# substeps   integration steps per control step, derived from the scenes
# jacobian   d(observation)/d(state), constant because `observe` is affine
# variant    which task these scenes were built for
Batch = namedtuple("Batch", "scenes angles state targets substeps jacobian variant")


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


def clip_action(actions, variant):
    """Clamp each component to its own entry in the variant's limit.

    Applied inside `to_wrench`, which is the only place an action becomes
    physics, so the limit cannot be bypassed by a caller that forgets it.
    Beware when measuring anything against force: a sweep that routes
    through `to_wrench` silently makes every value above the limit the
    same experiment.

    A hard clip has zero gradient once saturated, and the converged
    baseline sits on its limit 2.1% of steps on box-only and 5.9% on two
    pushers -- enough that a policy optimizing through it would spend real
    time somewhere it cannot be improved.

    That is already handled upstream rather than here: `policy.Actor`
    emits `limit * tanh(...)`, so a policy's actions are feasible by
    construction and this clip is a no-op for them. It still runs, because
    it is the one place the limit is enforced for callers that are NOT a
    policy -- `check_trajopt` optimizes a raw control sequence and needs
    projecting back onto the box every iteration.

    If a trained policy still lives at its limit, that is a statement
    about the force budget rather than about the clip. **Never widen the
    limit** -- it is what keeps the bodies on the ramp.
    """
    return np.clip(np.asarray(actions, dtype=float), -variant.limit, variant.limit)


def to_wrench(actions, ramp_angles, variant):
    """Ramp-frame actions (..., N, actuated, 2) -> GRIP wrenches (..., N, bodies, 3).

    The columns of the map are `uphill` and `normal`, which is the rotation
    by the ramp angle. Unactuated bodies get zero rows, and the torque row
    is zero for everything.
    """
    actions = clip_action(actions, variant)
    angles = np.atleast_1d(np.asarray(ramp_angles, dtype=float))

    # The [:, None, :] inserts an axis for the actuated slot, so one angle per
    # environment broadcasts across however many bodies that environment drives.
    up, out = ramp.uphill(angles)[:, None, :], ramp.normal(angles)[:, None, :]

    # The rotation itself, written as a linear combination rather than a matrix
    # product: the map's columns ARE uphill and normal. Keeping the 0:1 / 1:2
    # slices (rather than 0 / 1) preserves the trailing axis so each scalar
    # component broadcasts against a 2-vector.
    force = actions[..., 0:1] * up + actions[..., 1:2] * out

    # Scatter each action slot onto the body it drives. Unactuated bodies keep
    # their zero row, and column 2 -- the torque -- is never written by anything.
    wrench = np.zeros(actions.shape[:-2] + (variant.bodies, 3))
    for slot, body in enumerate(variant.actuated):
        wrench[..., body, 0:2] = force[..., slot, :]
    return wrench


def to_action_gradient(dJ_dU, ramp_angles, variant):
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
        # Gather this body's force gradient, dropping the torque column since
        # no action ever wrote it.
        force = dJ_dU[..., body, 0:2]

        # "...ni,ni->...n" is a per-environment dot product: contract the
        # 2-vector axis i against that environment's basis vector, keeping any
        # leading step axes. Two of them stacked give (tangential, normal),
        # which is the transpose of `to_wrench`'s rotation -- correct as the
        # pullback precisely because that map is orthogonal.
        gradients.append(np.stack([np.einsum("...ni,ni->...n", force, up), np.einsum("...ni,ni->...n", force, out)], axis=-1))

    # axis=-2 rebuilds the actuated-slot axis, so this comes back shaped like
    # the actions that went in.
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
        # Body order is downhill-to-uphill, so a higher index means this pusher
        # sits above the box and pushes down.
        uphill_of_box = body > variant.box

        # Centre-to-centre distance at first touch: half the box plus however
        # far this pusher's contact vertex sticks out toward it. An uphill
        # pusher is the mirrored shape and reaches with its -x vertex, hence
        # `toward_uphill` being the negation.
        reach = 0.5 * ramp.BOX_SIDE + ramp.contact_reach(bodies[body], toward_uphill=not uphill_of_box)

        # Signed so that positive always means "still apart", whichever side
        # the pusher is on: subtracting the touching distance from the actual
        # separation leaves the gap.
        sign = 1.0 if uphill_of_box else -1.0
        gaps.append(sign * (xi[..., body] - xi[..., variant.box]) - reach)
    return gaps


def reward(states, controls, ramp_angles, targets, variant, weights=None, shaping=False):
    """Per-step reward, shaped (steps, environments).

        r_t = -w_pos * ((xi_box - xi*)/side)^2  -  w_ctrl * sum_i |f_i|^2 / scale^2

    Both terms are normalized, which is what makes the weights order one
    and comparable. In raw units the ratio is about 1e-5 -- metres squared
    against newtons squared -- a number that tells a reader nothing and
    invites someone to "fix" it later.

    Only the box's position is scored by the task objective. Where the
    pushers end up is their own business, and pricing it would be deciding
    for the policy how to use its second body. (`shaping=True` does score
    a pusher's distance from the box, but one-sidedly and only while it is
    away -- see `approach_penalty`.)

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

    # [1:] drops the initial state: u_t is scored against the state it
    # produces, Z_{t+1}, and no control reaches Z_0.
    xi = ramp.along_ramp(states, ramp_angles)[1:, :, variant.box]

    # Normalized by the box side, so the position term is in box-widths and
    # the two weights end up comparable rather than differing by ~1e5.
    error = (xi - np.asarray(targets)) / ramp.BOX_SIDE

    # Squared force magnitude summed over actuated bodies; 0:2 drops the torque
    # column, which no action writes.
    effort = sum((controls[..., body, 0:2] ** 2).sum(axis=-1) for body in variant.actuated)

    total = -weights["position"] * error ** 2 - weights["control"] * effort / variant.scale ** 2

    if shaping:
        # `approach_penalty` returns a positive cost over ALL steps, so it is
        # subtracted and sliced the same way the position term was.
        total = total - approach_penalty(states, ramp_angles, variant, weights)[1:]
    return total


def approach_penalty(states, ramp_angles, variant, weights=None):
    """REWARD SHAPING: the cost of a pusher not being at the box yet.

    Named as such on purpose. This is not part of the task objective; it
    is a term added to the *training* objective to remove a region where
    the task objective has no gradient at all.

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


def reward_seeds(states, controls, ramp_angles, targets, variant, weights=None, shaping=False):
    """dl_dZ and dl_dU for `adjoint_batch`, matching `reward` term for term.

    Deliberately adjacent to the reward it differentiates. The two have to
    agree, and the cheapest guarantee of that is a change to one being
    visibly next to the other -- a wrong seed raises nothing, it just
    quietly trains for something else.

    Unshaped, the state enters only through the box's xi = p . uphill, so

        dr/d(x, y)_box = -2*w_pos*(xi - xi*)/side^2 * uphill

    and every other entry is zero: the task objective does not see
    orientation, it does not see velocity, and it does not see where the
    pushers are. For each actuated body,

        dr/df_i = -2*w_ctrl*f_i / scale^2

    with the torque row zero. Both are partials of the stage cost only --
    GRIP supplies everything that makes them total derivatives.

    With `shaping=True` the pushers DO enter, through the separation term,
    and each one writes two entries rather than one -- its own position and
    the box's, with opposite signs, because a separation is a difference.
    That is the block at the bottom of this function, and it is why the
    seed check runs shaped and unshaped separately.

    For a *policy* these seeds are not the whole story. One `adjoint_batch`
    call over a window gives the open-loop gradient, which is right for a
    fixed control sequence and wrong for state feedback; see
    `tests/check_closed_loop.py`.
    """
    weights = variant.weights if weights is None else weights
    states, controls = np.asarray(states), np.asarray(controls)
    angles = np.atleast_1d(np.asarray(ramp_angles, dtype=float))

    xi = ramp.along_ramp(states, angles)[..., variant.box]

    # d/d(xi) of -w_pos*((xi - xi*)/side)^2. One side length comes from the
    # normalization inside the square, the other from differentiating it.
    coefficient = -2.0 * weights["position"] * (xi - np.asarray(targets)) / ramp.BOX_SIDE ** 2

    # Chain that through xi = position . uphill, whose derivative w.r.t. the
    # (x, y) position is just `uphill`. [:, :, None] opens an axis for the two
    # position components; [None, :, :] opens one for steps. [1:] again because
    # Z_0 is fixed. Every other entry stays zero: unshaped, the objective sees
    # no orientation, no velocity, and none of the pushers.
    dl_dZ = np.zeros(states.shape)
    dl_dZ[1:, :, variant.box, 0:2] = coefficient[1:, :, None] * ramp.uphill(angles)[None, :, :]

    # d/df of -w_ctrl*|f|^2/scale^2, written straight onto each actuated body.
    # The torque column is left at zero.
    dl_dU = np.zeros(controls.shape)
    for body in variant.actuated:
        dl_dU[..., body, 0:2] = -2.0 * weights["control"] * controls[..., body, 0:2] / variant.scale ** 2

    if shaping:
        # d/d(sep) of -w*max(0, sep/side)^2, chained through
        # sep = sign*(xi_body - xi_box) - reach and xi = p . uphill.
        up = ramp.uphill(angles)[None, :, :]
        for body, gap in zip(pushing_bodies(variant), separations(states, angles, variant)):
            sign = 1.0 if body > variant.box else -1.0

            # max(0, gap) rather than gap: the one-sided term has zero
            # derivative once the bodies touch, which is what makes the shaping
            # vanish on contact instead of distorting the task there.
            coefficient = -2.0 * weights["position"] * np.maximum(0.0, gap) / ramp.BOX_SIDE ** 2

            # A separation is a DIFFERENCE of two positions, so it writes two
            # entries with opposite signs -- moving the pusher toward the box
            # and moving the box toward the pusher shrink the same gap.
            # `+=` because several pushers can score against the same box.
            dl_dZ[1:, :, body, 0:2] += (sign * coefficient[1:])[:, :, None] * up
            dl_dZ[1:, :, variant.box, 0:2] += (-sign * coefficient[1:])[:, :, None] * up
    return dl_dZ, dl_dU


def observe(states, ramp_angles, targets, variant):
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

    `policy.Actor` consumes this and `check_policy_gradient` differentiates
    through it, so the frame and the gradient path are exercised. What is
    NOT exercised is whether these are the right channels -- whether a
    policy can actually control the box from them, and whether any of them
    is dead weight. Only training answers that.

    Being affine in the state is a property worth preserving rather than an
    accident: it is what makes `observation_jacobian` a constant matrix
    built once per batch instead of a derivative recomputed every step. A
    channel with a square or a norm in it would quietly cost that.
    """
    states = np.asarray(states)
    angles = np.atleast_1d(np.asarray(ramp_angles, dtype=float))
    targets = np.asarray(targets)
    up, out = ramp.uphill(angles), ramp.normal(angles)

    # GRIP packs a state as (x, y, theta, vx, vy, omega), so 0:2 is position
    # and 3:5 is linear velocity.
    position, velocity = states[..., 0:2], states[..., 3:5]

    # "...nbi,ni->...nb": for environment n and body b, dot the 2-vector i
    # against that environment's own basis vector. Leading step axes ride
    # along untouched, so a single state and a whole trajectory both work.
    # This is the projection into the ramp frame, and it is the reason every
    # channel below reads the same at any slope.
    xi = np.einsum("...nbi,ni->...nb", position, up)
    eta = np.einsum("...nbi,ni->...nb", position, out)
    v_up = np.einsum("...nbi,ni->...nb", velocity, up)
    v_out = np.einsum("...nbi,ni->...nb", velocity, out)

    # Body angle relative to its own ramp, so a body sitting flush reads zero.
    # angles[:, None] opens a body axis for the per-environment slope.
    tilt = states[..., 2] - angles[:, None]

    # Height above the surface, measured from where each body's centre of mass
    # sits when it rests flush -- so this is zero at rest for every shape,
    # including the pusher whose centroid is not at half its height.
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
    # Every other body is described RELATIVE to the box, which is what makes
    # the observation independent of where on the ramp the whole scene sits.
    # Four channels each, against the box's seven.
    for body in range(variant.bodies):
        if body != box:
            channels += [(xi[..., body] - xi[..., box]) / ramp.BOX_SIDE, v_up[..., body], gap[..., body], tilt[..., body]]

    # axis=-1 makes features the last axis, which is what torch's Linear wants.
    return np.stack(channels, axis=-1)


def observation_jacobian(ramp_angles, variant):
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

    # An affine map is f(Z) = A Z + c. Evaluating at Z = 0 isolates c...
    base = observe(np.zeros(shape), angles, targets, variant)

    # ...and f(e_k) - c is then exactly column k of A. One unit state per
    # (body, component) sweeps out the whole matrix in bodies*6 evaluations.
    # This is NOT a finite difference: for an affine map it is exact, with no
    # step size to choose. `check_task` asserts the affinity that licenses it.
    jacobian = np.zeros(base.shape + (variant.bodies, 6))
    for body in range(variant.bodies):
        for component in range(6):
            probe = np.zeros(shape)
            probe[:, body, component] = 1.0
            jacobian[..., body, component] = observe(probe, angles, targets, variant) - base
    return jacobian


def state_gradient(dJ_dobs, jacobian):
    """Pull an adjoint on the observation back to an adjoint on the state.

    The chain rule dJ/dZ = (dJ/d(obs)) . (d(obs)/dZ), contracted over the
    feature axis. Used twice in the backward sweep -- once for the critic's
    dV/dZ and once per step for the path through the policy.
    """
    # "...nf,nfbc->...nbc": sum over feature f, keeping environment n and
    # opening the state's (body, component) axes so the result is shaped like
    # a state and can be added straight to GRIP's dJ_dZ0.
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


def placement(box_xi, gaps, variant):
    """Where each body starts along the ramp, from the box and the approach gaps.

    Each pusher is set back so its contact vertex sits `gap` short of the
    box face it will meet. Zero starts them touching; a few centimetres
    buys an approach and an impact, which is the second thing the task doc
    wants instrumented and the one thing a standing start cannot show.

    The reach is read off the vertex list rather than written down, since
    the 3 degree trim moves it, and it is read from the correct side --
    the uphill pusher is mirrored, so its contact vertex is at -x.

    Loops over the PUSHING bodies rather than the actuated ones, which is
    what makes this total. Under BOX_ONLY the box is the actuated body,
    there is nothing to set back, and the loop simply does not run -- so
    callers no longer branch on the body count to avoid placing the box
    relative to itself.

    `gaps` broadcasts against (..., pushers), so a single number puts every
    pusher at the same standoff on every slope.
    """
    bodies = bodies_for(variant)
    pushers = pushing_bodies(variant)
    box_xi = np.asarray(box_xi, dtype=float)

    # Accepts a scalar, one gap per pusher, or one per environment per pusher.
    # Broadcasting here means the callers do not each build a full array.
    gaps = np.broadcast_to(np.asarray(gaps, dtype=float), box_xi.shape + (len(pushers),))

    positions = np.zeros(box_xi.shape + (variant.bodies,))
    positions[..., variant.box] = box_xi
    for slot, body in enumerate(pushers):
        # The exact inverse of what `separations` measures: centre-to-centre
        # at first touch, plus the standoff asked for. If these two ever
        # disagree, a "0 cm gap" start stops being a 0 mm separation.
        uphill_of_box = body > variant.box
        reach = ramp.contact_reach(bodies[body], toward_uphill=not uphill_of_box)
        offset = 0.5 * ramp.BOX_SIDE + reach + gaps[..., slot]
        positions[..., body] = box_xi + offset if uphill_of_box else box_xi - offset
    return positions


def build_batch(ramp_angles, start, offset, gaps, variant):
    """Scenes, a settled state and a target, assembled into one `Batch`.

    The single place a batch is put together, so `fixed_batch` and
    `sample_batch` differ only in where their numbers come from and cannot
    drift on the mechanics -- which body count the scenes get, whether the
    state was settled, or which direction the target offset is measured in.

    Offsets are measured from where the box actually ends up after
    settling, not from where it was placed, since everything creeps a
    millimetre or two during the window.
    """
    angles = np.atleast_1d(np.asarray(ramp_angles, dtype=float))
    bodies = bodies_for(variant)

    scenes = ramp.make_scenes(angles, bodies=bodies)
    substeps = substeps_for(scenes[0])

    # `start` broadcasts to one box position per environment, so a scalar puts
    # the box at the same point on every slope.
    positions = placement(np.broadcast_to(np.asarray(start, dtype=float), angles.shape), gaps, variant)

    # Flush placement, then let the contact spring ring down. Scoring through
    # that transient would make the opening of every episode a disturbance.
    state = settle(scenes, ramp.resting_state(positions, angles, bodies=bodies), substeps)

    # Measured from the SETTLED position, not the placed one -- everything
    # creeps a millimetre or two during the settle window.
    targets = ramp.along_ramp(state, angles)[:, variant.box] + offset

    # The Jacobian is constant because `observe` is affine, so it is built once
    # here and never recomputed inside a backward sweep.
    return Batch(scenes, angles, state, targets, substeps, observation_jacobian(angles, variant), variant)


def fixed_batch(ramp_angles, variant, start=0.0, offset=0.0, gaps=0.0):
    """A batch with every number stated rather than sampled.

    The deterministic sibling of `sample_batch`, and what the checks use.
    Each of them wants a specific slope and a specific target so its
    printed number means the same thing run to run; three of them had
    grown their own copy of the make-scenes-place-settle-target sequence,
    which is exactly the code that must not differ between what is checked
    and what is trained.
    """
    return build_batch(ramp_angles, start, offset, gaps, variant)


def sample_batch(rng, n_envs, variant):
    """One randomized episode setup per environment, already settled.

    Targets sit a fixed distance to either side of the start, not always
    uphill. Uphill-only would let a policy score well by learning "push
    hard uphill" with no representation of the target at all.
    """
    angles = rng.uniform(*RAMP_ANGLE_RANGE, size=n_envs)
    start = rng.uniform(*START_RANGE, size=n_envs)

    # Magnitude and direction drawn separately, so the target is never closer
    # than TARGET_OFFSET_RANGE[0] and lands on either side with equal odds.
    offset = rng.uniform(*TARGET_OFFSET_RANGE, size=n_envs) * rng.choice([-1.0, 1.0], size=n_envs)

    # Width 0 for BOX_ONLY, which has no pushing bodies -- `placement` then
    # never indexes it.
    gaps = rng.uniform(*APPROACH_GAP_RANGE, size=(n_envs, len(pushing_bodies(variant))))

    return build_batch(angles, start, offset, gaps, variant)
