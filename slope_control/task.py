"""What we are asking the controller to do: push a box to a target on a ramp.

Everything here decides what to ask for. Nothing here decides how bodies
move -- that is GRIP's half. So the reward, its derivatives, the action
limits, the observation frame and the episode length all live here.

Actions are forces, two components per actuated body, given in the RAMP
frame as (tangential, normal) and rotated into GRIP's world frame on the
way in. Ramp frame because the slope is different in every environment, so
a world-frame action would mean something different in each one. A
positive tangential action always pushes toward a larger `xi`. Nothing
ever applies a torque.

Two variants, because the force limit depends on what is being driven:

    TWO_PUSHERS   Pusher, box, pusher. The box is unactuated, so every
                  newton reaching it crosses a contact. Two pushers rather
                  than one because a pusher can only push, and one of them
                  can never undo an overshoot. This is the task.

    BOX_ONLY      One box, pushed directly from its centre. Physically
                  fictional, and kept only while SHAC is being brought up.
                  It is the same problem with the contact removed, so if a
                  policy fails on pushers and works here, the fault is the
                  contact and not the gradient path. Delete it once step 5
                  lands; CLAUDE.md has the decision.

No function defaults its `variant`. One of the two is fictional, and a
caller who forgets the argument must not silently get it.

The reward has two terms, position error and control effort. That pair is
the task objective, it is what gets reported, and it is the same in both
the penalty and the NCP column. A third term, `approach_penalty`, is
reward shaping -- off unless asked for, added only when training, never
included in a reported number.

Notation
--------
Two symbols, taken from GRIP's `docs/derivations/notation.md`, which is
canonical for anything the two repositories share:

    l       one step's reward. GRIP calls this slot the stage cost.
            Written `ell` there, ASCII `l` here.
    J       the whole objective being differentiated: the sum of `l` over
            a window, plus the critic's estimate of what follows it.

The distinction is partial against total, and it is the reason both exist:

    dl_dZ, dl_dU     PARTIAL derivatives of one step's reward. These go
                     INTO `adjoint_batch` as seeds. `reward_seeds` makes
                     them.
    dJ_dZ0, dJ_dU    TOTAL derivatives of the objective. These come OUT
                     of `adjoint_batch`. Converting the first into the
                     second is the entire job of the adjoint.

There is no separate `r`. If a comment needs the per-step quantity, it is
`l` or the English word "reward".

**`l` holds a reward, not a cost, so everything here MAXIMIZES.** That is
a deviation worth stating rather than leaving to be discovered. Optimal
control conventionally calls `ell` a stage cost and minimizes it. This
reward is negative-definite instead, and `check_trajopt` ascends it:

    actions += learning_rate * ...

Consistent throughout, so no arithmetic is wrong. But anyone reading
`dl_dZ` with the usual convention in mind will expect the opposite sign,
which is a trap in the most gradient-sensitive code here. If a future
change makes something a genuine cost, flip it everywhere at once and say
so, rather than letting the two conventions coexist.
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
    """Derive the control weight from a position tolerance.

    Answers one question: how far off target may the box sit before holding
    it there stops being worth the force? Set the two reward terms equal at
    that error and solve for the weight:

        w_ctrl * sum over actuated bodies of (hold force / scale)^2
            = (tolerance / side)^2

    The sum runs over every actuated body because under penalty contact
    each one needs continuous force just to stay put.

    Stating a tolerance rather than picking a weight means the number stays
    meaningful when the force scale or the set of bodies changes.
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
    """How many integration steps make up one control step.

    Works out to 20 at penalty contact's dt of 5e-4, giving 100 Hz control.

    Derived from the scene rather than hardcoded, because the two contact
    models integrate at different rates. Pinning the control rate instead
    of the substep count is what keeps them the same control problem, and
    therefore comparable.
    """
    return int(round(1.0 / (CONTROL_HZ * scene.dt)))


def clip_action(actions, variant):
    """Clamp each force component to the variant's limit.

    Tangential and normal are clamped separately, against their own
    entries, because they are bounded by different physics.

    This runs inside `to_wrench`, the only place an action becomes a force,
    so no caller can skip it. Two consequences:

    A force sweep that goes through `to_wrench` makes every value above the
    limit the same experiment. Build wrenches directly when measuring
    against force, or the results will silently be identical.

    `policy.Actor` already emits `limit * tanh(...)`, so for a policy this
    clip does nothing. It matters for callers that optimize a raw control
    sequence, such as `check_trajopt`, which needs its iterate pushed back
    inside the limit every iteration.

    Never widen the limit to stop a policy saturating. The limit is what
    keeps the bodies on the ramp.
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
    """Convert dJ/d(wrench) from GRIP into dJ/d(action).

    The inverse direction of `to_wrench`. That map is a rotation, so
    undoing it is just the transpose: take each actuated body's force
    gradient and dot it with uphill and with normal. Torque entries are
    dropped, since no action wrote one.

    Saturation is not handled here. A clipped action has zero derivative
    and the caller has to mask it if that matters. Leaving it out keeps
    this function to one job.
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
    """Distance from each pusher to the box face it will hit.

    Positive means still apart, zero means touching, negative means
    overlapping. Returns one array per pushing body, empty for BOX_ONLY.

    The exact inverse of `placement`, which sets a body up at a requested
    gap. If the two ever disagree, a start requested at 0 cm stops actually
    being 0 mm apart.
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
    """Score each step. Returns (steps, environments).

        l = -w_pos * ((box position - target) / side)^2
            -w_ctrl * (total force)^2 / scale^2

    `l` is one step's reward, the stage-cost slot in GRIP's notation. It
    is negative, and it is MAXIMIZED. See the module docstring.

    Control u_t is scored against the state it produces, Z_{t+1}. Nothing
    reaches Z_0, so it is left out and the result has one fewer entry than
    `states`.

    Both terms are divided by a natural scale, the box side and the force
    limit, so the two weights come out around one and can be compared. In
    raw metres and newtons the ratio would be about 1e-5, which tells a
    reader nothing.

    Only the box is scored. Where the pushers end up is their own problem,
    and pricing it would be deciding for the policy how to use its second
    body.

    The control term is not boilerplate. Under penalty contact, friction
    only appears when something is sliding, so a body held still gets no
    help from the surface and the controller pays mg*sin(a) for as long as
    it holds. Under a rigid solve, holding is free once the box has
    arrived. Same reward, two very different bills -- which is the thing
    the 1.0 and 2.0 columns are there to compare.

    `shaping=True` adds `approach_penalty`. Train with it on, report with
    it off. The two mistakes are not symmetric: training without it fails
    loudly, with the driving pusher stuck at 0.00 N, while reporting with
    it on fails silently and makes numbers incomparable between columns.
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
    """Penalize a pusher for not having reached the box yet.

        cost = w_pos * sum over pushers of max(0, separation / side)^2

    Positive, and returned for every step including the initial state, so
    callers slice it the way they slice the position term.

    This is reward shaping. It is not part of the task objective and it is
    off unless `reward(..., shaping=True)` asks for it. Train with it on,
    report with it off.

    It exists because the task objective has a region with no gradient at
    all. A pusher that is not touching the box has no effect on the box, so
    d(box position)/d(pusher action) is exactly zero. Starting from zero
    force with a 5 cm gap, trajectory optimization left the driving pusher
    at 0.00 N for all 300 iterations and the box 52 cm short. Gradients
    cannot find a contact that does not exist.

    The fix works because the pushers are actuated directly, so
    d(pusher position)/d(pusher action) is never zero, contact or not.

    Three properties worth keeping if this is ever changed:

      One-sided. `max(0, gap)` is exactly zero once the bodies touch, so
      the term cannot distort behaviour once the task objective takes
      over. Squaring keeps the derivative continuous at the kink.

      No new weight to tune. It reuses w_pos and the same normalization by
      the box side, which reads as: being away from the box costs the same
      as the box being away from its target, but only while it is away.

      Formulation-agnostic. Approaching is the same problem under penalty
      contact and under a rigid solve, so this belongs in both columns
      unchanged. Under a rigid solve it is not just convenient but
      required: penalty creep happens to close one of the two gaps by
      itself, and nothing creeps under a solve.

    Measured with it on, same setup as the failure above: both contacts
    form and the box lands 1.8-2.4 cm from target. The reward measured
    WITHOUT shaping improves from -1193 to -93, so the term is not buying
    its result by moving the goalposts.

    Still to check once a policy optimizes against this rather than an
    open-loop sequence. A policy has more freedom to find a degenerate way
    to satisfy a shaped term.
    """
    weights = variant.weights if weights is None else weights
    gaps = separations(states, ramp_angles, variant)
    if not gaps:
        return np.zeros(np.asarray(states).shape[:-2])
    return weights["position"] * sum(np.maximum(0.0, gap / ramp.BOX_SIDE) ** 2 for gap in gaps)


def reward_seeds(states, controls, ramp_angles, targets, variant, weights=None, shaping=False):
    """Differentiate `reward`, giving the two seed arrays `adjoint_batch` wants.

    Returns dl/d(state) and dl/d(control) -- the arrays the code calls
    `dl_dZ` and `dl_dU` -- each shaped like what it differentiates
    against.

    These are PARTIAL derivatives of one step's reward. `adjoint_batch`
    takes them as seeds and returns total derivatives of the whole
    objective. That partial-against-total split is what `l` and `J` mean
    here; the module docstring has the table.

    Kept next to `reward` on purpose. The two must agree term for term, and
    a seed that disagrees raises nothing -- it just trains for a different
    objective. Editing one where you can see the other is the cheapest
    protection against that.

    Unshaped, the state enters only through the box's position along the
    ramp, xi = position . uphill:

        dl/d(box x, y) = -2 * w_pos * (xi - target) / side^2 * uphill

    Everything else is zero. The task objective does not see orientation,
    velocity, or where the pushers are. For each actuated body:

        dl/d(force) = -2 * w_ctrl * force / scale^2

    with the torque entry zero, since no action writes one.

    With `shaping=True` the pushers do enter, through the separation term.
    Each pusher writes two entries instead of one, its own position and the
    box's, with opposite signs, because a separation is a difference of the
    two. `check_task` therefore checks the shaped and unshaped seeds
    separately.

    These seeds are not the whole gradient for a policy. Contracting them
    with one `adjoint_batch` call over a window gives the open-loop
    gradient, which is correct for a fixed control sequence and wrong for
    state feedback. See `policy.policy_gradient`.
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
    """Build what the policy sees. Returns (..., environments, features).

    Seven numbers for the box:

        (xi - target) / side     how far off target, in box widths
        v_uphill                 speed along the ramp
        gap / side               height above the surface
        v_normal                 speed away from the surface
        theta - a                tilt relative to the ramp
        omega                    spin
        sin(a)                   the slope itself

    Then four more for each other body, measured relative to the box:

        (xi_body - xi_box) / side, v_uphill, gap / side, theta - a

    Everything is in the ramp frame, so it reads the same at any slope, and
    it matches the frame the actions are in -- a positive first action
    always pushes toward a larger first observation.

    sin(a) is the one channel that is not slope-invariant, and it is here
    on purpose. It sets how hard gravity is pulling, which the controller
    has to fight. Without it the policy could only infer the slope by
    watching how fast things slide.

    `observe` is affine in the state, and that is worth preserving. It is
    what lets `observation_jacobian` be a constant matrix built once per
    batch instead of a derivative recomputed at every step of every
    backward sweep. Adding a channel with a square or a norm in it would
    quietly cost that.

    These channels are exercised for shape and for their gradient, but
    nothing yet says they are the RIGHT channels -- whether a policy can
    control the box from them, or whether any is dead weight. Only training
    answers that.
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

    Constant for a given batch, so build it once and reuse it. That is only
    valid because `observe` is affine in the state: every channel is either
    a projection onto uphill or normal, a difference of two of those, or a
    state component passed through. The one nonlinear-looking channel,
    sin(a), does not depend on the state at all.

    The backward sweep needs d(observation)/d(state) at every step to carry
    the adjoint through the policy. The other half of that chain, the
    network's own input gradient, has to be recomputed each step. This half
    does not, which is why it lives on the batch.

    Built by evaluating `observe` on unit states, not by finite
    differences. For an affine map, f(e_k) - f(0) is exactly column k, with
    no step size to choose and no truncation error. `check_task` asserts
    the affinity this depends on.
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


def placement(box_xi, gaps, variant):
    """Place every body along the ramp, given where the box goes and how far
    back each pusher should start.

    A gap of zero starts a pusher touching the box. A few centimetres buys
    an approach and an impact, which a standing start cannot show.

    `gaps` broadcasts, so one number puts every pusher at the same standoff
    on every slope.

    Works for BOX_ONLY too: it has no pushing bodies, the loop does not
    run, and the box is placed where asked.
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
    """Assemble a `Batch`: build the scenes, place the bodies, settle them,
    and work out the targets.

    The one place this sequence lives, so `fixed_batch` and `sample_batch`
    differ only in where their numbers come from. They cannot drift apart
    on the mechanics -- body count, whether the state was settled, or where
    the target offset is measured from.

    `offset` is measured from where the box ends up AFTER settling, not
    where it was placed. Everything creeps a millimetre or two while
    settling.
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
    """Build a batch from stated numbers rather than sampled ones.

    The deterministic counterpart to `sample_batch`, used by the checks, so
    that a printed number means the same thing on every run.
    """
    return build_batch(ramp_angles, start, offset, gaps, variant)


def sample_batch(rng, n_envs, variant):
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

    # Width 0 for BOX_ONLY, which has no pushing bodies -- `placement` then
    # never indexes it.
    gaps = rng.uniform(*APPROACH_GAP_RANGE, size=(n_envs, len(pushing_bodies(variant))))

    return build_batch(angles, start, offset, gaps, variant)
