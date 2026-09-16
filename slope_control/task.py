"""What we are asking the controller to do: push a box to a target on a ramp.

Everything in this module and its three siblings decides what to ask for.
Nothing here decides how bodies move -- that is GRIP's half.

This module holds what the other three share: the scene layout, the action
limit and frame, and the episode length. The rest is split by job:

    objective     the reward, its shaping term, and their gradient seeds
    observation   what the policy sees, and its constant Jacobian
    batches       building a `Batch`: scenes, placement, settling, targets

Actions are forces, two components per actuated body, given in the RAMP
frame as (tangential, normal) and rotated into GRIP's world frame on the
way in. Ramp frame because the slope is different in every environment, so
a world-frame action would mean something different in each one. A
positive tangential action always pushes toward a larger `xi`. Nothing
ever applies a torque.

The scene is pusher, box, pusher. The box is unactuated, so every newton
reaching it crosses a contact. Two pushers rather than one because a pusher
can only push, and one of them can never undo an overshoot.
"""

import numpy as np

from . import ramp

CONTROL_HZ = 100.0
EPISODE_SECONDS = 4.0
EPISODE_STEPS = int(round(EPISODE_SECONDS * CONTROL_HZ))

# The scene: pusher, box, pusher, downhill to uphill.
BODIES = ramp.TWO_PUSHER_BODIES
BODY_COUNT = len(BODIES)

# Which body the reward scores, and which take an action, in action order.
BOX = 1
ACTUATED = (0, 2)

# Actuated bodies that are not the scored one -- the ones that approach.
PUSHING_BODIES = [body for body in ACTUATED if body != BOX]

# Per pushing body: the sign that makes a separation positive while it is
# still apart, and the centre-to-centre distance at which it first touches
# the box.
#
# The sign is +1 for a pusher sitting uphill of the box. That pusher is the
# mirrored shape and reaches with its -x vertex, which is why
# `toward_uphill` is the opposite of the side it sits on.
#
# Both depend only on geometry. `objective.separations` and
# `batches.placement` are inverses of each other and this is the arithmetic
# they have to agree on, so they read it from here rather than each deriving
# it -- which makes them agree by construction instead of by assertion.
TOUCHING = [(1.0 if body > BOX else -1.0, 0.5 * ramp.BOX_SIDE + ramp.contact_reach(BODIES[body], toward_uphill=body < BOX))
            for body in PUSHING_BODIES]

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
# The driving pusher must move itself and the box, 24.67 N at the steepest
# slope, against its own 18.19 N normal load. 30 N is deliberately short of
# the 41.11 N that would shove a passive partner along, so the trailing
# pusher has to be driven out of the way rather than bulldozed -- which is
# the reason for having two of them. Coordinated, the numbers are
# comfortable: 24.67 N to drive the box, 16.45 N for the trailing pusher to
# carry itself, both inside 30.
#
# Tipping is NOT a constraint, though an earlier draft claimed it was. A
# wrench acts at the centre of mass so exerts no moment there, and
# friction's couple is balanced by the centre of pressure shifting within
# the base. Measured: 2 mrad on the box at 25 N, 4 mrad on the pusher at
# 60 N.
LIMIT = np.array([30.0, 12.0])


def substeps_for(scene):
    """How many integration steps make up one control step.

    Works out to 20 at penalty contact's dt of 5e-4, giving 100 Hz control.

    Derived from the scene rather than hardcoded, because the two contact
    models integrate at different rates. Pinning the control rate instead
    of the substep count is what keeps them the same control problem, and
    therefore comparable.
    """
    return int(round(1.0 / (CONTROL_HZ * scene.dt)))


def clip_action(actions):
    """Clamp each force component to `LIMIT`.

    Tangential and normal are clamped separately, against their own
    entries, because they are bounded by different physics.

    Callers apply this; `to_wrench` does not. An optimizer working on a raw
    control sequence has to project its iterate back inside the limit every
    iteration, and `check_trajopt` does. `policy.Actor` needs nothing: it
    emits `limit * tanh(...)`, so every action it produces is already
    feasible and this would be a no-op.

    Clipping inside `to_wrench` instead would be worse than useless. It
    would silently make every value above the limit the same experiment, so
    a force sweep run through the conversion would return identical results
    and look like physics.

    Never widen the limit to stop a policy saturating. The limit is what
    keeps the bodies on the ramp.
    """
    return np.clip(np.asarray(actions, dtype=float), -LIMIT, LIMIT)


def to_wrench(actions, ramp_angles):
    """Ramp-frame actions (..., N, actuated, 2) -> GRIP wrenches (..., N, bodies, 3).

    The columns of the map are `uphill` and `normal`, which is the rotation
    by the ramp angle. Unactuated bodies get zero rows, and the torque row
    is zero for everything.

    A pure conversion: it does not clip. See `clip_action` for why that
    would be worse than useless here.
    """
    actions = np.asarray(actions, dtype=float)
    angles = ramp.as_angles(ramp_angles)

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
    wrench = np.zeros(actions.shape[:-2] + (BODY_COUNT, 3))
    for slot, body in enumerate(ACTUATED):
        wrench[..., body, 0:2] = force[..., slot, :]
    return wrench


def to_action_gradient(dJ_dU, ramp_angles):
    """Convert dJ/d(wrench) from GRIP into dJ/d(action).

    The inverse direction of `to_wrench`. That map is a rotation, so
    undoing it is just the transpose: take each actuated body's force
    gradient and dot it with uphill and with normal. Torque entries are
    dropped, since no action wrote one.

    Saturation is not handled here. A clipped action has zero derivative
    and the caller has to mask it if that matters. Leaving it out keeps
    this function to one job.
    """
    angles = ramp.as_angles(ramp_angles)
    up, out = ramp.uphill(angles), ramp.normal(angles)
    dJ_dU = np.asarray(dJ_dU)

    gradients = []
    for body in ACTUATED:
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
