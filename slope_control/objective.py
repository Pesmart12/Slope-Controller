"""The task objective: the reward, its shaping term, and their gradient seeds.

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

import numpy as np

from . import ramp, task

# How far off target the policy may sit before it is worth spending the
# steady holding force to close the gap. Rather under the 4.75 cm that
# drift.py measures for doing nothing at all, so the controller is being
# asked for something a released box does not already achieve.
POSITION_TOLERANCE = 0.02


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


# The isotropic force scale the control cost divides by.
SCALE = 30.0

# Position and control, the latter derived from POSITION_TOLERANCE. The hold
# forces are what each actuated body pays to stay put: the driving pusher
# carries itself and the box, the trailing one only itself.
WEIGHTS = dict(position=1.0, control=control_weight(SCALE, [ramp.hold_force(ramp.DEFAULT_RAMP_ANGLE, mass=ramp.PUSHER_MASS + ramp.BOX_MASS), ramp.hold_force(ramp.DEFAULT_RAMP_ANGLE, mass=ramp.PUSHER_MASS)]))


def separations(states, ramp_angles):
    """Distance from each pusher to the box face it will hit.

    Positive means still apart, zero means touching, negative means
    overlapping. Returns one array per pushing body.

    The exact inverse of `batches.placement`, which sets a body up at a
    requested gap. Both read their geometry from `task.TOUCHING`, so they
    cannot disagree.
    """
    xi = ramp.along_ramp(states, ramp_angles)

    # Signing the centre-to-centre distance before subtracting the touching
    # distance is what makes positive mean "still apart" on either side.
    return [sign * (xi[..., body] - xi[..., task.BOX]) - touching
            for body, (sign, touching) in zip(task.PUSHING_BODIES, task.TOUCHING)]


def check_trajectory(states, controls):
    """Raise unless `states` and `controls` are one trajectory with its controls.

    Expects states (steps + 1, environments, bodies, 6) and controls
    (steps, environments, bodies, 3). `reward` and `reward_seeds` slice
    [1:] off the first axis, so a single state of shape (environments,
    bodies, 6) would otherwise lose its first environment and return
    wrong numbers with no error.
    """
    if states.ndim != 4 or controls.shape[:-1] != (states.shape[0] - 1,) + states.shape[1:3]:
        raise ValueError(f"expected states (steps + 1, environments, bodies, 6) and controls (steps, environments, bodies, 3), got {states.shape} and {controls.shape}")


def reward(states, controls, ramp_angles, targets, shaping=False):
    """Score each step. Returns (steps, environments).

        l = -w_pos * ((box position - target) / side)^2
            -w_ctrl * (total force)^2 / scale^2

    `l` is one step's reward, the stage-cost slot in GRIP's notation. It
    is negative, and it is MAXIMIZED. See the module docstring.

    Control u_t is scored against the state it produces, Z_{t+1}. Nothing
    reaches Z_0, so it is left out and the result has one fewer entry than
    `states`. That makes a leading step axis required, unlike
    `observation.observe`, and `check_trajectory` raises without one.

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
    states, controls = np.asarray(states), np.asarray(controls)
    check_trajectory(states, controls)

    # [1:] drops the initial state: u_t is scored against the state it
    # produces, Z_{t+1}, and no control reaches Z_0.
    xi = ramp.along_ramp(states, ramp_angles)[1:, :, task.BOX]

    # Normalized by the box side, so the position term is in box-widths and
    # the two weights end up comparable rather than differing by ~1e5.
    error = (xi - np.asarray(targets)) / ramp.BOX_SIDE

    # Squared force magnitude summed over actuated bodies; 0:2 drops the torque
    # column, which no action writes. These are world-frame wrenches, but
    # `task.to_wrench` is a rotation and |f|^2 is invariant under one, so this
    # is the same number the ramp-frame action would give.
    effort = sum((controls[..., body, 0:2] ** 2).sum(axis=-1) for body in task.ACTUATED)

    total = -WEIGHTS["position"] * error ** 2 - WEIGHTS["control"] * effort / SCALE ** 2

    if shaping:
        # `approach_penalty` returns a positive cost over ALL steps, so it is
        # subtracted and sliced the same way the position term was.
        total = total - approach_penalty(states, ramp_angles)[1:]
    return total


def approach_penalty(states, ramp_angles):
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
    form and the box lands within a centimetre at every sampled slope. The
    reward measured WITHOUT shaping improves from -1195 to -82, so the term
    is not buying its result by moving the goalposts.

    Still to check once a policy optimizes against this rather than an
    open-loop sequence. A policy has more freedom to find a degenerate way
    to satisfy a shaped term.
    """
    gaps = separations(states, ramp_angles)
    return WEIGHTS["position"] * sum(np.maximum(0.0, gap / ramp.BOX_SIDE) ** 2 for gap in gaps)


def reward_seeds(states, controls, ramp_angles, targets, shaping=False):
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
    state feedback. See `sweep.policy_gradient`.
    """
    states, controls = np.asarray(states), np.asarray(controls)
    check_trajectory(states, controls)
    angles = ramp.as_angles(ramp_angles)

    xi = ramp.along_ramp(states, angles)[..., task.BOX]

    # d/d(xi) of -w_pos*((xi - xi*)/side)^2. One side length comes from the
    # normalization inside the square, the other from differentiating it.
    coefficient = -2.0 * WEIGHTS["position"] * (xi - np.asarray(targets)) / ramp.BOX_SIDE ** 2

    # Chain that through xi = position . uphill, whose derivative w.r.t. the
    # (x, y) position is just `uphill`. [:, :, None] opens an axis for the two
    # position components; [None, :, :] opens one for steps. [1:] again because
    # Z_0 is fixed. Every other entry stays zero: unshaped, the objective sees
    # no orientation, no velocity, and none of the pushers.
    dl_dZ = np.zeros(states.shape)
    dl_dZ[1:, :, task.BOX, 0:2] = coefficient[1:, :, None] * ramp.uphill(angles)[None, :, :]

    # d/df of -w_ctrl*|f|^2/scale^2, written straight onto each actuated body.
    # The torque column is left at zero.
    dl_dU = np.zeros(controls.shape)
    for body in task.ACTUATED:
        dl_dU[..., body, 0:2] = -2.0 * WEIGHTS["control"] * controls[..., body, 0:2] / SCALE ** 2

    if shaping:
        # d/d(sep) of -w*max(0, sep/side)^2, chained through
        # sep = sign*(xi_body - xi_box) - reach and xi = p . uphill.
        up = ramp.uphill(angles)[None, :, :]
        for body, gap in zip(task.PUSHING_BODIES, separations(states, angles)):
            sign = 1.0 if body > task.BOX else -1.0

            # max(0, gap) rather than gap: the one-sided term has zero
            # derivative once the bodies touch, which is what makes the shaping
            # vanish on contact instead of distorting the task there.
            coefficient = -2.0 * WEIGHTS["position"] * np.maximum(0.0, gap) / ramp.BOX_SIDE ** 2

            # A separation is a DIFFERENCE of two positions, so it writes two
            # entries with opposite signs -- moving the pusher toward the box
            # and moving the box toward the pusher shrink the same gap.
            # `+=` because several pushers can score against the same box.
            dl_dZ[1:, :, body, 0:2] += (sign * coefficient[1:])[:, :, None] * up
            dl_dZ[1:, :, task.BOX, 0:2] += (-sign * coefficient[1:])[:, :, None] * up
    return dl_dZ, dl_dU
