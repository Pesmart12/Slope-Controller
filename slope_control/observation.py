"""What the policy sees: the observation, and its constant Jacobian."""

import numpy as np

from . import ramp, task

# How far each body's centre of mass sits from the surface when it rests
# flush, one entry per body. Built once because `observe` runs at every
# control step and this never changes.
RESTING_OFFSETS = np.array([ramp.resting_offset(body["vertices"]) for body in task.BODIES])


def observe(states, ramp_angles, targets):
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
    angles = ramp.as_angles(ramp_angles)
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
    gap = (eta - RESTING_OFFSETS) / ramp.BOX_SIDE

    channels = [
        (xi[..., task.BOX] - targets) / ramp.BOX_SIDE,
        v_up[..., task.BOX],
        gap[..., task.BOX],
        v_out[..., task.BOX],
        tilt[..., task.BOX],
        states[..., task.BOX, 5],
        np.broadcast_to(np.sin(angles), xi[..., task.BOX].shape),
    ]
    # Every other body is described RELATIVE to the box, which is what makes
    # the observation independent of where on the ramp the whole scene sits.
    # Four channels each, against the box's seven.
    for body in range(task.BODY_COUNT):
        if body != task.BOX:
            channels += [(xi[..., body] - xi[..., task.BOX]) / ramp.BOX_SIDE, v_up[..., body], gap[..., body], tilt[..., body]]

    # axis=-1 makes features the last axis, which is what torch's Linear wants.
    return np.stack(channels, axis=-1)


def observation_jacobian(ramp_angles):
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
    angles = ramp.as_angles(ramp_angles)
    shape = (angles.size, task.BODY_COUNT, 6)
    targets = np.zeros(angles.size)  # only shifts the constant term

    # An affine map is f(Z) = A Z + c. Evaluating at Z = 0 isolates c...
    base = observe(np.zeros(shape), angles, targets)

    # ...and f(e_k) - c is then exactly column k of A. One unit state per
    # (body, component) sweeps out the whole matrix in bodies*6 evaluations.
    # This is NOT a finite difference: for an affine map it is exact, with no
    # step size to choose. `check_task` asserts the affinity that licenses it.
    jacobian = np.zeros(base.shape + (task.BODY_COUNT, 6))
    for body in range(task.BODY_COUNT):
        for component in range(6):
            probe = np.zeros(shape)
            probe[:, body, component] = 1.0
            jacobian[..., body, component] = observe(probe, angles, targets) - base
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
