"""The ramp scene, shared by every experiment here.

Geometry conventions, fixed once so nothing downstream has to rederive
them:

    uphill  = ( cos a,  sin a)   the direction of increasing height
    normal  = (-sin a,  cos a)   into free space, perpendicular to uphill
    offset  = 0                  the ramp passes through the origin

At a = 0 the normal is (0, 1) and this reduces to flat ground, which is a
cheap check that the signs are right. Gravity (0, -g) projected onto
uphill gives -g*sin(a), so it pulls downhill as it should.

Position along the ramp is written `xi` throughout -- a single coordinate
is enough because a box resting on the surface has only one degree of
freedom that any task here cares about.

The slope is per environment. Every function below takes either a scalar
`ramp_angle` or an array of `ramp_angles`, one per environment, because
the task randomizes the slope across a batch and a single shared angle
would be a quiet way to get that wrong.
"""

import math

import numpy as np

import grip

# Below the friction angle atan(mu), where rigid physics says a released
# box holds its position for all time. Penalty contact violates that
# visibly; an NCP solve does not. Every angle used here stays below it.
DEFAULT_RAMP_ANGLE = math.radians(20.0)

# GRIP's demo constants, so these numbers describe a configuration that is
# already exercised by a running program rather than one invented here.
DEFAULT_PENALTY = dict(stiffness=1.0e4, damping=50.0, slip_damping=200.0, friction=0.5)
DEFAULT_TIMESTEP = 5.0e-4

BOX_SIDE = 0.3
BOX_MASS = 1.0

GRAVITY = 9.81


def friction_angle(mu):
    """The steepest slope a rigid box rests on. atan(0.5) is 26.6 degrees."""
    return math.atan(mu)


def uphill(ramp_angle):
    """(cos a, sin a). A scalar gives (2,), an array of N gives (N, 2)."""
    a = np.asarray(ramp_angle, dtype=float)
    return np.stack([np.cos(a), np.sin(a)], axis=-1)


def normal(ramp_angle):
    """(-sin a, cos a). A scalar gives (2,), an array of N gives (N, 2)."""
    a = np.asarray(ramp_angle, dtype=float)
    return np.stack([-np.sin(a), np.cos(a)], axis=-1)


def scene_angle(scene):
    """Recover a ramp angle from a built Scene's plane normal.

    Angles and scenes travel as separate values and nothing structurally
    stops them disagreeing, which would project a batch onto the wrong
    slopes while every shape still lined up. Reading the angle back out
    of the scene turns that silent failure into a check.
    """
    n = np.asarray(scene.plane.normal, dtype=float)
    return math.atan2(-n[0], n[1])


def square_vertices(side):
    """Counterclockwise, which GRIP's SAT path requires."""
    h = 0.5 * side
    return [[-h, -h], [h, -h], [h, h], [-h, h]]


def make_scene(ramp_angle=DEFAULT_RAMP_ANGLE, dt=DEFAULT_TIMESTEP, penalty=None, side=BOX_SIDE, mass=BOX_MASS):
    """One box on a tilted half-plane."""
    penalty = DEFAULT_PENALTY if penalty is None else penalty
    inertia = mass * side * side / 6.0  # a square about its centre
    return grip.Scene(
        params=[grip.RigidBodyParams(mass=mass, inertia=inertia)],
        shapes=[grip.BodyShape(square_vertices(side))],
        plane=grip.HalfPlane(normal=normal(ramp_angle).tolist(), offset=0.0),
        penalty=grip.PenaltyParams(**penalty),
        dt=dt,
        gravity=GRAVITY,
    )


def make_scenes(ramp_angles, dt=DEFAULT_TIMESTEP, penalty=None, side=BOX_SIDE, mass=BOX_MASS):
    """One Scene per environment, each carrying its own slope.

    GRIP requires only that the body count match across a batch, so the
    ramp angle randomizes for free. That is the whole reason the task's
    per-episode slope costs nothing.
    """
    return [make_scene(ramp_angle=a, dt=dt, penalty=penalty, side=side, mass=mass) for a in np.atleast_1d(ramp_angles)]


def resting_state(xi, ramp_angles, side=BOX_SIDE):
    """Boxes sitting flush on their ramps at `xi` along them, shaped (N, 1, 6).

    Flush means the bottom face is parallel to the surface, so the body
    angle is the ramp angle, and the centre of mass sits half a side out
    along the normal. The resulting gap is exactly zero.

    Deliberately *not* the penalty equilibrium, for two reasons. It is
    unreachable by a uniform offset, because friction's moment arm tilts
    the box and loads the downhill corner harder than the uphill one, so
    the two corners settle to different depths. And it is a
    penalty-contact fact, so starting there would hand 1.0 and 2.0
    different initial conditions and quietly spoil the comparison they
    exist for. Start flush and let `task.settle` ring it down instead.

    `xi` broadcasts against `ramp_angles`, so a scalar places every
    environment at the same point along its own slope.
    """
    angles = np.atleast_1d(np.asarray(ramp_angles, dtype=float))
    xi = np.broadcast_to(np.asarray(xi, dtype=float), angles.shape)
    state = np.zeros((angles.size, 1, 6))
    state[:, 0, 0:2] = xi[:, None] * uphill(angles) + 0.5 * side * normal(angles)
    state[:, 0, 2] = angles
    return state


def along_ramp(states, ramp_angles):
    """Project body positions onto each environment's uphill direction.

    Expects GRIP's layout with the environment axis at -3, so a batch
    (environments, bodies, 6) and a whole trajectory (steps,
    environments, bodies, 6) both work and the last axis drops.

    Unlike the single-angle version this cannot take a squeezed array.
    With one angle per environment there is no way left to tell which
    axis is which, and guessing is how a batch silently projects onto the
    wrong slopes.
    """
    states = np.asarray(states)
    return np.einsum("...nbi,ni->...nb", states[..., 0:2], uphill(np.atleast_1d(ramp_angles)))


def creep_rate(ramp_angle=DEFAULT_RAMP_ANGLE, penalty=None, mass=BOX_MASS):
    """The drift penalty contact cannot avoid: mg*sin(a) / (2*b_slip).

    Sticking under a penalty law is not s = 0 but s = -beta/b_slip, so a
    held box must slide at exactly the rate that generates the friction
    holding it up.

    The 2 is the number of contact points. A box resting on a face
    contacts at both bottom corners, each carrying its own friction, so
    the pair together supply 2*b_slip*s and each has to creep only half as
    fast as a single contact would. Measured exactly, to five figures,
    from 5 to 18 degrees.

    Above roughly 19 degrees this UNDERESTIMATES, by 13% at 20 and 48% at
    26. Friction acts at the contact points, 0.15 m below the centre of
    mass, so it tips the box by about a milliradian; the tilt redistributes
    normal force between the corners by k*w*dtheta, and the lightly loaded
    uphill corner saturates on its own cone bound mu*lambda. The downhill
    corner then has to make up the shortfall, and creeps faster than an
    even split would. Both corners saturating is the friction angle
    itself, where the box slides freely and none of this applies.

    So: exact while both corners stick, a lower bound once one does not.
    """
    penalty = DEFAULT_PENALTY if penalty is None else penalty
    return mass * GRAVITY * np.sin(ramp_angle) / (2.0 * penalty["slip_damping"])


def hold_force(ramp_angle=DEFAULT_RAMP_ANGLE, mass=BOX_MASS):
    """What a controller must supply to hold a box perfectly still: mg*sin(a).

    Under penalty contact friction is beta = -b_slip*s, so zero slip is
    zero friction. A motionless box gets no help at all from the surface
    and the controller pays the entire gravity component for as long as
    it holds. Rigid friction does the same job for free on any slope
    below the friction angle.

    This is what the reward's control term is pricing, and it is a floor
    rather than a tuning artifact: no feedback law beats it, because the
    mechanism that would let it stop pushing is the one penalty contact
    removes.
    """
    return mass * GRAVITY * np.sin(ramp_angle)


def break_free_force(ramp_angle=DEFAULT_RAMP_ANGLE, penalty=None, mass=BOX_MASS):
    """What it takes to actually move the box uphill: mg(sin a + mu*cos a).

    Holding and moving are different problems and the gap between them is
    the whole friction cone. `hold_force` only cancels gravity, leaving the
    box stationary; to slide it uphill you must also overrun friction at
    its bound mu*lambda, and lambda is the full normal load mg*cos(a).

    Measured exact at 15, 20 and 22 degrees: below this the box does not
    move at all -- what looks like motion is arrested creep, a couple of
    centimetres over a whole episode -- and a newton above it the box
    accelerates away. Coulomb friction has no gentle regime, which is what
    makes this a real control problem rather than a set-and-forget one.

    An earlier force limit was sized against `hold_force` alone and came
    out at 5 N, below this threshold at every sampled slope, so the task
    was literally unsolvable until `check_trajopt.py` said so.
    """
    penalty = DEFAULT_PENALTY if penalty is None else penalty
    return mass * GRAVITY * (np.sin(ramp_angle) + penalty["friction"] * np.cos(ramp_angle))


def normal_load(ramp_angle=DEFAULT_RAMP_ANGLE, mass=BOX_MASS):
    """mg*cos(a), the force pinning the box to the ramp.

    An outward push at or above this unloads the contact entirely and the
    body leaves the surface, which is the one thing the action limit
    exists to prevent.
    """
    return mass * GRAVITY * np.cos(ramp_angle)
