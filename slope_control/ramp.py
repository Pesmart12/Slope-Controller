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
    return np.array([math.cos(ramp_angle), math.sin(ramp_angle)])


def normal(ramp_angle):
    return np.array([-math.sin(ramp_angle), math.cos(ramp_angle)])


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


def resting_state(xi, ramp_angle=DEFAULT_RAMP_ANGLE, side=BOX_SIDE):
    """A box sitting flush on the ramp at position `xi` along it.

    Flush means the bottom face is parallel to the surface, so the body
    angle is the ramp angle, and the centre of mass sits half a side out
    along the normal. The resulting gap is exactly zero -- the box settles
    the further mg*cos(a)/2k into the surface within the first few
    milliseconds of any rollout.
    """
    centre = xi * uphill(ramp_angle) + 0.5 * side * normal(ramp_angle)
    state = np.zeros((1, 1, 6))
    state[0, 0, 0:2] = centre
    state[0, 0, 2] = ramp_angle
    return state


def along_ramp(states, ramp_angle=DEFAULT_RAMP_ANGLE):
    """Project body positions onto the uphill direction.

    Accepts any array whose last axis is GRIP's 6-vector, so it works on a
    single state, a batch, or a whole trajectory, and returns the matching
    shape with the last axis dropped.
    """
    return np.asarray(states)[..., 0:2] @ uphill(ramp_angle)


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
    return mass * GRAVITY * math.sin(ramp_angle) / (2.0 * penalty["slip_damping"])
