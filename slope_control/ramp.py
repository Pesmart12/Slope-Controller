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

# Wide and short, so it cannot tip and needs no orientation controller --
# measured at 4 mrad under 60 N. The box-facing edge is tilted, which is
# what keeps the pusher off GRIP's parallel-face tie-break: it meets the
# box at a vertex instead of flush.
#
# The height is half the box's side on purpose. The tilt only displaces
# vertices in x, so the contact vertex always sits a full body-height above
# the ramp, and at this height that is exactly the box's centre of mass --
# the push therefore exerts no tipping moment on the box, at any tilt.
PUSHER_WIDTH = 0.30
PUSHER_HEIGHT = 0.15
PUSHER_MASS = 2.0
PUSHER_FACE_TILT = math.radians(3.0)

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


def polygon_properties(vertices):
    """Area, centroid and second moment about the centroid, for a CCW polygon.

    Needed because the pusher is not a rectangle and its centroid is not
    where the rectangle's centre would be. GRIP takes `BodyShape` vertices
    in a frame centred on the centre of mass, so an un-recentred list puts
    a standing torque offset into every contact the body makes.
    """
    v = np.asarray(vertices, dtype=float)
    x, y = v[:, 0], v[:, 1]

    # roll(-1) pairs each vertex with the next one and wraps the last back to
    # the first, so `cross` holds one edge cross product per edge with no
    # special case for closing the loop.
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y

    # The shoelace formula. Positive area is the check that the winding is
    # counterclockwise, which GRIP's SAT path requires.
    area = 0.5 * cross.sum()

    # Standard polygon centroid: each edge contributes its midpoint weighted by
    # its cross product, and 6*area normalizes.
    centroid = np.array([((x + xn) * cross).sum(), ((y + yn) * cross).sum()]) / (6.0 * area)

    # Second moments about the ORIGIN, not the centroid. Note the axis naming
    # is the moment-of-area convention: ixx integrates y^2.
    ixx = ((y * y + y * yn + yn * yn) * cross).sum() / 12.0
    iyy = ((x * x + x * xn + xn * xn) * cross).sum() / 12.0

    # ixx + iyy is the polar moment about the origin, which is the only
    # combination a 2D rigid body needs. The subtraction is the parallel axis
    # theorem run backwards, shifting it from the origin to the centroid.
    return area, centroid, (ixx + iyy) - area * centroid.dot(centroid)


def pusher_vertices(width=PUSHER_WIDTH, height=PUSHER_HEIGHT, tilt=PUSHER_FACE_TILT):
    """The pusher's trapezoid, counterclockwise and centred on its centroid.

    The +x face is the one that meets the box. Leaning it outward puts the
    contact at the TOP vertex; leaning it the other way would put contact
    at the bottom vertex, down where the ramp contact already is.

    Derived rather than written down, because the 3 degree trim moves the
    centroid about 2 mm and shifts the inertia 2%, and a hardcoded vertex
    list is exactly the sort of thing that silently stops matching the
    constants above it.
    """
    half_w, half_h = 0.5 * width, 0.5 * height

    # How far the BOTTOM of the +x face is pulled back to lean it by `tilt`.
    # Only the bottom vertex moves, so the top one stays the contact point.
    trim = height * math.tan(tilt)

    # Counterclockwise from the bottom-left. The second entry is the trimmed
    # bottom-right; the third is the untouched top-right that meets the box.
    raw = [[-half_w, -half_h], [half_w - trim, -half_h], [half_w, half_h], [-half_w, half_h]]

    # GRIP takes vertices in a frame centred on the centre of mass, and the
    # trim moved the centroid ~2 mm off the rectangle's centre. Not recentring
    # would put a standing torque offset into every contact this body makes.
    _, centroid, _ = polygon_properties(raw)
    return [[vx - centroid[0], vy - centroid[1]] for vx, vy in raw]


def uniform_inertia(vertices, mass):
    """Second moment of a uniform-density polygon about its own centroid."""
    area, _, second = polygon_properties(vertices)
    return mass * second / area


def resting_offset(vertices):
    """How far a body's centre of mass sits from the surface when it rests flush.

    The lowest vertex touches, so this is just how far the shape extends
    below its own centroid -- which is not half the height once the
    centroid has moved.
    """
    return -min(vy for _, vy in vertices)


def mirrored_vertices(vertices):
    """Reflect a polygon across its own y axis, keeping the winding CCW.

    The uphill pusher pushes downhill, so its tilted face has to be on the
    other side. Rotating the body 180 degrees will not do it -- that puts
    the trimmed edge at the bottom, so the contact vertex ends up down at
    the ramp surface instead of at the box's centre of mass. Reflecting
    keeps the shape resting the same way up and moves only the face.

    Without this the uphill pusher meets the box flat, which is exactly the
    parallel-face tie-break the tilt exists to avoid.
    """
    # Negating x reflects the shape but also reverses the traversal direction,
    # so the list has to be reversed to put the winding back counterclockwise.
    reflected = [[-vx, vy] for vx, vy in reversed(vertices)]

    # Positive area is the counterclockwise test -- `polygon_properties`
    # returns a signed area, so this catches a winding mistake immediately
    # rather than leaving GRIP's SAT path to fail obscurely.
    assert polygon_properties(reflected)[0] > 0.0, "mirrored winding came out clockwise"
    return reflected


def body(vertices, mass):
    """A shape and its mass properties, with the inertia derived from the outline.

    Derived for every body rather than written down for some. The box's
    closed form m*s^2/6 agrees with this exactly -- `check_task.py` asserts
    that -- but a hardcoded formula only stays right for the shape it was
    written for, and the pusher is already a shape it was not written for.
    """
    return dict(vertices=vertices, mass=mass, inertia=uniform_inertia(vertices, mass))


BOX = body(square_vertices(BOX_SIDE), BOX_MASS)
PUSHER = body(pusher_vertices(), PUSHER_MASS)
# Reflection preserves both the centroid at the origin and the polar moment
# about it, so this rebuilds to the same inertia rather than assuming so.
PUSHER_MIRRORED = body(mirrored_vertices(PUSHER["vertices"]), PUSHER_MASS)

# Pusher, box, pusher. The box is unactuated and sits between them, which
# is what makes it drivable in both directions -- a single convex pusher
# only pushes, and which way is fixed by the side it starts on.
BOX_ONLY_BODIES = [BOX]
TWO_PUSHER_BODIES = [PUSHER, BOX, PUSHER_MIRRORED]


def contact_reach(body, toward_uphill):
    """How far a body's contact vertex extends toward the box it pushes."""
    # Read off the vertex list rather than written down, because the 3 degree
    # face trim moves it. Whichever extreme faces the box is the one that
    # touches first: +x for a body reaching uphill, -x for the mirrored one,
    # negated so both come back as a positive distance.
    xs = [vx for vx, _ in body["vertices"]]
    return max(xs) if toward_uphill else -min(xs)


def make_scene(ramp_angle=DEFAULT_RAMP_ANGLE, bodies=None, dt=DEFAULT_TIMESTEP, penalty=None):
    """Bodies on a tilted half-plane, in the order given.

    Defaults to the single box, which is what `drift.py` and the box-only
    task want. Pass `TWO_PUSHER_BODIES` for the manipulation scene.
    """
    bodies = BOX_ONLY_BODIES if bodies is None else bodies
    penalty = DEFAULT_PENALTY if penalty is None else penalty
    return grip.Scene(
        params=[grip.RigidBodyParams(mass=b["mass"], inertia=b["inertia"]) for b in bodies],
        shapes=[grip.BodyShape(b["vertices"]) for b in bodies],
        plane=grip.HalfPlane(normal=normal(ramp_angle).tolist(), offset=0.0),
        penalty=grip.PenaltyParams(**penalty),
        dt=dt,
        gravity=GRAVITY,
    )


def make_scenes(ramp_angles, bodies=None, dt=DEFAULT_TIMESTEP, penalty=None):
    """One Scene per environment, each carrying its own slope.

    GRIP requires only that the body count match across a batch, so the
    ramp angle randomizes for free. That is the whole reason the task's
    per-episode slope costs nothing.
    """
    return [make_scene(ramp_angle=a, bodies=bodies, dt=dt, penalty=penalty) for a in np.atleast_1d(ramp_angles)]


def resting_state(xi, ramp_angles, bodies=None):
    """Bodies sitting flush on their ramps, shaped (environments, bodies, 6).

    Flush means the bottom face is parallel to the surface, so the body
    angle is the ramp angle and the centre of mass sits `resting_offset`
    out along the normal -- how far the shape extends below its own
    centroid, which is NOT half its height once the centroid has moved.
    The box's 150.000 mm is half its side; the pusher's 75.332 mm is not
    half of 150, because the 3 degree face trim shifts the centroid. The
    resulting gap is exactly zero either way.

    Deliberately *not* the penalty equilibrium, for two reasons. It is
    unreachable by a uniform offset, because friction's moment arm tilts
    the box and loads the downhill corner harder than the uphill one, so
    the two corners settle to different depths. And it is a
    penalty-contact fact, so starting there would hand 1.0 and 2.0
    different initial conditions and quietly spoil the comparison they
    exist for. Start flush and let `task.settle` ring it down instead.

    `xi` gives each body's position along the ramp and broadcasts against
    (environments, bodies), so a scalar places everything at the same point
    on every slope.
    """
    bodies = BOX_ONLY_BODIES if bodies is None else bodies
    angles = np.atleast_1d(np.asarray(ramp_angles, dtype=float))

    # A bare (environments,) means one position per environment, shared by
    # every body -- not one per body. Numpy would align it to the trailing
    # axis and get that backwards, so say which axis it is.
    xi = np.asarray(xi, dtype=float)
    if xi.ndim == 1 and xi.size == angles.size:
        xi = xi[:, None]
    xi = np.broadcast_to(xi, (angles.size, len(bodies)))

    up, out = uphill(angles), normal(angles)
    state = np.zeros((angles.size, len(bodies), 6))
    for index, body in enumerate(bodies):
        # World position = travel along the surface + stand-off along its
        # normal. The [:, index, None] opens an axis so a per-environment
        # scalar multiplies a 2-vector. Velocities stay zero.
        state[:, index, 0:2] = xi[:, index, None] * up + resting_offset(body["vertices"]) * out

        # Flush means the body angle IS the ramp angle, so the bottom face
        # lies parallel to the surface.
        state[:, index, 2] = angles
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

    # "...nbi,ni->...nb": for environment n and body b, project the position
    # 2-vector i onto that environment's own uphill direction. Leading step
    # axes pass through untouched, which is why one state and a whole
    # trajectory both work and only the last axis disappears.
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
