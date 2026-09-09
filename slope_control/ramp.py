"""The world: a tilted surface, the bodies on it, and the closed-form physics.

Geometry conventions, fixed here so nothing downstream rederives them:

    uphill  = ( cos a,  sin a)   direction of increasing height
    normal  = (-sin a,  cos a)   away from the surface, perpendicular
    offset  = 0                  the ramp passes through the origin

At a = 0 the normal is (0, 1) and everything reduces to flat ground. That
is the cheap test that the signs are right; keep it working. Gravity
(0, -g) dotted with uphill gives -g*sin(a), so it pulls downhill.

`xi` means position along the ramp. One coordinate is enough, because a
body resting on the surface has only one degree of freedom any task here
cares about.

Every function takes either one angle or one angle per environment,
because the task randomizes the slope across a batch. A single shared
angle would be a quiet way to get that wrong.

The closed forms at the bottom -- `creep_rate`, `hold_force`,
`break_free_force`, `normal_load` -- are what the measurements are checked
against. They describe rigid Coulomb friction, so where penalty contact
disagrees with them, the difference is the modelling error.
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
    """Area, centroid, and polar second moment about that centroid, for a
    counterclockwise polygon.

    The area comes back signed, so a negative value means the winding is
    clockwise.

    Needed because the pusher is a trapezoid, not a rectangle, so its
    centroid is not where you would guess. GRIP expects `BodyShape`
    vertices measured from the centre of mass, and a list that is not
    recentred puts a permanent torque offset into every contact that body
    makes.
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

    # Shift every vertex so the centroid lands on the origin. GRIP takes
    # vertices in a frame centred on the centre of mass, and the trim moved
    # the centroid ~2 mm off the rectangle's centre; leaving it there would
    # put a standing torque offset into every contact this body makes.
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
# Build the mirrored pusher, deriving its mass properties again rather than
# copying them. Reflection preserves both the centroid at the origin and the
# polar moment about it, so the rebuild returns the same inertia.
PUSHER_MIRRORED = body(mirrored_vertices(PUSHER["vertices"]), PUSHER_MASS)

# Pusher, box, pusher. The box is unactuated and sits between them, which
# is what makes it drivable in both directions -- a single convex pusher
# only pushes, and which way is fixed by the side it starts on.
BOX_ONLY_BODIES = [BOX]
TWO_PUSHER_BODIES = [PUSHER, BOX, PUSHER_MIRRORED]


def contact_reach(body, toward_uphill):
    """How far a body's contact vertex extends toward the box it pushes."""
    # Take the extreme vertex on whichever side faces the box, since that is
    # the one that touches first: +x for a body reaching uphill, -x for the
    # mirrored one, negated so both come back as a positive distance. Read
    # off the vertex list rather than written down, because the 3 degree face
    # trim moves it.
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
    """Put bodies flush on their ramps. Returns (environments, bodies, 6).

    Flush means the bottom face lies parallel to the surface, touching it
    exactly, with no velocity. So each body's angle is its ramp angle, and
    its centre of mass sits `resting_offset` out along the normal.

    That offset is how far the shape extends below its own centroid, which
    is not half its height once the centroid has moved. The box sits at
    150.000 mm, half its side. The pusher sits at 75.332 mm, which is not
    half of 150, because the tilted face shifts its centroid.

    `xi` is each body's position along the ramp. It broadcasts against
    (environments, bodies), so one number places everything at the same
    point on every slope.

    Flush is not where penalty contact wants the body -- that is about half
    a millimetre lower. Starting there instead was rejected: it is a fact
    about penalty contact, so it would give the two columns different
    initial conditions. Start flush and let `task.settle` handle it.
    """
    bodies = BOX_ONLY_BODIES if bodies is None else bodies
    angles = np.atleast_1d(np.asarray(ramp_angles, dtype=float))

    # Open a body axis on a bare (environments,) input, giving
    # (environments, 1) so it broadcasts across bodies. Such an input means
    # one position per environment shared by every body, and numpy would
    # otherwise align it to the trailing axis and read it as one per body.
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

        # Set every body's angle to the ramp angle, which lays its bottom
        # face flush against the surface.
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
    """How fast a box slides on a slope it should be resting on.

        rate = m * g * sin(a) / (2 * b_slip)     metres per second

    Penalty contact cannot avoid this. Its friction law is
    beta = -b_slip * slip, so friction only exists when something is
    sliding. A box held up by friction must therefore be sliding at exactly
    the speed that generates the friction holding it up. Rigid physics says
    the box does not move at all below the friction angle, so this whole
    quantity is the modelling error.

    The 2 is the number of contact points. A box resting on its face
    touches at both bottom corners, and each carries its own friction, so
    the pair together supply twice as much and each need creep only half as
    fast.

    Exact to five figures from 5 to 18 degrees. Above about 19 degrees it
    UNDERESTIMATES -- 13% low at 20 degrees, 48% at 26. Friction acts at
    the corners, 0.15 m below the centre of mass, so it tilts the box by
    around a milliradian. The tilt shifts normal load off the uphill
    corner, which then hits its own friction limit and cannot carry its
    half. The downhill corner makes up the difference and creeps faster.

    So: exact while both corners grip, a lower bound once one does not.
    """
    penalty = DEFAULT_PENALTY if penalty is None else penalty
    return mass * GRAVITY * np.sin(ramp_angle) / (2.0 * penalty["slip_damping"])


def hold_force(ramp_angle=DEFAULT_RAMP_ANGLE, mass=BOX_MASS):
    """Force needed to hold a body perfectly still on the slope.

        force = m * g * sin(a)     newtons

    Under penalty contact, friction only appears when something slides, so
    a body that is not moving gets no help at all from the surface. The
    controller pays the whole gravity component for as long as it holds.
    Rigid friction does the same job for free on any slope below the
    friction angle.

    This is a floor, not a tuning artifact. No feedback law beats it,
    because the mechanism that would let a controller stop pushing is the
    one penalty contact removes. It is what the reward's control term
    prices.
    """
    return mass * GRAVITY * np.sin(ramp_angle)


def break_free_force(ramp_angle=DEFAULT_RAMP_ANGLE, penalty=None, mass=BOX_MASS):
    """Force needed to actually slide a body uphill.

        force = m * g * (sin(a) + mu * cos(a))     newtons

    Holding and moving are different problems, and the gap between them is
    the width of the friction cone. `hold_force` only cancels gravity,
    which leaves the body stationary. To move it you must also beat
    friction, which pushes back with up to mu times the normal load.

    Size an action limit against THIS, not against `hold_force`. A limit
    that can hold a box cannot necessarily move one, and a task posed that
    way is unsolvable while every gradient check still passes.

    Measured exact at 15, 20 and 22 degrees. A newton below it the body
    does not move -- what looks like motion is creep, a couple of
    centimetres over a whole episode. A newton above it the body
    accelerates away. Coulomb friction has no gentle regime in between,
    which is what makes this a real control problem.
    """
    penalty = DEFAULT_PENALTY if penalty is None else penalty
    return mass * GRAVITY * (np.sin(ramp_angle) + penalty["friction"] * np.cos(ramp_angle))


def normal_load(ramp_angle=DEFAULT_RAMP_ANGLE, mass=BOX_MASS):
    """Force pressing a body into the ramp.

        force = m * g * cos(a)     newtons

    Push outward this hard and the contact carries no load at all, so the
    body leaves the surface. The normal component of the action limit has
    to stay below this.
    """
    return mass * GRAVITY * np.cos(ramp_angle)
