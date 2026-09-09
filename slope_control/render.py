"""Draw a scene, and animate an episode.

Pure presentation. Nothing here computes anything the rest of the package
does not already return -- it takes a trajectory and turns it into
pictures, so a trained policy can be watched rather than only scored.

Matplotlib's Agg backend must be selected before pyplot is imported, which
is the caller's job, the same as in `experiments/drift.py`.
"""

import numpy as np

from . import ramp


# Muted enough that the box reads as the subject and the pushers as tools.
BOX_COLOR = "#2b6cb0"
PUSHER_COLOR = "#a0aec0"
RAMP_COLOR = "#4a5568"
TARGET_COLOR = "#c53030"


def body_corners(state, vertices):
    """Where one body's outline sits in the world, given its state.

    `state` is GRIP's (x, y, theta, vx, vy, omega). The vertices are in the
    body frame, centred on the centre of mass, so this rotates them by
    theta and shifts them to the body's position.
    """
    x, y, theta = state[0], state[1], state[2]
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.asarray(vertices) @ rotation.T + np.array([x, y])


def view_bounds(states, ramp_angle, target, bodies, margin=0.12):
    """The world rectangle worth drawing, taken from where things actually go.

    A fixed window wastes most of the canvas: the bodies travel along the
    ramp, so the interesting region is a diagonal band, and how long it is
    depends on the target. This walks every body's outline over the whole
    episode, adds the target, and returns the box around all of it.
    """
    corners = []
    for frame in states:
        for index, body in enumerate(bodies):
            corners.append(body_corners(frame[index], body["vertices"]))
    corners.append(np.atleast_2d(target * ramp.uphill(ramp_angle)))

    points = np.concatenate(corners, axis=0)
    low, high = points.min(axis=0) - margin, points.max(axis=0) + margin
    return low, high


def draw_frame(axes, state, ramp_angle, target, bodies, box_index, low, high):
    """Draw one instant: the ramp surface, every body, and the target mark."""
    up, out = ramp.uphill(ramp_angle), ramp.normal(ramp_angle)

    # Draw the ramp as a line through the origin along `uphill`, running
    # twice the view diagonal in each direction. That length guarantees it
    # crosses the whole frame at any slope, so it reads as a surface rather
    # than a line segment.
    reach = 2.0 * float(np.linalg.norm(high - low))
    ends = np.stack([-reach * up, reach * up])
    axes.plot(ends[:, 0], ends[:, 1], color=RAMP_COLOR, linewidth=2.0, zorder=2)

    # Fill the quad hanging below the surface, so which side is solid is
    # unambiguous.
    depth = 2.0 * (high[1] - low[1])
    below = np.stack([ends[0], ends[1], ends[1] - depth * out, ends[0] - depth * out])
    axes.fill(below[:, 0], below[:, 1], color=RAMP_COLOR, alpha=0.13, zorder=1)

    for index, body in enumerate(bodies):
        corners = body_corners(state[index], body["vertices"])
        color = BOX_COLOR if index == box_index else PUSHER_COLOR
        axes.fill(corners[:, 0], corners[:, 1], color=color, zorder=3, edgecolor="white", linewidth=0.9)

    # Put the target marker on the surface by scaling `uphill` by the
    # target's arc length, rather than leaving it somewhere in free space.
    foot = target * up
    axes.plot([foot[0]], [foot[1]], marker="v", markersize=10, color=TARGET_COLOR, zorder=4)


def animate(states, ramp_angle, target, bodies, box_index, path, fps=25, stride=4, title=None, width=7.0):
    """Write one environment's episode to an animated GIF.

    `states` is (steps + 1, bodies, 6). `stride` keeps every nth control
    step; at 100 Hz an untouched 400-step episode is far more frames than a
    watchable GIF needs, and 4 gives 25 fps in real time.

    The figure is shaped to the scene rather than the other way round, so a
    shallow ramp gets a wide frame and a steep one a taller frame, and the
    bodies fill the canvas in both.
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    frames = states[::stride]
    low, high = view_bounds(frames, ramp_angle, target, bodies)
    extent = high - low

    figure, axes = plt.subplots(figsize=(width, max(2.2, width * extent[1] / extent[0])), dpi=140)

    def update(index):
        axes.clear()
        axes.set_aspect("equal")
        axes.axis("off")
        axes.set_xlim(low[0], high[0])
        axes.set_ylim(low[1], high[1])
        if title:
            axes.set_title(title, fontsize=10, color="#2d3748", pad=4)

        draw_frame(axes, frames[index], ramp_angle, target, bodies, box_index, low, high)

        # Stamp the elapsed time in the corner, so a viewer can tell the
        # approach from the hold.
        axes.text(0.015, 0.93, f"{index * stride / 100.0:5.2f} s", transform=axes.transAxes,
                  fontsize=9, family="monospace", color="#4a5568")

    movie = animation.FuncAnimation(figure, update, frames=len(frames), blit=False)
    movie.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(figure)
    return len(frames)
