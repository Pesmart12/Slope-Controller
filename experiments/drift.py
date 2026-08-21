"""Artifact 1: how far a box drifts on a slope it should not drift on.

No policy, no training, no gradients. A box is placed flush on a ramp
below the friction angle and left alone for five seconds. Coulomb's law
says it must not move at all -- a closed-form fact about rigid bodies,
not a modelling opinion -- so the measured drift is exactly the error the
contact formulation introduces.

Worth keeping apart from anything learned. A reward curve needs caveats;
"the box slid 4.7 cm when the physics says 0" does not.

The right panel is a second finding that came out of checking the first.
Creep is mg*sin(a)/(2*b_slip) exactly -- the 2 being the two bottom
corners, each carrying its own friction -- but only while both corners
stick. Above about 19 degrees friction's moment arm tilts the box by
roughly a milliradian, normal force redistributes between the corners,
and the lightly loaded one saturates on its cone. The closed form becomes
a lower bound from there to the friction angle.

Run:  python experiments/drift.py
"""

import argparse
import math
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import grip  # noqa: E402
from slope_control import ramp  # noqa: E402

CONTROL_HZ = 100.0
DURATION = 5.0


def run(ramp_angle, duration=DURATION):
    """Roll out one box under zero control, returning its ramp coordinate."""
    scene = ramp.make_scene(ramp_angle=ramp_angle)
    substeps = int(round(1.0 / (CONTROL_HZ * scene.dt)))
    steps = int(round(duration * CONTROL_HZ))

    initial = ramp.resting_state(0.0, ramp_angle=ramp_angle)
    trajectory = grip.rollout_batch([scene], initial, np.zeros((steps, 1, 1, 3)), substeps=substeps)

    times = np.arange(steps + 1) / CONTROL_HZ
    xi = ramp.along_ramp(trajectory[:, 0, 0, :], ramp_angle=ramp_angle)
    return times, xi - xi[0]


def steady_rate(times, drift):
    """Downhill creep over the final second, after settling. Positive."""
    per_second = int(round(CONTROL_HZ))
    return -(drift[-1] - drift[-1 - per_second])


def plot_drift(axes, angles, limit):
    colours = plt.cm.viridis(np.linspace(0.15, 0.75, len(angles)))
    for degrees, colour in zip(angles, colours):
        times, drift = run(math.radians(degrees))
        axes.plot(times, 100.0 * drift, color=colour, linewidth=2.0, label=f"penalty, {degrees:.0f}°")
        print(f"  {degrees:4.1f}°   drift {100 * drift[-1]:6.2f} cm over {times[-1]:.0f} s")

    axes.axhline(0.0, color="#333333", linewidth=1.8, linestyle="--", label="rigid contact (Coulomb)")
    axes.set_xlabel("time (s)")
    axes.set_ylabel("drift along the ramp (cm)")
    axes.set_title(f"A box that should not move, moving\nevery slope below the {limit:.1f}° friction angle")
    axes.legend(loc="lower left", frameon=False)


def plot_crossover(axes, limit):
    degrees = np.arange(4.0, math.floor(limit) + 0.5, 1.0)
    measured, predicted = [], []
    for d in degrees:
        angle = math.radians(d)
        times, drift = run(angle, duration=3.0)
        measured.append(steady_rate(times, drift))
        predicted.append(ramp.creep_rate(angle))
    measured = 100.0 * np.asarray(measured)
    predicted = 100.0 * np.asarray(predicted)

    exact = np.isclose(measured, predicted, rtol=1e-3)
    breaks_at = degrees[~exact].min() if (~exact).any() else None
    print(f"\n  closed form exact up to {degrees[exact].max():.0f}°" + (f", departs at {breaks_at:.0f}°" if breaks_at else ""))
    print(f"  worst departure {100 * (measured / predicted - 1.0).max():.0f}% at {degrees[np.argmax(measured / predicted)]:.0f}°")

    axes.plot(degrees, predicted, color="#333333", linewidth=1.8, linestyle="--", label=r"$mg\sin\alpha\,/\,2b_{slip}$")
    axes.plot(degrees, measured, color="#c1442e", linewidth=2.0, marker="o", markersize=3.5, label="measured")
    if breaks_at is not None:
        axes.axvline(breaks_at, color="#888888", linewidth=1.0, linestyle=":")
        axes.annotate("uphill corner\nsaturates its cone", xy=(breaks_at, predicted[degrees == breaks_at][0]), xytext=(breaks_at - 9.5, measured.max() * 0.72), fontsize=9, color="#555555", arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0))
    axes.set_xlabel("ramp angle (degrees)")
    axes.set_ylabel("steady creep rate (cm/s)")
    axes.set_title("Where the closed form stops being exact\nboth corners stick, until one cannot")
    axes.legend(loc="upper left", frameon=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angles", type=float, nargs="+", default=[10.0, 15.0, 20.0])
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1] / "figures" / "drift.png")
    args = parser.parse_args()

    mu = ramp.DEFAULT_PENALTY["friction"]
    limit = math.degrees(ramp.friction_angle(mu))
    print(f"friction angle atan({mu}) = {limit:.1f}° -- every angle below it must hold still\n")

    for degrees in args.angles:
        if degrees >= limit:
            raise SystemExit(f"{degrees}° is at or above the friction angle {limit:.1f}°; the box is meant to slide there and the comparison says nothing")

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    plot_drift(left, args.angles, limit)
    plot_crossover(right, limit)
    for axes in (left, right):
        axes.grid(alpha=0.25)
        axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
