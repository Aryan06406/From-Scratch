"""
v01_vector_ops.py

Visual companion to m01_vector_ops.py.

Takes a pair of 2D vectors (for the plane operations) and a pair of 3D
vectors (for the cross product), runs every operation implemented in the Vector class, 
and renders each one as a labelled matplotlib panel so you can see what the operation
did geometrically.

Output: a single PNG saved next to this script (v01_vector_ops.png)
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from pathlib import Path
from linear_algebra.m01_vector_ops import Vector


# Small helpers for drawing
 
def draw_arrow(ax, vec, origin=(0, 0), color="tab:blue", label=None, lw=2.2, z=3):
    """Draw a 2D vector as an arrow from `origin`."""
    ax.annotate(
        "", xy=(origin[0] + vec[0], origin[1] + vec[1]), xytext=origin,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0),
        zorder=z,
    )
    if label:
        tip = (origin[0] + vec[0], origin[1] + vec[1])
        ax.annotate(label, tip, textcoords="offset points", xytext=(6, 6),
                    color=color, fontsize=10, fontweight="bold")
  
def setup_axes(ax, points, title=""):
    """Auto-scale a 2D axis around the given (x, y) points and add grid/origin lines.
 
    Padding scales with the spread of the points instead of being a fixed
    constant, so labels on large vectors get as much breathing room as
    labels on small ones (a fixed pad works for magnitude ~5 but clips
    labels/titles once vectors reach magnitude ~40).
    """
    xs = [p[0] for p in points] + [0.0]
    ys = [p[1] for p in points] + [0.0]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span = max(x_max - x_min, y_max - y_min, 2.0)
    pad = 0.3 * span
    lo_x, hi_x = x_min - pad, x_max + pad
    lo_y, hi_y = y_min - pad, y_max + pad
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(lo_y, hi_y)
    ax.axhline(0, color="gray", lw=0.8, zorder=0)
    ax.axvline(0, color="gray", lw=0.8, zorder=0)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(title, fontsize=11, fontweight="bold")
 
# Input handling
 
def read_vector(prompt, default):
    raw = input(f"{prompt} [default {default}]: ").strip()
    if not raw:
        return list(default)
    return [float(x) for x in raw.replace(",", " ").split()]
  
def get_inputs():
    print("=== Vector Operations Visualizer ===")
    print("Enter 2D vectors for the plane operations (space or comma separated).")
    v1_2d = read_vector("Vector A (2D)", (4, 2))
    v2_2d = read_vector("Vector B (2D)", (1, 3))
    print("\nEnter 3D vectors for the cross product demo.")
    v1_3d = read_vector("Vector A (3D)", (2, 0, 1))
    v2_3d = read_vector("Vector B (3D)", (0, 3, 1))
    scalar = input("\nScalar for scalar multiplication/division [default 2]: ").strip()
    scalar = float(scalar) if scalar else 2.0
    return Vector(v1_2d), Vector(v2_2d), Vector(v1_3d), Vector(v2_3d), scalar
 
# Panel builders — one function per operation
 
def panel_addition(ax, a, b):
    s = a + b
    draw_arrow(ax, a.elements, color="tab:blue", label="A")
    draw_arrow(ax, b.elements, color="tab:orange", label="B")
    draw_arrow(ax, b.elements, origin=a.elements, color="tab:orange", lw=1.2)
    draw_arrow(ax, a.elements, origin=b.elements, color="tab:blue", lw=1.2)
    draw_arrow(ax, s.elements, color="tab:green", label="A+B", lw=2.6)
    setup_axes(ax, [a.elements, b.elements, s.elements], title="Addition: A + B\n(parallelogram rule)")
 
def panel_subtraction(ax, a, b):
    d = a - b
    draw_arrow(ax, a.elements, color="tab:blue", label="A")
    draw_arrow(ax, b.elements, color="tab:orange", label="B")
    draw_arrow(ax, d.elements, origin=b.elements, color="tab:green", label="A-B", lw=2.6)
    setup_axes(ax, [a.elements, b.elements, d.elements], title="Subtraction: A - B\n(arrow from B to A)")
  
def panel_scalar(ax, a, scalar):
    scaled = a * scalar
    draw_arrow(ax, a.elements, color="tab:blue", label="A")
    draw_arrow(ax, scaled.elements, color="tab:red", label=f"{scalar:g}·A")
    setup_axes(ax, [a.elements, scaled.elements],
               title=f"Scalar Multiplication: {scalar:g} * A\n(stretches/flips A)")
  
def panel_negation(ax, a):
    neg = -a
    draw_arrow(ax, a.elements, color="tab:blue", label="A")
    draw_arrow(ax, neg.elements, color="tab:purple", label="-A")
    setup_axes(ax, [a.elements, neg.elements], title="Negation: -A\n(reverses direction)")
  
def panel_dot_projection(ax, a, b):
    dot = a.dot_product(b)
    b_mag_sq = sum(x**2 for x in b.elements)
    proj_scalar = dot / b_mag_sq if b_mag_sq != 0 else 0
    proj = b * proj_scalar
    draw_arrow(ax, a.elements, color="tab:blue", label="A")
    draw_arrow(ax, b.elements, color="tab:orange", label="B")
    draw_arrow(ax, proj.elements, color="tab:green", label="proj of A on B", lw=3)
    ax.plot([a.elements[0], proj.elements[0]], [a.elements[1], proj.elements[1]],
            linestyle=":", color="gray")
    setup_axes(ax, [a.elements, b.elements, proj.elements],
               title=f"Dot Product: A·B = {dot:.2f}\n(green = projection of A onto B)")
  
def panel_hadamard(ax, a, b):
    h = a.hadamard_product(b)
    idx = np.arange(len(a))
    width = 0.25
    ax.bar(idx - width, a.elements, width, label="A", color="tab:blue")
    ax.bar(idx, b.elements, width, label="B", color="tab:orange")
    ax.bar(idx + width, h.elements, width, label="A⊙B", color="tab:green")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"x{i+1}" for i in idx])
    ax.axhline(0, color="gray", lw=0.8)
    ax.legend(fontsize=8)
    ax.set_title("Hadamard Product: A ⊙ B\n(element-by-element, not geometric)", fontsize=11, fontweight="bold")
  
def panel_magnitude_unit(ax, a):
    unit = a.unit_vector()
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="lightgray", lw=1, label="unit circle")
    draw_arrow(ax, a.elements, color="tab:blue", label=f"A (||A||={a.magnitude():.2f})")
    draw_arrow(ax, unit.elements, color="tab:green", label="Â (unit vector)")
    setup_axes(ax, [a.elements, unit.elements, (1, 1), (-1, -1)],
               title="Magnitude & Unit Vector\n(Â lies on the unit circle)")
  
def panel_angle(ax, a, b):
    angle = a.angle_between_vectors(b)
    draw_arrow(ax, a.elements, color="tab:blue", label="A")
    draw_arrow(ax, b.elements, color="tab:orange", label="B")
    # small arc showing the angle at the origin
    ang_a = math.atan2(a.elements[1], a.elements[0])
    ang_b = math.atan2(b.elements[1], b.elements[0])
    r = 0.6 * min(a.magnitude(), b.magnitude())
    start, end = sorted([ang_a, ang_b])
    if end - start > math.pi:
        start, end = end, start + 2 * math.pi
    arc_theta = np.linspace(start, end, 60)
    ax.plot(r * np.cos(arc_theta), r * np.sin(arc_theta), color="tab:green", lw=2)
    mid = (start + end) / 2
    ax.annotate(f"{angle:.1f}°", (r * 1.3 * math.cos(mid), r * 1.3 * math.sin(mid)),
                color="tab:green", fontweight="bold")
    setup_axes(ax, [a.elements, b.elements], title=f"Angle Between A & B = {angle:.1f}°")
  
def panel_orthogonal_parallel(ax, a, b):
    orth = a.is_orthogonal(b)
    para = a.is_parallel(b)
    draw_arrow(ax, a.elements, color="tab:blue", label="A")
    draw_arrow(ax, b.elements, color="tab:orange", label="B")
    setup_axes(ax, [a.elements, b.elements], title="Orthogonality / Parallelism check")
    # Placed below the axes (not inside the plot area) so it can never sit on
    # top of a vector, regardless of which quadrant A/B happen to point into.
    ax.text(0.5, -0.16, f"Orthogonal: {orth}    |    Parallel: {para}",
            transform=ax.transAxes, ha="center", va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
            clip_on=False)
  
def panel_distance(ax, a, b):
    dist = a.distance_between_vectors(b)
    ax.plot(*a.elements, "o", color="tab:blue", markersize=10)
    ax.plot(*b.elements, "o", color="tab:orange", markersize=10)
    ax.annotate("A (as point)", a.elements, textcoords="offset points", xytext=(6, 6), color="tab:blue")
    ax.annotate("B (as point)", b.elements, textcoords="offset points", xytext=(6, 6), color="tab:orange")
    ax.plot([a.elements[0], b.elements[0]], [a.elements[1], b.elements[1]],
            linestyle="--", color="tab:green", lw=2)
    mid = ((a.elements[0] + b.elements[0]) / 2, (a.elements[1] + b.elements[1]) / 2)
    ax.annotate(f"dist = {dist:.2f}", mid, color="tab:green", fontweight="bold")
    setup_axes(ax, [a.elements, b.elements], title="Euclidean Distance ||A - B||\n(A, B treated as points)")
  
def panel_cross_product_3d(fig, slot, a3, b3):
    """Cross product needs a real 3D axis, so it gets its own subplot spec.
 
    AxB is often much longer (or shorter) than A and B themselves — e.g.
    A, B with magnitude ~9 can easily produce a cross product of magnitude
    ~30. Plotting true lengths then forces the axis to fit the biggest
    vector, shrinking the other two into an unreadable cluster at the
    origin. So directions are drawn at equal length here (this panel is
    about *orientation* — is AxB really perpendicular to both? — not
    magnitude), and the true component values are put in the labels
    instead of being implied by arrow length.
    """
    ax = fig.add_subplot(slot, projection="3d")
    c = a3.cross_product(b3)
 
    def unit(vec):
        n = math.sqrt(sum(x ** 2 for x in vec))
        return [x / n for x in vec] if n > 1e-12 else [0.0, 0.0, 0.0]
 
    def q(vec, color, label):
        d = unit(vec)
        ax.quiver(0, 0, 0, d[0], d[1], d[2], color=color, linewidth=2.5, arrow_length_ratio=0.15)
        rounded = [round(x, 2) for x in vec]
        ax.text(d[0] * 1.2, d[1] * 1.2, d[2] * 1.2, f"{label} {rounded}",
                color=color, fontweight="bold", fontsize=9)
 
    q(a3.elements, "tab:blue", "A")
    q(b3.elements, "tab:orange", "B")
    q(c.elements, "tab:green", "A×B")
 
    lim = 1.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_title(f"Cross Product: A×B = {[round(x, 2) for x in c.elements]}\n"
                 "(arrows shown as unit directions; green is orthogonal to A and B)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
 
# Main
 
def main():
    a2, b2, a3, b3, scalar = get_inputs()
 
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Vector Operations — Visualized", fontsize=18, fontweight="bold")
    grid = fig.add_gridspec(4, 3, hspace=0.55, wspace=0.35)
 
    panel_addition(fig.add_subplot(grid[0, 0]), a2, b2)
    panel_subtraction(fig.add_subplot(grid[0, 1]), a2, b2)
    panel_scalar(fig.add_subplot(grid[0, 2]), a2, scalar)
 
    panel_negation(fig.add_subplot(grid[1, 0]), a2)
    panel_dot_projection(fig.add_subplot(grid[1, 1]), a2, b2)
    panel_magnitude_unit(fig.add_subplot(grid[1, 2]), a2)
 
    panel_angle(fig.add_subplot(grid[2, 0]), a2, b2)
    panel_orthogonal_parallel(fig.add_subplot(grid[2, 1]), a2, b2)
    panel_distance(fig.add_subplot(grid[2, 2]), a2, b2)
 
    panel_hadamard(fig.add_subplot(grid[3, 0]), a2, b2)
    panel_cross_product_3d(fig, grid[3, 1:3], a3, b3)
 
    out_path = Path(__file__).parent / "figures"
    out_path.mkdir(parents=True, exist_ok=True) 
    figure_path = out_path / "v01_vector_ops.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to {figure_path}")

    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()