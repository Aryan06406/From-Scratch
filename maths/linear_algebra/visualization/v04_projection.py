"""
v04_projection.py

Visual companion to m04_projection.py.

Takes two 2D vectors (u is projected onto v) and renders every concept
in the Projection class as a dedicated, clearly-labelled panel.

Output: v04_projection.png
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
from maths.linear_algebra.core.m01_vector_ops import Vector
from maths.linear_algebra.core.m03_linear_transformation import LinearTransformation, Projection


# Helpers

def read_vector(prompt, default):
    raw = input(f"{prompt} [default {default}]: ").strip()
    if not raw:
        return Vector(list(default))
    return Vector([float(x) for x in raw.replace(",", " ").split()])

def get_inputs():
    print("=== Projection Visualizer ===")
    u = read_vector("Vector u (2D)", (3, 4))
    v = read_vector("Vector v  (2D, project u ONTO v)", (5, 1))
    return u, v

def arrow(ax, tip, origin=(0, 0), color="tab:blue", lw=2.2, ls="-", zorder=4):
    ax.annotate("", xy=tip, xytext=origin,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, shrinkA=0, shrinkB=0),
                zorder=zorder)

def label_tip(ax, tip, text, color, side=1, magnitude=13):
    """Perpendicular-to-direction offset so the text never sits on the arrow."""
    dx, dy = tip[0], tip[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        ox, oy = magnitude, magnitude
    else:
        ux, uy = dx / length, dy / length
        ox = -uy * magnitude * side
        oy =  ux * magnitude * side
    ax.annotate(text, tip, textcoords="offset points", xytext=(ox, oy),
                color=color, fontsize=9.5, fontweight="bold")

def setup(ax, points, title, pad_factor=0.35):
    xs = [p[0] for p in points] + [0.0]
    ys = [p[1] for p in points] + [0.0]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span = max(x_max - x_min, y_max - y_min, 2.0)
    pad = pad_factor * span
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.axhline(0, color="gray", lw=0.7, zorder=0)
    ax.axvline(0, color="gray", lw=0.7, zorder=0)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_title(title, fontsize=11, fontweight="bold")

def caption(ax, text):
    """Pin the legend text below the axes — never inside the data area."""
    ax.text(0.5, -0.14, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, bbox=dict(boxstyle="round", facecolor="white",
                                    edgecolor="gray", alpha=0.92),
            clip_on=False)

def right_angle_marker(ax, corner, d1, d2, size=0.15):
    """Draw a small square at `corner` between directions d1 and d2."""
    scale = size * max(math.hypot(*d1), math.hypot(*d2))
    n1 = (d1[0] / math.hypot(*d1), d1[1] / math.hypot(*d1))
    n2 = (d2[0] / math.hypot(*d2), d2[1] / math.hypot(*d2))
    p1 = (corner[0] + scale * n1[0], corner[1] + scale * n1[1])
    p2 = (corner[0] + scale * n2[0], corner[1] + scale * n2[1])
    p3 = (p1[0] + scale * n2[0], p1[1] + scale * n2[1])
    sq = plt.Polygon([corner, p1, p3, p2], fill=False, edgecolor="gray", lw=1.2)
    ax.add_patch(sq)

# Panel builders

def panel_overview(ax, u, v):
    """The master diagram: u, v, proj, rejection, scalar comp — all in one."""
    proj = Projection.vector_projection(u, v)
    rej  = Projection.rejection(u, v)
    sc   = Projection.scalar_projection(u, v)

    proj_tip  = tuple(proj.elements)
    rej_tip   = tuple(rej.elements)
    u_tip     = tuple(u.elements)
    v_unit    = (v.elements[0] / v.magnitude(), v.elements[1] / v.magnitude())
    v_long    = (v_unit[0] * v.magnitude() * 2.5, v_unit[1] * v.magnitude() * 2.5)

    # faint extended v line to show the subspace
    ax.plot([-v_long[0], v_long[0]], [-v_long[1], v_long[1]],
            color="tab:orange", lw=1, linestyle=":", alpha=0.5, zorder=1)

    arrow(ax, u_tip,    color="tab:blue",   lw=2.4)
    arrow(ax, tuple(v.elements), color="tab:orange", lw=2.4)
    arrow(ax, proj_tip, color="tab:green",  lw=2.6)

    # rejection arrow: from proj_tip to u_tip
    arrow(ax, u_tip, origin=proj_tip, color="tab:red", lw=2.2, ls="--")

    # right-angle marker at the rejection foot
    right_angle_marker(ax, proj_tip,
                       (rej.elements[0], rej.elements[1]),
                       (v.elements[0],   v.elements[1]))

    # scalar comp tick on v axis
    ax.plot(*proj_tip, "D", color="tab:green", markersize=7, zorder=5)

    label_tip(ax, u_tip,    "u",                "tab:blue",   side=1)
    label_tip(ax, tuple(v.elements), "v", "tab:orange",  side=-1)
    label_tip(ax, proj_tip, f"proj = {[round(x,2) for x in proj.elements]}", "tab:green",  side=-1)
    label_tip(ax, u_tip,    f"rej = {[round(x,2) for x in rej.elements]}",  "tab:red",    side=1)

    setup(ax, [u_tip, tuple(v.elements), proj_tip, rej_tip],
          f"Projection Overview\nscalar_projection(u,v) = {sc:.3f}")
    caption(ax, "blue = u   |   orange = v   |   green = proj_v(u)   |   red dashed = rejection   |   square = 90°")

def panel_scalar_projection(ax, u, v):
    """Geometric meaning of the scalar projection — signed length on v's axis."""
    sc = Projection.scalar_projection(u, v)
    proj = Projection.vector_projection(u, v)
    proj_tip = tuple(proj.elements)
    u_tip    = tuple(u.elements)
    v_hat    = Vector([x / v.magnitude() for x in v.elements])

    # draw unit vector of v
    arrow(ax, tuple(v_hat.elements), color="tab:orange", lw=2)
    arrow(ax, u_tip, color="tab:blue", lw=2.4)

    # dashed drop from u to proj
    ax.plot([u_tip[0], proj_tip[0]], [u_tip[1], proj_tip[1]],
            color="gray", lw=1.4, linestyle=":")
    ax.plot(*proj_tip, "D", color="tab:green", markersize=8, zorder=5)

    # annotate the scalar length along v_hat
    mid = (proj_tip[0] / 2, proj_tip[1] / 2)
    ax.annotate(f"comp = {sc:.3f}", mid, fontsize=9, color="tab:green",
                ha="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    label_tip(ax, u_tip, "u", "tab:blue", side=1)
    label_tip(ax, tuple(v_hat.elements), "v̂ (unit of v)", "tab:orange", side=-1)

    setup(ax, [u_tip, tuple(v.elements), proj_tip],
          f"Scalar Projection: comp_v(u) = (u·v)/‖v‖\n= {sc:.3f}  (signed length of u along v)")
    caption(ax, "diamond = foot of perpendicular from u onto v   |   gray dotted = the drop")

def panel_vector_projection(ax, u, v):
    """proj_v(u) drawn as a vector on v's line, alongside u."""
    proj     = Projection.vector_projection(u, v)
    proj_tip = tuple(proj.elements)
    u_tip    = tuple(u.elements)

    v_long_scale = 1.6
    v_ext = (v.elements[0] * v_long_scale, v.elements[1] * v_long_scale)
    ax.plot([-v_ext[0], v_ext[0]], [-v_ext[1], v_ext[1]],
            color="tab:orange", lw=1, linestyle=":", alpha=0.5)

    arrow(ax, u_tip,    color="tab:blue",  lw=2.4)
    arrow(ax, tuple(v.elements), color="tab:orange", lw=2.2)
    arrow(ax, proj_tip, color="tab:green", lw=3.0)

    ax.plot([u_tip[0], proj_tip[0]], [u_tip[1], proj_tip[1]],
            color="gray", lw=1.4, linestyle=":")
    right_angle_marker(ax, proj_tip,
                       (u_tip[0] - proj_tip[0], u_tip[1] - proj_tip[1]),
                       (v.elements[0], v.elements[1]))

    label_tip(ax, u_tip, "u", "tab:blue", side=1)
    label_tip(ax, tuple(v.elements), "v", "tab:orange", side=-1)
    label_tip(ax, proj_tip, f"proj_v(u)\n{[round(x,2) for x in proj.elements]}",
              "tab:green", side=-1)

    sc = u.dot_product(v) / v.dot_product(v)
    setup(ax, [u_tip, tuple(v.elements), proj_tip],
          f"Vector Projection: proj_v(u) = ((u·v)/(v·v))·v\nscalar factor = {sc:.3f}")
    caption(ax, "green = proj_v(u) — lies exactly on v's line   |   square = 90° at foot")

def panel_rejection(ax, u, v):
    """Decompose u = proj + rejection visually."""
    proj = Projection.vector_projection(u, v)
    rej  = Projection.rejection(u, v)
    proj_tip = tuple(proj.elements)
    rej_tip  = tuple(rej.elements)
    u_tip    = tuple(u.elements)

    # draw the parallelogram: proj + rej = u
    ax.fill([0, proj_tip[0], u_tip[0], rej_tip[0]],
            [0, proj_tip[1], u_tip[1], rej_tip[1]],
            alpha=0.08, color="tab:purple")
    ax.plot([0, proj_tip[0], u_tip[0], rej_tip[0], 0],
            [0, proj_tip[1], u_tip[1], rej_tip[1], 0],
            color="tab:purple", lw=0.8, linestyle="--")

    arrow(ax, u_tip,    color="tab:blue",   lw=2.8)
    arrow(ax, proj_tip, color="tab:green",  lw=2.4)
    arrow(ax, rej_tip,  color="tab:red",    lw=2.4)

    # rej from proj_tip to u_tip
    arrow(ax, u_tip, origin=proj_tip, color="tab:red", lw=2, ls="--")
    right_angle_marker(ax, proj_tip,
                       (rej.elements[0], rej.elements[1]),
                       (proj.elements[0], proj.elements[1]))

    label_tip(ax, u_tip,    "u = proj + rej",   "tab:blue",  side=1)
    label_tip(ax, proj_tip, f"proj {[round(x,2) for x in proj.elements]}", "tab:green", side=-1)
    label_tip(ax, rej_tip,  f"rej {[round(x,2) for x in rej.elements]}",  "tab:red",   side=1)

    setup(ax, [u_tip, proj_tip, rej_tip],
          "Rejection: rej_v(u) = u − proj_v(u)\nDecomposition: u = proj + rej")
    caption(ax, "green + red = blue (vector addition)   |   purple parallelogram = geometric proof   |   square = 90°")

def panel_orthogonal_component(ax, u, v):
    """orth_comp is identical to rejection — visualize its orthogonality explicitly."""
    orth = Projection.orthogonal_component(u, v)
    proj = Projection.vector_projection(u, v)
    u_tip    = tuple(u.elements)
    proj_tip = tuple(proj.elements)
    orth_tip = tuple(orth.elements)

    arrow(ax, u_tip,    color="tab:blue",   lw=2.4)
    arrow(ax, tuple(v.elements), color="tab:orange", lw=2.2)
    arrow(ax, proj_tip, color="tab:green",  lw=2.4)
    arrow(ax, orth_tip, color="tab:red",    lw=2.6)

    # show angle between orth and v
    dot_val = abs(orth.dot_product(v))
    angle_deg = orth.angle_between_vectors(v) if orth.magnitude() > 1e-9 else 0
    right_angle_marker(ax, (0, 0),
                       (orth.elements[0], orth.elements[1]),
                       (v.elements[0],    v.elements[1]))

    label_tip(ax, u_tip,    "u", "tab:blue",  side=1)
    label_tip(ax, tuple(v.elements), "v", "tab:orange", side=-1)
    label_tip(ax, proj_tip, "proj", "tab:green", side=-1)
    label_tip(ax, orth_tip, f"orth_comp\n{[round(x,2) for x in orth.elements]}",
              "tab:red", side=1)

    setup(ax, [u_tip, tuple(v.elements), proj_tip, orth_tip],
          f"Orthogonal Component: orth_v(u) = u − proj_v(u)\north · v = {dot_val:.2e}  ≈ 0  (angle = {angle_deg:.1f}°)")
    caption(ax, "red = orth_comp (identical to rejection)   |   square at origin = 90° between orth_comp and v")

def panel_projection_matrix(ax, v):
    """Show P = vvᵀ/(vᵀv) by applying it to a fan of vectors."""
    P = Projection.projection_matrix(v)
    v_hat = Vector([x / v.magnitude() for x in v.elements])

    # fan of input vectors
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    r = max(v.magnitude(), 2.0) * 0.9
    tips_in, tips_out = [], []
    for a in angles:
        inp = Vector([r * math.cos(a), r * math.sin(a)])
        out = P.apply_transformation(inp)
        tips_in.append(inp.elements)
        tips_out.append(out.elements)

    # draw faint input arrows and projected output arrows
    for tin, tout in zip(tips_in, tips_out):
        arrow(ax, tuple(tin), color="tab:blue", lw=0.9, zorder=2)
        ax.plot([tin[0], tout[0]], [tin[1], tout[1]],
                color="gray", lw=0.7, linestyle=":", zorder=1)
        arrow(ax, tuple(tout), color="tab:green", lw=1.5, zorder=3)

    arrow(ax, tuple(v.elements), color="tab:orange", lw=2.6, zorder=5)
    label_tip(ax, tuple(v.elements), "v", "tab:orange", side=-1)

    # annotate the matrix
    mat = [[round(P[i][j], 3) for j in range(2)] for i in range(2)]
    ax.text(0.02, 0.97, f"P =\n{mat[0]}\n{mat[1]}",
            transform=ax.transAxes, va="top", fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))

    all_pts = tips_in + tips_out + [v.elements]
    setup(ax, all_pts,
          "Projection Matrix P = vvᵀ/(vᵀv)\n(every blue input arrow maps to a green arrow on v's line)")
    caption(ax, "all green arrows are collinear with v   |   P·P = P (idempotent) — applying P twice gives the same result")

def panel_idempotency(ax, v):
    """P² = P: applying the projection matrix twice gives the same result."""
    P  = Projection.projection_matrix(v)
    PP = P * P

    angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    r = max(v.magnitude(), 2.0) * 0.85
    all_tips = []
    for a in angles:
        inp = Vector([r * math.cos(a), r * math.sin(a)])
        p1  = P.apply_transformation(inp)
        p2  = PP.apply_transformation(inp)
        all_tips += [inp.elements, p1.elements, p2.elements]

        arrow(ax, tuple(inp.elements), color="tab:blue",   lw=0.9)
        arrow(ax, tuple(p1.elements),  color="tab:green",  lw=2.2)
        # p2 drawn slightly offset so it doesn't obscure p1
        ax.plot(*p2.elements, "x", color="tab:red", markersize=8, markeredgewidth=2, zorder=5)

    arrow(ax, tuple(v.elements), color="tab:orange", lw=2.6, zorder=6)
    label_tip(ax, tuple(v.elements), "v", "tab:orange", side=-1)

    # verify numerically
    diff = max(abs(P[i][j] - PP[i][j]) for i in range(2) for j in range(2))
    setup(ax, all_tips + [v.elements],
          f"Idempotency: P² = P\n‖P² − P‖_max = {diff:.2e}  (should be ≈ 0)")
    caption(ax, "blue = input   |   green arrow = P·x   |   red cross = P²·x  (lands exactly on green = P² = P)")

def panel_scalar_vs_vector_comparison(ax, u, v):
    """Side-by-side bar chart: scalar projection and the norms of proj & rej."""
    sc      = Projection.scalar_projection(u, v)
    proj    = Projection.vector_projection(u, v)
    rej     = Projection.rejection(u, v)

    labels = [
        "scalar proj\n(comp_v(u))",
        "‖proj_v(u)‖",
        "‖rej_v(u)‖",
        "‖u‖",
        "Pythagorean\n√(proj²+rej²)",
    ]
    pyth = math.sqrt(proj.magnitude()**2 + rej.magnitude()**2)
    values = [sc, proj.magnitude(), rej.magnitude(), u.magnitude(), pyth]
    colors = ["tab:green", "tab:green", "tab:red", "tab:blue", "tab:purple"]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.55)
    for bar, val in zip(bars, values):
        ax.annotate(f"{val:.3f}",
                    (bar.get_x() + bar.get_width() / 2, max(val, 0)),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8.5)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylim(min(values + [0]) - 0.6, max(values) + 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_title("Scalar vs Vector Projection — Magnitude Comparison\n"
                 "(Pythagorean check: ‖proj‖² + ‖rej‖² = ‖u‖²)",
                 fontsize=11, fontweight="bold")
    caption(ax, "green = along v   |   red = perpendicular to v   |   purple should equal blue (Pythagorean theorem)")

def panel_projection_onto_axes(ax, u):
    """Visualise the trivial (but instructive) case: project u onto x and y axes."""
    e1 = Vector([1.0, 0.0])
    e2 = Vector([0.0, 1.0])
    p1 = Projection.vector_projection(u, e1)
    p2 = Projection.vector_projection(u, e2)
    u_tip = tuple(u.elements)

    arrow(ax, u_tip,             color="tab:blue",   lw=2.6)
    arrow(ax, tuple(p1.elements), color="tab:green",  lw=2.4)
    arrow(ax, tuple(p2.elements), color="tab:red",    lw=2.4)

    # dotted drops to axes
    ax.plot([u_tip[0], p1.elements[0]], [u_tip[1], p1.elements[1]],
            color="tab:green", lw=1.2, linestyle=":")
    ax.plot([u_tip[0], p2.elements[0]], [u_tip[1], p2.elements[1]],
            color="tab:red",   lw=1.2, linestyle=":")

    label_tip(ax, u_tip,             "u", "tab:blue",  side=1)
    label_tip(ax, tuple(p1.elements), f"proj_x = ({p1[0]:.2f}, 0)", "tab:green", side=-1)
    label_tip(ax, tuple(p2.elements), f"proj_y = (0, {p2[1]:.2f})", "tab:red",   side=1)

    setup(ax, [u_tip, p1.elements, p2.elements],
          "Projection onto Coordinate Axes (special case)\n"
          "proj onto x-axis = (ux, 0),  proj onto y-axis = (0, uy)")
    caption(ax, "this is how coordinates work — they ARE projections onto the basis vectors")

# Main

def main():
    u, v = get_inputs()

    fig = plt.figure(figsize=(20, 24))
    fig.suptitle("Vector Projections — Visualized", fontsize=20, fontweight="bold")
    grid = fig.add_gridspec(4, 2, hspace=0.55, wspace=0.32)

    panel_overview(              fig.add_subplot(grid[0, 0]), u, v)
    panel_scalar_projection(     fig.add_subplot(grid[0, 1]), u, v)

    panel_vector_projection(     fig.add_subplot(grid[1, 0]), u, v)
    panel_rejection(             fig.add_subplot(grid[1, 1]), u, v)

    panel_orthogonal_component(  fig.add_subplot(grid[2, 0]), u, v)
    panel_projection_matrix(     fig.add_subplot(grid[2, 1]), v)

    panel_idempotency(           fig.add_subplot(grid[3, 0]), v)
    panel_scalar_vs_vector_comparison(fig.add_subplot(grid[3, 1]), u, v)

    out_path = Path(__file__).parent / "figures"
    out_path.mkdir(parents=True, exist_ok=True) 
    figure_path = out_path / "v04_projection.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to {figure_path}")

    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()