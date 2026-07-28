"""
v03_linear_transformation.py

Visual companion to m03_linear_transformation.py.

Linear transformations only have a clean *picture* in 2D (grids, unit
squares, circles), so the geometric panels here all use 2x2 matrices —
A is the main subject, B is a second 2x2 matrix used for the composition
check, and v is a 2D vector used for the "transform a specific vector"
panel. Trace/determinant/invertibility etc. are read straight from your
LinearTransformation class, not recomputed.

Output: v03_linear_transformation.png saved next to this
script, plus an interactive window if your environment supports one.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from pathlib import Path
from linear_algebra.m01_vector_ops import Vector
from linear_algebra.m03_linear_transformation import LinearTransformation


# Input handling

def read_matrix_2x2(prompt, default):
    raw = input(f"{prompt} rows as 'a b; c d' [default {default}]: ").strip()
    if not raw:
        return LinearTransformation([list(default[0]), list(default[1])])
    rows = raw.split(";")
    matrix = [[float(x) for x in row.replace(",", " ").split()] for row in rows]
    return LinearTransformation(matrix)
  
def read_vector_2d(prompt, default):
    raw = input(f"{prompt} [default {default}]: ").strip()
    if not raw:
        return Vector(list(default))
    return Vector([float(x) for x in raw.replace(",", " ").split()])
  
def get_inputs():
    print("=== Linear Transformation Visualizer (2x2 matrices) ===")
    a = read_matrix_2x2("Matrix A", ((2, 1), (0, 3)))
    b = read_matrix_2x2("Matrix B", ((0, -1), (1, 0)))
    v = read_vector_2d("Vector v (2D)", (1, 1))
    return a, b, v

# Geometry helpers

def apply(matrix, xy):
    """Apply a LinearTransformation to a plain (x, y) point, return (x, y)."""
    result = matrix.apply_transformation(Vector([xy[0], xy[1]]))
    return result.elements[0], result.elements[1]
  
def draw_grid(ax, matrix, color, alpha=0.5, extent=2, n_lines=5, lw=0.9, label=None):
    """Draw a grid of lines warped through `matrix` (identity matrix = a plain grid)."""
    coords = np.linspace(-extent, extent, n_lines)
    samples = np.linspace(-extent, extent, 30)
    first = True
    for c in coords:
        xs = [apply(matrix, (c, y))[0] for y in samples]
        ys = [apply(matrix, (c, y))[1] for y in samples]
        ax.plot(xs, ys, color=color, alpha=alpha, lw=lw,
                 label=label if first else None)
        first = False
        xs = [apply(matrix, (x, c))[0] for x in samples]
        ys = [apply(matrix, (x, c))[1] for x in samples]
        ax.plot(xs, ys, color=color, alpha=alpha, lw=lw)
  
def draw_basis(ax, matrix, colors=("tab:red", "tab:green")):
    i_img = apply(matrix, (1, 0))
    j_img = apply(matrix, (0, 1))
    for vec, color, label in [(i_img, colors[0], "A·i"), (j_img, colors[1], "A·j")]:
        ax.annotate("", xy=vec, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.4))
        ax.annotate(label, vec, textcoords="offset points", xytext=(6, 6),
                    color=color, fontsize=9, fontweight="bold")
  
def unit_square_polygon(matrix, color, alpha=0.3, ls="-"):
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    warped = [apply(matrix, c) for c in corners]
    return Polygon(warped, closed=True, facecolor=color, edgecolor=color,
                    alpha=alpha, linewidth=2, linestyle=ls)
 
def compute_extent(points, minimum=2.0, pad_factor=1.35):
    """Smallest half-width that comfortably fits every (x, y) point given."""
    if not points:
        return minimum
    max_abs = max(abs(c) for p in points for c in p)
    return max(minimum, pad_factor * max_abs)
  
def setup_axes(ax, extent, title=""):
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.axhline(0, color="gray", lw=0.8, zorder=0)
    ax.axvline(0, color="gray", lw=0.8, zorder=0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10.5, fontweight="bold")
  
def perp_offset(dx, dy, magnitude=12, side=1):
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return (magnitude, magnitude)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    return (px * magnitude * side, py * magnitude * side)
 
def add_caption(ax, text):
    ax.text(0.5, -0.14, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=8, bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
            clip_on=False)
 
# Panels

def panel_grid_and_area(ax, a):
    identity = LinearTransformation.identity(2)
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    warped_corners = [apply(a, c) for c in corners]
    i_img, j_img = apply(a, (1, 0)), apply(a, (0, 1))
    extent = compute_extent(warped_corners + [i_img, j_img])
 
    draw_grid(ax, identity, color="lightgray", alpha=0.6, extent=extent, n_lines=7)
    draw_grid(ax, a, color="tab:blue", alpha=0.55, extent=extent, n_lines=7)
    ax.add_patch(unit_square_polygon(identity, "gray", alpha=0.15))
    ax.add_patch(unit_square_polygon(a, "tab:blue", alpha=0.35))
    draw_basis(ax, a)
    det = a.determinant_optimised()
    orient = "orientation flipped (det < 0)" if det < 0 else "orientation preserved"
    setup_axes(ax, extent, f"Grid Transformation by A\narea scales by |det(A)| = {abs(det):.2f} — {orient}")
  
def panel_vector_transform(ax, a, v):
    identity = LinearTransformation.identity(2)
    v_img = apply(a, v.elements)
    extent = compute_extent([tuple(v.elements), v_img])
 
    draw_grid(ax, identity, color="lightgray", alpha=0.4, extent=extent, n_lines=7)
    draw_grid(ax, a, color="tab:blue", alpha=0.35, extent=extent, n_lines=7)
    ax.annotate("", xy=v.elements, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="tab:purple", lw=2.4, linestyle="--"))
    ax.annotate(f"v = {v.elements}", v.elements, textcoords="offset points",
                xytext=perp_offset(v.elements[0], v.elements[1], magnitude=12, side=1),
                color="tab:purple", fontsize=9, fontweight="bold")
    ax.annotate("", xy=v_img, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="tab:green", lw=2.6))
    ax.annotate(f"A·v = ({v_img[0]:.2f}, {v_img[1]:.2f})", v_img, textcoords="offset points",
                xytext=perp_offset(v_img[0], v_img[1], magnitude=12, side=-1),
                color="tab:green", fontsize=9, fontweight="bold")
    setup_axes(ax, extent, "Applying A to a Specific Vector\n(dashed = v, solid = A·v, faint grid shows the warp)")
  
def panel_composition(ax, a, b):
    identity = LinearTransformation.identity(2)
    ab = a * b
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    b_only = [apply(b, c) for c in corners]
    a_after_b = [apply(a, c) for c in b_only]
    direct_ab = [apply(ab, c) for c in corners]
    extent = compute_extent(b_only + a_after_b + direct_ab)
 
    ax.add_patch(Polygon([apply(identity, c) for c in corners], closed=True,
                          facecolor="none", edgecolor="gray", linestyle=":", linewidth=1.5))
    ax.add_patch(Polygon(b_only, closed=True, facecolor="tab:orange", edgecolor="tab:orange",
                          alpha=0.25, linewidth=1.5, label="after B"))
    ax.add_patch(Polygon(a_after_b, closed=True, facecolor="tab:blue", edgecolor="tab:blue",
                          alpha=0.35, linewidth=2, label="A(B(square))"))
    ax.plot(*zip(*(direct_ab + [direct_ab[0]])), color="tab:red", linestyle="--",
            lw=2)
    setup_axes(ax, extent, "Composition Check: A(B(x)) vs (A·B)x\n(red dashed outline should trace the blue shape exactly)")
    add_caption(ax, "gray dotted = original square   |   orange = after B   |   blue = A(B(square))   |   red dashed = (A·B)(square)")
  
def panel_transpose(ax, a):
    at = a.transpose()
    identity = LinearTransformation.identity(2)
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    a_corners = [apply(a, c) for c in corners]
    at_corners = [apply(at, c) for c in corners]
    extent = compute_extent(a_corners + at_corners)
 
    ax.add_patch(unit_square_polygon(identity, "gray", alpha=0.12))
    ax.add_patch(unit_square_polygon(a, "tab:blue", alpha=0.3))
    ax.add_patch(unit_square_polygon(at, "tab:orange", alpha=0.3))
    draw_basis(ax, a, colors=("tab:blue", "tab:blue"))
    i_img, j_img = apply(at, (1, 0)), apply(at, (0, 1))
    for vec in (i_img, j_img):
        ax.annotate("", xy=vec, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="tab:orange", lw=2.2))
    setup_axes(ax, extent, "A (blue) vs Aᵀ (orange)\n(transpose reflects how rows/columns map basis vectors)")
  
def panel_inverse_roundtrip(ax, a):
    identity = LinearTransformation.identity(2)
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    a_corners = [apply(a, c) for c in corners]
    extent = compute_extent(a_corners)
 
    ax.add_patch(unit_square_polygon(identity, "gray", alpha=0.15))
    ax.add_patch(unit_square_polygon(a, "tab:blue", alpha=0.35))
    if a.is_invertible():
        a_inv = a.inverse_optimised()
        after_a = a_corners
        roundtrip = [apply(a_inv, c) for c in after_a]
        ax.add_patch(Polygon(roundtrip, closed=True, facecolor="none",
                              edgecolor="tab:green", linewidth=2.5, linestyle="--"))
        add_caption(ax, "gray = original square   |   blue = A(square)   |   green dashed = A\u207b\u00b9(A(square))")
        title = "Inverse Round-Trip: A⁻¹(A(x)) = x\n(green dashed should land back on the gray square)"
    else:
        ax.text(0.5, 0.5, "Matrix A is singular\n(det = 0) — no inverse exists",
                transform=ax.transAxes, ha="center", va="center", fontsize=11,
                color="tab:red", fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="tab:red"))
        title = "Inverse Round-Trip: A⁻¹(A(x)) = x"
    setup_axes(ax, extent, title)
  
def panel_orthogonality(ax, a):
    theta = np.linspace(0, 2 * np.pi, 200)
    circle_pts = np.array([(np.cos(t), np.sin(t)) for t in theta])
    warped = np.array([apply(a, pt) for pt in circle_pts])
    extent = compute_extent([tuple(p) for p in warped])
 
    ax.add_patch(Circle((0, 0), 1, facecolor="none", edgecolor="gray", linestyle=":", linewidth=1.5))
    ax.plot(warped[:, 0], warped[:, 1], color="tab:blue", lw=2.2)
    draw_basis(ax, a)
 
    orth = a.is_orthogonal()
    verdict = "circle stays a circle -> orthogonal (lengths & angles preserved)" if orth \
        else "circle becomes an ellipse -> NOT orthogonal (lengths/angles distorted)"
    setup_axes(ax, extent, f"Unit Circle Transformed by A\nis_orthogonal() = {orth} — {verdict}")
  
def panel_identity_zero(ax, a):
    identity = LinearTransformation.identity(2)
    zero = LinearTransformation.zeros(2, 2)
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    ax.add_patch(Polygon([apply(identity, c) for c in corners], closed=True,
                          facecolor="none", edgecolor="gray", linewidth=2))
    ax.add_patch(Polygon([apply(identity, c) for c in corners], closed=True,
                         facecolor="tab:green", edgecolor="tab:green",
                         alpha=0.25, linewidth=2, linestyle="--"))
    zero_pt = apply(zero, (0.5, 0.5))
    ax.plot(*zero_pt, "o", color="tab:red", markersize=10)
    setup_axes(ax, 2, "Identity vs Zero Matrix\n(identity changes nothing, zero collapses everything to the origin)")
    add_caption(ax, "gray/green = Identity(square) — unchanged   |   red dot = Zero(any point) — collapses to origin")
  
def panel_dashboard(ax, a, b):
    ax.axis("off")
    lines = [f"Matrix A, shape {a.shape}:"]
    for row in a.matrix:
        lines.append("  " + str(row))
    lines.append("")
    is_sq = a.shape[0] == a.shape[1]
    if is_sq:
        lines.append(f"trace(A)               = {a.trace():.3f}")
        lines.append(f"determinant(A)         = {a.determinant_optimised():.3f}")
        lines.append(f"is_invertible(A)        = {a.is_invertible()}")
        lines.append(f"is_symmetric(A)         = {a.is_symmetric()}")
        lines.append(f"is_orthogonal(A)        = {a.is_orthogonal()}")
    else:
        lines.append("A is not square — trace/determinant/inverse are undefined.")
    lines.append("")
    lines.append(f"Matrix B, shape {b.shape}:")
    for row in b.matrix:
        lines.append("  " + str(row))
    if a.shape[1] == b.shape[0]:
        lines.append("")
        lines.append("A * B =")
        for row in (a * b).matrix:
            lines.append("  " + str([round(x, 2) for x in row]))
 
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=9.5, family="monospace",
            bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="gray"))
    ax.set_title("Structural Properties Dashboard", fontsize=10.5, fontweight="bold")
 
# Main

def main():
    a, b, v = get_inputs()
 
    fig = plt.figure(figsize=(15, 22))
    fig.suptitle("Linear Transformations — Visualized", fontsize=18, fontweight="bold")
    grid = fig.add_gridspec(4, 2, hspace=0.6, wspace=0.3)
 
    panel_grid_and_area(fig.add_subplot(grid[0, 0]), a)
    panel_vector_transform(fig.add_subplot(grid[0, 1]), a, v)
 
    panel_composition(fig.add_subplot(grid[1, 0]), a, b)
    panel_transpose(fig.add_subplot(grid[1, 1]), a)
 
    panel_inverse_roundtrip(fig.add_subplot(grid[2, 0]), a)
    panel_orthogonality(fig.add_subplot(grid[2, 1]), a)
 
    panel_identity_zero(fig.add_subplot(grid[3, 0]), a)
    panel_dashboard(fig.add_subplot(grid[3, 1]), a, b)
 
    out_path = Path(__file__).parent / "figures"
    out_path.mkdir(parents=True, exist_ok=True) 
    figure_path = out_path / "v03_linear_transformation.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to {figure_path}")
    
    try:
        plt.show()
    except Exception:
        pass
  
if __name__ == "__main__":
    main()