"""
v06_fundamental_subspaces.py

Visual companion to m06_fundamental_subspaces.py.

Fundamental subspaces only have a clean *picture* when the matrix is
small, so this script asks for a 2x3 matrix (2 rows, 3 columns) — that
gives:
  - Column Space & Left Null Space living in R^2 (easy 2D plot)
  - Row Space & Null Space living in R^3 (3D plot)
and, crucially, both pairs are orthogonal complements of each other —
which is the single most important fact about the four subspaces and
the whole reason this module exists.


Output: v06_fundamental_subspaces.png
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from linear_algebra.m01_vector_ops import Vector
from linear_algebra.m03_linear_transformation import LinearTransformation
from linear_algebra.m06_fundamental_subspaces import FundamentalSubspaces

# input

def read_matrix_2x3(prompt, default):
    raw = input(f"{prompt} rows as 'a b c; d e f' [default {default}]: ").strip()
    if not raw:
        return LinearTransformation([list(default[0]), list(default[1])])
    rows = raw.split(";")
    matrix = [[float(x) for x in row.replace(",", " ").split()] for row in rows]
    return LinearTransformation(matrix)

def get_inputs():
    print("=== Fundamental Subspaces V06r (2x3 matrices) ===")
    a = read_matrix_2x3("Matrix A", ((1, 2, -1), (2, 4, -2)))
    return a

# 2D helpers

def setup_2d(ax, points, title, pad=0.4):
    xs = [p[0] for p in points] + [0.0]
    ys = [p[1] for p in points] + [0.0]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 2.0)
    p = pad * span
    ax.set_xlim(min(xs) - p, max(xs) + p)
    ax.set_ylim(min(ys) - p, max(ys) + p)
    ax.axhline(0, color="gray", lw=0.7, zorder=0)
    ax.axvline(0, color="gray", lw=0.7, zorder=0)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_title(title, fontsize=10.5, fontweight="bold")

def caption(ax, text, y=-0.14):
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, bbox=dict(boxstyle="round", facecolor="white",
                                    edgecolor="gray", alpha=0.92),
            clip_on=False)

def draw_line_2d(ax, direction, color, extent=6, lw=2.2, ls="-", label=None, alpha=1.0):
    n = np.hypot(direction[0], direction[1])
    if n < 1e-9:
        return
    ux, uy = direction[0] / n, direction[1] / n
    ax.plot([-extent * ux, extent * ux], [-extent * uy, extent * uy],
            color=color, lw=lw, linestyle=ls, alpha=alpha, zorder=2, label=label)

def arrow_2d(ax, tip, color, lw=2.4, label=None):
    ax.annotate("", xy=tip, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0),
                zorder=4)
    if label:
        ax.annotate(label, tip, textcoords="offset points", xytext=(8, 8),
                    color=color, fontsize=9.5, fontweight="bold")

# panels

def panel_matrix_rref(ax, a, fs):
    ax.axis("off")
    lines = [f"Matrix A, shape {a.shape}:"]
    for row in a.matrix:
        lines.append("  " + str([round(x, 3) for x in row]))
    lines.append("")
    lines.append("RREF(A):")
    for row in fs._rref:
        lines.append("  " + str([round(x, 3) for x in row]))
    lines.append("")
    lines.append(f"Pivot columns : {fs._pivot_cols}")
    lines.append(f"rank(A)       : {fs.rank()}")
    lines.append(f"nullity(A)    : {fs.nullity()}")
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="gray"))
    ax.set_title("Matrix & Reduced Row Echelon Form", fontsize=10.5, fontweight="bold")

def panel_col_and_left_null(ax, a, fs):
    """Column Space and Left Null Space both live in R^2 (m rows) — and are
    orthogonal complements of each other."""
    col_basis = fs.column_space()
    lns_basis = fs.left_null_space()

    all_pts = [(6, 6), (-6, -6)]
    for v in col_basis:
        draw_line_2d(ax, v.elements, "tab:blue", ls=":", alpha=0.5)
        arrow_2d(ax, tuple(v.elements), "tab:blue", label=f"col basis {[round(x,2) for x in v.elements]}")
        all_pts.append(tuple(v.elements))
    for v in lns_basis:
        draw_line_2d(ax, v.elements, "tab:red", ls=":", alpha=0.5)
        arrow_2d(ax, tuple(v.elements), "tab:red", label=f"left-null basis {[round(x,2) for x in v.elements]}")
        all_pts.append(tuple(v.elements))

    dim_c, dim_l = len(col_basis), len(lns_basis)
    dot_check = ""
    if dim_c == 1 and dim_l == 1:
        d = col_basis[0].dot_product(lns_basis[0])
        dot_check = f"\ncol·left-null = {d:.2e} ≈ 0 (perpendicular)"

    setup_2d(ax, all_pts,
             f"Column Space (blue) & Left Null Space (red) in R^{a.shape[0]}\n"
             f"dim(col)={dim_c}, dim(left-null)={dim_l}{dot_check}")
    caption(ax, "blue = Col(A) — spanned by pivot columns   |   red = Left Null Space — solutions to Aᵀy=0")

def panel_row_and_null_3d(ax_slot, fig, a, fs):
    """Row Space and Null Space both live in R^n (n columns) — and are
    orthogonal complements of each other. Only handles n=3 (required input shape)."""
    ax = fig.add_subplot(ax_slot, projection="3d")
    row_basis = fs.row_space()
    null_basis = fs.null_space()

    extent = 3.0

    # Null space: could be dim 0 (origin only), 1 (line), 2 (plane), or 3 (all of R^3)
    if len(null_basis) == 1:
        b = null_basis[0].elements
        n = np.linalg.norm(b)
        t = np.linspace(-extent, extent, 2)
        pts = np.outer(t, np.array(b) / n) if n > 1e-9 else np.zeros((2, 3))
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="tab:green", lw=3, label="Null Space (line)")
    elif len(null_basis) == 2:
        b1 = np.array(null_basis[0].elements)
        b2 = np.array(null_basis[1].elements)
        s = np.linspace(-1.4, 1.4, 8)
        t = np.linspace(-1.4, 1.4, 8)
        S, T = np.meshgrid(s, t)
        surface = (S[..., None] * b1 + T[..., None] * b2)
        corners = [
            (-1.4 * b1 - 1.4 * b2), (1.4 * b1 - 1.4 * b2),
            (1.4 * b1 + 1.4 * b2), (-1.4 * b1 + 1.4 * b2),
        ]
        poly = Poly3DCollection([corners], alpha=0.25, facecolor="tab:green", edgecolor="tab:green")
        ax.add_collection3d(poly)

    # Row space: usually dim 1 here, but handle dim 2 as a plane too
    if len(row_basis) == 1:
        b = row_basis[0].elements
        n = np.linalg.norm(b)
        t = np.linspace(-extent, extent, 2)
        pts = np.outer(t, np.array(b) / n) if n > 1e-9 else np.zeros((2, 3))
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="tab:orange", lw=3, label="Row Space (line)")
        ax.quiver(0, 0, 0, b[0], b[1], b[2], color="tab:orange", linewidth=2, arrow_length_ratio=0.15)
    elif len(row_basis) == 2:
        b1 = np.array(row_basis[0].elements)
        b2 = np.array(row_basis[1].elements)
        corners = [
            (-1.2 * b1 - 1.2 * b2), (1.2 * b1 - 1.2 * b2),
            (1.2 * b1 + 1.2 * b2), (-1.2 * b1 + 1.2 * b2),
        ]
        poly = Poly3DCollection([corners], alpha=0.25, facecolor="tab:orange", edgecolor="tab:orange")
        ax.add_collection3d(poly)

    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(-extent, extent)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    dim_r, dim_n = len(row_basis), len(null_basis)
    perp_note = ""
    if dim_r == 1 and dim_n >= 1:
        # dot product of row basis with each null basis vector should be ~0
        checks = [round(row_basis[0].dot_product(nb), 8) for nb in null_basis]
        perp_note = f"\nrow·null checks ≈ 0: {checks}"
    ax.set_title(f"Row Space (orange) & Null Space (green) in R^{a.shape[1]}\n"
                 f"dim(row)={dim_r}, dim(null)={dim_n}{perp_note}",
                 fontsize=10.5, fontweight="bold")

def panel_dimension_summary(ax, a, fs):
    m, n = a.shape
    dims = {
        "rank\n(=dim Col)": fs.rank(),
        "nullity\n(=dim Null)": fs.nullity(),
        "dim\nRow Space": len(fs.row_space()),
        "dim\nLeft Null": len(fs.left_null_space()),
    }
    labels = list(dims.keys())
    values = list(dims.values())
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]
    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.annotate(str(v), (bar.get_x() + bar.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(values + [1]) + 1.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_title(f"Dimension Summary  (A is {m}×{n})\n"
                 "rank(Col)=rank(Row) always — that's the Rank Theorem",
                 fontsize=10.5, fontweight="bold")
    caption(ax, "note dim(Col)=dim(Row)=rank(A) even though they live in different spaces (R^m vs R^n)")

def panel_rank_nullity_theorem(ax, a, fs):
    m, n = a.shape
    rank = fs.rank()
    nullity = fs.nullity()
    lns_dim = len(fs.left_null_space())

    categories = ["rank + nullity\n(should = n)", "rank + dim(LeftNull)\n(should = m)"]
    computed = [rank + nullity, rank + lns_dim]
    targets = [n, m]

    idx = np.arange(len(categories))
    width = 0.32
    ax.bar(idx - width / 2, computed, width, label="computed", color="tab:purple")
    ax.bar(idx + width / 2, targets, width, label="target (n or m)", color="gray", alpha=0.6)
    for i, (c, t) in enumerate(zip(computed, targets)):
        ax.annotate(f"{c}", (idx[i] - width/2, c), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9)
        ax.annotate(f"{t}", (idx[i] + width/2, t), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9)
    ax.set_xticks(idx)
    ax.set_xticklabels(categories, fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(computed + targets) + 1.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_title("Rank-Nullity Theorem Verification\n(purple bar should exactly match the gray bar)",
                 fontsize=10.5, fontweight="bold")

def panel_null_space_sanity(ax, a, fs):
    null_basis = fs.null_space()
    if not null_basis:
        ax.axis("off")
        ax.text(0.5, 0.5, "Null space is trivial ({0})\nno free variables — nothing to check",
                transform=ax.transAxes, ha="center", va="center", fontsize=10.5)
        ax.set_title("Null Space Sanity Check: A·nᵢ = 0", fontsize=10.5, fontweight="bold")
        return

    labels = [f"‖A·n{i+1}‖" for i in range(len(null_basis))]
    residuals = [a.apply_transformation(v).magnitude() for v in null_basis]
    bars = ax.bar(labels, residuals, color="tab:green")
    for bar, r in zip(bars, residuals):
        ax.annotate(f"{r:.2e}", (bar.get_x() + bar.get_width() / 2, r),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8.5)
    ax.set_ylim(0, max(residuals + [1e-9]) * 3 + 1e-9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_title("Null Space Sanity Check: A·nᵢ = 0\n(every bar should be ≈ 0)",
                 fontsize=10.5, fontweight="bold")
    caption(ax, "each nᵢ is a basis vector of Null(A) — multiplying by A should return the zero vector")

def panel_left_null_sanity(ax, a, fs):
    lns_basis = fs.left_null_space()
    at = a.transpose()
    if not lns_basis:
        ax.axis("off")
        ax.text(0.5, 0.5, "Left null space is trivial ({0})\nA has full row rank — nothing to check",
                transform=ax.transAxes, ha="center", va="center", fontsize=10.5)
        ax.set_title("Left Null Space Sanity Check: Aᵀ·lᵢ = 0", fontsize=10.5, fontweight="bold")
        return

    labels = [f"‖Aᵀ·l{i+1}‖" for i in range(len(lns_basis))]
    residuals = [at.apply_transformation(v).magnitude() for v in lns_basis]
    bars = ax.bar(labels, residuals, color="tab:red")
    for bar, r in zip(bars, residuals):
        ax.annotate(f"{r:.2e}", (bar.get_x() + bar.get_width() / 2, r),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8.5)
    ax.set_ylim(0, max(residuals + [1e-9]) * 3 + 1e-9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_title("Left Null Space Sanity Check: Aᵀ·lᵢ = 0\n(every bar should be ≈ 0)",
                 fontsize=10.5, fontweight="bold")
    caption(ax, "each lᵢ is a basis vector of Null(Aᵀ) — multiplying by Aᵀ should return the zero vector")

def panel_all_bases_dashboard(ax, a, fs):
    ax.axis("off")
    lines = ["Four Fundamental Subspaces — Basis Vectors", "=" * 46, ""]

    def fmt_list(vecs, name):
        out = [f"{name} (dim = {len(vecs)}):"]
        if not vecs:
            out.append("  {0}  (trivial)")
        for i, v in enumerate(vecs):
            out.append(f"  {name[0].lower()}{i+1} = {[round(x, 3) for x in v.elements]}")
        return out

    lines += fmt_list(fs.column_space(), "Column Space")
    lines.append("")
    lines += fmt_list(fs.row_space(), "Row Space")
    lines.append("")
    lines += fmt_list(fs.null_space(), "Null Space")
    lines.append("")
    lines += fmt_list(fs.left_null_space(), "Left Null Space")

    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=9.3, family="monospace",
            bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="gray"))
    ax.set_title("All Basis Vectors", fontsize=10.5, fontweight="bold")

# main

def main():
    a = get_inputs()
    if a.shape != (2, 3):
        print(f"Note: geometric panels are designed for a 2x3 matrix; got {a.shape}."
              " Row/Null space panel may not render correctly for other shapes.")
    fs = FundamentalSubspaces(a)

    fig = plt.figure(figsize=(20, 24))
    fig.suptitle("Fundamental Subspaces — V06d", fontsize=20, fontweight="bold")
    grid = fig.add_gridspec(4, 2, hspace=0.55, wspace=0.32)

    panel_matrix_rref(          fig.add_subplot(grid[0, 0]), a, fs)
    panel_col_and_left_null(    fig.add_subplot(grid[0, 1]), a, fs)

    panel_row_and_null_3d(grid[1, 0], fig, a, fs)
    panel_dimension_summary(    fig.add_subplot(grid[1, 1]), a, fs)

    panel_rank_nullity_theorem( fig.add_subplot(grid[2, 0]), a, fs)
    panel_null_space_sanity(    fig.add_subplot(grid[2, 1]), a, fs)

    panel_left_null_sanity(     fig.add_subplot(grid[3, 0]), a, fs)
    panel_all_bases_dashboard(  fig.add_subplot(grid[3, 1]), a, fs)

    out_path = Path(__file__).parent / "figures"
    out_path.mkdir(parents=True, exist_ok=True)
    figure_path = out_path / "v06_fundamental_spaces.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to {figure_path}")
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()