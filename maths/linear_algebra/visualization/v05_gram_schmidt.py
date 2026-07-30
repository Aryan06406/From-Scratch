"""
v05_gram_schmidt.py

Visualizes the Gram-Schmidt process step-by-step, driven by the SAME
algorithm as m05_gram_schmidt.py (via traced_gram_schmidt.py, which
narrates every projection/subtraction/normalization).
 
Supports 2D and 3D input vectors.

Uses the real Vector, Projection, and GramSchmidt classes from this
package -- no reimplementation of any vector/projection math.

Run as a script (from the parent directory of the package):
    python -m mypackage.v05_gram_schmidt
"""

from __future__ import annotations
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from linear_algebra.m01_vector_ops import Vector
from linear_algebra.m04_projection import Projection
from linear_algebra.m05_gram_schmidt import GramSchmidt

COLOR_INPUT = "#4C72B0"     # blue   - original input vectors
COLOR_BASIS = "#55A868"     # green  - orthogonal basis
COLOR_PROJ = "#C44E52"      # red    - projection subtracted
COLOR_RESID = "#8172B2"     # purple - residual / orthogonalized result
COLOR_NORM = "#CCB974"      # gold   - orthonormal (unit) vectors
COLOR_GRID = "#DDDDDD"


# Small plotting helpers (2D only -- Gram Schmidt dashboard is 2D by design,
# matching the norms dashboard's flat, at a glance layout)

def _arrow(ax, v: Vector, color, label=None, origin=(0, 0), lw=2.2, alpha=1.0, style="-"):
    ax.annotate(
        "", xy=(origin[0] + v[0], origin[1] + v[1]), xytext=origin,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha,
                         linestyle=style, shrinkA=0, shrinkB=0),
    )
    if label:
        tx, ty = origin[0] + v[0], origin[1] + v[1]
        ax.text(tx * 1.08 + 0.05, ty * 1.08 + 0.05, label, color=color,
                 fontsize=9, fontweight="bold")

def _setup_2d(ax, pad, title, subtitle):
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.axhline(0, color=COLOR_GRID, lw=1, zorder=0)
    ax.axvline(0, color=COLOR_GRID, lw=1, zorder=0)
    ax.set_aspect("equal")
    ax.grid(True, color=COLOR_GRID, lw=0.5)
    ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold")

def _pad_for(*vecs, minimum=1.0, ratio=1.4):
    m = minimum
    for v in vecs:
        for e in v:
            m = max(m, abs(e))
    return m * ratio


# Panel builders

def panel_inputs(ax, vectors):
    pad = _pad_for(*vectors)
    _setup_2d(ax, pad, "Input Vectors", "(what we start with, before orthogonalizing)")
    for i, v in enumerate(vectors):
        _arrow(ax, v, COLOR_INPUT, label=f"v{i+1}")

def panel_projection_removed(ax, vectors, basis_after_v1):
    """Shows v2's projection onto b1 being subtracted -- the core GS operation."""
    v1, v2 = vectors[0], vectors[1]
    b1 = basis_after_v1[0]
    proj = Projection.vector_projection(v2, b1)
    residual = GramSchmidt.orthogonalize_vector(v2, basis_after_v1)

    pad = _pad_for(v1, v2, proj, residual)
    _setup_2d(ax, pad, "Removing the Projection",
              "proj_b1(v2) is subtracted from v2, leaving the orthogonal residual")
    _arrow(ax, b1, COLOR_BASIS, label="b1", alpha=0.6)
    _arrow(ax, v2, COLOR_INPUT, label="v2", alpha=0.85)
    _arrow(ax, proj, COLOR_PROJ, label="proj", style="--")
    _arrow(ax, residual, COLOR_RESID, label="residual", lw=2.6)
    ax.plot([v2[0], residual[0]], [v2[1], residual[1]], linestyle=":", color=COLOR_RESID, lw=1.4)

def panel_orthogonal_basis(ax, vectors, ortho_basis):
    pad = _pad_for(*vectors, *ortho_basis)
    _setup_2d(ax, pad, "Orthogonal Basis",
              "b1, b2, ... span the same space as v1, v2, ... but are mutually perpendicular")
    for i, v in enumerate(vectors):
        _arrow(ax, v, COLOR_INPUT, label=f"v{i+1}", alpha=0.35)
    for i, b in enumerate(ortho_basis):
        _arrow(ax, b, COLOR_BASIS, label=f"b{i+1}", lw=2.6)

def panel_orthonormal_basis(ax, ortho_basis, orthonormal_basis):
    pad = _pad_for(*ortho_basis, minimum=1.3)
    _setup_2d(ax, pad, "Orthonormal Basis",
              "each b_i is rescaled to unit length: e_i = b_i / ||b_i||")
    theta = [t / 200 * 2 * math.pi for t in range(201)]
    ax.plot([math.cos(t) for t in theta], [math.sin(t) for t in theta],
            color=COLOR_NORM, lw=0.8, alpha=0.4, linestyle=":")
    for i, b in enumerate(ortho_basis):
        _arrow(ax, b, COLOR_BASIS, label=f"b{i+1}", alpha=0.4)
    for i, e in enumerate(orthonormal_basis):
        _arrow(ax, e, COLOR_NORM, label=f"e{i+1}", lw=2.6)

def panel_orthogonality_check(ax, ortho_basis):
    """Bar chart of pairwise dot products -- should all be ~0."""
    n = len(ortho_basis)
    pairs, dots = [], []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(f"b{i+1}\u00b7b{j+1}")
            dots.append(ortho_basis[i].dot_product(ortho_basis[j]))
    ax.set_title("Orthogonality Check\n(pairwise dot products, should all be \u2248 0)",
                  fontsize=11, fontweight="bold")
    if not pairs:
        ax.text(0.5, 0.5, "Only one basis vector\n(no pairs to check)",
                 ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        return
    bars = ax.bar(pairs, dots, color=COLOR_BASIS)
    ax.axhline(0, color="black", lw=0.8)
    ymax = max(1e-8, max(abs(d) for d in dots)) * 1.6
    ax.set_ylim(-ymax, ymax)
    for rect, d in zip(bars, dots):
        ax.text(rect.get_x() + rect.get_width() / 2, d, f"{d:.1e}",
                 ha="center", va="bottom" if d >= 0 else "top", fontsize=8)

def panel_norm_check(ax, orthonormal_basis):
    """Bar chart of unit-vector magnitudes -- should all be 1."""
    labels = [f"e{i+1}" for i in range(len(orthonormal_basis))]
    mags = [e.magnitude() for e in orthonormal_basis]
    ax.set_title("Unit Length Check\n(||e_i|| for each orthonormal vector, should all be = 1)",
                  fontsize=11, fontweight="bold")
    bars = ax.bar(labels, mags, color=COLOR_NORM)
    ax.axhline(1.0, color="red", linestyle="--", lw=1.2, label="||e|| = 1")
    ax.set_ylim(0, max(1.3, max(mags) * 1.2))
    ax.legend(fontsize=8)
    for rect, m in zip(bars, mags):
        ax.text(rect.get_x() + rect.get_width() / 2, m, f"{m:.4f}",
                 ha="center", va="bottom", fontsize=8)

# Main

def build_figure(vectors: list[Vector]) -> plt.Figure:
    ortho_basis = GramSchmidt.orthogonal_basis(vectors)
    orthonormal = GramSchmidt.orthonormal_basis(vectors)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Gram-Schmidt Process \u2014 Visualized", fontsize=20, fontweight="bold")

    panel_inputs(axes[0, 0], vectors)

    if len(vectors) >= 2:
        basis_after_v1 = GramSchmidt.orthogonal_basis(vectors[:1])
        panel_projection_removed(axes[0, 1], vectors, basis_after_v1)
    else:
        axes[0, 1].axis("off")
        axes[0, 1].text(0.5, 0.5, "Need \u2265 2 input vectors\nto show a projection step",
                         ha="center", va="center", transform=axes[0, 1].transAxes)

    panel_orthogonal_basis(axes[0, 2], vectors, ortho_basis)
    panel_orthonormal_basis(axes[1, 0], ortho_basis, orthonormal)
    panel_orthogonality_check(axes[1, 1], ortho_basis)
    panel_norm_check(axes[1, 2], orthonormal)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def main():
    # Demo input -- replace with your own Vector list as needed.
    vectors = [Vector([3, 1]), Vector([2, 2])]

    fig = build_figure(vectors)

    out_path = Path(__file__).parent / "figures"
    out_path.mkdir(parents=True, exist_ok=True)
    figure_path = out_path / "v05_gram_schmidt.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to {figure_path}")


if __name__ == "__main__":
    main()