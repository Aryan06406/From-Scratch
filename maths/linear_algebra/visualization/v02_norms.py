"""
v02_norms.py

Visual companion to m02_norms.py.

Takes two 2D vectors (A is the main subject, B is there so the norms can
be compared against a second vector too — triangle inequality, "which
vector is bigger", etc.) and a custom p, then renders every norm and a
few norm-vs-norm comparisons as labelled matplotlib panels.

Output: vector_norms_visualization.png saved next to this script, plus
an interactive window if your environment supports one.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from linear_algebra.m01_vector_ops import Vector
from linear_algebra.m02_norms import Norms

# Input handling

def read_vector(prompt, default):
    raw = input(f"{prompt} [default {default}]: ").strip()
    if not raw:
        return list(default)
    return [float(x) for x in raw.replace(",", " ").split()]

def get_inputs():
    print("=== Vector Norms Visualizer ===")
    a = read_vector("Vector A (2D)", (3, 4))
    b = read_vector("Vector B (2D, used for comparisons)", (1, -2))
    p_raw = input("\nCustom p for the p-norm [default 3]: ").strip()
    p = float(p_raw) if p_raw else 3.0
    return Vector(a), Vector(b), p

# Shared helpers

def ball_radius(theta, p):
    """Radius of the unit p-norm ball at angle theta: solves ||r*(cosθ,sinθ)||_p = 1.

    Same equation Norms.p_norm / Norms.infinity_norm implement — written
    directly here (instead of instantiating a Vector per angle) purely so
    the whole boundary curve can be swept with numpy in one shot.
    """
    c, s = np.abs(np.cos(theta)), np.abs(np.sin(theta))
    if p == np.inf:
        denom = np.maximum(c, s)
    else:
        denom = (c ** p + s ** p) ** (1 / p)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return 1.0 / denom

def setup_axes(ax, points, title=""):
    xs = [p[0] for p in points] + [0.0]
    ys = [p[1] for p in points] + [0.0]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span = max(x_max - x_min, y_max - y_min, 2.0)
    pad = 0.3 * span
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.axhline(0, color="gray", lw=0.8, zorder=0)
    ax.axvline(0, color="gray", lw=0.8, zorder=0)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(title, fontsize=11, fontweight="bold")

def draw_arrow(ax, vec, origin=(0, 0), color="tab:blue", label=None, lw=2.2, z=3):
    ax.annotate(
        "", xy=(origin[0] + vec[0], origin[1] + vec[1]), xytext=origin,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0),
        zorder=z,
    )
    if label:
        tip = (origin[0] + vec[0], origin[1] + vec[1])
        ax.annotate(label, tip, textcoords="offset points", xytext=(6, 6),
                    color=color, fontsize=10, fontweight="bold")

# Panels

def panel_norm_bar_comparison(ax, a, p):
    norms = Norms(a)
    labels = ["L1", "L2", f"L{p:g}", "L∞"]
    values = [norms.l1_norm(), norms.l2_norm(), norms.p_norm(p), norms.infinity_norm()]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    ax.set_title(f"Norm Comparison for A = {a.elements}\n(same vector, different notions of \"size\")",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.25)

def panel_unit_ball_family(ax, a):
    theta = np.linspace(0, 2 * np.pi, 400)
    p_values = [1, 1.5, 2, 3, 5, 10, np.inf]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(p_values)))
    for p, color in zip(p_values, colors):
        r = ball_radius(theta, p)
        x, y = r * np.cos(theta), r * np.sin(theta)
        label = "L∞ (square)" if p == np.inf else f"p={p:g}"
        ax.plot(x, y, color=color, lw=1.8, label=label)

    # Ray through A's direction, with markers showing where each ball's
    # boundary crosses it — i.e. where A/||A||_p would land for that norm.
    ang = math.atan2(a.elements[1], a.elements[0])
    ray_len = 1.6
    ax.plot([0, ray_len * math.cos(ang)], [0, ray_len * math.sin(ang)],
            color="gray", lw=1, linestyle=":", zorder=1)
    for p, color in zip(p_values, colors):
        r = ball_radius(np.array([ang]), p)[0]
        ax.plot(r * math.cos(ang), r * math.sin(ang), "o", color=color, markersize=5, zorder=4)

    setup_axes(ax, [(1.6, 1.6), (-1.6, -1.6)],
               title="Unit Ball Family Across Norms\n(dots: where A/‖A‖ₚ lands, along A's direction, for each p)")
    ax.legend(fontsize=7, loc="upper right", ncol=1)

def panel_p_curve(ax, a):
    norms = Norms(a)
    p_range = np.linspace(0.5, 15, 200)
    values = [norms.p_norm(p) for p in p_range]
    inf_val = norms.infinity_norm()

    ax.plot(p_range, values, color="tab:blue", lw=2)
    ax.axhline(inf_val, color="tab:red", linestyle="--", lw=1.5, label=f"L∞ = {inf_val:.2f}")
    for p_mark, color in [(1, "tab:purple"), (2, "tab:orange"), (3, "tab:green")]:
        ax.plot(p_mark, norms.p_norm(p_mark), "o", color=color, markersize=7)
        ax.annotate(f"p={p_mark}\n{norms.p_norm(p_mark):.2f}", (p_mark, norms.p_norm(p_mark)),
                    textcoords="offset points", xytext=(8, 8), fontsize=8, color=color)
    ax.set_xlabel("p")
    ax.set_ylabel("‖A‖ₚ")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title("p-norm as a Function of p\n(monotonically decreasing, approaches L∞)",
                 fontsize=11, fontweight="bold")

def panel_component_breakdown(ax, a):
    norms = Norms(a)
    abs_vals = [abs(x) for x in a.elements]
    idx = np.arange(len(a))
    ax.bar(idx, abs_vals, color="tab:blue", width=0.5, label="|xi| (raw components)")
    ax.axhline(norms.l1_norm(), color="tab:orange", linestyle="--", lw=1.5,
               label=f"L1 = Σ|xi| = {norms.l1_norm():.2f}")
    ax.axhline(norms.l2_norm(), color="tab:green", linestyle="--", lw=1.5,
               label=f"L2 = √Σxi² = {norms.l2_norm():.2f}")
    ax.axhline(norms.infinity_norm(), color="tab:red", linestyle="--", lw=1.5,
               label=f"L∞ = max|xi| = {norms.infinity_norm():.2f}")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"x{i+1}" for i in idx])
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Component Contributions vs Aggregated Norms\n"
                 "(L1 adds them all up, L2 grows slower, L∞ only cares about the biggest)",
                 fontsize=11, fontweight="bold")

def panel_triangle_inequality(ax, a, b):
    s = a + b
    norms_a, norms_b, norms_s = Norms(a), Norms(b), Norms(s)
    categories = ["L1", "L2", "L∞"]
    a_vals = [norms_a.l1_norm(), norms_a.l2_norm(), norms_a.infinity_norm()]
    b_vals = [norms_b.l1_norm(), norms_b.l2_norm(), norms_b.infinity_norm()]
    s_vals = [norms_s.l1_norm(), norms_s.l2_norm(), norms_s.infinity_norm()]
    sum_vals = [x + y for x, y in zip(a_vals, b_vals)]

    idx = np.arange(len(categories))
    width = 0.25
    ax.bar(idx - width, a_vals, width, label="‖A‖", color="tab:blue")
    ax.bar(idx, b_vals, width, label="‖B‖", color="tab:orange")
    ax.bar(idx + width, s_vals, width, label="‖A+B‖", color="tab:green")

    for i, (sv, cat_sum) in enumerate(zip(s_vals, sum_vals)):
        ax.plot([i - 1.5 * width, i + 1.5 * width], [cat_sum, cat_sum],
                color="black", linestyle="--", lw=1.3,
                label="‖A‖+‖B‖ (triangle-inequality ceiling)" if i == 0 else None)

    ax.set_xticks(idx)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Triangle Inequality: ‖A+B‖ ≤ ‖A‖ + ‖B‖\n(green bar should never cross the dashed line)",
                 fontsize=11, fontweight="bold")

def panel_ab_comparison(ax, a, b, p):
    norms_a, norms_b = Norms(a), Norms(b)
    categories = ["L1", "L2", f"L{p:g}", "L∞"]
    a_vals = [norms_a.l1_norm(), norms_a.l2_norm(), norms_a.p_norm(p), norms_a.infinity_norm()]
    b_vals = [norms_b.l1_norm(), norms_b.l2_norm(), norms_b.p_norm(p), norms_b.infinity_norm()]

    idx = np.arange(len(categories))
    width = 0.35
    ax.bar(idx - width / 2, a_vals, width, label=f"A = {a.elements}", color="tab:blue")
    ax.bar(idx + width / 2, b_vals, width, label=f"B = {b.elements}", color="tab:orange")
    ax.set_xticks(idx)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=8)
    ax.set_title("A vs B: Which Vector Is \"Bigger\"?\n(the answer can flip depending on which norm you pick)",
                 fontsize=11, fontweight="bold")

# Main

def main():
    a, b, p = get_inputs()

    fig = plt.figure(figsize=(19, 12))
    fig.suptitle("Vector Norms — Visualized", fontsize=18, fontweight="bold")
    grid = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.35)

    panel_norm_bar_comparison(fig.add_subplot(grid[0, 0]), a, p)
    panel_unit_ball_family(fig.add_subplot(grid[0, 1]), a)
    panel_p_curve(fig.add_subplot(grid[0, 2]), a)

    panel_component_breakdown(fig.add_subplot(grid[1, 0]), a)
    panel_triangle_inequality(fig.add_subplot(grid[1, 1]), a, b)
    panel_ab_comparison(fig.add_subplot(grid[1, 2]), a, b, p)

    out_path = Path(__file__).parent / "figures"
    out_path.mkdir(parents=True, exist_ok=True) 
    figure_path = out_path / "v02_norms.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to {figure_path}")
    
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()