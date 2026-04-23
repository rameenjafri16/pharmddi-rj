"""
generate_results_figures.py
Run from pharmddi-rj repo root: python scripts/generate_results_figures.py
Generates 8 figures to figures/
"""

import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

matplotlib.rcParams["figure.dpi"] = 100

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

TANIMOTO_COLOR = "#5B8DB8"
PATHWAY_COLOR  = "#C0392B"
CORRECT_COLOR  = "#27AE60"
WRONG_COLOR    = "#E74C3C"
AMBIG_COLOR    = "#F39C12"
UNSTATED_COLOR = "#BDC3C7"
BG_COLOR       = "#FAFAFA"
GRID_COLOR     = "#E8E8E8"

CONDITION_COLORS = {
    "Tanimoto baseline":       "#5B8DB8",
    "All fixes (RJ)":          "#C0392B",
    "No Fix 1 (no PK/PD)":    "#8E44AD",
    "No Fix 2 (no prodrug)":   "#16A085",
    "No Fix 4 (no severity)":  "#E67E22",
    "No Fix 5 (no pathway)":   "#2C3E50",
}

COND_NAMES = [
    "Tanimoto baseline",
    "All fixes (RJ)",
    "No Fix 1 (no PK/PD)",
    "No Fix 2 (no prodrug)",
    "No Fix 4 (no severity)",
    "No Fix 5 (no pathway)",
]

COND_SHORT = {
    "Tanimoto baseline":       "Tanimoto\nbaseline",
    "All fixes (RJ)":          "All fixes\n(RJ)",
    "No Fix 1 (no PK/PD)":    "No Fix 1\n(no PK/PD)",
    "No Fix 2 (no prodrug)":   "No Fix 2\n(no prodrug)",
    "No Fix 4 (no severity)":  "No Fix 4\n(no severity)",
    "No Fix 5 (no pathway)":   "No Fix 5\n(no pathway)",
}


def parse_direction_report(path):
    with open(path) as f:
        text = f.read()

    def extract_metrics(block):
        m = {}
        patterns = [
            ("correct",           r"Overall:.*?Correct:\s+[\d,]+\s+\(([\d.]+)%\)"),
            ("wrong",             r"Overall:.*?Wrong:\s+[\d,]+\s+\(([\d.]+)%\)"),
            ("ambiguous",         r"Overall:.*?Ambiguous:\s+[\d,]+\s+\(([\d.]+)%\)"),
            ("not_stated",        r"Overall:.*?Not stated:\s+[\d,]+\s+\(([\d.]+)%\)"),
            ("prodrug_correct",   r"Prodrug pairs.*?Correct:\s+[\d,]+\s+\(([\d.]+)%\)"),
            ("prodrug_wrong",     r"Prodrug pairs.*?Wrong:\s+[\d,]+\s+\(([\d.]+)%\)"),
            ("prodrug_ambiguous", r"Prodrug pairs.*?Ambiguous:\s+[\d,]+\s+\(([\d.]+)%\)"),
            ("metabolism_correct",r"metabolism\s+n=\d+\s+correct=\d+\s+\((\d+)%\)"),
        ]
        for key, pat in patterns:
            hit = re.search(pat, block, re.DOTALL)
            if hit:
                m[key] = float(hit.group(1))
        return m

    parts = text.split("PATHWAY + RJ prompts")
    tan_block  = parts[0] if len(parts) > 0 else ""
    path_block = parts[1] if len(parts) > 1 else ""

    result = {
        "tanimoto": extract_metrics(tan_block),
        "pathway":  extract_metrics(path_block),
    }
    hit = re.search(r"Pathway only:\s+[\d,]+\s+\(([\d.]+)%\)", text)
    if hit: result["pathway_wins"] = float(hit.group(1))
    hit = re.search(r"Tanimoto only:\s+[\d,]+\s+\(([\d.]+)%\)", text)
    if hit: result["tanimoto_wins"] = float(hit.group(1))
    return result


FILE_MAP = {
    "All fixes (RJ)":         "pilot_all_fixes",
    "No Fix 1 (no PK/PD)":   "ablation_no_fix1",
    "No Fix 2 (no prodrug)":  "ablation_no_fix2",
    "No Fix 4 (no severity)": "ablation_no_fix4",
    "No Fix 5 (no pathway)":  "ablation_no_fix5",
}

raw = {}
for cname, fname in FILE_MAP.items():
    p = RESULTS_DIR / f"{fname}_direction.txt"
    if p.exists():
        raw[cname] = parse_direction_report(p)
    else:
        print(f"  Missing: {p}")

if "All fixes (RJ)" in raw:
    raw["Tanimoto baseline"] = {"pathway": raw["All fixes (RJ)"]["tanimoto"],
                                 "tanimoto": raw["All fixes (RJ)"]["tanimoto"]}

print(f"Loaded {len(raw)} conditions")


def get(cname, key, condition="pathway"):
    return raw.get(cname, {}).get(condition, {}).get(key, 0)


def style(ax):
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)


# ── Fig 1: Ablation bar chart ─────────────────────────────────────────────────

def fig_ablation_direction():
    metrics = [
        ("Overall correct",    "correct"),
        ("Metabolism correct", "metabolism_correct"),
        ("Prodrug correct",    "prodrug_correct"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=BG_COLOR)
    fig.suptitle("Direction Accuracy by Ablation Condition",
                 fontsize=15, fontweight="bold", y=1.01)

    for ax, (label, key) in zip(axes, metrics):
        ax.set_facecolor(BG_COLOR)
        vals   = [get(c, key) for c in COND_NAMES]
        colors = [CONDITION_COLORS[c] for c in COND_NAMES]
        x      = np.arange(len(COND_NAMES))
        bars   = ax.bar(x, vals, color=colors, width=0.65,
                        edgecolor="white", linewidth=1.5, zorder=3)
        best = int(np.argmax(vals))
        bars[best].set_edgecolor("#1A252F")
        bars[best].set_linewidth(2.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.8,
                        f"{val:.1f}%", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold", color="#1A252F")
        ax.set_title(label, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([COND_SHORT[c] for c in COND_NAMES],
                           fontsize=8, rotation=15, ha="right")
        ax.set_ylabel("Direction Accuracy (%)", fontsize=10)
        ax.set_ylim(0, max(vals) * 1.18 + 3)
        style(ax)

    plt.tight_layout()
    out = FIGURES_DIR / "fig_ablation_direction_accuracy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"Saved: {out}")


# ── Fig 2: Stacked breakdown ──────────────────────────────────────────────────

def fig_direction_breakdown():
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    corr, wrong, ambig, unstated, valid = [], [], [], [], []
    pilot_tan = raw.get("All fixes (RJ)", {}).get("tanimoto", {})

    for c in COND_NAMES:
        if c not in raw:
            continue
        if c == "Tanimoto baseline":
            corr.append(pilot_tan.get("correct", 0))
            wrong.append(pilot_tan.get("wrong", 0))
            ambig.append(pilot_tan.get("ambiguous", 0))
            unstated.append(pilot_tan.get("not_stated", 0))
        else:
            corr.append(get(c, "correct", "pathway"))
            wrong.append(get(c, "wrong", "pathway"))
            ambig.append(get(c, "ambiguous", "pathway"))
            unstated.append(get(c, "not_stated", "pathway"))
        valid.append(c)

    x = np.arange(len(valid))
    w = 0.65
    ax.bar(x, corr,     w, label="Correct",    color=CORRECT_COLOR,   edgecolor="white", zorder=3)
    ax.bar(x, wrong,    w, bottom=corr,          label="Wrong",          color=WRONG_COLOR,    edgecolor="white", zorder=3)
    b3 = [c+ww for c,ww in zip(corr,wrong)]
    ax.bar(x, ambig,    w, bottom=b3,             label="Ambiguous",      color=AMBIG_COLOR,    edgecolor="white", zorder=3)
    b4 = [c+ww+a for c,ww,a in zip(corr,wrong,ambig)]
    ax.bar(x, unstated, w, bottom=b4,             label="Not stated",     color=UNSTATED_COLOR, edgecolor="white", zorder=3)

    for i, val in enumerate(corr):
        if val > 5:
            ax.text(x[i], val/2, f"{val:.1f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")

    ax.set_title("Direction Accuracy Breakdown by Condition", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_SHORT[c] for c in valid], fontsize=9)
    ax.set_ylabel("Percentage of Traces (%)", fontsize=11)
    ax.set_ylim(0, 105)
    style(ax)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)

    plt.tight_layout()
    out = FIGURES_DIR / "fig_ablation_breakdown.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"Saved: {out}")


# ── Fig 3: Pilot grouped bar ──────────────────────────────────────────────────

def fig_pilot_comparison():
    if "All fixes (RJ)" not in raw: return
    categories = ["Overall\nCorrect", "Metabolism\nCorrect", "Prodrug\nCorrect"]
    tan_vals   = [get("All fixes (RJ)", k, "tanimoto")
                  for k in ["correct","metabolism_correct","prodrug_correct"]]
    path_vals  = [get("All fixes (RJ)", k, "pathway")
                  for k in ["correct","metabolism_correct","prodrug_correct"]]

    x = np.arange(len(categories))
    w = 0.32
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    bars1 = ax.bar(x - w/2, tan_vals,  w, label="Tanimoto + original prompts",
                   color=TANIMOTO_COLOR, edgecolor="white", linewidth=1.5, zorder=3)
    bars2 = ax.bar(x + w/2, path_vals, w, label="Pathway + RJ prompts",
                   color=PATHWAY_COLOR,  edgecolor="white", linewidth=1.5, zorder=3)

    for bar, val in zip(bars1, tan_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9.5, color="#2C3E50")
    for bar, val in zip(bars2, path_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9.5, color="#2C3E50")

    for i, (t, p) in enumerate(zip(tan_vals, path_vals)):
        delta = p - t
        color = "#1E8449" if delta >= 0 else "#C0392B"
        sign  = "+" if delta >= 0 else ""
        ypos  = max(t, p) + 3.5
        ax.text(x[i], ypos, f"{sign}{delta:.1f}pp",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=color)
        ax.annotate("", xy=(x[i] + w/2, max(t,p) + 1.5),
                    xytext=(x[i] - w/2, max(t,p) + 1.5),
                    arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=1))

    ax.set_title("Pilot Experiment: Direction Accuracy\nTanimoto Baseline vs RJ Configuration",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel("Direction Accuracy (%)", fontsize=11)
    ax.set_ylim(0, max(max(tan_vals), max(path_vals)) * 1.3 + 5)
    style(ax)
    ax.legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()
    out = FIGURES_DIR / "fig_pilot_direction_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"Saved: {out}")


# ── Fig 4: Prodrug across conditions ─────────────────────────────────────────

def fig_prodrug_across_conditions():
    tan_vals, path_vals, valid = [], [], []
    for c in COND_NAMES:
        if c not in raw: continue
        tan_vals.append(get(c, "prodrug_correct", "tanimoto"))
        path_vals.append(get(c, "prodrug_correct", "pathway"))
        valid.append(c)

    x = np.arange(len(valid))
    w = 0.35
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    ax.bar(x - w/2, tan_vals,  w, label="Tanimoto condition",
           color=TANIMOTO_COLOR, alpha=0.88, edgecolor="white", zorder=3)
    ax.bar(x + w/2, path_vals, w, label="Pathway + RJ condition",
           color=PATHWAY_COLOR,  alpha=0.88, edgecolor="white", zorder=3)
    ax.axhline(y=51.8, color=TANIMOTO_COLOR, linestyle="--",
               linewidth=1.3, alpha=0.5, label="Original baseline (51.8%)")

    for i, (t, p) in enumerate(zip(tan_vals, path_vals)):
        if t > 1:
            ax.text(x[i]-w/2, t+0.4, f"{t:.1f}%",
                    ha="center", va="bottom", fontsize=8.5, color="#2C3E50")
        if p > 1:
            ax.text(x[i]+w/2, p+0.4, f"{p:.1f}%",
                    ha="center", va="bottom", fontsize=8.5, color="#2C3E50")

    ax.set_title("Prodrug Direction Accuracy Across All Conditions",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_SHORT[c] for c in valid], fontsize=9)
    ax.set_ylabel("Prodrug Direction Accuracy (%)", fontsize=11)
    ax.set_ylim(48, 72)
    style(ax)
    ax.legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()
    out = FIGURES_DIR / "fig_prodrug_across_conditions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"Saved: {out}")


# ── Fig 5: Delta heatmap ──────────────────────────────────────────────────────

def fig_delta_heatmap():
    metrics = ["correct","metabolism_correct","prodrug_correct",
               "prodrug_ambiguous","wrong"]
    metric_labels = ["Overall\ncorrect","Metabolism\ncorrect","Prodrug\ncorrect",
                     "Prodrug\nambiguous","Overall\nwrong"]

    baseline = {m: get("Tanimoto baseline", m, "tanimoto") for m in metrics}
    conds    = [c for c in COND_NAMES if c != "Tanimoto baseline" and c in raw]
    matrix   = np.zeros((len(conds), len(metrics)))

    for i, c in enumerate(conds):
        for j, m in enumerate(metrics):
            delta = get(c, m) - baseline[m]
            matrix[i, j] = -delta if m in ("prodrug_ambiguous","wrong") else delta

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-8, vmax=8)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([COND_SHORT[c].replace("\n"," ") for c in conds], fontsize=10)

    for i in range(len(conds)):
        for j, m in enumerate(metrics):
            raw_delta = get(conds[i], m) - baseline[m]
            if m in ("prodrug_ambiguous","wrong"):
                raw_delta = -raw_delta
            sign  = "+" if raw_delta >= 0 else ""
            color = "white" if abs(matrix[i,j]) > 5 else "#1A252F"
            ax.text(j, i, f"{sign}{raw_delta:.1f}pp",
                    ha="center", va="center", fontsize=9, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, label="Delta from baseline (pp)\nGreen = better")
    ax.set_title("Improvement Over Tanimoto Baseline by Condition and Metric",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Metric (for ambiguous/wrong: negative delta = improvement)", fontsize=9)

    plt.tight_layout()
    out = FIGURES_DIR / "fig_delta_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"Saved: {out}")


# ── Fig 6: Radar chart ────────────────────────────────────────────────────────

def fig_radar():
    metrics       = ["correct","metabolism_correct","prodrug_correct"]
    metric_labels = ["Overall\ncorrect","Metabolism\ncorrect","Prodrug\ncorrect"]
    N      = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    for c in COND_NAMES:
        if c not in raw: continue
        vals  = [get(c, m) for m in metrics] + [get(c, metrics[0])]
        color = CONDITION_COLORS[c]
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=c, alpha=0.85)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 75)
    ax.set_yticks([20, 40, 60])
    ax.set_yticklabels(["20%","40%","60%"], fontsize=8, color="#888888")
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.set_title("Direction Accuracy Profile by Condition",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out = FIGURES_DIR / "fig_radar_accuracy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"Saved: {out}")


# ── Fig 7: Scatter overall vs prodrug ─────────────────────────────────────────

def fig_scatter():
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    for c in COND_NAMES:
        if c not in raw: continue
        cond  = "tanimoto" if c == "Tanimoto baseline" else "pathway"
        xv    = get(c, "correct", cond)
        yv    = get(c, "prodrug_correct", cond)
        color = CONDITION_COLORS[c]
        ax.scatter(xv, yv, s=200, color=color, zorder=5,
                   edgecolors="white", linewidth=2)
        ax.annotate(COND_SHORT[c].replace("\n", " "),
                    xy=(xv, yv), xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=8.5, color=color, fontweight="bold")

    bx = get("Tanimoto baseline", "correct", "tanimoto")
    by = get("Tanimoto baseline", "prodrug_correct", "tanimoto")
    ax.axhline(y=by, color=TANIMOTO_COLOR, linestyle="--", linewidth=1, alpha=0.4)
    ax.axvline(x=bx, color=TANIMOTO_COLOR, linestyle="--", linewidth=1, alpha=0.4)
    ax.text(64.8, by + 0.3, "Better on\nboth", fontsize=8,
            color="#27AE60", ha="right", va="bottom", style="italic")

    ax.set_xlabel("Overall Direction Accuracy (%)", fontsize=11)
    ax.set_ylabel("Prodrug Direction Accuracy (%)", fontsize=11)
    ax.set_title("Overall vs Prodrug Direction Accuracy by Condition",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(59.5, 65.0)
    ax.set_ylim(54.5, 64.5)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_visible(False)

    out = FIGURES_DIR / "fig_scatter_overall_vs_prodrug.png"
    fig.savefig(out, dpi=100, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Fig 8: Fix contribution horizontal bar ────────────────────────────────────

def fig_waterfall():
    allfixes_val = get("All fixes (RJ)", "correct", "pathway")
    baseline_val = get("Tanimoto baseline", "correct", "pathway")

    fix_labels = [
        "Fix 5: No-shared-pathway note",
        "Fix 4: Severity classifier",
        "Fix 2: Prodrug warning",
        "Fix 1: PK/PD flag",
    ]
    fix_keys = [
        "No Fix 5 (no pathway)",
        "No Fix 4 (no severity)",
        "No Fix 2 (no prodrug)",
        "No Fix 1 (no PK/PD)",
    ]
    deltas = [allfixes_val - get(k, "correct", "pathway") for k in fix_keys]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    colors = ["#27AE60" if d >= 0 else "#E74C3C" for d in deltas]
    y = np.arange(len(fix_labels))

    bars = ax.barh(y, deltas, 0.55, color=colors,
                   edgecolor="white", linewidth=1.5, zorder=3, alpha=0.88)

    for bar, delta in zip(bars, deltas):
        sign  = "+" if delta >= 0 else ""
        xpos  = delta + 0.03 if delta >= 0 else delta - 0.03
        ha    = "left" if delta >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f"{sign}{delta:.2f}pp",
                ha=ha, va="center", fontsize=11,
                fontweight="bold", color="#1A252F")

    ax.axvline(x=0, color="#888888", linewidth=1.2, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(fix_labels, fontsize=11)
    ax.set_xlabel("Marginal Contribution to Direction Accuracy (pp)", fontsize=11)
    ax.set_title(
        f"Marginal Contribution of Each Fix\n"
        f"Baseline: {baseline_val:.1f}%  |  All fixes (RJ): {allfixes_val:.1f}%  |  "
        f"Net gain: +{allfixes_val - baseline_val:.1f}pp",
        fontsize=12, fontweight="bold", pad=12
    )

    xpad = max(abs(d) for d in deltas) * 0.35
    ax.set_xlim(min(deltas) - xpad, max(deltas) + xpad)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)

    pos_patch = mpatches.Patch(color="#27AE60", label="Helps (positive contribution)")
    neg_patch = mpatches.Patch(color="#E74C3C", label="Hurts (negative contribution)")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=10,
              framealpha=0.9, loc="upper right")

    plt.tight_layout()
    out = FIGURES_DIR / "fig_fix_contribution_waterfall.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Generating 8 figures...")
    fig_ablation_direction()
    fig_direction_breakdown()
    fig_pilot_comparison()
    fig_prodrug_across_conditions()
    fig_delta_heatmap()
    fig_radar()
    fig_scatter()
    fig_waterfall()
    print(f"\nDone. Figures saved to {FIGURES_DIR}/")
