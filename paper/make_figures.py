"""Data figures of the paper (Figure_3 to Figure_6 of the paper, Figure_S1 of the supplement), drawn from the
result tables of the repository.

Run from anywhere:  python paper/make_figures.py
Output: paper/output/. Figures 1 and 2 of the paper are diagrams without data.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
OUTDIR = HERE / "output"
T5 = RES / "tables_test_five_configs"
TCV = RES / "tables_test_five_configs_calibration_cv"
TGT = RES / "tables_calibration_gemma_think"
TMS = RES / "tables_method_study"

ORDER = ["claude-haiku", "claude-haiku-nothink", "gpt-luna-think", "gpt-luna", "gemma", "gemma-think"]
LABEL = {"claude-haiku": "Haiku-R", "claude-haiku-nothink": "Haiku-N", "gpt-luna-think": "Luna-R",
         "gpt-luna": "Luna-N", "gemma": "Gemma-N", "gemma-think": "Gemma-R"}
# Okabe-Ito palette (colour-blind safe); filled marker = configuration with a reasoning step
COL = {"claude-haiku": "#0072B2", "claude-haiku-nothink": "#0072B2", "gpt-luna-think": "#D55E00",
       "gpt-luna": "#D55E00", "gemma": "#009E73", "gemma-think": "#009E73"}
MRK = {"claude-haiku": "o", "claude-haiku-nothink": "o", "gpt-luna-think": "s", "gpt-luna": "s",
       "gemma": "^", "gemma-think": "^"}
FILLED = {"claude-haiku", "gpt-luna-think", "gemma-think"}
BOUND = 5 / 12
PROPOSED = "F_vote_scores|R_myopic|A_none"
LITE = "F_verbal_scores|R_myopic|A_none"
VOTE = "E_vote|R_myopic|A_none"
STATED = "E_verbal|R_myopic|A_none"
SETS = [("household_test", "Household test split"), ("new_domains_pooled", "New domains (pooled)"),
        ("trustnav", "TRUST-NAV")]

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})


def mstyle(m):
    return dict(marker=MRK[m], color=COL[m], mfc=COL[m] if m in FILLED else "white", mec=COL[m], ms=4.5,
                ls="none", mew=1.0)


def legend_handles(models):
    return [plt.Line2D([], [], label=LABEL[m], **mstyle(m)) for m in models]


def save(fig, name):
    OUTDIR.mkdir(exist_ok=True)
    fig.savefig(OUTDIR / f"{name}.pdf")
    fig.savefig(OUTDIR / f"{name}.png", dpi=300)
    plt.close(fig)


def policies():
    pol = pd.read_csv(T5 / "policies.csv")
    return pol[pol.block_rule.fillna("").str.startswith("block") | pol.block_rule.isna()]


# ----------------------------------------------------------------------------------------- Figure 3
# per-configuration cost of the gates (method-study variants) and of the three prompted/agent baselines

ROWS_FIG3 = [("CATWAY", "ms", PROPOSED), ("CATWAY-Lite", "ms", LITE), ("Vote only", "ms", VOTE),
             ("Stated only", "ms", STATED), ("Direct prompt", "pol", "P1_direct_prompt"),
             ("Cost-aware prompt", "pol", "P1c_direct_prompt_with_costs"),
             ("Unguarded agent", "pol", "P8_unguarded_agent")]


def fig3():
    pol = policies()
    ms = pd.read_csv(TMS / "test_by_config.csv")
    models = [m for m in ORDER if m in set(ms.model)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    for ax, (s, title) in zip(axes, SETS):
        for i, (lab, src, key) in enumerate(ROWS_FIG3):
            y = len(ROWS_FIG3) - 1 - i
            for j, m in enumerate(models):
                if src == "ms":
                    v = ms[(ms.variant == key) & (ms.model == m) & (ms.set == s)].cost
                else:
                    v = pol[(pol.policy == key) & (pol.model == m) & (pol.set == s)].cost
                if len(v):
                    ax.plot(v.iloc[0], y + (j - 2) * 0.11, **mstyle(m))
        ax.axhline(2.5, color="grey", lw=0.5, ls=":")
        if s == SETS[-1][0]:
            ax.text(1.18, 2.62, "CATWAY and ablations", ha="right", va="bottom", fontsize=6, color="#555555")
            ax.text(1.18, 2.38, "baselines", ha="right", va="top", fontsize=6, color="#555555")
        ax.set_title(title)
        ax.set_xlim(-0.03, 1.2)
        ax.set_xlabel("decision cost per item")
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    axes[0].set_yticks(range(len(ROWS_FIG3)))
    axes[0].set_yticklabels([r[0] for r in ROWS_FIG3][::-1])
    fig.legend(handles=legend_handles(models), loc="lower center", ncol=len(models), frameon=False,
               bbox_to_anchor=(0.55, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "Figure_4")  # cost per configuration


# ----------------------------------------------------------------------------------------- Figure 4

def fig4():
    fi = pd.concat([pd.read_csv(TCV / "fit_info_calibration_cv.csv"), pd.read_csv(TGT / "fit_info_calibration_cv.csv")])
    fi = fi[fi.source == "p_sample"].drop_duplicates("model")
    ue = pd.read_csv(T5 / "unanimous_errors.csv")
    ue = ue[(ue.set == "household_calibration") & (ue.voted_condition == "all")].set_index("model")
    vb = pd.read_csv(TGT / "vote_by_source_condition_calibration.csv")
    vb = vb[(vb.model == "gemma-think") & (vb["round"] == 1)]
    share = {m: ue.loc[m, "unanimous_share"] for m in ue.index}
    share["gemma-think"] = vb.unanimous_sets.sum() / vb.n.sum()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for _, r in fi.iterrows():
        m = r.model
        lo, hi = [float(x) for x in r.cv_lam_range.strip("[]").split(",")]
        ax.errorbar(share[m], r.lam, yerr=[[r.lam - lo], [hi - r.lam]], capsize=2, elinewidth=0.8, **mstyle(m))
        off = {"gpt-luna": (6, -9), "gemma": (5, -11)}.get(m, (6, 2))
        ax.annotate(LABEL[m], (share[m], r.lam), textcoords="offset points", xytext=off, fontsize=6.5)
    ax.axhspan(BOUND, 1.0, color="#D55E00", alpha=0.08, lw=0)
    ax.axhline(BOUND, color="black", lw=0.7, ls="--")
    ax.text(0.725, 0.85, "vote-only gate asks about\nevery unanimous clear vote", fontsize=5.8, va="top", color="#7A3300")
    ax.text(0.83, BOUND * 0.93, r"$\nu=5/12$", fontsize=5.8, va="top")
    ax.set_yscale("log")
    ax.set_ylim(0.03, 1.0)
    ax.set_xlim(0.72, 1.0)
    ax.set_xlabel("share of unanimous ten-sample votes (calibration split)")
    ax.set_ylabel(r"fitted pseudo-count $\hat\nu$ (fold range)")
    fig.tight_layout()
    save(fig, "Figure_3")  # unanimity against fitted pseudo-count


# ----------------------------------------------------------------------------------------- Figure 5

def fig5():
    lp = pd.read_csv(T5 / "lambda_profile.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    models = [m for m in ORDER if m in set(lp.model)]
    for ax, (s, title) in zip(axes, SETS):
        d = lp[lp.set == s]
        for m in models:
            x = d[d.model == m].sort_values("lambda")
            st = mstyle(m)
            ax.plot(x["lambda"], x.cost_P5, color=st["color"], lw=1.0, ls="-" if m in FILLED else "--")
            fr = x[x.is_frozen]
            if len(fr):
                ax.plot(fr["lambda"].iloc[0], fr.cost_P5.iloc[0], **st)
        ax.axvspan(BOUND, 1.2, color="#D55E00", alpha=0.08, lw=0)
        ax.axvline(BOUND, color="black", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(0.008, 1.2)
        ax.set_title(title)
        ax.set_xlabel(r"pseudo-count $\nu$")
        ax.grid(lw=0.3, alpha=0.5)
    axes[0].set_ylabel("cost of the vote-only gate per item")
    fig.legend(handles=legend_handles(models), loc="lower center", ncol=len(models), frameon=False,
               bbox_to_anchor=(0.55, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "Figure_5")  # pseudo-count profile


# ----------------------------------------------------------------------------------------- Figure 6
# generations per decision against mean cost over configurations, every variant that is defined for all
# configurations; the frontier and the named gates are highlighted

def fig6():
    ef = pd.read_csv(TMS / "efficiency_frontier.csv")
    sel = pd.read_csv(TMS / "selection_on_calibration.csv")
    sel = sel[sel.eligible_all_configs]
    pol = policies()
    panels = [("household_calibration_cv", "Calibration split (cross-validated)"),
              ("household_test", "Household test split"), ("new_domains_pooled", "New domains (pooled)")]
    named = {PROPOSED: ("CATWAY", "#0072B2", "o"), LITE: ("CATWAY-Lite", "#D55E00", "s"),
             VOTE: ("vote only", "#000000", "^"), STATED: ("stated only", "#7F7F7F", "v")}
    base = [("direct prompt", "P1_direct_prompt"), ("cost-aware prompt", "P1c_direct_prompt_with_costs"),
            ("unguarded agent", "P8_unguarded_agent")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, (s, title) in zip(axes, panels):
        d = ef[ef.set == s]
        ax.scatter(d.generations_per_decision, d.mean_cost, s=6, color="#BBBBBB", lw=0, zorder=1)
        # step line: at each budget, the variant with the lowest cross-validated cost among those with at most
        # that many generations (what selection on calibration data would choose), at its cost on this panel
        xs, ys = [], []
        for b in sorted(sel.generations_per_decision.unique()):
            v = sel[sel.generations_per_decision <= b].sort_values("cv_cost_mean").variant.iloc[0]
            r = d[d.variant == v]
            if len(r):
                xs.append(b)
                ys.append(r.mean_cost.iloc[0])
        ax.step(xs, ys, where="post", color="#555555", lw=0.7, zorder=2)
        for v, (lab, col, mk) in named.items():
            r = d[d.variant == v]
            if len(r):
                gx, gy = r.generations_per_decision.iloc[0], r.mean_cost.iloc[0]
                ax.plot(gx, gy, mk, color=col, ms=5, zorder=3)
                off = {"CATWAY": (-5, 6), "CATWAY-Lite": (4, -9), "vote only": (-6, 5), "stated only": (5, 4)}[lab]
                ax.annotate(lab, (gx, gy), textcoords="offset points", xytext=off, fontsize=6, color=col,
                            ha="right" if off[0] < 0 else "left", zorder=4)
        if s != "household_calibration_cv":
            for lab, p in base:
                c = pol[(pol.set == s) & (pol.policy == p)].cost.mean()
                ax.plot(1, c, "x", color="#AA3377", ms=4, zorder=3)
            ax.plot([], [], "x", color="#AA3377", ms=4, label="prompt / agent baselines")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20])
        ax.set_xticklabels(["1", "2", "5", "10", "20"])
        ax.set_xlabel("model generations per decision")
        ax.set_title(title)
        ax.grid(lw=0.3, alpha=0.5)
    axes[0].set_ylabel("mean cost over configurations")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "Figure_6")  # efficiency


# ----------------------------------------------------------------------------------------- Figure 7

def fig7():
    ar = pd.read_csv(TMS / "ablation_reasoning.csv")
    rp = pd.read_csv(TGT / "reasoning_pairs_calibration_cv.csv")
    groups = [("Claude Haiku, household", "claude-haiku-5-5", "household_test"),
              ("Claude Haiku, new domains", "claude-haiku-5-5", "new_domains_pooled"),
              ("GPT-6 Luna, household", "gpt-6-luna", "household_test"),
              ("GPT-6 Luna, new domains", "gpt-6-luna", "new_domains_pooled"),
              ("Gemma 4, calibration (CV)", "gemma", None)]
    gates = [("VOTE-ONLY", "cost:P5_three_way_calibrated", "vote only", "#000000", "^", -0.15),
             ("PROPOSED", "cost:P7_learned", "CATWAY", "#0072B2", "o", 0.15)]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for i, (lab, model, s) in enumerate(groups):
        y = len(groups) - 1 - i
        for g, q, glab, col, mk, dy in gates:
            if s is None:
                r = rp[rp.quantity == q].iloc[0]
            else:
                r = ar[(ar.gate == g) & (ar.model == model) & (ar.set == s)].iloc[0]
            ax.errorbar(r.diff_on_minus_off, y + dy, xerr=[[r.diff_on_minus_off - r.ci_low], [r.ci_high - r.diff_on_minus_off]],
                        fmt=mk, color=col, capsize=2, ms=4, label=glab if i == 0 else None)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[0] for g in groups][::-1])
    ax.set_xlabel("cost with reasoning minus cost without")
    ax.grid(axis="x", lw=0.3, alpha=0.5)
    ax.legend(frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.4, 1.0))
    fig.tight_layout()
    save(fig, "Figure_S1")  # reasoning (supplement)


if __name__ == "__main__":
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    print("written:", sorted(p.name for p in OUTDIR.glob("Figure_*.pdf")))
