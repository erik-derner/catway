"""Print the LaTeX rows of the result tables of the paper (Tables 5 to 8) from the result tables.

Run from anywhere:  python paper/make_tables.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
OUTDIR = HERE / "output"
T5 = RES / "tables_test_six_configs"
TCV = RES / "tables_calibration_six_configs"
TGT = RES / "tables_calibration_six_configs"   # the calibration tables of all six configurations
TMS = RES / "tables_method_study"
CONF = ["claude-haiku", "claude-haiku-nothink", "gpt-luna-think", "gpt-luna", "gemma", "gemma-think"]
LAB = {"claude-haiku": "Haiku-R", "claude-haiku-nothink": "Haiku", "gpt-luna-think": "Luna-R",
       "gpt-luna": "Luna", "gemma": "Gemma", "gemma-think": "Gemma-R"}
SETS = ["household_test", "new_domains_pooled", "trustnav"]
PROPOSED = "F_vote_scores|R_myopic|A_none"
LITE = "F_verbal_scores|R_myopic|A_none"


def f3(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def table5():
    print("% Table 5: calibration of the vote on the calibration split")
    fi = pd.concat([pd.read_csv(TCV / "fit_info_calibration_cv.csv"), pd.read_csv(TGT / "fit_info_calibration_cv.csv")])
    fi = fi[fi.source == "p_sample"].drop_duplicates("model").set_index("model")
    cal = pd.concat([pd.read_csv(TCV / "calibration_calibration_cv.csv"), pd.read_csv(TGT / "calibration_calibration_cv.csv")])
    cal = cal[cal.source == "p_sample"].drop_duplicates(["model", "version"])
    ue = pd.read_csv(T5 / "unanimous_errors.csv")
    ue = ue[(ue.set == "household_calibration") & (ue.voted_condition == "all")].set_index("model")
    vb = pd.read_csv(TGT / "vote_by_source_condition_calibration.csv")
    vb = vb[(vb.model == "gemma-think") & (vb["round"] == 1)]
    ms = pd.read_csv(TMS / "cv_by_config.csv").set_index(["variant", "model"])
    for m in CONF:
        r = fi.loc[m]
        raw = cal[(cal.model == m) & cal.version.str.startswith("raw")].iloc[0]
        cc = cal[(cal.model == m) & cal.version.str.startswith("calibrated")].iloc[0]
        un = ue.loc[m, "unanimous_share"] if m in ue.index else vb.unanimous_sets.sum() / vb.n.sum()
        lo, hi = [float(x) for x in r.cv_lam_range.strip("[]").split(",")]
        print(f"{LAB[m]} & {r.lam:.3f} & {lo:.3f}--{hi:.3f} & {un:.3f} & {raw.ece:.3f} & {cc.ece:.3f} & "
              f"{raw.accuracy_4class:.3f} & {ms.loc[('E_vote|R_myopic|A_none', m), 'cost']:.3f} & "
              f"{ms.loc[(PROPOSED, m), 'cost']:.3f} \\\\")


def summary_rows():
    """mean / worst over the six configurations for gates (method study) and baselines (policies)."""
    ts = pd.read_csv(TMS / "test_summary.csv").set_index(["variant", "set"])
    pol = pd.read_csv(T5 / "policies.csv")
    pol = pol[pol.block_rule.fillna("").str.startswith("block") | pol.block_rule.isna()]
    tb = pd.read_csv(TMS / "test_by_config.csv")
    gates = [("CATWAY (vote + scores)", PROPOSED, 11), ("CATWAY-Lite (stated + scores)", LITE, 2),
             ("Vote only", "E_vote|R_myopic|A_none", 10), ("Stated probabilities only", "E_verbal|R_myopic|A_none", 1)]
    base = [("Conformal set rule", "P4_conformal", 10), ("Top-probability threshold", "P4b_chow_threshold", 10),
            ("Fuzzy rule base", "P6_fuzzy", 1), ("Rule cascade", "P2_cascade", 3),
            ("TRUST-NAV gate", "P3r_trustnav_runner_config", 1), ("Direct prompt", "P1_direct_prompt", 1),
            ("Cost-aware prompt", "P1c_direct_prompt_with_costs", 1), ("Unguarded agent", "P8_unguarded_agent", 1),
            ("Agent with checker", "P9_agent_plus_checker", 1)]
    print("% Table 6: mean / worst cost over six configurations; unsafe executed (count over configurations, household + new)")
    nuns = {"household_test": 96, "new_domains_pooled": 46}
    for lab, v, g in gates:
        cells = []
        for s in SETS:
            r = ts.loc[(v, s)]
            cells.append(f"{r.mean_cost:.3f} & {r.worst_cost:.3f}")
        x = tb[(tb.variant == v) & tb.set.isin(nuns)]
        cnt = sum(round(r.unsafe_executed * nuns[r.set]) for _, r in x.iterrows())
        print(f"{lab} & {g} & " + " & ".join(cells) + f" & {cnt} \\\\")
    for lab, p, g in base:
        cells = []
        for s in SETS:
            c = pol[(pol.set == s) & (pol.policy == p) & pol.model.isin(CONF)].cost
            cells.append(f"{c.mean():.3f} & {c.max():.3f}")
        x = pol[(pol.policy == p) & pol.set.isin(nuns) & pol.model.isin(CONF)]
        cnt = sum(round(r.unsafe_executed * nuns[r.set]) for _, r in x.iterrows())
        print(f"{lab} & {g} & " + " & ".join(cells) + f" & {cnt} \\\\")


def table7():
    print("% Table 7: CATWAY minus baseline effects (baseline cost minus CATWAY cost), [95% CI], Holm marker")
    t = pd.read_csv(TMS / "test_comparisons.csv")
    t = t[t.variant == PROPOSED]
    names = {"P1_direct_prompt": "direct", "P1c_direct_prompt_with_costs": "cost-aware", "P8_unguarded_agent": "agent"}
    for m in CONF:
        for b, bl in names.items():
            cells = []
            for s in SETS:
                r = t[(t.model == m) & (t.set == s) & (t.comparator == b)].iloc[0]
                star = "$^\\ast$" if r.p_holm_selected < 0.05 else ""
                cells.append(f"{r.diff_comparator_minus_variant:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]{star}")
            print(f"{LAB[m] if b == 'P1_direct_prompt' else ''} & {bl} & " + " & ".join(cells) + " \\\\")
    for v in (PROPOSED, LITE):
        x = pd.read_csv(TMS / "test_comparisons.csv")
        x = x[(x.variant == v) & x.comparator.isin(names)]
        for s in SETS:
            y = x[x.set == s]
            print(v, s, "better:", int(((y.p_holm_selected < 0.05) & (y.diff_comparator_minus_variant > 0)).sum()),
                  "worse:", int(((y.p_holm_selected < 0.05) & (y.diff_comparator_minus_variant < 0)).sum()), "of", len(y))


def table8():
    print("% Table 8: ablations; calibration CV mean/worst over six configurations; test mean/worst over six")
    sel = pd.read_csv(TMS / "selection_on_calibration.csv").set_index("variant")
    ts = pd.read_csv(TMS / "test_summary.csv").set_index(["variant", "set"])
    rows = [("Vote", "E_vote|R_myopic|A_none"), ("Stated probabilities", "E_verbal|R_myopic|A_none"),
            ("Token probabilities$^\\dagger$", "E_logit|R_myopic|A_none"),
            ("Average of calibrated sources", "E_average|R_myopic|A_none"),
            ("Source selected per configuration", "E_select|R_myopic|A_none"),
            ("Vote + scores (CATWAY)", PROPOSED), ("Stated + scores (CATWAY-Lite)", LITE),
            ("Stated + scores + yes/no", "F_verbal_scores_yesno|R_myopic|A_none"),
            ("All probability probes", "F_all|R_myopic|A_none"),
            ("All probes + action answers", "F_all_actions|R_myopic|A_none"),
            ("sequential (estimated ask row)", "F_vote_scores|R_sequential|A_none"),
            ("conformal set", "F_vote_scores|R_conformal|A_none"),
            ("risk cap on $p_{\\mathrm{un}}$", "F_vote_scores|R_riskcap|A_none"),
            ("prior re-estimation (EM)", "F_vote_scores|R_myopic|A_em"),
            ("agreement temperature", "F_vote_scores|R_myopic|A_agreement")]
    for lab, v in rows:
        r = sel.loc[v]
        cells = [f"{r.cv_cost_mean:.3f} & {r.cv_cost_worst:.3f}"]
        for s in SETS:
            q = ts.loc[(v, s)]
            cells.append(f"{q.mean_cost:.3f} & {q.worst_cost:.3f}")
        print(f"{lab} & {int(r.generations_per_decision)} & " + " & ".join(cells) + " \\\\")
    ab = pd.read_csv(TMS / "ablations.csv")
    ab = ab[(ab.gate == "PROPOSED") & (ab.ablation == "remove input group")]
    for st in ("without scores", "without vote"):
        x = ab[ab.setting == st]
        cells = []
        for s in ["household_calibration_cv"] + SETS:
            c = x[x.set == s].cost
            cells.append(f"{c.mean():.3f} & {c.max():.3f}")
        print(f"{st} & -- & " + " & ".join(cells) + " \\\\")


BASE3 = ["P1_direct_prompt", "P1c_direct_prompt_with_costs", "P8_unguarded_agent"]


def direction_counts(variant, s):
    """significantly cheaper, cheaper (n.s.), equal, costlier (n.s.), significantly costlier"""
    t = pd.read_csv(TMS / "test_comparisons.csv")
    x = t[(t.variant == variant) & (t.set == s) & t.comparator.isin(BASE3)]
    dd = x.diff_comparator_minus_variant.round(6)
    sig = x.p_holm_selected < 0.05
    return [int((sig & (dd > 0)).sum()), int((~sig & (dd > 0)).sum()), int((dd == 0).sum()),
            int((~sig & (dd < 0)).sum()), int((sig & (dd < 0)).sum())]


def table_directions():
    print("% Table: direction and significance of the baseline comparisons (main text)")
    names = {"household_test": "Household test", "new_domains_pooled": "New domains", "trustnav": "TRUST-NAV"}
    for v, lab in ((PROPOSED, "\\name{}"), (LITE, "\\lite{}")):
        pass
    tot = {PROPOSED: [0] * 5, LITE: [0] * 5}
    for s_, nm in names.items():
        for k, (v, lab) in enumerate(((PROPOSED, r"\name{}"), (LITE, r"\lite{}"))):
            c = direction_counts(v, s_)
            tot[v] = [a + b for a, b in zip(tot[v], c)]
            print(f"{nm if k == 0 else ''} & {lab} & " + " & ".join(map(str, c)) + " \\\\")
    print("\\midrule")
    for k, (v, lab) in enumerate(((PROPOSED, r"\name{}"), (LITE, r"\lite{}"))):
        print(f"{'All three sets' if k == 0 else ''} & {lab} & " + " & ".join(map(str, tot[v])) + " \\\\")


if __name__ == "__main__":
    table5()
    summary_rows()
    table7()
    table8()
    table_directions()
