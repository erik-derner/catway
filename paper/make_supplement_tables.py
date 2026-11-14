"""Write the result tables of Supplementary S3 (paper/output/supp_tables.tex) from the result tables.

Run from anywhere:  python paper/make_supplement_tables.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "output" / "supp_tables.tex"
RES = HERE.parent / "results"
OUTDIR = HERE / "output"
T5 = RES / "tables_test_five_configs"
TCV = RES / "tables_test_five_configs_calibration_cv"
TGT = RES / "tables_calibration_gemma_think"
TMS = RES / "tables_method_study"
CONF = ["claude-haiku", "claude-haiku-nothink", "gpt-luna-think", "gpt-luna", "gemma"]
CONF6 = CONF + ["gemma-think"]
LAB = {"claude-haiku": "Haiku-R", "claude-haiku-nothink": "Haiku-N", "gpt-luna-think": "Luna-R",
       "gpt-luna": "Luna-N", "gemma": "Gemma-N", "gemma-think": "Gemma-R"}
SETNAME = {"household_test": "Household test", "new_domains_pooled": "New domains (pooled)",
           "production_cell": "Production cell", "mobile_robot": "Mobile robot", "trustnav": "TRUST-NAV",
           "trustnav_complete": "TRUST-NAV + completeness sentence", "household_calibration_cv": "Calibration (CV)"}
EST = {"E_vote": "vote", "E_verbal": "stated", "E_logit": "token", "E_average": "average", "E_select": "selected source",
       "F_vote_scores": "vote+scores", "F_verbal_scores": "stated+scores", "F_verbal_scores_yesno": "stated+scores+yes/no",
       "F_all": "all probes", "F_all_actions": "all probes+actions"}
RULE = {"R_myopic": "min. loss", "R_sequential": "sequential", "R_conformal": "conformal", "R_riskcap": "risk cap"}
ADAPT = {"A_none": "none", "A_em": "EM priors", "A_agreement": "agreement T"}
POL = {"P1_direct_prompt": "Direct prompt", "P1c_direct_prompt_with_costs": "Cost-aware prompt",
       "P8_unguarded_agent": "Unguarded agent", "P9_agent_plus_checker": "Agent with checker",
       "P2_cascade": "Rule cascade", "P3r_trustnav_runner_config": "TRUST-NAV gate (authors' experiment configuration)",
       "P3_trustnav_thresholds": "TRUST-NAV gate (released decision-module defaults)",
       "P4_conformal": "Conformal set rule", "P4b_chow_threshold": "Top-probability threshold",
       "P6_fuzzy": "Fuzzy rule base", "P0_always_execute": "Always execute"}
GATES = {"F_vote_scores|R_myopic|A_none": "CATWAY", "F_verbal_scores|R_myopic|A_none": "CATWAY-Lite",
         "E_vote|R_myopic|A_none": "Vote only", "E_verbal|R_myopic|A_none": "Stated only",
         "E_logit|R_myopic|A_none": "Token only"}

L = []


def f3(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def longtable(caption, label, colspec, header, rows):
    L.append(f"\\begin{{longtable}}{{{colspec}}}")
    L.append(f"\\caption{{{caption}}}\\label{{{label}}}\\\\")
    L.append("\\toprule " + header + " \\\\ \\midrule\\endfirsthead")
    L.append("\\toprule " + header + " \\\\ \\midrule\\endhead")
    L.append("\\bottomrule\\endfoot")
    L.extend(r + " \\\\" for r in rows)
    L.append("\\end{longtable}")


def variants():
    sel = pd.read_csv(TMS / "selection_on_calibration.csv").sort_values("rank")
    ts = pd.read_csv(TMS / "test_summary.csv").set_index(["variant", "set"])
    rows = []
    for _, r in sel.iterrows():
        cells = [str(int(r["rank"])), EST[r.estimator], RULE[r.rule], ADAPT[r.adaptation], str(int(r.generations_per_decision)),
                 f"{r.cv_cost_mean:.3f}", f"{r.cv_cost_worst:.3f}"]
        for s in ["household_test", "new_domains_pooled", "trustnav"]:
            q = ts.loc[(r.variant, s)] if (r.variant, s) in ts.index else None
            cells.append("--" if q is None else f"{q.mean_cost:.3f}/{q.worst_cost:.3f}")
        mark = {"PROPOSED": " (CATWAY)", "LOW-COST": " (Lite)"}.get(str(r.selected), "")
        cells[1] += mark
        rows.append(" & ".join(cells))
    L.append(r"{\scriptsize")
    longtable("All 120 variants, ranked by mean cross-validated cost on the calibration split (six configurations). "
              "Gen.: generations per decision. Test: mean/worst cost over five configurations. Token-probability "
              "variants exist for Gemma only and were not eligible for selection.", "tab:s-variants",
              "@{}rlllrrrlll@{}",
              "Rank & Estimator & Rule & Adaptation & Gen. & CV mean & CV worst & Household & New domains & TRUST-NAV", rows)
    L.append("}")


def per_config():
    pol = pd.read_csv(T5 / "policies.csv")
    pol = pol[pol.block_rule.fillna("").str.startswith("block") | pol.block_rule.isna()]
    tb = pd.read_csv(TMS / "test_by_config.csv")
    for s in ["household_test", "new_domains_pooled", "production_cell", "mobile_robot", "trustnav", "trustnav_complete"]:
        rows = []
        for v, lab in GATES.items():
            x = tb[(tb.variant == v) & (tb.set == s)].set_index("model")
            cells = [f3(x.cost.get(m)) for m in CONF]
            ep = [f3(x.episode_cost.get(m)) for m in CONF]
            rows.append(lab + " & " + " & ".join(cells) + " & " + " & ".join(ep))
        for p, lab in POL.items():
            x = pol[(pol.policy == p) & (pol.set == s)].set_index("model")
            cells = [f3(x.cost.get(m)) for m in CONF]
            ep = [f3(x.episode_cost.get(m)) for m in CONF]
            rows.append(lab + " & " + " & ".join(cells) + " & " + " & ".join(ep))
        n = int(pol[(pol.set == s)].n.iloc[0])
        hdr = "Gate or baseline & " + " & ".join(LAB[m] for m in CONF) + " & " + " & ".join(LAB[m] for m in CONF)
        longtable(f"{SETNAME[s]} ($n={n}$): decision cost per item (first five columns) and episode cost with one "
                  "clarification round (last five; -- where the set has no clarification round).",
                  f"tab:s-{s}", "@{}p{3.7cm}rrrrrrrrrr@{}", hdr, rows)


def comparisons():
    t = pd.read_csv(TMS / "test_comparisons.csv")
    names = {"P1_direct_prompt": "direct", "P1c_direct_prompt_with_costs": "cost-aware", "P8_unguarded_agent": "agent",
             "vote_only_gate": "vote only"}
    groups = [["household_test", "new_domains_pooled", "trustnav"],
              ["production_cell", "mobile_robot", "trustnav_complete"]]
    for k, sets in enumerate(groups):
        rows = []
        for v in ["F_vote_scores|R_myopic|A_none", "F_verbal_scores|R_myopic|A_none"]:
            for m in CONF:
                for b, bl in names.items():
                    cells = []
                    for s_ in sets:
                        r = t[(t.variant == v) & (t.model == m) & (t.set == s_) & (t.comparator == b)].iloc[0]
                        p = r.p_holm_all_variants if b == "vote_only_gate" else r.p_holm_selected
                        cells.append(f"{r.diff_comparator_minus_variant:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}] ({p:.3f})")
                    rows.append(f"{GATES[v]} & {LAB[m]} & {bl} & " + " & ".join(cells))
        hdr = "Gate & Config. & Comparator & " + " & ".join(SETNAME[x] for x in sets)
        longtable(("CATWAY and CATWAY-Lite against the baselines that use the model directly and against the vote-only gate"
                   + (" (household test split, new domains, TRUST-NAV)" if k == 0 else
                      " (single new domains and TRUST-NAV with the completeness sentence; with 25 quadruples per domain these comparisons have little power)")
                   + ": comparator cost minus gate cost per item, 95\\% cluster-bootstrap interval and Holm-adjusted $p$. "
                   "Against the baselines, $p$ is adjusted over the 30 comparisons of the two selected gates with the three "
                   "baselines in the five configurations on that set; against the vote-only gate, over all 2,208 comparisons "
                   "made on that set (every variant and configuration against the three baselines and the vote-only gate)."),
                  f"tab:s-compare-{k + 1}", "@{}lllp{3.6cm}p{3.6cm}p{3.6cm}@{}", hdr, rows)


def ablations():
    ab = pd.read_csv(TMS / "ablations.csv")
    sets = ["household_calibration_cv", "household_test", "new_domains_pooled", "trustnav"]
    rows = []
    for g, gl in [("PROPOSED", "CATWAY"), ("LOW-COST", "CATWAY-Lite")]:
        x = ab[ab.gate == g]
        for (a, st), y in x.groupby(["ablation", "setting"], sort=False):
            cells = [f"{y[y.set == s].cost.mean():.3f}" if (y.set == s).any() else "--" for s in sets]
            rows.append(f"{gl} & {a} & {st.replace('_', ' ')} & " + " & ".join(cells))
    longtable("Ablations of the selected gates: mean cost over configurations with one component changed. Loss-matrix "
              "rows give the cost under that matrix (unsafe-execution cost, ask row scaled, or an alternative matrix).",
              "tab:s-ablations", "@{}lllrrrr@{}", "Gate & Ablation & Setting & Calibration (CV) & Household & New domains & TRUST-NAV", rows)


def calibration():
    rows = []
    for d, ms in [(TCV, CONF), (TGT, ["gemma-think"])]:
        c = pd.read_csv(d / "calibration_calibration_cv.csv")
        for m in ms:
            for _, r in c[c.model == m].iterrows():
                src = {"p_sample": "vote", "p_verbal": "stated", "p_logit": "token"}[r.source]
                ver = "raw" if r.version.startswith("raw") else "fitted"
                par = f"$\\nu={r.lam:.3f}$" if r.source == "p_sample" else f"$T={r.temperature:.3f}$"
                rows.append(f"{LAB[m]} & {src} & {ver} & {par if ver == 'fitted' else '--'} & {int(r.n)} & {r.ece:.3f} & {r.brier:.3f} & {r.nll:.3f} & {r.accuracy_4class:.3f}")
    longtable("Calibration of each source on the household calibration split (out-of-fold, five folds; the parameter shown is the median over folds).",
              "tab:s-calibration", "@{}lllrrrrrr@{}", "Config. & Source & Version & Parameter & $n$ & ECE & Brier & NLL & Accuracy", rows)


def calfit():
    fi = pd.concat([pd.read_csv(TCV / "fit_info_calibration_cv.csv"), pd.read_csv(TGT / "fit_info_calibration_cv.csv")])
    fi = fi[fi.source == "p_sample"].drop_duplicates("model").set_index("model")
    cal = pd.concat([pd.read_csv(TCV / "calibration_calibration_cv.csv"), pd.read_csv(TGT / "calibration_calibration_cv.csv")])
    cal = cal[cal.source == "p_sample"].drop_duplicates(["model", "version"])
    ue = pd.read_csv(T5 / "unanimous_errors.csv")
    ue = ue[(ue.set == "household_calibration") & (ue.voted_condition == "all")].set_index("model")
    vb = pd.read_csv(TGT / "vote_by_source_condition_calibration.csv")
    vb = vb[(vb.model == "gemma-think") & (vb["round"] == 1)]
    ms = pd.read_csv(TMS / "cv_by_config.csv").set_index(["variant", "model"])
    rows = []
    for m in CONF6:
        r = fi.loc[m]
        raw = cal[(cal.model == m) & cal.version.str.startswith("raw")].iloc[0]
        cc = cal[(cal.model == m) & cal.version.str.startswith("calibrated")].iloc[0]
        un = ue.loc[m, "unanimous_share"] if m in ue.index else vb.unanimous_sets.sum() / vb.n.sum()
        lo, hi = [float(x) for x in r.cv_lam_range.strip("[]").split(",")]
        pv = ("E_vote|R_myopic|A_none", m)
        pc = ("F_vote_scores|R_myopic|A_none", m)
        rows.append(f"{LAB[m]} & {r.lam:.3f} & {lo:.3f}--{hi:.3f} & {un:.3f} & {raw.ece:.3f} & {cc.ece:.3f} & "
                    f"{raw.accuracy_4class:.3f} & {ms.loc[pv, 'nll']:.3f} & {ms.loc[pc, 'nll']:.3f} & "
                    f"{ms.loc[pv, 'cost']:.3f} & {ms.loc[pc, 'cost']:.3f}")
    longtable("Calibration of the vote on the household calibration split (432 items; Gemma-R 419 items with a complete vote) "
              "and cross-validated negative log-likelihood (NLL) and cost of the vote-only gate and CATWAY. "
              r"$\hat\nu$: fitted pseudo-count (bound $5/12\approx0.417$); fold range over five folds; ECE: out-of-fold "
              r"expected calibration error of the vote before ($\nu=1/2$) and after fitting; accuracy: four-way accuracy of the vote.",
              "tab:s-calfit", "@{}lrrrrrrrrrr@{}",
              r"Config. & $\hat\nu$ & fold range & unanimous & ECE raw & ECE fitted & accuracy & NLL vote & NLL CATWAY & cost vote & cost CATWAY", rows)


def transfer():
    t = pd.read_csv(T5 / "transfer_noninferiority.csv")
    t = t[t.row.str.startswith("T1")]
    rows = []
    for _, r in t.iterrows():
        src = {"p_sample": "vote", "p_verbal": "stated", "p_logit": "token"}[r.source]
        rows.append(f"{LAB[r.model]} & {src} & {SETNAME.get(r.set, r.set)} & {r.cost_frozen_P5:.3f} & {r.cost_crossfit_refit_P5:.3f} & "
                    f"{r.delta_cv_frozen_minus_refit:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}] & {r.verdict.split(' (')[0]}")
    longtable("Frozen calibration against a parameter refitted on the target by cross-fitting (single-source gates; margin 0.10 per item).",
              "tab:s-transfer", "@{}lllrrll@{}", "Config. & Source & Set & Frozen & Refit & Difference [95\\% CI] & Verdict", rows)
    p = pd.read_csv(T5 / "prior_decomposition.csv")
    p = p[p.set == "new_domains_pooled"]
    rows = []
    qn = {"P5_three_way_calibrated": "vote-only gate", "P1_direct_prompt": "direct prompt", "P8_unguarded_agent": "unguarded agent"}
    for _, r in p.iterrows():
        if r.quantity in qn:
            rows.append(f"{LAB[r.model]} & {qn[r.quantity]} & {r.change_target_minus_household:.3f} & {r.prior_component:.3f} & {r.within_condition_component:.3f}")
    longtable("Change in cost from the household test split to the new domains, decomposed into a class-mix and a within-condition component.",
              "tab:s-decomp", "@{}llrrr@{}", "Config. & Policy & Change & Class mix & Within condition", rows)
    u = pd.read_csv(TMS / "ablation_unanimous_cell.csv")
    rows = []
    for _, r in u[u.set.isin(["household_test", "new_domains_pooled", "trustnav"])].iterrows():
        rows.append(f"{ {'PROPOSED': 'CATWAY', 'LOW-COST': 'CATWAY-Lite', 'VOTE-ONLY': 'Vote only'}[r.gate]} & {LAB[r.model]} & {SETNAME[r.set]} & {int(r.n_unanimous_clear)} & {r.share_truly_clear:.3f} & {r.execute_share:.3f} & {r.cost:.3f}")
    longtable("Items with ten ``clear'' votes: number, share truly clear, share executed by the gate, and cost of the gate on these items.",
              "tab:s-unanimous", "@{}lllrrrr@{}", "Gate & Config. & Set & $n$ & Truly clear & Executed & Cost", rows)


def reasoning():
    a = pd.read_csv(TMS / "ablation_reasoning.csv")
    rows = []
    for _, r in a.iterrows():
        g = {"PROPOSED": "CATWAY", "LOW-COST": "CATWAY-Lite", "VOTE-ONLY": "Vote only"}[r.gate]
        mdl = {"claude-haiku-5-5": "Claude Haiku", "gpt-6-luna": "GPT-6 Luna"}.get(r.model, r.model)
        rows.append(f"{g} & {mdl} & {SETNAME[r.set]} & {r.cost_on:.3f} & {r.cost_off:.3f} & {r.diff_on_minus_off:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]")
    longtable("Reasoning on against off on the same items (cost per item; 95\\% cluster-bootstrap interval).",
              "tab:s-reasoning", "@{}lllrrl@{}", "Gate & Model & Set & On & Off & On minus off", rows)


def trustnav_configs():
    t = pd.read_csv(T5 / "trustnav_gate_configs.csv")
    rows = []
    for _, r in t.iterrows():
        rows.append(f"{POL.get(r.policy, r.policy)} & {r.weights_ambiguity_hallucination_conflict} & {r.execute_threshold} & {r.clarify_threshold} & {r.hallucination_ceiling} & {r.conflict_ceiling} & {r.ambiguity_ceiling}")
    longtable("The two TRUST-NAV gate configurations applied to our graded scores (ambiguity, infeasibility, risk in place of ambiguity, hallucination, semantic conflict).",
              "tab:s-trustnav-configs", "@{}p{5.5cm}lrrrrr@{}", "Configuration & Weights & Execute $\\le$ & Ask $\\le$ & Infeas. ceiling & Risk ceiling & Amb. ceiling", rows)


if __name__ == "__main__":
    variants()
    per_config()
    comparisons()
    ablations()
    calibration()
    calfit()
    transfer()
    reasoning()
    trustnav_configs()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf8")
    print("written", OUT)
