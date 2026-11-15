"""Supplementary analyses (Supplementary Tables S20-S27; exploratory, added after the main analyses).
No model calls: every number is recomputed from the stored answers with the functions of analyze.py,
method_study.py and build_data.py.

    python src/supplementary_analyses.py                      # default inputs, writes results/supplementary_analyses/
    python src/supplementary_analyses.py --raw DIR [--raw DIR ...] --out DIR [--boot 2000] [--seed 20261115]
                                         [--skip-selection]

Inputs: the same answer folders as method_study.py (later folders win; default results/answers), loaded
with method_study.load_configs (the same loading and dropping rule as analyze.py --unblind), and the built
household items data/processed/household.jsonl (README.md, "Rebuild the benchmark"). The stored tables in
results/tables_method_study/ and results/tables_test_six_configs/ are read only to cross-check the
recomputation (validation.csv) and, for the 2,640-comparison Holm family of M2, to supply the p-values of the
variants not recomputed here.

How decisions are made (nothing is re-implemented):
  CATWAY (F_vote_scores|R_myopic|A_none), CATWAY-Lite (F_verbal_scores|R_myopic|A_none), the vote-only gate
  (E_vote|R_myopic|A_none) and the other method-study variants: method_study.run_variant, fitted on the
  household calibration split and frozen for the test sets.
  Fitted and prompted baselines (conformal set rule P4_conformal, top-probability threshold P4b_chow_threshold,
  fuzzy rule base P6_fuzzy, TRUST-NAV gate P3r_trustnav_runner_config, rule cascade P2_cascade, direct prompt
  P1_direct_prompt, cost-aware prompt P1c_direct_prompt_with_costs, unguarded agent P8_unguarded_agent, agent
  with checker P9_agent_plus_checker): analyze.fit_base / fit_cost_params / decide on the same rows, exactly
  as analyze.analyse_model does (Table 4 of the paper).
Tests (as in the paper): differences are comparator minus gate (positive = the gate is cheaper), 95%
percentile intervals of a cluster bootstrap over matched groups (analyze.paired_bootstrap, 10,000 resamples,
seed 0), two-sided sign-flip test on group sums, exact by complete enumeration (analyze.exact_sign_flip_p),
Holm (analyze.holm) within the family named in each table.

Outputs (results/supplementary_analyses/, every table has a `status` column):
  m5_catway_identity.csv          P7_learned (analyze) against CATWAY (method_study), item level, rounds 1 and 2
  m5_fitted_gates_vs_catway.csv   CATWAY against the four fitted gates; Holm over the 24 rows of each set
  m5_summary.csv                  counts per set
  m1_vote_rules.csv               Table 7 style rows: vote under min-loss, conformal and sequential rules (+ refs)
  m2_trustnav_cells.csv           CATWAY / CATWAY-Lite against the vote-only gate, with Holm over 2,640 and over 12
  m4_holm12.csv, m4_group_counts.csv, m4_p_methods.csv
  m3_pseudocount_bootstrap.csv, m3_failing_configurations.csv
  m6_per_domain_comparisons.csv, m6_per_domain_directions.csv
  m8_unsafe_by_config.csv, m8_unsafe_totals.csv, m8_upper_bounds.csv
  m9_cost_under_matrices.csv, m9_matrix_summary.csv, m9_directions.csv, m9_selection_per_matrix.csv
  m10_class_mix.csv, m10_summary.csv
  m11_calibration_metrics.csv
  m12_selection_primary.csv, m12_summary.csv
  c_lite_gap_closed.csv, c_unanimous_cell.csv, c_scores_ablation.csv, c_forced_calibration_groups.csv,
  c_secondary_metrics.csv, c_vote_only_costlier_question.csv
  key_numbers.csv                 every number the revision quotes, with the CSV and column it comes from
  validation.csv                  recomputed values against the stored pipeline tables
  README.txt
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402
import build_data as BD  # noqa: E402
import method_study as M  # noqa: E402

ROOT = A.ROOT
DEFAULT_RAW = [ROOT / "results" / "answers"]
OUT = ROOT / "results" / "supplementary_analyses"
STORED_MS = ROOT / "results" / "tables_method_study"
STORED_AN = ROOT / "results" / "tables_test_six_configs"
STATUS = "supplementary analyses (exploratory): recomputed from the stored answers"
DEFAULT_SEED = 20261115
DEFAULT_BOOT = A.N_BOOT_NESTED                       # 2000: the project's convention for bootstraps with a refit inside

LABEL = {"claude-haiku": "Haiku-R", "claude-haiku-nothink": "Haiku-N", "gpt-luna-think": "Luna-R",
         "gpt-luna": "Luna-N", "gemma": "Gemma-N", "gemma-think": "Gemma-R"}
CATWAY = "F_vote_scores|R_myopic|A_none"
LITE = "F_verbal_scores|R_myopic|A_none"
VOTE = M.VOTE_ONLY                                   # E_vote|R_myopic|A_none
VOTE_CONF = "E_vote|R_conformal|A_none"
VOTE_SEQ = "E_vote|R_sequential|A_none"
STATED = "E_verbal|R_myopic|A_none"
FUSION5 = "F_verbal_scores_yesno|R_myopic|A_none"
GATE_NAMES = {CATWAY: "CATWAY", LITE: "CATWAY-Lite", VOTE: "vote-only gate", VOTE_CONF: "vote, conformal set rule",
              VOTE_SEQ: "vote, sequential rule", STATED: "stated probabilities", FUSION5: "stated + scores + yes/no"}
FITTED = ("P4_conformal", "P4b_chow_threshold", "P6_fuzzy", "P3r_trustnav_runner_config")
BASELINES = M.BASELINES                              # P1, P1c, P8
P_NAMES = {"P4_conformal": "conformal set rule", "P4b_chow_threshold": "top-probability threshold",
           "P6_fuzzy": "fuzzy rule base", "P3r_trustnav_runner_config": "TRUST-NAV gate", "P2_cascade": "rule cascade",
           "P1_direct_prompt": "direct prompt", "P1c_direct_prompt_with_costs": "cost-aware prompt",
           "P8_unguarded_agent": "unguarded agent", "P9_agent_plus_checker": "agent with checker",
           "P5_three_way_calibrated": "vote-only gate (analyze P5)", "P7_learned": "P7_learned (analyze)"}
NAME = {**GATE_NAMES, **P_NAMES}
MAIN_SETS = ("household_test", "new_domains_pooled", "trustnav")
ALL_SETS = M.SETS_TEST
UNSAFE_SETS = ("household_test", "new_domains_pooled")
SECONDARY_POLICIES = (CATWAY, LITE, VOTE, "P4_conformal", "P4b_chow_threshold", "P6_fuzzy", "P2_cascade",
                      "P3r_trustnav_runner_config", "P1_direct_prompt", "P1c_direct_prompt_with_costs",
                      "P8_unguarded_agent", "P9_agent_plus_checker")
SWEEP_POLICIES = ("P1_direct_prompt", "P1c_direct_prompt_with_costs", "P2_cascade", "P3r_trustnav_runner_config",
                  "P4_conformal", "P4b_chow_threshold", "P5_three_way_calibrated", "P6_fuzzy", "P7_learned",
                  "P8_unguarded_agent", "P9_agent_plus_checker")
ALPHA = 0.05

KEY = []           # rows of key_numbers.csv
CHECKS = []        # rows of validation.csv


def key(qid, quantity, value, csv, column, where=""):
    KEY.append({"id": qid, "quantity": quantity, "value": value, "csv": csv, "column": column, "row_filter": where})


def check(name, new, stored, tol=1e-9, note=""):
    new, stored = np.asarray(new, float), np.asarray(stored, float)
    both = ~(np.isnan(new) & np.isnan(stored))
    diff = float(np.nanmax(np.abs(new[both] - stored[both]))) if both.any() else 0.0
    nan_mismatch = int((np.isnan(new) != np.isnan(stored)).sum())
    CHECKS.append({"check": name, "n_values": int(new.size), "max_abs_diff": diff, "nan_mismatches": nan_mismatch,
                   "tolerance": tol, "ok": bool(diff <= tol and nan_mismatch == 0), "note": note})


def write(out, name, df):
    df = df.copy()
    if "status" not in df:
        df.insert(0, "status", STATUS)
    df.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")   # same bytes on every OS
    return df


def compare(cb, cg, grp):
    diff, lo, hi = A.paired_bootstrap(cb, cg, grp)
    p, how = A.exact_sign_flip_p(cb - cg, grp)
    return {"diff_comparator_minus_gate": diff, "ci_low": lo, "ci_high": hi, "p_signflip": p, "p_method": how}


# ---------------------------------------------------------------- data and decisions

def load(raw):
    configs = M.load_configs(raw)
    for c in configs.values():
        d = c.d
        c.base = A.fit_base(d, d.cal, "p_sample")
        c.cp = A.fit_cost_params(d, c.base, d.cal, A.COST)
        c.apol, c.ar2 = A.decide(d, c.base, c.cp, A.COST)
        c.gates = {}
    return configs


def gate(c, variant, groups=None, cost=None, tag=None):
    """(cv metrics, per-item cv cost, tests) of a method-study variant, cached per configuration."""
    k = tag or variant
    if cost is None and k in c.gates:
        return c.gates[k]
    e, r, ad = variant.split("|")
    res = M.run_variant(c, e, r, ad, cost=cost, groups=groups)
    if cost is None:
        c.gates[k] = res
    return res


def acts(c, name, s):
    """Round-one (int) and round-two (float) actions of a gate or a baseline, over every row, for set s."""
    if "|" in name:
        _, a, a2, _, _ = gate(c, name)[2][s]
        return a, a2
    return c.apol[name], c.ar2[name]


def rows_of(c, s):
    return np.flatnonzero(c.parts[s]) if s in c.parts else None


# ---------------------------------------------------------------- M5

def m5(out, configs):
    ident, comp = [], []
    for m, c in configs.items():
        d = c.d
        for s in ALL_SETS:
            rws = rows_of(c, s)
            if rws is None:
                continue
            a, a2 = acts(c, CATWAY, s)
            p7, p72 = c.apol["P7_learned"], c.ar2["P7_learned"]
            j = d.r2pos[rws]
            j = j[j >= 0]
            y = d.y[rws]
            ident.append({"model": m, "config": LABEL[m], "set": s, "n": len(rws), "n_round_two_rows": len(j),
                          "identical_round_one": bool((a[rws] == p7[rws]).all()),
                          "n_round_one_differences": int((a[rws] != p7[rws]).sum()),
                          "identical_round_two": bool(np.array_equal(np.nan_to_num(a2[j], nan=-9),
                                                                     np.nan_to_num(p72[j], nan=-9))),
                          "cost_catway": float(A.COST[a[rws], y].mean()), "cost_p7_learned": float(A.COST[p7[rws], y].mean())})
            cg = A.COST[a[rws], y]
            for b in FITTED:
                cb = A.COST[c.apol[b][rws], y]
                comp.append({"model": m, "config": LABEL[m], "set": s, "comparator": b, "comparator_name": P_NAMES[b],
                             "n": len(rws), "n_clusters": len(np.unique(d.group[rws])), "cost_catway": float(cg.mean()),
                             "cost_comparator": float(cb.mean()), **compare(cb, cg, d.group[rws])})
    ident = write(out, "m5_catway_identity", pd.DataFrame(ident))
    cdf = pd.DataFrame(comp)
    cdf["holm_family"] = "the 24 comparisons of the set (6 configurations x 4 fitted gates)"
    cdf["p_holm_set"] = np.nan
    for s in cdf.set.unique():
        ix = cdf.index[cdf.set == s]
        cdf.loc[ix, "p_holm_set"] = A.holm(cdf.loc[ix, "p_signflip"].to_numpy())
    cdf = write(out, "m5_fitted_gates_vs_catway", cdf)
    summ = []
    for s, g in cdf.groupby("set", sort=False):
        sig = g.p_holm_set < ALPHA
        fav_gate = g[(g.ci_high < 0) & ~sig]
        summ.append({"set": s, "n_comparisons": len(g),
                     "sig_catway_cheaper": int((sig & (g.diff_comparator_minus_gate > 0)).sum()),
                     "sig_catway_costlier": int((sig & (g.diff_comparator_minus_gate < 0)).sum()),
                     "ci_wholly_favours_fitted_gate_not_sig": len(fav_gate),
                     "which_favour_fitted_gate": "; ".join(f"{r.config} {r.comparator_name} {r.diff_comparator_minus_gate:.3f} "
                                                           f"[{r.ci_low:.3f}, {r.ci_high:.3f}]" for r in fav_gate.itertuples()),
                     "which_sig_cheaper": "; ".join(f"{r.config} {r.comparator_name} {r.diff_comparator_minus_gate:.3f}"
                                                    for r in g[sig & (g.diff_comparator_minus_gate > 0)].itertuples())})
    summ = write(out, "m5_summary", pd.DataFrame(summ))
    all_id = bool(ident.identical_round_one.all() and ident.identical_round_two.all())
    key("M5.identity", "P7_learned identical to CATWAY in every item (round 1 and 2), all configs and sets", all_id,
        "m5_catway_identity.csv", "identical_round_one, identical_round_two", "all rows")
    for s in MAIN_SETS:
        r = summ[summ.set == s].iloc[0]
        key(f"M5.{s}.sig_cheaper", f"CATWAY significantly cheaper than a fitted gate, {s} (of 24)", int(r.sig_catway_cheaper),
            "m5_summary.csv", "sig_catway_cheaper", f"set={s}")
        key(f"M5.{s}.sig_costlier", f"CATWAY significantly costlier than a fitted gate, {s} (of 24)", int(r.sig_catway_costlier),
            "m5_summary.csv", "sig_catway_costlier", f"set={s}")
        key(f"M5.{s}.ci_favour_gate", f"intervals wholly in a fitted gate's favour, not significant, {s}",
            r.which_favour_fitted_gate or "none", "m5_summary.csv", "which_favour_fitted_gate", f"set={s}")
    return cdf


# ---------------------------------------------------------------- M1 (and Table 7 reference rows)

def m1(out, configs):
    rows = []
    for v in (VOTE, VOTE_CONF, VOTE_SEQ, STATED, CATWAY, LITE, FUSION5):
        e, r, ad = v.split("|")
        cv = [gate(c, v)[0]["cost"] for c in configs.values()]
        row = {"variant": v, "name": GATE_NAMES[v],
               "generations": max(M.deployed_generations(c, e, ad) for c in configs.values()),
               "cv_mean": float(np.mean(cv)), "cv_worst": float(np.max(cv))}
        for s in MAIN_SETS:
            cs = []
            for c in configs.values():
                rws = rows_of(c, s)
                a, _ = acts(c, v, s)
                cs.append(float(A.COST[a[rws], c.d.y[rws]].mean()))
            row[f"{s}_mean"], row[f"{s}_worst"] = float(np.mean(cs)), float(np.max(cs))
        rows.append(row)
    df = write(out, "m1_vote_rules", pd.DataFrame(rows))
    for v in (VOTE, VOTE_CONF, VOTE_SEQ):
        r = df[df.variant == v].iloc[0]
        key(f"M1.{v}", f"Table 7 row '{GATE_NAMES[v]}': gen, CV mean/worst, household, new domains, TRUST-NAV mean/worst",
            " & ".join([str(int(r.generations))] + [f"{r[c]:.3f}" for c in ("cv_mean", "cv_worst", "household_test_mean",
                        "household_test_worst", "new_domains_pooled_mean", "new_domains_pooled_worst", "trustnav_mean",
                        "trustnav_worst")]), "m1_vote_rules.csv", "generations..trustnav_worst", f"variant={v}")
    return df


# ---------------------------------------------------------------- M2 and M4 (gate against the vote-only gate)

def vote_comparisons(configs):
    rows = []
    for m, c in configs.items():
        d = c.d
        for s in ALL_SETS:
            rws = rows_of(c, s)
            if rws is None:
                continue
            y, grp = d.y[rws], d.group[rws]
            cb = A.COST[acts(c, VOTE, s)[0][rws], y]
            for v in (CATWAY, LITE):
                cg = A.COST[acts(c, v, s)[0][rws], y]
                rows.append({"variant": v, "gate": GATE_NAMES[v], "model": m, "config": LABEL[m], "set": s,
                             "comparator": "vote_only_gate", "n": len(rws), "n_clusters": len(np.unique(grp)),
                             "cost_gate": float(cg.mean()), "cost_vote_only": float(cb.mean()), **compare(cb, cg, grp)})
    return pd.DataFrame(rows)


def m2_m4(out, configs):
    vc = vote_comparisons(configs)
    stored = pd.read_csv(STORED_MS / "test_comparisons.csv")
    # Holm over 12: CATWAY and CATWAY-Lite against the vote-only gate, six configurations, per set.
    vc["p_holm_12"] = np.nan
    for s in vc.set.unique():
        ix = vc.index[vc.set == s]
        vc.loc[ix, "p_holm_12"] = A.holm(vc.loc[ix, "p_signflip"].to_numpy())
    # Holm over all rows of the set in the stored method-study table (2,640 per set), with the rows
    # recomputed here checked against it first.
    mg = vc.merge(stored, on=["variant", "model", "set", "comparator"], how="left", suffixes=("", "_stored"))
    check("M2: recomputed gate-vs-vote-only p_signflip = stored test_comparisons.csv", mg.p_signflip, mg.p_signflip_stored)
    check("M2: recomputed gate-vs-vote-only difference = stored", mg.diff_comparator_minus_gate,
          mg.diff_comparator_minus_variant)
    check("M2: recomputed gate-vs-vote-only CI = stored", np.r_[mg.ci_low, mg.ci_high],
          np.r_[mg.ci_low_stored, mg.ci_high_stored])
    fam_sizes = stored.groupby("set").size()
    rh = []
    for s, g in stored.groupby("set"):
        rh.append(pd.Series(A.holm(g.p_signflip.to_numpy()), index=g.index))
    check("M2: Holm over the stored 2,640-row family reproduces p_holm_all_variants", pd.concat(rh).sort_index(),
          stored.p_holm_all_variants.sort_index())
    vc["holm_family_size_all_variants"] = vc.set.map(fam_sizes)
    vc["p_holm_all_variants"] = mg.p_holm_all_variants.to_numpy()
    vc = write(out, "m2_trustnav_cells", vc)
    for cfgm in ("claude-haiku-nothink", "gemma"):
        r = vc[(vc.variant == CATWAY) & (vc.model == cfgm) & (vc.set == "trustnav")].iloc[0]
        key(f"M2.{LABEL[cfgm]}.trustnav", f"TRUST-NAV {LABEL[cfgm]}: vote-only cost / CATWAY cost / Holm(2,640) p",
            f"{r.cost_vote_only:.3f} / {r.cost_gate:.3f} / {r.p_holm_all_variants:.2g}", "m2_trustnav_cells.csv",
            "cost_vote_only, cost_gate, p_holm_all_variants", f"variant={CATWAY}, model={cfgm}, set=trustnav")
    r = vc[(vc.variant == CATWAY) & (vc.model == "claude-haiku-nothink") & (vc.set == "new_domains_pooled")].iloc[0]
    key("M2.HaikuN.newdomains", "Haiku-N new domains, vote-only minus CATWAY [CI], exact p, Holm(2,640), Holm(12)",
        f"{r.diff_comparator_minus_gate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}], p={r.p_signflip:.2g}, "
        f"Holm2640={r.p_holm_all_variants:.3f}, Holm12={r.p_holm_12:.3f}", "m2_trustnav_cells.csv",
        "diff_comparator_minus_gate, ci_low, ci_high, p_signflip, p_holm_all_variants, p_holm_12",
        f"variant={CATWAY}, model=claude-haiku-nothink, set=new_domains_pooled")

    # M4: the 12-comparison family, as a labelled sensitivity analysis.
    h12 = vc[vc.set.isin(MAIN_SETS)][["set", "gate", "config", "cost_gate", "cost_vote_only", "diff_comparator_minus_gate",
                                      "ci_low", "ci_high", "p_signflip", "p_holm_12", "p_holm_all_variants"]].copy()
    h12["significant_holm_12"] = h12.p_holm_12 < ALPHA
    h12 = write(out, "m4_holm12", h12)
    sig = h12[h12.significant_holm_12]
    key("M4.holm12.significant", "Significant under Holm over 12 (gate vs vote-only; household, new domains, TRUST-NAV)",
        "; ".join(f"{s}: " + ", ".join(sorted(f"{r.gate}/{r.config}" for r in g.itertuples())) for s, g in sig.groupby("set")),
        "m4_holm12.csv", "significant_holm_12", "significant_holm_12=True")
    for s in MAIN_SETS:
        for cfgm in ("Haiku-N", "Gemma-N"):
            r = h12[(h12.set == s) & (h12.config == cfgm) & (h12.gate == "CATWAY")].iloc[0]
            key(f"M4.holm12.{s}.{cfgm}", f"Holm(12) p, CATWAY vs vote-only, {cfgm}, {s}", f"{r.p_holm_12:.2g}",
                "m4_holm12.csv", "p_holm_12", f"set={s}, config={cfgm}, gate=CATWAY")
    gc = []
    for s in ALL_SETS:
        ns = [(len(np.unique(c.d.group[rows_of(c, s)])), len(rows_of(c, s))) for c in configs.values() if s in c.parts]
        gc.append({"set": s, "n_clusters_min": min(x[0] for x in ns), "n_clusters_max": max(x[0] for x in ns),
                   "n_items_min": min(x[1] for x in ns), "n_items_max": max(x[1] for x in ns)})
    gc = write(out, "m4_group_counts", pd.DataFrame(gc))
    key("M4.groups", "Matched groups: household test / new domains pooled / production cell / mobile robot / TRUST-NAV",
        " / ".join(str(int(gc[gc.set == s].n_clusters_min.iloc[0])) + ("" if gc[gc.set == s].n_clusters_min.iloc[0] ==
                   gc[gc.set == s].n_clusters_max.iloc[0] else "*") for s in ("household_test", "new_domains_pooled",
                                                                              "production_cell", "mobile_robot", "trustnav")),
        "m4_group_counts.csv", "n_clusters_min (= n_clusters_max)", "set in household_test, new_domains_pooled, "
                                                                  "production_cell, mobile_robot, trustnav")
    pm = pd.concat([stored.p_method.value_counts().rename("stored_test_comparisons"),
                    vc.p_method.value_counts().rename("recomputed_here")], axis=1).fillna(0).astype(int)
    pm.index.name = "p_method"
    pm = write(out, "m4_p_methods", pm.reset_index())
    key("M4.exact", "All p-values of the stored 15,840-row test_comparisons.csv exact", f"{int((stored.p_method == 'exact').sum())} of {len(stored)}",
        "m4_p_methods.csv", "stored_test_comparisons", "p_method=exact")
    return vc


# ---------------------------------------------------------------- M3

def m3(out, configs, B, seed):
    rng = np.random.default_rng(seed)
    labels = sorted(set().union(*[set(c.d.group[c.d.cal]) for c in configs.values()]))
    pos = {g: k for k, g in enumerate(labels)}
    K = len(labels)
    Wc = np.zeros((B, K))
    np.add.at(Wc, (np.arange(B)[:, None], rng.integers(0, K, (B, K))), 1.0)
    rows, above_all = [], {}
    fit_info = pd.read_csv(STORED_AN / "fit_info.csv")
    fit_info = fit_info[(fit_info.role == "main")].set_index("model")
    for m, c in configs.items():
        d = c.d
        det = SimpleNamespace(model=m, d=d, src="p_sample")
        f = A._Fitter(det)
        cal = np.flatnonzero(d.cal)
        full = f.value(f.fit(d.cal.astype(float)))
        cols = [pos[g] for g in d.group[cal]]
        lam = []
        for k in range(0, B, 2000):                    # chunks keep the weight matrix small for large B
            W = np.zeros((min(2000, B - k), len(d.y)))
            W[:, cal] = Wc[k:k + 2000][:, cols]
            lam.append(f.grid[f.fit_many(W)])
        lam = np.concatenate(lam)
        above = lam > A.LAMBDA_BOUNDARY
        above_all[m] = above
        vals, freq = np.unique(np.round(lam, 3), return_counts=True)
        C = np.stack(d.df["counts"].to_list())[cal]
        complete = C.sum(axis=1) > 0
        g398 = int(np.argmin(np.abs(f.grid[:-1] - 0.398)))
        g447 = int(np.argmin(np.abs(f.grid[:-1] - 0.447)))
        gfz = int(np.argmin(np.abs(f.grid[:-1] - full)))
        row = {"model": m, "config": LABEL[m], "n_calibration_items": len(cal),
               "n_calibration_groups": len(np.unique(d.group[cal])), "n_groups_joint_resample": K,
               "fitted_lambda_full_sample": full, "fitted_lambda_fit_info_csv": float(fit_info.loc[m, "lam"]),
               "unanimous_share_calibration": float((C[complete].max(1) == C[complete].sum(1)).mean()),
               "lambda_boundary": A.LAMBDA_BOUNDARY, "B": B, "seed": seed,
               "share_bootstrap_above_boundary": float(above.mean()),
               "mc_se_share": float(np.sqrt(above.mean() * (1 - above.mean()) / B)),
               "bootstrap_lambda_distribution": json.dumps(dict(zip(map(float, vals), map(int, freq))))}
        for s in MAIN_SETS:
            rws = rows_of(c, s)
            for lab, g in (("0.398", g398), ("0.447", g447), ("frozen", gfz)):
                row[f"vote_only_cost_{s}_lambda_{lab}"] = float(f.Cost[rws, g].mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    check("M3: full-sample pseudo-count refit (_Fitter) = fit_info.csv lam", df.fitted_lambda_full_sample,
          df.fitted_lambda_fit_info_csv, tol=1e-12)
    df = write(out, "m3_pseudocount_bootstrap", df)
    n_fail = np.sum([above_all[m] for m in configs], axis=0)
    dist = pd.DataFrame({"n_configurations_above_boundary": np.arange(len(configs) + 1),
                         "share_of_resamples": [float((n_fail == k).mean()) for k in range(len(configs) + 1)]})
    dist["expected_number"] = float(n_fail.mean())
    dist["B"], dist["seed"] = B, seed
    dist["note"] = "joint cluster bootstrap: the same resampled calibration groups for all six configurations"
    dist = write(out, "m3_failing_configurations", dist)
    for cfgm in ("claude-haiku-nothink", "gemma", "gpt-luna"):
        r = df[df.model == cfgm].iloc[0]
        key(f"M3.share.{LABEL[cfgm]}", f"share of bootstrap fits above 5/12, {LABEL[cfgm]} (B={B}, seed {seed})",
            f"{r.share_bootstrap_above_boundary:.3f} (MC SE {r.mc_se_share:.3f})", "m3_pseudocount_bootstrap.csv",
            "share_bootstrap_above_boundary", f"model={cfgm}")
    key("M3.share.reasoning", "share above 5/12 for Haiku-R, Luna-R, Gemma-R",
        ", ".join(f"{df[df.model == m].share_bootstrap_above_boundary.iloc[0]:.3f}" for m in ("claude-haiku", "gpt-luna-think", "gemma-think")),
        "m3_pseudocount_bootstrap.csv", "share_bootstrap_above_boundary", "model in claude-haiku, gpt-luna-think, gemma-think")
    key("M3.expected", "expected number of configurations on the expensive side (of 6)", f"{float(n_fail.mean()):.2f}",
        "m3_failing_configurations.csv", "expected_number", "any row")
    key("M3.fitted", "full-sample fitted pseudo-counts Haiku-N / Gemma-N / Luna-N",
        " / ".join(f"{df[df.model == m].fitted_lambda_full_sample.iloc[0]:.3f}" for m in ("claude-haiku-nothink", "gemma", "gpt-luna")),
        "m3_pseudocount_bootstrap.csv", "fitted_lambda_full_sample", "")
    g = df[df.model == "gemma"].iloc[0]
    key("M3.gemma.lower", "Gemma-N vote-only cost had lambda been 0.398: new domains / TRUST-NAV",
        f"{g['vote_only_cost_new_domains_pooled_lambda_0.398']:.3f} / "
        f"{g['vote_only_cost_trustnav_lambda_0.398']:.3f}", "m3_pseudocount_bootstrap.csv",
        "vote_only_cost_<set>_lambda_0.398", "model=gemma")
    g = df[df.model == "gpt-luna"].iloc[0]
    key("M3.luna.higher", "Luna-N vote-only cost had lambda been 0.447: new domains / TRUST-NAV",
        f"{g['vote_only_cost_new_domains_pooled_lambda_0.447']:.3f} / {g['vote_only_cost_trustnav_lambda_0.447']:.3f}",
        "m3_pseudocount_bootstrap.csv", "vote_only_cost_<set>_lambda_0.447", "model=gpt-luna")
    return df


# ---------------------------------------------------------------- M6

def m6(out, configs):
    rows = []
    for m, c in configs.items():
        d = c.d
        for s in ALL_SETS:
            rws = rows_of(c, s)
            if rws is None:
                continue
            y, grp = d.y[rws], d.group[rws]
            for v in (CATWAY, LITE):
                cg = A.COST[acts(c, v, s)[0][rws], y]
                for b in BASELINES:
                    cb = A.COST[c.apol[b][rws], y]
                    rows.append({"gate": GATE_NAMES[v], "variant": v, "model": m, "config": LABEL[m], "set": s,
                                 "comparator": b, "comparator_name": P_NAMES[b], "n": len(rws),
                                 "n_clusters": len(np.unique(grp)), **compare(cb, cg, grp)})
    df = pd.DataFrame(rows)
    df["holm_family"] = "the 36 comparisons of the set (CATWAY and CATWAY-Lite x 3 baselines x 6 configurations)"
    df["p_holm_36"] = np.nan
    for s in df.set.unique():
        ix = df.index[df.set == s]
        df.loc[ix, "p_holm_36"] = A.holm(df.loc[ix, "p_signflip"].to_numpy())
    stored = pd.read_csv(STORED_MS / "test_comparisons.csv")
    mg = df.merge(stored, on=["variant", "model", "set", "comparator"], how="left")
    check("M6: recomputed p_holm over 36 = stored p_holm_selected", mg.p_holm_36, mg.p_holm_selected)
    df = write(out, "m6_per_domain_comparisons", df)
    summ = []
    for (s, gname), g in df.groupby(["set", "gate"], sort=False):
        sig = g.p_holm_36 < ALPHA
        dd = g.diff_comparator_minus_gate
        summ.append({"set": s, "gate": gname, "n": len(g), "sig_cheaper": int((sig & (dd > 0)).sum()),
                     "cheaper_not_sig": int((~sig & (dd > 1e-12)).sum()), "equal": int((dd.abs() <= 1e-12).sum()),
                     "costlier_not_sig": int((~sig & (dd < -1e-12)).sum()), "sig_costlier": int((sig & (dd < 0)).sum()),
                     "ci_excludes_zero_favouring_gate": int((g.ci_low > 0).sum()),
                     "ci_below_zero": "; ".join(f"{r.config} vs {r.comparator_name} {r.diff_comparator_minus_gate:.3f} "
                                                f"[{r.ci_low:.3f}, {r.ci_high:.3f}]" for r in g[g.ci_high < 0].itertuples())})
    summ = write(out, "m6_per_domain_directions", pd.DataFrame(summ))
    for s in ("production_cell", "mobile_robot", "trustnav_complete"):
        r = summ[(summ.set == s) & (summ.gate == "CATWAY")].iloc[0]
        key(f"M6.{s}", f"CATWAY vs 3 baselines, {s}: sig cheaper / cheaper n.s. / equal / costlier n.s. / sig costlier; "
            "CIs excluding zero in CATWAY's favour; CIs below zero",
            f"{r.sig_cheaper} / {r.cheaper_not_sig} / {r.equal} / {r.costlier_not_sig} / {r.sig_costlier}; "
            f"{r.ci_excludes_zero_favouring_gate}; {r.ci_below_zero or 'none'}", "m6_per_domain_directions.csv",
            "sig_cheaper..ci_below_zero", f"set={s}, gate=CATWAY")
    return summ


# ---------------------------------------------------------------- M8

def m8(out, configs):
    pols = (CATWAY, LITE, VOTE, "P4_conformal", "P4b_chow_threshold", "P6_fuzzy", "P2_cascade",
            "P3r_trustnav_runner_config", "P1_direct_prompt", "P1c_direct_prompt_with_costs", "P8_unguarded_agent",
            "P9_agent_plus_checker")
    rows = []
    for name in pols:
        for m, c in configs.items():
            d = c.d
            for s in UNSAFE_SETS:
                rws = rows_of(c, s)
                a, a2 = acts(c, name, s)
                y = d.y[rws]
                a1 = a[rws]
                un = y == 3
                j = d.r2pos[rws]
                sel = un & (a1 == 1) & (j >= 0)
                sec = np.asarray(a2, float)[j[sel]]
                rows.append({"policy": name, "name": NAME[name], "model": m, "config": LABEL[m], "set": s,
                             "n_unsafe_items": int(un.sum()), "executed_round_one": int((a1[un] == 0).sum()),
                             "asked_round_one": int((a1[un] == 1).sum()), "refused_round_one": int((a1[un] == 2).sum()),
                             "asked_without_round_two_item": int((un & (a1 == 1) & (j < 0)).sum()),
                             "executed_after_confirmation": int(np.nansum(sec == 0)),
                             "asked_again_round_two": int(np.nansum(sec == 1)),
                             "round_two_unusable": int(np.isnan(sec).sum())})
    df = pd.DataFrame(rows)
    df["executed_either_round"] = df.executed_round_one + df.executed_after_confirmation
    df = write(out, "m8_unsafe_by_config", df)
    cols = ["n_unsafe_items", "executed_round_one", "asked_round_one", "refused_round_one", "executed_after_confirmation",
            "round_two_unusable", "executed_either_round"]
    tot = df.groupby(["policy", "name"], sort=False)[cols].sum().reset_index().rename(columns={"n_unsafe_items": "n_unsafe_decisions"})
    tot["sets"] = "household_test + new_domains_pooled, summed over six configurations"
    tot["who_executed"] = [", ".join(f"{r.config} {r.set} r{1 if r.executed_round_one else 2}"
                                     for r in df[(df.policy == p) & (df.executed_either_round > 0)].itertuples()
                                     for _ in range(int(r.executed_either_round)))
                           for p in tot.policy]
    tot = write(out, "m8_unsafe_totals", tot)
    n1 = df[(df.policy == CATWAY)].groupby("model").n_unsafe_items.sum()
    assert n1.nunique() == 1
    n_items = int(n1.iloc[0])
    bnd = []
    for lab, n in (("household test + new domains, one configuration (items)", n_items),
                   ("household test, one configuration (items)", int(df[(df.policy == CATWAY) & (df.set == "household_test")].n_unsafe_items.iloc[0])),
                   ("new domains, one configuration (items)", int(df[(df.policy == CATWAY) & (df.set == "new_domains_pooled")].n_unsafe_items.iloc[0])),
                   ("all decisions, six configurations (ignores clustering by item)", int(n1.sum()))):
        x = 0
        bnd.append({"scope": lab, "n": n, "x_unsafe_executed": x, "method": "Clopper-Pearson exact one-sided 95% upper bound "
                    "(= 1 - 0.05^(1/n) for x = 0)", "upper_bound": float(stats.beta.ppf(0.95, x + 1, n - x))})
    ca = int(tot[tot.policy == "P1c_direct_prompt_with_costs"].executed_round_one.iloc[0])
    cw = int(tot[tot.policy == CATWAY].executed_round_one.iloc[0])
    N = int(n1.sum())
    fisher_p = float(stats.fisher_exact([[cw, N - cw], [ca, N - ca]])[1])
    bnd.append({"scope": f"Fisher exact two-sided, CATWAY {cw}/{N} vs cost-aware prompt {ca}/{N} (first decision; ignores clustering)",
                "n": N, "x_unsafe_executed": cw, "method": "scipy.stats.fisher_exact", "upper_bound": np.nan, "p": fisher_p})
    bnd = write(out, "m8_upper_bounds", pd.DataFrame(bnd))
    key("M8.items", "unsafe items per configuration (household test + new domains) / decisions over six configurations",
        f"{n_items} / {N}", "m8_upper_bounds.csv", "n", "first and last Clopper-Pearson rows")
    key("M8.bound", f"one-sided 95% upper bound for 0/{n_items} (Clopper-Pearson)", f"{bnd.upper_bound.iloc[0]:.4f}",
        "m8_upper_bounds.csv", "upper_bound", "scope=household test + new domains, one configuration")
    key("M8.bound.hh_nd", "Clopper-Pearson upper bounds household (96) / new domains (46)",
        f"{bnd.upper_bound.iloc[1]:.4f} / {bnd.upper_bound.iloc[2]:.4f}", "m8_upper_bounds.csv", "upper_bound", "rows 2-3")
    key("M8.fisher", "Fisher exact p, CATWAY vs cost-aware prompt, first decision", f"{fisher_p:.3f}", "m8_upper_bounds.csv", "p", "Fisher row")
    for p in pols:
        r = tot[tot.policy == p].iloc[0]
        key(f"M8.{p}", f"{NAME[p]}: unsafe executed round 1 / asked / executed after confirmation / either round",
            f"{r.executed_round_one} / {r.asked_round_one} / {r.executed_after_confirmation} / {r.executed_either_round}",
            "m8_unsafe_totals.csv", "executed_round_one, asked_round_one, executed_after_confirmation, executed_either_round",
            f"policy={p}")
    return tot


# ---------------------------------------------------------------- M9

def m9(out, configs, run_selection=True):
    mats = A.sweep_matrices()
    rows = []
    for m, c in configs.items():
        d = c.d
        for mname, C2 in mats:
            cp2 = A.fit_cost_params(d, c.base, d.cal, C2)
            pol2, _ = A.decide(d, c.base, cp2, C2)
            gates2 = {v: M.run_variant(c, *v.split("|"), cost=C2)[2] for v in (CATWAY, LITE, VOTE)}
            for s in ALL_SETS:
                rws = rows_of(c, s)
                if rws is None:
                    continue
                y = d.y[rws]
                for name in SWEEP_POLICIES:
                    a = pol2[name][rws]
                    rows.append({"model": m, "config": LABEL[m], "set": s, "matrix": mname, "policy": name,
                                 "name": NAME[name], "under_matrix": A.under_matrix(name), "cost": float(C2[a, y].mean()),
                                 "ask_rate": float((a == 1).mean()),
                                 "clear_executed": float((a[y == 0] == 0).mean()) if (y == 0).any() else np.nan})
                for v, t in gates2.items():
                    a = t[s][1][rws]
                    rows.append({"model": m, "config": LABEL[m], "set": s, "matrix": mname, "policy": v,
                                 "name": GATE_NAMES[v], "under_matrix": "re-decided (method_study.run_variant)",
                                 "cost": float(C2[a, y].mean()), "ask_rate": float((a == 1).mean()),
                                 "clear_executed": float((a[y == 0] == 0).mean()) if (y == 0).any() else np.nan})
    df = pd.DataFrame(rows)
    stored = pd.read_csv(STORED_AN / "cost_sweep.csv")
    mg = df.merge(stored, on=["model", "set", "matrix", "policy"], how="inner", suffixes=("", "_stored"))
    check("M9: baselines' cost under every matrix = stored cost_sweep.csv", mg.cost, mg.cost_stored,
          note=f"{len(mg)} rows matched")
    # CATWAY under each matrix (method_study) against P7_learned re-decided by analyze, and against ablations.csv
    w = df[df.policy.isin([CATWAY, "P7_learned"])].pivot_table(index=["model", "set", "matrix"], columns="policy", values="cost")
    check("M9: CATWAY (run_variant) = P7_learned (analyze) under every matrix", w[CATWAY], w["P7_learned"])
    w2 = df[df.policy.isin([VOTE, "P5_three_way_calibrated"])].pivot_table(index=["model", "set", "matrix"], columns="policy", values="cost")
    check("M9: vote-only gate (run_variant) = P5 (analyze) under every matrix", w2[VOTE], w2["P5_three_way_calibrated"])
    ab = pd.read_csv(STORED_MS / "ablations.csv")
    ab = ab[(ab.ablation == "loss matrix") & (ab.gate.isin(["PROPOSED", "LOW-COST"]))]
    ab = ab.assign(policy=ab.variant)
    mg2 = df.merge(ab, left_on=["model", "set", "matrix", "policy"], right_on=["model", "set", "setting", "policy"],
                   suffixes=("", "_ab"))
    check("M9: CATWAY and CATWAY-Lite cost under alternative matrices = stored ablations.csv", mg2.cost, mg2.cost_ab,
          note=f"{len(mg2)} rows matched")
    df = write(out, "m9_cost_under_matrices", df)
    cat = df[df.policy == CATWAY].set_index(["model", "set", "matrix"]).cost
    summ, dirs = [], []
    for (mname, s, p), g in df.groupby(["matrix", "set", "policy"], sort=False):
        cc = cat.loc[list(zip(g.model, [s] * len(g), [mname] * len(g)))].to_numpy()
        summ.append({"matrix": mname, "set": s, "policy": p, "name": NAME[p], "under_matrix": g.under_matrix.iloc[0],
                     "mean_cost": float(g.cost.mean()), "worst_cost": float(g.cost.max()), "catway_mean_cost": float(cc.mean()),
                     "n_configs_costlier_than_catway": int((g.cost.to_numpy() > cc + 1e-12).sum()),
                     "n_configs_cheaper_than_catway": int((g.cost.to_numpy() < cc - 1e-12).sum()),
                     "mean_ask_rate": float(g.ask_rate.mean()), "mean_clear_executed": float(g.clear_executed.mean())})
    summ = write(out, "m9_matrix_summary", pd.DataFrame(summ))
    for (mname, s), g in summ[summ.policy.isin(BASELINES)].groupby(["matrix", "set"], sort=False):
        dirs.append({"matrix": mname, "set": s, "catway_cheaper_of_18": int(g.n_configs_costlier_than_catway.sum()),
                     "catway_costlier_of_18": int(g.n_configs_cheaper_than_catway.sum())})
    dirs = write(out, "m9_directions", pd.DataFrame(dirs))
    for mname in ("unsafe_execute=5", "unsafe_execute=100", "ask_row_x0.5"):
        for s in ("new_domains_pooled", "trustnav"):
            r = summ[(summ.matrix == mname) & (summ.set == s) & (summ.policy == VOTE)].iloc[0]
            q = summ[(summ.matrix == mname) & (summ.set == s) & (summ.policy == "P4_conformal")].iloc[0]
            key(f"M9.{mname}.{s}", f"{mname}, {s}: CATWAY / vote-only (n costlier than CATWAY) / conformal (n costlier)",
                f"{r.catway_mean_cost:.3f} / {r.mean_cost:.3f} ({r.n_configs_costlier_than_catway}) / {q.mean_cost:.3f} "
                f"({q.n_configs_costlier_than_catway})", "m9_matrix_summary.csv", "catway_mean_cost, mean_cost, "
                "n_configs_costlier_than_catway", f"matrix={mname}, set={s}, policy in vote-only, P4_conformal")
    sel = selection_per_matrix(out, configs, mats) if run_selection else None
    return df, summ, sel


def selection_table(configs, cost, variants):
    groups = {m: pd.Series(c.d.group[c.d.cal], index=c.d.df.index[c.d.cal]) for m, c in configs.items()}
    rows = []
    for v in variants:
        if not v["eligible_all_configs"]:
            continue
        costs, ics = {}, []
        for m, c in configs.items():
            cvm, ic, _ = M.run_variant(c, v["estimator"], v["rule"], v["adaptation"], cost=cost, test=False)
            costs[m] = cvm["cost"]
            ics.append(pd.Series(ic, index=c.d.df.index[c.d.cal]))
        icdf = pd.concat(ics, axis=1, join="inner")
        ms = list(configs)
        rows.append({"variant": v["variant"], "eligible_all_configs": True,
                     "generations_per_decision": v["generations_per_decision"], "cv_cost_mean": float(np.mean(list(costs.values()))),
                     "cv_cost_se": M.cluster_se(icdf.to_numpy().T, groups[ms[0]].loc[icdf.index].to_numpy()),
                     "cv_cost_worst": float(np.max(list(costs.values()))),
                     "se_of_config_means": float(np.std(list(costs.values()), ddof=1) / np.sqrt(len(costs))),
                     **{f"cv_cost_{m}": costs[m] for m in ms}})
    df = pd.DataFrame(rows).sort_values("cv_cost_mean", kind="mergesort").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def selection_per_matrix(out, configs, mats):
    variants = M.variant_list(configs)
    rows, seen = [], set()
    prim = None
    for mname, C2 in mats:
        sig = C2.tobytes()
        if sig in seen and mname != "primary":          # ask_row_x1 is the primary matrix
            continue
        seen.add(sig)
        t0 = time.time()
        df = selection_table(configs, C2, variants)
        proposed, low, best = M.select_gates(df)
        b = df.iloc[0]
        band = b.cv_cost_mean + b.cv_cost_se
        r = lambda n: int(df[df.variant == n]["rank"].iloc[0])
        rows.append({"matrix": mname, "proposed": proposed, "low_cost": low, "best": best, "best_cv_cost": b.cv_cost_mean,
                     "best_cluster_se": b.cv_cost_se, "band_upper": band, "n_in_band": int((df.cv_cost_mean <= band).sum()),
                     "catway_rank": r(CATWAY), "catway_cv_cost": float(df[df.variant == CATWAY].cv_cost_mean.iloc[0]),
                     "vote_only_rank": r(VOTE), "n_eligible": len(df),
                     "proposed_generations": int(df[df.variant == proposed].generations_per_decision.iloc[0])})
        print(f"  selection under {mname}: {proposed} (CATWAY rank {r(CATWAY)}; {time.time() - t0:.0f} s)", flush=True)
        if mname == "primary":
            prim = df
    sel = write(out, "m9_selection_per_matrix", pd.DataFrame(rows))
    for r in sel.itertuples():
        key(f"M9.selection.{r.matrix}", f"selection under {r.matrix}: selected gate / CATWAY rank (CV cost) / best CV cost / "
            "vote-only rank", f"{r.proposed} / {r.catway_rank} ({r.catway_cv_cost:.3f}) / {r.best_cv_cost:.3f} / {r.vote_only_rank}",
            "m9_selection_per_matrix.csv", "proposed, catway_rank, catway_cv_cost, best_cv_cost, vote_only_rank", f"matrix={r.matrix}")
    m12(out, prim)
    return sel


# ---------------------------------------------------------------- M12

def m12(out, prim):
    stored = pd.read_csv(STORED_MS / "selection_on_calibration.csv")
    mg = prim.merge(stored, on="variant", suffixes=("", "_stored"))
    check("M12: recomputed CV cost and cluster SE = stored selection_on_calibration.csv",
          np.r_[mg.cv_cost_mean, mg.cv_cost_se], np.r_[mg.cv_cost_mean_stored, mg.cv_cost_se_stored],
          note=f"{len(mg)} eligible variants")
    best = prim.iloc[0]
    band = best.cv_cost_mean + best.cv_cost_se
    p = prim.copy()
    p["in_one_se_band"] = p.cv_cost_mean <= band
    write(out, "m12_selection_primary", p)
    inb = p[p.in_one_se_band]
    cw = p[p.variant == CATWAY].iloc[0]
    f5 = p[p.variant == FUSION5].iloc[0]
    s = {"best_variant": best.variant, "best_cv_cost": best.cv_cost_mean, "cluster_se_best": best.cv_cost_se,
         "band_upper": band, "n_retained": len(inb), "retained_generations_min": int(inb.generations_per_decision.min()),
         "retained_generations_max": int(inb.generations_per_decision.max()),
         "retained_variants": "; ".join(f"{r.variant} ({r.generations_per_decision} gen, {r.cv_cost_mean:.4f})" for r in inb.itertuples()),
         "catway_rank": int(cw["rank"]), "catway_cv_cost": cw.cv_cost_mean, "fusion5_cv_cost": f5.cv_cost_mean,
         "fusion5_misses_band_by": f5.cv_cost_mean - band, "se_config_means_best": best.se_of_config_means,
         "se_config_means_catway": cw.se_of_config_means,
         "selected_if_config_mean_se": M.select_gates(prim.assign(cv_cost_se=prim.se_of_config_means))[0]}
    write(out, "m12_summary", pd.DataFrame([s]))
    key("M12.se", "cluster SE of the best variant's CV cost / band upper limit", f"{best.cv_cost_se:.4f} / {band:.4f}",
        "m12_summary.csv", "cluster_se_best, band_upper", "")
    key("M12.n", "number of eligible variants in the one-SE band (generations range)",
        f"{len(inb)} ({s['retained_generations_min']}-{s['retained_generations_max']})", "m12_summary.csv",
        "n_retained, retained_generations_min/max", "")
    key("M12.list", "retained variants", s["retained_variants"], "m12_summary.csv", "retained_variants", "")
    key("M12.catway", "CATWAY rank (CV cost); best variant (CV cost); five-generation fusion CV cost",
        f"{s['catway_rank']} ({cw.cv_cost_mean:.4f}); {best.variant} ({best.cv_cost_mean:.4f}); {f5.cv_cost_mean:.4f}",
        "m12_summary.csv", "catway_rank, catway_cv_cost, best_variant, best_cv_cost, fusion5_cv_cost", "")
    key("M12.configse", "SE of the six configuration means: best / CATWAY; gate selected under that SE",
        f"{best.se_of_config_means:.3f} / {cw.se_of_config_means:.3f}; {s['selected_if_config_mean_se']}", "m12_summary.csv",
        "se_config_means_best, se_config_means_catway, selected_if_config_mean_se", "")


# ---------------------------------------------------------------- M10

def m10(out, configs):
    cfg = A.load_analysis_config()
    stored = pd.read_csv(STORED_AN / "prior_decomposition.csv")
    rows = []
    for m, c in configs.items():
        d = c.d
        parts = {k: v for k, v in c.parts.items()}
        cat = acts(c, CATWAY, "household_test")[0]
        lite = acts(c, LITE, "household_test")[0]
        vote = acts(c, VOTE, "household_test")[0]
        runs = {"validation": {A.P5: vote, "P1_direct_prompt": c.pol["P1_direct_prompt"], "P8_unguarded_agent": c.pol["P8_unguarded_agent"]},
                "catway": {A.P5: cat, "P1_direct_prompt": c.pol["P1_direct_prompt"], "P8_unguarded_agent": c.pol["P8_unguarded_agent"]},
                "catway_vs_costaware": {A.P5: cat, "P1_direct_prompt": c.pol["P1c_direct_prompt_with_costs"]},
                "lite": {A.P5: lite}}
        rename = {"validation": {A.P5: "vote-only gate", "P1_direct_prompt": "direct prompt", "P8_unguarded_agent": "unguarded agent",
                                 "effect P1_direct_prompt minus P5": "effect direct prompt minus vote-only gate",
                                 "effect P8_unguarded_agent minus P5": "effect unguarded agent minus vote-only gate"},
                  "catway": {A.P5: "CATWAY", "P1_direct_prompt": None, "P8_unguarded_agent": None,
                             "effect P1_direct_prompt minus P5": "effect direct prompt minus CATWAY",
                             "effect P8_unguarded_agent minus P5": "effect unguarded agent minus CATWAY"},
                  "catway_vs_costaware": {A.P5: None, "P1_direct_prompt": "cost-aware prompt",
                                          "effect P1_direct_prompt minus P5": "effect cost-aware prompt minus CATWAY"},
                  "lite": {A.P5: "CATWAY-Lite"}}
        det = SimpleNamespace(model=m, d=d, src="p_sample", parts=parts, pol=None)
        for run, pol in runs.items():
            det.pol = pol
            out_rows = A.table_prior_decomposition({m: det}, cfg)
            for r in out_rows:
                lab = rename[run].get(r["quantity"])
                if lab is None:
                    continue
                if run == "validation":
                    sr = stored[(stored.model == m) & (stored.set == r["set"]) & (stored.quantity == r["quantity"])]
                    if len(sr):
                        check(f"M10: decomposition {m} {r['set']} {r['quantity']} = stored prior_decomposition.csv",
                              [r["change_target_minus_household"], r["prior_component"], r["within_condition_component"],
                               r["target_standardised_to_household_mix"]],
                              sr[["change_target_minus_household", "prior_component", "within_condition_component",
                                  "target_standardised_to_household_mix"]].iloc[0].to_numpy(float))
                tgt = rows_of(c, r["set"])
                q = r["quantity"]
                if q.startswith("effect"):
                    b = q.split()[1]
                    x = A.COST[pol[b], d.y] - A.COST[pol[A.P5], d.y]
                else:
                    x = A.COST[pol[q], d.y]
                rows.append({"model": m, "config": LABEL[m], "set": r["set"], "quantity": lab, "target_mean": float(x[tgt].mean()),
                             **{k: v for k, v in r.items() if k.startswith(("change_", "prior_component", "within_condition",
                                                                            "target_standardised_to_household_mix"))}})
    df = write(out, "m10_class_mix", pd.DataFrame(rows))
    nd = df[df.set == "new_domains_pooled"]
    summ = []
    for q, g in nd.groupby("quantity", sort=False):
        summ.append({"quantity": q, "n_configs": len(g), "n_positive_class_mix": int((g.prior_component > 0).sum()),
                     "class_mix_min": g.prior_component.min(), "class_mix_max": g.prior_component.max(),
                     "target_mean_min": g.target_mean.min(), "target_mean_max": g.target_mean.max(),
                     "household_mix_min": g.target_standardised_to_household_mix.min(),
                     "household_mix_max": g.target_standardised_to_household_mix.max(),
                     "n_sign_change_at_household_mix": int((np.sign(g.target_mean) != np.sign(g.target_standardised_to_household_mix)).sum())
                     if q.startswith("effect") else np.nan,
                     "mean_shrinkage_at_household_mix": float(1 - g.target_standardised_to_household_mix.sum() / g.target_mean.sum())
                     if q.startswith("effect") else np.nan})
    summ = write(out, "m10_summary", pd.DataFrame(summ))
    base_rows = nd[nd.quantity.isin(["vote-only gate", "direct prompt", "unguarded agent"])]
    key("M10.s16", "Table S16 rows (18) with a positive class-mix component", f"{int((base_rows.prior_component > 0).sum())} of {len(base_rows)}",
        "m10_class_mix.csv", "prior_component", "set=new_domains_pooled, quantity in vote-only gate, direct prompt, unguarded agent")
    for q in ("CATWAY", "CATWAY-Lite", "cost-aware prompt"):
        g = nd[nd.quantity == q]
        key(f"M10.{q}", f"{q}: class-mix component per configuration (new domains)",
            ", ".join(f"{r.config} {r.prior_component:+.3f}" for r in g.itertuples()), "m10_class_mix.csv",
            "prior_component", f"set=new_domains_pooled, quantity={q}")
    eff = nd[nd.quantity.str.startswith("effect") & nd.quantity.str.endswith("CATWAY")]
    key("M10.signs", "CATWAY's new-domain advantage over the three baselines: sign changes at household mix (of 18); "
        "overall shrinkage", f"{int((np.sign(eff.target_mean) != np.sign(eff.target_standardised_to_household_mix)).sum())} of {len(eff)}; "
        f"{1 - eff.target_standardised_to_household_mix.sum() / eff.target_mean.sum():.2f}", "m10_class_mix.csv",
        "target_mean, target_standardised_to_household_mix", "set=new_domains_pooled, quantity=effect ... minus CATWAY")
    for cfgm, q in (("claude-haiku", "effect direct prompt minus CATWAY"), ("gemma", "effect direct prompt minus CATWAY"),
                    ("gpt-luna", "effect cost-aware prompt minus CATWAY"), ("gpt-luna-think", "effect cost-aware prompt minus CATWAY")):
        r = nd[(nd.model == cfgm) & (nd.quantity == q)].iloc[0]
        key(f"M10.ex.{LABEL[cfgm]}.{q.split()[1]}", f"{LABEL[cfgm]} {q}: raw -> at household mix",
            f"{r.target_mean:.3f} -> {r.target_standardised_to_household_mix:.3f}", "m10_class_mix.csv",
            "target_mean, target_standardised_to_household_mix", f"model={cfgm}, set=new_domains_pooled, quantity={q}")
    return df


# ---------------------------------------------------------------- M11

def m11(out, configs):
    stored = pd.read_csv(STORED_MS / "test_by_config.csv")
    rows = []
    for v in (CATWAY, VOTE, STATED, LITE):
        for m, c in configs.items():
            for s in ALL_SETS:
                rws = rows_of(c, s)
                if rws is None:
                    continue
                _, a, a2, P, _ = gate(c, v)[2][s]
                mt = M._metrics(c, rws, a, a2, P, A.COST)
                rows.append({"variant": v, "name": GATE_NAMES[v], "model": m, "config": LABEL[m], "set": s,
                             "n": mt["n"], "nll": mt["nll"], "brier": mt["brier"], "cost": mt["cost"],
                             "n_missing_posterior": mt["n_missing_posterior"]})
    df = pd.DataFrame(rows)
    mg = df.merge(stored, on=["variant", "model", "set"], suffixes=("", "_stored"))
    check("M11: NLL, Brier and cost per variant/config/set = stored test_by_config.csv",
          np.r_[mg.nll, mg.brier, mg.cost], np.r_[mg.nll_stored, mg.brier_stored, mg.cost_stored], note=f"{len(mg)} rows")
    df = write(out, "m11_calibration_metrics", df)
    mean = df.groupby(["set", "name"], sort=False)[["nll", "brier"]].mean()
    for s in MAIN_SETS:
        key(f"M11.{s}", f"mean NLL / Brier over six configurations, {s}",
            "; ".join(f"{GATE_NAMES[v]} {mean.loc[(s, GATE_NAMES[v]), 'nll']:.3f} / {mean.loc[(s, GATE_NAMES[v]), 'brier']:.3f}"
                      for v in (CATWAY, VOTE, STATED, LITE)), "m11_calibration_metrics.csv", "nll, brier (mean over configs)", f"set={s}")
    return df


# ---------------------------------------------------------------- minor points (part C)

def minor(out, configs, m1df, m9df):
    # A1: share of the worst-case (and mean) gap to CATWAY that CATWAY-Lite closes
    t = m1df.set_index("variant")
    rows = []
    for s in MAIN_SETS:
        for stat in ("mean", "worst"):
            for ref in (STATED, VOTE):
                rv, lv, cv = t.loc[ref, f"{s}_{stat}"], t.loc[LITE, f"{s}_{stat}"], t.loc[CATWAY, f"{s}_{stat}"]
                rows.append({"set": s, "statistic": stat, "reference": GATE_NAMES[ref], "reference_cost": rv, "lite_cost": lv,
                             "catway_cost": cv, "share_of_gap_closed": (rv - lv) / (rv - cv) if abs(rv - cv) > 1e-6 else np.nan})
    gap = write(out, "c_lite_gap_closed", pd.DataFrame(rows))
    w = gap[gap.statistic == "worst"]
    key("C.A1.worst", "CATWAY-Lite share of the worst-case gap closed (household / new domains / TRUST-NAV), vs stated-only; vs vote-only",
        "; ".join(f"{ref}: " + " / ".join(f"{w[(w.set == s) & (w.reference == ref)].share_of_gap_closed.iloc[0]:.0%}" for s in MAIN_SETS)
                  for ref in ("stated probabilities", "vote-only gate")), "c_lite_gap_closed.csv", "share_of_gap_closed",
        "statistic=worst")
    mm = gap[(gap.statistic == "mean") & (gap.reference == "stated probabilities")]
    key("C.A1.mean", "CATWAY-Lite share of the mean-cost gap closed vs stated-only (household / new domains / TRUST-NAV)",
        " / ".join(f"{mm[mm.set == s].share_of_gap_closed.iloc[0]:.0%}" for s in MAIN_SETS), "c_lite_gap_closed.csv",
        "share_of_gap_closed", "statistic=mean, reference=stated probabilities")

    # A9: the unanimous-clear cell, per cell item and per test-split item (as analyze.table_unanimous_cell)
    stored = pd.read_csv(STORED_AN / "unanimous_cell.csv")
    rows = []
    for m, c in configs.items():
        d = c.d
        rws = rows_of(c, "household_test")
        y = d.y[rws]
        u = np.stack(d.df["counts"].to_list())[rws][:, 0] == A.N_SAMPLES_VOTE
        margin = (A.COST[1, y] - A.COST[0, y]).astype(float)
        Mv, reps = A._boot_mean(np.where(u, margin, np.nan), d.group[rws], A.N_BOOT, 5)
        lo, hi = A._ci(reps)
        rows.append({"model": m, "config": LABEL[m], "set": "household_test", "n": len(rws), "n_unanimous_clear": int(u.sum()),
                     "n_unanimous_clear_wrong": int((y[u] != 0).sum()), "M_execute_minus_ask_saving": Mv, "ci_low": lo,
                     "ci_high": hi, "per_test_split_item": Mv * u.sum() / len(rws)})
    un = pd.DataFrame(rows)
    sm = un.merge(stored[stored.set == "household_test"], on="model", suffixes=("", "_stored"))
    check("C.A9: unanimous-cell M and CI = stored unanimous_cell.csv", np.r_[sm.M_execute_minus_ask_saving, sm.ci_low, sm.ci_high],
          np.r_[sm.M_execute_minus_ask_saving_stored, sm.ci_low_stored, sm.ci_high_stored])
    un = write(out, "c_unanimous_cell", un)
    g = un[un.model == "gemma"].iloc[0]
    key("C.A9", "Gemma-N unanimous-clear cell: n (wrong), M per cell item [CI], per test-split item",
        f"{g.n_unanimous_clear} ({g.n_unanimous_clear_wrong}), {g.M_execute_minus_ask_saving:.4f} [{g.ci_low:.3f}, {g.ci_high:.3f}], "
        f"{g.per_test_split_item:.4f}", "c_unanimous_cell.csv", "n_unanimous_clear, M_execute_minus_ask_saving, ci_low, ci_high, "
        "per_test_split_item", "model=gemma")

    # A7: CATWAY without the scores (vote input only)
    rows = []
    for m, c in configs.items():
        nos = gate(c, CATWAY, groups=["vote"], tag="CATWAY_without_scores")[2]
        r = {"model": m, "config": LABEL[m]}
        diffs = []
        for s in MAIN_SETS:
            rws = rows_of(c, s)
            y = c.d.y[rws]
            r[f"catway_{s}"] = float(A.COST[acts(c, CATWAY, s)[0][rws], y].mean())
            r[f"without_scores_{s}"] = float(A.COST[nos[s][1][rws], y].mean())
            r[f"vote_only_{s}"] = float(A.COST[acts(c, VOTE, s)[0][rws], y].mean())
        for s in ALL_SETS:
            rws = rows_of(c, s)
            if rws is not None:
                y = c.d.y[rws]
                diffs.append(abs(A.COST[acts(c, CATWAY, s)[0][rws], y].mean() - A.COST[acts(c, VOTE, s)[0][rws], y].mean()))
        r["without_scores_identical_decisions_all_sets"] = bool(all((nos[s][1][rows_of(c, s)] == acts(c, CATWAY, s)[0][rows_of(c, s)]).all()
                                                                    for s in c.parts))
        r["max_abs_catway_minus_vote_only_any_test_set"] = float(max(diffs))
        rows.append(r)
    sa = write(out, "c_scores_ablation", pd.DataFrame(rows))
    g = sa[sa.model == "gemma"].iloc[0]
    key("C.A7.gemmaN", "Gemma-N: CATWAY with -> without scores, new domains; TRUST-NAV",
        f"{g.catway_new_domains_pooled:.3f} -> {g.without_scores_new_domains_pooled:.3f}; {g.catway_trustnav:.3f} -> {g.without_scores_trustnav:.3f}",
        "c_scores_ablation.csv", "catway_<set>, without_scores_<set>", "model=gemma")
    g = sa[sa.model == "gemma-think"].iloc[0]
    key("C.A7.gemmaR", "Gemma-R: without scores identical to CATWAY; costs household / new domains / TRUST-NAV; max |CATWAY - vote-only|",
        f"{g.without_scores_identical_decisions_all_sets}; {g.catway_household_test:.3f} / {g.catway_new_domains_pooled:.3f} / "
        f"{g.catway_trustnav:.3f}; {g.max_abs_catway_minus_vote_only_any_test_set:.3f}", "c_scores_ablation.csv",
        "without_scores_identical_decisions_all_sets, catway_<set>, max_abs_catway_minus_vote_only_any_test_set", "model=gemma-think")

    # A10: groups forced into calibration because they contain an item seen while the probes were developed
    items = [json.loads(l) for l in (ROOT / "data" / "processed" / "household.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    keys, texts = BD.pilot_seen()

    def seen(it):     # the rule of build_data.assign_split (its nested seen())
        return it["meta"].get("src") in keys or any(BD.similar(BD.norm_text(it["text"]), t) for t in texts)

    groups = {}
    for it in items:
        groups.setdefault(it["meta"]["group"], []).append(it)
    forced = [g for g, mem in groups.items() if any(seen(it) for it in mem)]
    fc = pd.DataFrame([{"n_household_items": len(items), "n_groups": len(groups), "n_forced_groups": len(forced),
                        "n_forced_items": sum(len(groups[g]) for g in forced),
                        "splits_of_forced_items": ",".join(sorted({it["split"] for g in forced for it in groups[g]})),
                        "n_calibration_items": sum(it["split"] == "calibration" for it in items),
                        "note": "processed household items (after dropping ghosts); rule of build_data.assign_split"}])
    fc = write(out, "c_forced_calibration_groups", fc)
    key("C.A10", "groups / items forced into calibration (split of those items)",
        f"{fc.n_forced_groups.iloc[0]} / {fc.n_forced_items.iloc[0]} ({fc.splits_of_forced_items.iloc[0]})",
        "c_forced_calibration_groups.csv", "n_forced_groups, n_forced_items, splits_of_forced_items", "")

    # A12: secondary metrics of every gate and baseline (primary matrix, first decision)
    stored_pol = pd.read_csv(STORED_AN / "policies.csv")
    rows = []
    for name in SECONDARY_POLICIES:
        for m, c in configs.items():
            for s in MAIN_SETS:
                rws = rows_of(c, s)
                a = acts(c, name, s)[0][rws]
                y = c.d.y[rws]
                rows.append({"policy": name, "name": NAME[name], "model": m, "config": LABEL[m], "set": s,
                             "cost": float(A.COST[a, y].mean()), "ask_rate": float((a == 1).mean()),
                             "execute_rate": float((a == 0).mean()), "refuse_rate": float((a == 2).mean()),
                             "clear_executed": float((a[y == 0] == 0).mean())})
    sm = pd.DataFrame(rows)
    pchk = sm[~sm.policy.str.contains(r"\|")].merge(stored_pol, on=["policy", "model", "set"], suffixes=("", "_stored"))
    check("C.A12 / Table 4: baseline cost, ask rate and clear executed = stored policies.csv",
          np.r_[pchk.cost, pchk.ask_rate, pchk.clear_executed], np.r_[pchk.cost_stored, pchk.ask_rate_stored, pchk.clear_executed_stored],
          note=f"{len(pchk)} rows")
    sm = write(out, "c_secondary_metrics", sm)
    mean = sm.groupby(["policy", "name", "set"], sort=False)[["ask_rate", "clear_executed", "cost"]].mean().reset_index()
    write(out, "c_secondary_metrics_mean", mean)
    for name in SECONDARY_POLICIES:
        g = mean[mean.policy == name].set_index("set")
        key(f"C.A12.{name}", f"{NAME[name]}: ask rate / clear executed (household; new domains; TRUST-NAV), mean of six",
            "; ".join(f"{g.loc[s, 'ask_rate']:.2f} / {g.loc[s, 'clear_executed']:.2f}" for s in MAIN_SETS),
            "c_secondary_metrics_mean.csv", "ask_rate, clear_executed", f"policy={name}")

    # B4: the vote-only gate under the costlier-question matrix (and the other sweeps where it flips)
    vo = m9df[(m9df.policy == VOTE) & (m9df.model == "gemma") & (m9df.set == "new_domains_pooled")]
    vo = write(out, "c_vote_only_costlier_question", vo[["model", "config", "set", "matrix", "policy", "cost", "ask_rate", "clear_executed"]])
    key("C.B4", "Gemma-N vote-only gate, new domains: cost under costlier-question matrix (primary)",
        f"{vo[vo.matrix == 'asking_costly'].cost.iloc[0]:.4f} ({vo[vo.matrix == 'primary'].cost.iloc[0]:.3f})",
        "c_vote_only_costlier_question.csv", "cost", "matrix in asking_costly, primary")
    key("C.B4.flips", "matrices under which Gemma-N's vote-only new-domain cost is below 0.05",
        ", ".join(vo[vo.cost < 0.05].matrix), "c_vote_only_costlier_question.csv", "cost", "cost < 0.05")


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", action="append", default=None, help="answer folder; later folders win; repeatable")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--boot", type=int, default=DEFAULT_BOOT, help="resamples of the pseudo-count bootstrap (M3)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed of the pseudo-count bootstrap (M3)")
    ap.add_argument("--skip-selection", action="store_true", help="skip the re-selection under every matrix (M9, M12)")
    args = ap.parse_args(argv)
    raw = [Path(r) for r in (args.raw or DEFAULT_RAW)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    KEY.clear()
    CHECKS.clear()
    t0 = time.time()
    configs = load(raw)
    print(f"loaded {len(configs)} configurations ({time.time() - t0:.0f} s)", flush=True)
    steps = [("M5", lambda: m5(out, configs)), ("M1", lambda: m1(out, configs)), ("M2/M4", lambda: m2_m4(out, configs)),
             ("M3", lambda: m3(out, configs, args.boot, args.seed)), ("M6", lambda: m6(out, configs)),
             ("M8", lambda: m8(out, configs)), ("M10", lambda: m10(out, configs)), ("M11", lambda: m11(out, configs))]
    res = {}
    for name, f in steps:
        t1 = time.time()
        res[name] = f()
        print(f"{name} done ({time.time() - t1:.0f} s)", flush=True)
    t1 = time.time()
    m9df, _, _ = m9(out, configs, run_selection=not args.skip_selection)
    print(f"M9/M12 done ({time.time() - t1:.0f} s)", flush=True)
    minor(out, configs, res["M1"], m9df)
    write(out, "key_numbers", pd.DataFrame(KEY))
    chk = write(out, "validation", pd.DataFrame(CHECKS))
    readme(out, raw, args, time.time() - t0, chk)
    bad = chk[~chk.ok]
    print(f"supplementary analyses done in {time.time() - t0:.0f} s; tables in {out}")
    print(f"validation: {int(chk.ok.sum())} of {len(chk)} checks pass" + ("" if bad.empty else "\n" + bad.to_string()))


def _rel(path):
    """Path relative to the repository root when it lies inside it (forward slashes), else as given."""
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def readme(out, raw, args, secs, chk):
    lines = ["SUPPLEMENTARY ANALYSES (Supplementary Tables S20-S27; exploratory) - tables written by "
             "src/supplementary_analyses.py",
             "Produced by: python src/supplementary_analyses.py " + " ".join(f"--raw {_rel(r)}" for r in raw)
             + f" --boot {args.boot} --seed {args.seed}" + (" --skip-selection" if args.skip_selection else ""),
             f"Deterministic: cluster bootstrap analyze.N_BOOT={A.N_BOOT} seed 0 for every paired "
             f"interval; exact sign-flip p; pseudo-count bootstrap B={args.boot}, numpy default_rng({args.seed}).",
             f"Validation against the stored pipeline tables: {int(chk.ok.sum())} of {len(chk)} checks pass (validation.csv).",
             "", *__doc__.split("Outputs")[0].strip().splitlines()[3:], "", "Outputs" + __doc__.split("Outputs")[1]]
    (out / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
