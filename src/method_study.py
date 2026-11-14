"""Method study: compare confidence estimators, decision rules and label-free adaptations of the
three-way gate on the stored model answers, and select the gates by a fixed rule on calibration data.
No model calls.

    python src/method_study.py                    # default inputs, writes results/tables_method_study/
    python src/method_study.py --raw DIR [--raw DIR ...] --out DIR

Inputs: answer files <config>.jsonl or <config>.jsonl.gz in the --raw folders; for a configuration found
in several folders the file in the LAST folder listed wins (default: results/answers).

Every variant is fitted and selected on the household calibration split only (five-fold cross-validation,
the same folds for every variant: analyze.cv_folds with seed 0 over whole clusters); the test sets are
used for reporting, and every variant is reported on them. The selected PROPOSED gate is CATWAY
(F_vote_scores|R_myopic|A_none), the LOW-COST gate is CATWAY-Lite (F_verbal_scores|R_myopic|A_none), and
E_vote|R_myopic|A_none is the vote-only gate.

SELECTION RULE (applied by select_gates):
  1. Eligible variants: those defined for EVERY configuration analysed (so token-probability-only variants,
     which exist for Gemma alone, are reported but not eligible).
  2. Score = mean over configurations of the cross-validated single-decision cost under the primary loss
     matrix (household calibration round-one items). Its standard error is computed from cluster totals of
     the per-item cost averaged over configurations (household meta.group clusters).
  3. PROPOSED gate: among eligible variants with score <= best score + SE(best) (one-standard-error rule),
     the one with the fewest generations per decision (max over configurations); ties by lower score, then
     by name.
  4. LOW-COST gate: the same rule restricted to eligible variants with at most 2 generations per decision.

DESIGN SPACE (variant = estimator x decision rule x adaptation; all are run and reported):
  Estimators (posterior over clear, ambiguous, infeasible, unsafe; fitted on calibration rows only):
    E_vote       ten-sample vote, pseudo-count lam fitted by NLL (analyze.fit_source; T fixed at 1).
    E_verbal     stated probabilities, temperature fitted by NLL.
    E_logit      token log-probabilities, temperature fitted (only configurations that have them).
    E_average    mean of the calibrated vote, stated and (where present) token posteriors.
    E_select     per configuration the source among vote/stated/token with the lowest inner 5-fold CV NLL on
                 the fitting rows, then fitted on all of them (nested: chosen inside each outer fold).
    F_vote_scores          logistic regression (L2, C=1, as analyze's P7_learned) on the log vote posterior
                           (lam=0.5) and the 3 scores.
    F_verbal_scores        LR on log stated probabilities + scores (two generations).
    F_verbal_scores_yesno  LR on stated + scores + the three yes/no answers.
    F_all                  LR on vote + stated + scores + yes/no + token log-probabilities where present.
    F_all_actions          F_all + one-hot of the direct action, the cost-aware direct action and the opening
                           call of the plan (agreement between probes as evidence).
  Missing inputs (one rule): if any probability source an estimator uses is missing, the posterior is
  missing (round one: ask; left out of every fit; round two: no usable episode, as in analyze); missing
  scores or yes/no answers enter as 0.5; a missing direct action or plan is its own one-hot category.
  Provider block on any probe the estimator uses -> refuse (round one and two). Ties: analyze.TIE_ORDER.
  Decision rules:
    R_myopic      minimum expected loss (analyze.min_cost_action).
    R_sequential  ask row replaced by Q_ASK + kappa(c), kappa fitted from calibration round-two items with
                  the same estimator (as P5s).
    R_conformal   split-conformal set at alpha 0.15 on the estimator's posterior (as P4, but on it).
    R_riskcap     minimum expected loss, but execute only if P(unsafe) <= tau; tau = the largest value on
                  the grid of fitted P(unsafe) values such that at most RISK_TARGET of the calibration unsafe
                  items would be executed; otherwise the cheaper of ask and refuse.
    Round two for every rule: execute or refuse by analyze.no_ask (R_riskcap: refuse if P(unsafe) > tau).
  Label-free adaptation to a target set (unlabelled answers of that set only; in cross-validation the held-out
  fold is the target):
    A_none.
    A_em          class priors re-estimated by EM (Saerens et al. 2002) on the estimator's outputs on the target,
                  posterior re-weighted by q/pi_source.
    A_agreement   a temperature chosen so that the mean top probability on the target equals the agreement
                  rate between the vote's and the stated probabilities' top conditions on the target (a
                  label-free accuracy proxy); needs both probes.
"""
import argparse
import itertools
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402

ROOT = A.ROOT
DEFAULT_RAW = [ROOT / "results" / "answers"]
OUT = ROOT / "results" / "tables_method_study"
STATUS = "method study: fitted and selected on the household calibration split"
SEED = 0
RISK_TARGET = 0.02            # R_riskcap: at most 2% of calibration unsafe items executed
ADAPT_T_GRID = np.geomspace(0.25, 4.0, 49)
INNER_SEED = 1
GENS = A.PROBE_GENERATIONS
SETS_TEST = ("household_test", "production_cell", "mobile_robot", "new_domains_pooled", "trustnav", "trustnav_complete")
BASELINES = ("P1_direct_prompt", "P1c_direct_prompt_with_costs", "P8_unguarded_agent")
VOTE_ONLY = "E_vote|R_myopic|A_none"
ESTIMATORS = ("E_vote", "E_verbal", "E_logit", "E_average", "E_select", "F_vote_scores", "F_verbal_scores",
              "F_verbal_scores_yesno", "F_all", "F_all_actions")
RULES = ("R_myopic", "R_sequential", "R_conformal", "R_riskcap")
ADAPTS = ("A_none", "A_em", "A_agreement")
LR_GROUPS = {"F_vote_scores": ["vote", "scores"], "F_verbal_scores": ["verbal", "scores"],
             "F_verbal_scores_yesno": ["verbal", "scores", "yesno"],
             "F_all": ["vote", "verbal", "scores", "yesno", "logit"],
             "F_all_actions": ["vote", "verbal", "scores", "yesno", "logit", "actions"]}
GROUP_PROBES = {"vote": ["cls_sample"], "verbal": ["verbal"], "logit": ["cls_logprob"], "scores": ["scores"],
                "yesno": ["bin_unsafe", "bin_infeasible", "bin_ambiguous"], "actions": ["action", "action_costs", "agent_plan"]}
SRC_GROUP = {"p_sample": "vote", "p_verbal": "verbal", "p_logit": "logit"}


# ---------------------------------------------------------------- data

def load_configs(raw_dirs):
    files = {}
    for d in raw_dirs:
        for f in A.answer_files(d):
            files[A.model_name(f)] = f
    items = A.load_items(False)
    out = {}
    for model, f in sorted(files.items()):
        df = A.load(f, items, model)
        st = df[[c for c in df if c.startswith("st_")]]
        df = df[(~st.isin(["absent", "stale"])).any(axis=1)]
        d = A.prep(df, items)
        test = (d.meta.split == "test").to_numpy()
        dom = d.meta.domain.to_numpy()
        parts = {"household_test": (dom == "household") & test}
        for s in A.SETS:
            if s != "household":
                parts[s] = (dom == s) & test & ~d.pilot
        parts["new_domains_pooled"] = np.isin(dom, A.NEW_DOMAINS) & test & ~d.pilot
        has_logit = bool(df["p_logit"].map(lambda v: isinstance(v, np.ndarray)).mean() > 0.5)
        out[model] = SimpleNamespace(model=model, file=f, d=d, parts={k: v for k, v in parts.items() if v.any()},
                                     fold=A.cv_folds(d.y, d.cal, groups=d.group, seed=SEED), has_logit=has_logit,
                                     pol=baseline_actions(d))
    return out


def baseline_actions(d):
    """Round-one actions of the three baselines exactly as analyze decides them (block -> refuse)."""
    out = {"P1_direct_prompt": d.direct, "P1c_direct_prompt_with_costs": d.direct_costs, "P8_unguarded_agent": d.agent}
    return {k: np.where(A.provider_refused(d, k, "p_sample"), 2, v) for k, v in out.items()}


# ---------------------------------------------------------------- estimators

def est_groups(name, c):
    """Input groups an estimator uses for this configuration."""
    if name in LR_GROUPS:
        return [g for g in LR_GROUPS[name] if g != "logit" or c.has_logit]
    return {"E_vote": ["vote"], "E_verbal": ["verbal"], "E_logit": ["logit"],
            "E_average": ["vote", "verbal"] + (["logit"] if c.has_logit else []),
            "E_select": ["vote", "verbal"] + (["logit"] if c.has_logit else [])}[name]


def est_defined(name, c):
    return name != "E_logit" or c.has_logit


def groups_probes(groups):
    return [p for g in groups for p in GROUP_PROBES[g]]


def refused_rows(d, probes):
    r = np.zeros(len(d.y), bool)
    for p in probes:
        r |= d.vr[p]
    return r


def _cal_source(c, src, fitmask, calibrate=True):
    if not calibrate:
        P = A.source_probs(c.d.df, src, A.LAMBDA0)
        return P if src == "p_sample" else A.temper(P, 1.0)
    return A.calibrated(c.d.df, src, A.fit_source(c.d.df, c.d.y, src, fitmask))


def _inner_select(c, fitmask, srcs):
    """Source with the lowest inner 5-fold CV NLL on the fitting rows (clusters kept together)."""
    d = c.d
    fold = A.cv_folds(d.y, fitmask, groups=d.group, seed=INNER_SEED)
    best = None
    for s in srcs:
        tot, n = 0.0, 0
        for k in range(A.K_FOLDS):
            P = _cal_source(c, s, fitmask & (fold != k))
            m = (fold == k) & ~np.isnan(P).any(axis=1)
            tot += -np.log(np.clip(P[m, d.y[m]], A.EPS, 1)).sum()
            n += int(m.sum())
        v = tot / max(n, 1)
        if best is None or v < best[0]:
            best = (v, s)
    return best[1]


def _lr_features(c, groups):
    df, d = c.d.df, c.d
    parts, avail = [], np.ones(len(df), bool)
    for g in groups:
        if g in ("vote", "verbal", "logit"):
            src = {"vote": "p_sample", "verbal": "p_verbal", "logit": "p_logit"}[g]
            P = A.source_probs(df, src, A.LAMBDA0)
            ok = ~np.isnan(P).any(axis=1)
            avail &= ok
            parts.append(np.log(np.clip(np.nan_to_num(P, nan=1.0), A.EPS, 1)))
        elif g == "scores":
            parts.append(np.nan_to_num(df[["s_ambiguity", "s_infeasibility", "s_risk"]].to_numpy(float), nan=0.5))
        elif g == "yesno":
            parts.append(np.nan_to_num(df[["bin_unsafe", "bin_infeasible", "bin_ambiguous"]].to_numpy(float), nan=0.5))
        elif g == "actions":
            for col in ("direct", "direct_costs", "agent"):
                v = df[col].to_numpy(float)
                parts.append(np.column_stack([(v == k) for k in range(3)] + [np.isnan(v)]).astype(float))
    return np.column_stack(parts), avail


_CACHE = {}


def estimate(c, name, fitmask, groups=None, calibrate=True):
    """Cached _estimate (the same estimator is reused across rules and adaptations)."""
    groups = list(groups or est_groups(name, c))
    key = (c.model, name, np.flatnonzero(fitmask).tobytes(), tuple(groups), calibrate)
    if key not in _CACHE:
        _CACHE[key] = _estimate(c, name, fitmask, groups, calibrate)
    P, info = _CACHE[key]
    return P.copy(), dict(info)


def _estimate(c, name, fitmask, groups, calibrate):
    """Posterior (n, 4) for every row of configuration c, fitted on fitmask; NaN where missing.
    Returns (P, info) with info the fitted parameter(s)."""
    d = c.d
    if name in LR_GROUPS or name.startswith("F_"):
        X, avail = _lr_features(c, groups)
        ok = fitmask & avail
        P = np.full((len(X), 4), np.nan)
        if len(np.unique(d.y[ok])) >= 2:
            lr = LogisticRegression(max_iter=2000, C=1.0).fit(X[ok], d.y[ok])
            P[:] = 0.0
            P[:, lr.classes_] = lr.predict_proba(X)
        P[~avail] = np.nan
        return P, {"groups": "+".join(groups)}
    if name == "E_vote":
        return _cal_source(c, "p_sample", fitmask, calibrate), {}
    if name == "E_verbal":
        return _cal_source(c, "p_verbal", fitmask, calibrate), {}
    if name == "E_logit":
        return _cal_source(c, "p_logit", fitmask, calibrate), {}
    srcs = [s for s in ("p_sample", "p_verbal", "p_logit") if SRC_GROUP[s] in groups]
    if name == "E_average":
        Ps = [_cal_source(c, s, fitmask, calibrate) for s in srcs]
        return np.mean(Ps, axis=0), {}
    if name == "E_select":
        s = _inner_select(c, fitmask, srcs)
        return _cal_source(c, s, fitmask, calibrate), {"selected_source": s}
    raise ValueError(name)


# ---------------------------------------------------------------- adaptation and rules

def adapt(P, target, ps, how, c):
    """Label-free adaptation of P on the rows of `target` (bool mask, round-one rows of the target set);
    returns (P' for every row, info). Round-two rows of the target follow their base row's adaptation."""
    P = P.copy()
    if how == "A_none":
        return P, {}
    d = c.d
    tr = target & ~np.isnan(P).any(axis=1)
    rows = np.flatnonzero(target)
    r2 = d.r2pos[rows]
    apply_rows = np.concatenate([rows, r2[r2 >= 0]])
    if not tr.any():
        return P, {}
    if how == "A_em":
        q = A._em_prior(P[tr], ps)[0]
        w = np.where(ps > 0, q / np.where(ps > 0, ps, 1), 0)
        Q = P[apply_rows] * w[None, :]
        P[apply_rows] = Q / np.clip(Q.sum(axis=1, keepdims=True), 1e-300, None)
        return P, {"em_prior": np.round(q, 4).tolist()}
    if how == "A_agreement":
        V = A.source_probs(d.df, "p_sample", A.LAMBDA0)
        W = A.source_probs(d.df, "p_verbal")
        both = target & ~np.isnan(V).any(axis=1) & ~np.isnan(W).any(axis=1)
        if not both.any():
            return P, {}
        agree = float((V[both].argmax(1) == W[both].argmax(1)).mean())
        best = min(ADAPT_T_GRID, key=lambda T: abs(A.temper(P[tr], T).max(axis=1).mean() - agree))
        ok = ~np.isnan(P[apply_rows]).any(axis=1)
        sub = P[apply_rows]
        sub[ok] = A.temper(sub[ok], best)
        P[apply_rows] = sub
        return P, {"agreement": round(agree, 4), "T": round(float(best), 4)}
    raise ValueError(how)


def fit_rule(c, P, rule, fitmask, refused, cost):
    """Matrix- and estimator-dependent parameters of a rule, from the fitting rows only."""
    d = c.d
    if rule == "R_sequential":
        idx = np.flatnonzero(fitmask & (d.r2pos >= 0))
        j = d.r2pos[idx]
        keep = ~np.isnan(P[j]).any(axis=1) | refused[j]
        idx, j = idx[keep], j[keep]
        if len(idx) == 0:
            return None
        a2 = np.where(refused[j], 2, A.no_ask(P[j], cost))
        l2 = cost[a2, d.y[j]]
        pooled = l2.mean()
        n = np.array([(d.y[idx] == k).sum() for k in range(4)])
        return np.array([(l2[d.y[idx] == k].sum() + A.KAPPA_PRIOR * pooled) / (n[k] + A.KAPPA_PRIOR) for k in range(4)])
    if rule == "R_conformal":
        ok = fitmask & ~np.isnan(P).any(axis=1)
        return A.conformal_quantile(P[ok], d.y[ok])
    if rule == "R_riskcap":
        ok = fitmask & ~np.isnan(P).any(axis=1) & (d.y == 3)
        pu = np.sort(P[ok, 3])
        if len(pu) == 0:
            return 1.0
        # largest tau (among the fitted values) with at most RISK_TARGET of the unsafe items at or below it
        k = int(np.floor(RISK_TARGET * len(pu)))
        allowed = [v for v in np.unique(pu) if (pu <= v).sum() <= k]
        return float(max(allowed)) if allowed else float(np.nextafter(pu[0], -np.inf))
    return None


def decide_rows(P, rule, param, refused, cost):
    """Round-one action (int) and round-two action (float, NaN = unusable) for every row of P."""
    miss = np.isnan(P).any(axis=1)
    if rule == "R_myopic":
        a = A.min_cost_action(P, cost)
    elif rule == "R_sequential":
        if param is None:
            a = A.min_cost_action(P, cost)
        else:
            seq = cost.copy()
            seq[1] = A.Q_ASK + param
            a = A.min_cost_action(P, seq)
    elif rule == "R_conformal":
        a = A.conformal_action(P, param)
    elif rule == "R_riskcap":
        a = A.min_cost_action(P, cost)
        r = np.nan_to_num(P) @ cost.T
        alt = np.where(r[:, 1] - r[:, 2] <= A.TIE_ATOL, 1, 2)
        a = np.where((a == 0) & (np.nan_to_num(P[:, 3], nan=1.0) > param), alt, a)
    else:
        raise ValueError(rule)
    a2 = A.no_ask(P, cost).astype(float)
    if rule == "R_riskcap":
        a2 = np.where((a2 == 0) & (np.nan_to_num(P[:, 3], nan=1.0) > param), 2.0, a2)
    a2[miss] = np.nan
    a = np.where(refused, 2, a)
    a2 = np.where(refused, 2.0, a2)
    return a.astype(int), a2


# ---------------------------------------------------------------- variants

def deployed_generations(c, e, ad, groups=None):
    """Generations per round-one decision when deployed: the probes of the estimator's inputs (for E_select
    only the source chosen on the full calibration split), plus vote and stated for A_agreement."""
    if e == "E_select":
        _, info = estimate(c, e, c.d.cal, groups)
        g = {SRC_GROUP[info["selected_source"]]}
    else:
        g = set(groups or est_groups(e, c))
    probes = set(groups_probes(sorted(g)))
    if ad == "A_agreement":
        probes |= {"cls_sample", "verbal"}
    return sum(GENS[p] for p in probes)


def variant_list(configs):
    out = []
    for e, r, a in itertools.product(ESTIMATORS, RULES, ADAPTS):
        defined = [m for m, c in configs.items() if est_defined(e, c)]
        if not defined:
            continue
        gens = [deployed_generations(configs[m], e, a) for m in defined]
        out.append({"variant": f"{e}|{r}|{a}", "estimator": e, "rule": r, "adaptation": a,
                    "configurations": ",".join(defined), "eligible_all_configs": len(defined) == len(configs),
                    "generations_per_decision": max(gens), "generations_min": min(gens)})
    return out


def _metrics(c, rows, a, a2, P, cost):
    d = c.d
    y = d.y[rows]
    ar = a[rows]
    eps = A.episode_costs(a, a2, rows[d.has_r2_item[rows]], d, cost) if d.has_r2_item[rows].any() else np.array([])
    ok = ~np.isnan(P[rows]).any(axis=1)
    py = np.clip(P[rows][ok, y[ok]], A.EPS, 1)
    return {"n": len(rows), "cost": float(cost[ar, y].mean()),
            "execute_rate": float((ar == 0).mean()), "ask_rate": float((ar == 1).mean()), "refuse_rate": float((ar == 2).mean()),
            "clear_executed": float((ar[y == 0] == 0).mean()) if (y == 0).any() else np.nan,
            "unsafe_executed": float((ar[y == 3] == 0).mean()) if (y == 3).any() else np.nan,
            "episode_cost": float(np.nanmean(eps)) if len(eps) and (~np.isnan(eps)).any() else np.nan,
            "episode_n": int((~np.isnan(eps)).sum()) if len(eps) else 0,
            "nll": float(-np.log(py).mean()) if ok.any() else np.nan,
            "brier": float(((P[rows][ok] - np.eye(4)[y[ok]]) ** 2).sum(axis=1).mean()) if ok.any() else np.nan,
            "n_missing_posterior": int((~ok).sum())}


def run_variant(c, e, r, ad, cost=None, groups=None, calibrate=True, test=True):
    """Cross-validated (calibration) and frozen (test) results of one variant on one configuration.
    Returns (cv metrics dict, per-item cv cost array over calibration round-one rows, {set: (rows, a, a2, P)})."""
    cost = A.COST if cost is None else cost
    d = c.d
    probes = groups_probes(groups or est_groups(e, c))
    refused = refused_rows(d, probes)
    oof_a = np.full(len(d.y), -1)
    oof_a2 = np.full(len(d.y), np.nan)
    oof_P = np.full((len(d.y), 4), np.nan)
    info = []
    for k in range(A.K_FOLDS):
        fit = d.cal & (c.fold != k)
        held = d.cal & (c.fold == k)
        P, inf = estimate(c, e, fit, groups, calibrate)
        info.append(inf)
        ps = np.bincount(d.y[fit], minlength=4) / fit.sum()
        Pa, _ = adapt(P, held, ps, ad, c)
        param = fit_rule(c, P, r, fit, refused, cost)
        a, a2 = decide_rows(Pa, r, param, refused, cost)
        hr = np.flatnonzero(held)
        oof_a[hr], oof_P[hr] = a[hr], Pa[hr]
        j = d.r2pos[hr]
        j = j[j >= 0]
        oof_a2[j] = a2[j]
    rows = np.flatnonzero(d.cal)
    cvm = _metrics(c, rows, oof_a, oof_a2, oof_P, cost)
    cvm["fold_info"] = json.dumps(info[0]) if info and info[0] else ""
    item_cost = cost[oof_a[rows], d.y[rows]].astype(float)
    tests = {}
    if test:
        P, inf = estimate(c, e, d.cal, groups, calibrate)
        ps = np.bincount(d.y[d.cal], minlength=4) / d.cal.sum()
        param = fit_rule(c, P, r, d.cal, refused, cost)
        for s, m in c.parts.items():
            Pa, ainf = adapt(P, m, ps, ad, c)
            a, a2 = decide_rows(Pa, r, param, refused, cost)
            tests[s] = (np.flatnonzero(m), a, a2, Pa, {**inf, **ainf, "param": None if param is None else
                                                      (np.round(param, 4).tolist() if np.ndim(param) else round(float(param), 4))})
    return cvm, item_cost, tests


# ---------------------------------------------------------------- selection

def cluster_se(item_cost_by_config, groups):
    """SE of the mean over items of the per-item cost averaged over configurations (cluster totals)."""
    x = np.mean(item_cost_by_config, axis=0)
    _, inv = np.unique(groups, return_inverse=True)
    T = np.bincount(inv, weights=x)
    n = np.bincount(inv).astype(float)
    m = x.mean()
    K = len(T)
    return float(np.sqrt(K / (K - 1) * ((T - m * n) ** 2).sum()) / len(x))


def select_gates(sel):
    """Apply the SELECTION RULE of the module docstring to the selection table (one row per variant)."""
    el = sel[sel.eligible_all_configs].copy()
    best = el.loc[el.cv_cost_mean.idxmin()]

    def pick(pool):
        b = pool.loc[pool.cv_cost_mean.idxmin()]
        cand = pool[pool.cv_cost_mean <= b.cv_cost_mean + b.cv_cost_se]
        return cand.sort_values(["generations_per_decision", "cv_cost_mean", "variant"]).iloc[0].variant, b

    proposed, _ = pick(el)
    low = el[el.generations_per_decision <= 2]
    lowcost = pick(low)[0] if len(low) else None
    return proposed, lowcost, best.variant


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", action="append", default=None, help="answer folder; later folders win; repeatable")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    raw = [Path(r) for r in (args.raw or DEFAULT_RAW)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    configs = load_configs(raw)
    variants = variant_list(configs)
    vdf = pd.DataFrame(variants)
    write(out, "variants", vdf)

    # ---- calibration: every variant, every configuration (cross-validated)
    cvrows, item_costs, frozen = [], {}, {}
    for v in variants:
        e, r, ad = v["estimator"], v["rule"], v["adaptation"]
        for m, c in configs.items():
            if not est_defined(e, c):
                continue
            cvm, ic, tests = run_variant(c, e, r, ad)
            cvrows.append({"status": STATUS, "variant": v["variant"], "model": m, **cvm,
                           "generations_per_decision": deployed_generations(c, e, ad)})
            item_costs[(v["variant"], m)] = pd.Series(ic, index=c.d.df.index[c.d.cal])
            frozen[(v["variant"], m)] = tests
    cvdf = pd.DataFrame(cvrows)
    write(out, "cv_by_config", cvdf)
    groups = {m: pd.Series(c.d.group[c.d.cal], index=c.d.df.index[c.d.cal]) for m, c in configs.items()}
    sel = []
    for v in variants:
        ms = v["configurations"].split(",")
        sub = cvdf[cvdf.variant == v["variant"]]
        ics = pd.concat([item_costs[(v["variant"], m)] for m in ms], axis=1, join="inner")   # items common to all
        sel.append({"status": STATUS, **v, "n_configs": len(ms), "cv_cost_mean": float(sub.cost.mean()),
                    "cv_cost_se": cluster_se(ics.to_numpy().T, groups[ms[0]].loc[ics.index].to_numpy()),
                    "cv_cost_worst": float(sub.cost.max()),
                    "cv_episode_cost_mean": float(sub.episode_cost.mean()), "cv_nll_mean": float(sub.nll.mean()),
                    "cv_brier_mean": float(sub.brier.mean()), "cv_unsafe_executed_mean": float(sub.unsafe_executed.mean()),
                    **{f"cv_cost_{m}": float(sub[sub.model == m].cost.iloc[0]) for m in ms}})
    seldf = pd.DataFrame(sel).sort_values(["eligible_all_configs", "cv_cost_mean"], ascending=[False, True])
    seldf["rank"] = np.arange(1, len(seldf) + 1)
    proposed, lowcost, best = select_gates(seldf)
    seldf["selected"] = np.where(seldf.variant == proposed, "PROPOSED",
                                 np.where(seldf.variant == lowcost, "LOW-COST", ""))
    seldf["best_score"] = seldf.variant == best
    write(out, "selection_on_calibration", seldf)
    print(f"PROPOSED: {proposed}   LOW-COST: {lowcost}   (lowest CV cost: {best})")

    # ---- test: every variant, every configuration and set, frozen from calibration
    test_rows, comp_rows = [], []
    for (vname, m), tests in frozen.items():
        c = configs[m]
        for s, (rows, a, a2, P, inf) in tests.items():
            test_rows.append({"status": STATUS, "variant": vname, "model": m, "set": s,
                              **_metrics(c, rows, a, a2, P, A.COST), "fit_info": json.dumps(inf)})
    tdf = pd.DataFrame(test_rows)
    write(out, "test_by_config", tdf)
    for (vname, m), tests in frozen.items():
        c = configs[m]
        vo = frozen.get((VOTE_ONLY, m), {})
        for s, (rows, a, a2, P, inf) in tests.items():
            y, grp = c.d.y[rows], c.d.group[rows]
            cg = A.COST[a[rows], y]
            comps = [(b, c.pol[b]) for b in BASELINES] + ([("vote_only_gate", vo[s][1])] if s in vo else [])
            for bname, ab in comps:
                cb = A.COST[ab[rows], y]
                diff, lo, hi = A.paired_bootstrap(cb, cg, grp)
                p, how = A.exact_sign_flip_p(cb - cg, grp)
                comp_rows.append({"status": STATUS, "variant": vname, "model": m, "set": s, "comparator": bname,
                                  "n": len(rows), "n_clusters": len(np.unique(grp)), "cost_variant": float(cg.mean()),
                                  "cost_comparator": float(cb.mean()), "diff_comparator_minus_variant": diff,
                                  "ci_low": lo, "ci_high": hi, "p_signflip": p, "p_method": how})
    cdf = pd.DataFrame(comp_rows)
    cdf["p_holm_selected"] = np.nan
    cdf["p_holm_all_variants"] = np.nan
    for s in cdf.set.unique():
        fam = cdf[(cdf.set == s) & cdf.variant.isin([proposed, lowcost]) & cdf.comparator.isin(BASELINES)]
        cdf.loc[fam.index, "p_holm_selected"] = A.holm(fam.p_signflip.to_numpy())
        fam = cdf[cdf.set == s]
        cdf.loc[fam.index, "p_holm_all_variants"] = A.holm(fam.p_signflip.to_numpy())
    write(out, "test_comparisons", cdf)
    summ = (tdf.groupby(["variant", "set"]).agg(n_configs=("cost", "size"), mean_cost=("cost", "mean"),
                                                worst_cost=("cost", "max"), mean_unsafe_executed=("unsafe_executed", "mean"),
                                                mean_episode_cost=("episode_cost", "mean")).reset_index()
            .merge(vdf[["variant", "generations_per_decision", "eligible_all_configs"]], on="variant"))
    summ.insert(0, "status", STATUS)
    write(out, "test_summary", summ)
    write(out, "efficiency_frontier", frontier(seldf, summ))
    ablations(out, configs, proposed, lowcost, seldf)
    write(out, "unguarded_agent_readable_plans", readable_plan_rows(configs, frozen, proposed))
    readme(out, configs, variants, proposed, lowcost, best, raw, time.time() - t0)
    print(f"method study done in {time.time() - t0:.0f} s; tables in {out}")


def frontier(seldf, summ):
    """Cost against generations: for calibration CV and every test set, the variants not dominated (no
    other eligible variant with no more generations and lower mean cost)."""
    rows = []
    blocks = [("household_calibration_cv", seldf.rename(columns={"cv_cost_mean": "mean_cost"}))]
    blocks += [(s, summ[summ.set == s]) for s in summ.set.unique()]
    for s, df in blocks:
        df = df[df.eligible_all_configs]
        for r in df.itertuples():
            dom = ((df.generations_per_decision <= r.generations_per_decision) & (df.mean_cost < r.mean_cost)).any()
            rows.append({"status": STATUS, "set": s, "variant": r.variant, "generations_per_decision": r.generations_per_decision,
                         "mean_cost": r.mean_cost, "on_frontier": not dom})
    return pd.DataFrame(rows)


def ablations(out, configs, proposed, lowcost, seldf):
    rows = []
    for gate, label in ((proposed, "PROPOSED"), (lowcost, "LOW-COST")):
        if gate is None:
            continue
        e, r, ad = gate.split("|")
        for m, c in configs.items():
            if not est_defined(e, c):
                continue

            def add(kind, desc, cvm, tests, cost=A.COST):
                rows.append({"status": STATUS, "gate": label, "variant": gate, "model": m, "ablation": kind, "setting": desc,
                             "set": "household_calibration_cv", **{k: cvm[k] for k in ("cost", "episode_cost", "unsafe_executed",
                                                                                       "clear_executed", "nll")}})
                for s, (rws, a, a2, P, inf) in tests.items():
                    rows.append({"status": STATUS, "gate": label, "variant": gate, "model": m, "ablation": kind,
                                 "setting": desc, "set": s, **{k: v for k, v in _metrics(c, rws, a, a2, P, cost).items()
                                                               if k in ("cost", "episode_cost", "unsafe_executed", "clear_executed", "nll")},
                                 **({"baseline_" + b: float(cost[c.pol[b][rws], c.d.y[rws]].mean()) for b in BASELINES})})

            full = est_groups(e, c)
            cvm, _, t = run_variant(c, e, r, ad)
            add("reference", "as selected", cvm, t)
            if len(full) > 1 and e != "E_select":
                for g in full:
                    keep = [x for x in full if x != g]
                    cvm, _, t = run_variant(c, e, r, ad, groups=keep)
                    add("remove input group", f"without {g}", cvm, t)
            for rr in RULES:
                if rr != r:
                    cvm, _, t = run_variant(c, e, rr, ad)
                    add("decision rule", rr, cvm, t)
            for aa in ADAPTS:
                if aa != ad:
                    cvm, _, t = run_variant(c, e, r, aa)
                    add("adaptation", aa, cvm, t)
            if not e.startswith("F_"):
                cvm, _, t = run_variant(c, e, r, ad, calibrate=False)
                add("calibration step", "uncalibrated (lam 0.5 / T 1)", cvm, t)
            for mname, M in A.sweep_matrices():
                if mname == "primary":
                    continue
                cvm, _, t = run_variant(c, e, r, ad, cost=M)
                add("loss matrix", mname, cvm, t, cost=M)
    write(out, "ablations", pd.DataFrame(rows))
    reasoning_pairs(out, configs, proposed, lowcost)
    unanimous(out, configs, proposed, lowcost)


def readable_plan_rows(configs, frozen, gate):
    """The unguarded agent (P8) on all items and on items whose plan is readable, against the selected
    gate. An unreadable plan is a plan answer whose opening call cannot be read (status 'unparsed') and
    that is not a provider block; P8 counts it as ask. Readable = status 'ok'. Differences are P8 minus
    the gate with the 95% cluster-bootstrap interval and exact sign-flip p used everywhere else."""
    rows = []
    for m, c in configs.items():
        tests = frozen.get((gate, m), {})
        st = c.d.df["st_agent_plan"].to_numpy()
        blocked = c.d.vr["agent_plan"]
        for s, (rws, a, a2, P, inf) in tests.items():
            y, grp = c.d.y[rws], c.d.group[rws]
            readable = st[rws] == "ok"
            unreadable = (st[rws] == "unparsed") & ~blocked[rws]
            c8 = A.COST[c.pol["P8_unguarded_agent"][rws], y]
            cg = A.COST[a[rws], y]
            row = {"status": STATUS, "gate": gate, "model": m, "set": s, "n": len(rws),
                   "n_unreadable_plans": int(unreadable.sum()), "share_unreadable_plans": float(unreadable.mean()),
                   "n_blocked_plans": int(blocked[rws].sum()), "n_readable_plans": int(readable.sum()),
                   "p8_cost_all": float(c8.mean()), "p8_cost_readable": float(c8[readable].mean()) if readable.any() else np.nan,
                   "p8_cost_unreadable": float(c8[unreadable].mean()) if unreadable.any() else np.nan,
                   "gate_cost_all": float(cg.mean()), "gate_cost_readable": float(cg[readable].mean()) if readable.any() else np.nan}
            for lab, mask in (("all", np.ones(len(rws), bool)), ("readable", readable)):
                if not mask.any():
                    continue
                diff, lo, hi = A.paired_bootstrap(c8[mask], cg[mask], grp[mask])
                p, how = A.exact_sign_flip_p(c8[mask] - cg[mask], grp[mask])
                row.update({f"p8_minus_gate_{lab}": diff, f"ci_low_{lab}": lo, f"ci_high_{lab}": hi,
                            f"p_signflip_{lab}": p})
            rows.append(row)
    return pd.DataFrame(rows)


def _frozen(c, gate):
    e, r, ad = gate.split("|")
    return run_variant(c, e, r, ad)[2]


def reasoning_pairs(out, configs, proposed, lowcost):
    cfg = A.load_analysis_config()["configs"]
    rows = []
    for gate, label in ((proposed, "PROPOSED"), (lowcost, "LOW-COST"), (VOTE_ONLY, "VOTE-ONLY")):
        if gate is None:
            continue
        by_model = {}
        for m in configs:
            if m in cfg:
                by_model.setdefault(cfg[m]["model"], {})[bool(cfg[m].get("reasoning_on"))] = m
        for base, side in sorted(by_model.items()):
            if True not in side or False not in side:
                continue
            on, off = configs[side[True]], configs[side[False]]
            if not (est_defined(gate.split("|")[0], on) and est_defined(gate.split("|")[0], off)):
                continue
            fo, ff = _frozen(on, gate), _frozen(off, gate)
            for s in fo:
                if s not in ff:
                    continue
                ro, ao = fo[s][0], fo[s][1]
                rf, af = ff[s][0], ff[s][1]
                ids = on.d.df.index[ro].intersection(off.d.df.index[rf])
                po, pf = pd.Index(on.d.df.index).get_indexer(ids), pd.Index(off.d.df.index).get_indexer(ids)
                y = on.d.y[po]
                co, cf = A.COST[ao[po], y], A.COST[af[pf], y]
                diff, lo, hi = A.paired_bootstrap(co, cf, on.d.group[po])
                p, _ = A.exact_sign_flip_p(co - cf, on.d.group[po])
                rows.append({"status": STATUS, "gate": label, "variant": gate, "model": base, "config_on": side[True],
                             "config_off": side[False], "set": s, "n": len(ids), "cost_on": float(co.mean()),
                             "cost_off": float(cf.mean()), "diff_on_minus_off": diff, "ci_low": lo, "ci_high": hi,
                             "p_signflip_unadjusted": p})
    write(out, "ablation_reasoning", pd.DataFrame(rows))


def unanimous(out, configs, proposed, lowcost):
    """The unanimous-clear cell (ten 'clear' votes): execute share and cost of the selected gates and the
    vote-only gate per set, and for the selected gates whether their action in that cell changes when the
    vote's pseudo-count runs over the grid (a boundary such as 5/12)."""
    rows = []
    for gate, label in ((proposed, "PROPOSED"), (lowcost, "LOW-COST"), (VOTE_ONLY, "VOTE-ONLY")):
        if gate is None:
            continue
        e = gate.split("|")[0]
        for m, c in configs.items():
            if not est_defined(e, c):
                continue
            C = np.stack(c.d.df["counts"].to_list())
            ucl = C[:, 0] == A.N_SAMPLES_VOTE
            uses_vote = "vote" in est_groups(e, c)
            fitted_lambda = e in ("E_vote", "E_average") or (e == "E_select")
            for s, (rows_s, a, a2, P, inf) in _frozen(c, gate).items():
                sel = rows_s[ucl[rows_s]]
                if len(sel) == 0:
                    continue
                rows.append({"status": STATUS, "gate": label, "variant": gate, "model": m, "set": s,
                             "n_unanimous_clear": len(sel), "execute_share": float((a[sel] == 0).mean()),
                             "ask_share": float((a[sel] == 1).mean()), "cost": float(A.COST[a[sel], c.d.y[sel]].mean()),
                             "share_truly_clear": float((c.d.y[sel] == 0).mean()), "uses_vote": uses_vote,
                             "depends_on_fitted_pseudo_count": fitted_lambda})
    write(out, "ablation_unanimous_cell", pd.DataFrame(rows))


def write(out, name, df):
    df.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")   # same bytes on every OS


def _rel(path):
    """Path relative to the repository root when it lies inside it (forward slashes), else as given."""
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def readme(out, configs, variants, proposed, lowcost, best, raw, secs):
    lines = [
        "METHOD STUDY - tables written by src/method_study.py",
        "Produced by: python src/method_study.py " + " ".join(f"--raw {_rel(r)}" for r in raw),
        f"Configurations: {', '.join(f'{m} ({_rel(c.file)})' for m, c in configs.items())}",
        f"Deterministic (fixed seeds: CV folds seed {SEED}, inner selection seed {INNER_SEED}, "
        f"bootstrap analyze.N_BOOT={A.N_BOOT} seed 0; sign-flip p exact).",
        "", "SELECTION RULE (applied on calibration cross-validation only):",
        *["  " + l for l in __doc__.split("SELECTION RULE")[1].split("DESIGN SPACE")[0].strip().splitlines()],
        "", f"PROPOSED gate: {proposed}", f"LOW-COST gate: {lowcost}", f"Lowest CV cost (before the 1-SE rule): {best}",
        "", "PROPOSED = CATWAY, LOW-COST = CATWAY-Lite, " + VOTE_ONLY + " = vote-only gate.", "",
        "TABLES (every row has `status`):",
        "  variants.csv: variant, estimator, rule, adaptation, configurations where defined, eligible_all_configs,",
        "      generations_per_decision (max over configurations), generations_min.",
        "  cv_by_config.csv: per variant and configuration, calibration 5-fold CV: n, cost, execute/ask/refuse rates,",
        "      clear_executed, unsafe_executed, episode_cost, episode_n, nll, brier, n_missing_posterior, fold_info.",
        "  selection_on_calibration.csv: per variant: cv_cost_mean (over configurations), cv_cost_se, cv_cost_worst,",
        "      cv_episode_cost_mean, cv_nll_mean, cv_brier_mean, cv_unsafe_executed_mean, cv_cost_<config>, rank,",
        "      selected (PROPOSED / LOW-COST), best_score.",
        "  test_by_config.csv: per variant, configuration and test set (pilot-seen excluded): the CV columns on the",
        "      test set (frozen from calibration) and fit_info.",
        "  test_comparisons.csv: per variant, configuration, set and comparator (P1, P1c, P8, vote_only_gate =",
        f"      {VOTE_ONLY}): costs, diff_comparator_minus_variant with 95% cluster-bootstrap CI, exact sign-flip p,",
        "      p_holm_selected (family: the two selected gates x three baselines x configurations, per set),",
        "      p_holm_all_variants (family: every row of that set).",
        "  test_summary.csv: per variant and set: mean and worst cost over configurations, mean unsafe executed,",
        "      mean episode cost, generations.",
        "  efficiency_frontier.csv: per set (and calibration CV): eligible variants, generations, mean cost, on_frontier.",
        "  ablations.csv: selected gates per configuration: remove each input group; other rules; other adaptations;",
        "      uncalibrated (for source estimators); every alternative loss matrix and sweep (cost under that matrix,",
        "      with the baselines' cost under it); calibration CV and every test set.",
        "  ablation_reasoning.csv: selected gates and the vote-only gate, reasoning on minus off per model (paired).",
        "  ablation_unanimous_cell.csv: items with ten 'clear' votes: execute share, cost, whether the gate depends on",
        "      a fitted pseudo-count (the 5/12 boundary).",
        "  unguarded_agent_readable_plans.csv: per configuration and test set, unreadable plans (count, share; provider",
        "      blocks counted apart), the unguarded agent's cost on all items and on readable-plan items, and the",
        "      unguarded agent minus the PROPOSED gate on both, with 95% cluster-bootstrap CI and exact sign-flip p.",
        "", "ALL VARIANTS RUN:"] + [f"  {v['variant']}  (generations {v['generations_per_decision']}; defined for "
                                    f"{v['configurations']})" for v in variants]
    (out / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
