"""Ten-sample vote on the household calibration split, per configuration, round, source and condition.

    python src/vote_by_source.py --raw results/answers/calibration_gemma_think \
        --out results/tables_calibration_gemma_think/vote_by_source_condition_calibration.csv

Only complete sample sets (ten parsed answers, none stopped by the provider) are counted. Round 1: the
household calibration items; round 2: their clarification-round versions (an ambiguous item is clear
after the reply). Columns:
  n                                      items with a complete sample set
  share_correct_majority                 the condition named by more samples than any other is the true
                                         one (a tie for the most votes is not a majority)
  n_ties                                 items with a tie for the most votes
  share_correct_argmax_first_condition   the most-voted condition, ties broken by the order clear,
                                         ambiguous, infeasible, unsafe, is the true one
  unanimous_sets                         all ten samples name one condition
  unanimous_and_wrong                    unanimous and that condition is not the true one
  voted_ambiguous                        the majority condition is ambiguous
Rows with source and condition "all" pool each configuration and round.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402


def rows_for(model_file, items):
    model = A.model_name(model_file)
    df = A.load(model_file, items, model)
    byid = {it["id"]: it for it in items}
    out = []
    for iid, r in df.iterrows():
        c = np.asarray(r["counts"], float)
        if c.sum() != A.PROBES["cls_sample"][2]:
            continue
        it = byid[iid]
        y = A.CONDITIONS.index(it["condition"])
        top = c.max()
        tie = int((c == top).sum() > 1)
        maj = -1 if tie else int(np.argmax(c))
        out.append({"model": model, "round": 2 if it["split"] in A.R2_SPLITS else 1, "source": it["source"],
                    "condition": it["condition"], "n": 1, "correct_majority": int(maj == y), "n_ties": tie,
                    "correct_argmax": int(int(np.argmax(c)) == y), "unanimous_sets": int(top == c.sum()),
                    "unanimous_and_wrong": int(top == c.sum() and int(np.argmax(c)) != y),
                    "voted_ambiguous": int(maj == 1)})
    return out


def summarise(df, keys):
    g = df.groupby(keys, sort=True).sum(numeric_only=True).reset_index()
    return pd.DataFrame({
        **{k: g[k] for k in keys},
        "n": g["n"].astype(float),
        "share_correct_majority": (g["correct_majority"] / g["n"]).round(4),
        "n_ties": g["n_ties"].astype(float),
        "share_correct_argmax_first_condition": (g["correct_argmax"] / g["n"]).round(4),
        "unanimous_sets": g["unanimous_sets"].astype(float),
        "unanimous_and_wrong": g["unanimous_and_wrong"].astype(float),
        "voted_ambiguous": g["voted_ambiguous"].astype(float)})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", required=True, help="folder with <config>.jsonl or <config>.jsonl.gz answer files")
    ap.add_argument("--out", required=True, help="output CSV")
    args = ap.parse_args(argv)
    items = A.load_items(True)
    rows = [r for f in A.answer_files(args.raw) for r in rows_for(f, items)]
    df = pd.DataFrame(rows)
    by = summarise(df, ["model", "round", "source", "condition"])
    tot = summarise(df, ["model", "round"])
    tot.insert(2, "source", "all")
    tot.insert(3, "condition", "all")
    pd.concat([by, tot], ignore_index=True).to_csv(args.out, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
