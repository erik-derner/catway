# CATWAY: Cost-sensitive agentic three-way robot decisions with calibrated language-model confidence

Code, data, model answers and result tables for the paper *CATWAY: Cost-sensitive agentic three-way robot
decisions with calibrated language-model confidence*. A robot agent driven by a language model
must decide whether to execute an instruction, ask a clarifying question or refuse. CATWAY is a
cost-sensitive three-way decision gate: a loss matrix over four instruction conditions (clear, ambiguous,
infeasible, unsafe) fixes execute, ask and refuse regions on the probability simplex, with the clarifying
question as the information-gathering step, and the gate acts by the minimum-expected-loss rule. The
probabilities come from a logistic fusion of a vote over ten sampled answers with three graded scores of
ambiguity, infeasibility and risk; CATWAY-Lite replaces the vote by one stated probability vector. The
gate is fitted on household instructions, frozen, and tested in-domain, in two new robot domains and on
an external benchmark, against prompted, agent and fitted baselines. The repository contains the
benchmark (household instructions built from public sources, two authored robot domains and the
external TRUST-NAV set), the code that queries the models, all stored model answers, and the analysis
that produces every table and figure of the paper and its supplement from them.

## Layout

```
src/
  download_sources.py   fetch the public sources into data/raw/ (pinned revisions, SHA-256 checked)
  build_data.py         build the benchmark items in data/processed/
  elicit.py             query a model with every probe and store its answers
  checker.py            static checker of the skill calls in an agent's plan
  analyze.py            gates and baselines, calibration, statistics, transfer analyses (tables)
  method_study.py       the 120 gate variants, their selection on calibration data, ablations (tables)
  vote_by_source.py     vote statistics per source and condition on the calibration split
  supplementary_analyses.py   supplementary analyses (Supplementary Tables S20-S27; tables)
paper/                  scripts for the figures and the LaTeX tables of the paper (output in paper/output/)
config/                 models, loss matrices, item sets, analysed configurations, excluded items
data/
  domains/              domain specifications (robot, skills, locations, people, safety rules)
  items/                authored items: production cell, mobile robot, household infeasible
  review/               labels and review decisions for the public items (AmbiK: identifiers only)
  processed/            built items of the new domains and TRUST-NAV; SHA256SUMS of all nine built files
  object_names.yaml, reconciliation.yaml, text_patches.yaml   inputs of build_data.py
results/
  answers/                           answers of the six configurations on all sets (gzip JSON lines)
  tables_test_six_configs/           test-set tables (analyze.py --unblind)
  tables_calibration_six_configs/    calibration-split tables (cross-validated) and vote statistics
  tables_method_study/               variants, selection, test comparisons, ablations (README.txt there)
  supplementary_analyses/            supplementary analyses, Supplementary Tables S20-S27 (README.txt there)
tests/                  test suite (python -m pytest)
licenses/               licences of the redistributed third-party data
```

## Install

Python 3.12. `python -m pip install -r requirements.txt` (pinned versions; `httpx`, `anthropic` and
`openai` are needed only to query models).

## Rebuild the benchmark

AmbiK states no licence, so the household items, which contain AmbiK instructions and scenes, are not in
the repository. Download the four public sources from their official locations at the revisions used
and build all item files:

```
python src/download_sources.py      # data/raw/, six files, SHA-256 checked
python src/build_data.py            # data/processed/*.jsonl
python -m pytest tests/test_benchmark.py   # checks every built file against data/processed/SHA256SUMS
```

The build is deterministic; `data/processed/SHA256SUMS` lists the checksums of all nine built files
(`sha256sum -c SHA256SUMS` in `data/processed/` does the same check).

## Reproduce the tables and figures

The analysis needs the built items (previous section) and the stored answers; no model is queried.
Each command rewrites one folder of `results/`; afterwards `git status` shows no change.

```
python src/analyze.py --unblind --allow-missing-round2 --raw results/answers --tables results/tables_test_six_configs
python src/analyze.py --calibration-only --allow-missing-round2 --raw results/answers --tables results/tables_calibration_six_configs
python src/vote_by_source.py --raw results/answers --out results/tables_calibration_six_configs/vote_by_source_condition_calibration.csv
python src/method_study.py
python src/supplementary_analyses.py    # results/supplementary_analyses/ (next section)
python paper/make_figures.py            # Figures 3 to 6 and S1 -> paper/output/
python paper/make_tables.py             # LaTeX rows of the result tables of the main text (printed)
python paper/make_supplement_tables.py  # tables of Supplementary S3 -> paper/output/supp_tables.tex
```

The first command takes about an hour, the method study about 7 minutes, the supplementary analyses
about 5 minutes, the others a few minutes.
All resampling uses fixed seeds. `--allow-missing-round2` is needed for Gemma with reasoning, many of
whose clarification-round plans were cut off at the output limit; the affected episode-cost rows (its
unguarded agent and agent with checker on every set with a clarification round, and the uncalibrated
vote gate on the household test split) are marked `episode_incomplete`.

The six configurations are evaluated on every set. `config/analysis.yaml` gives Gemma with reasoning
the role `descriptive`: it is analysed and tabulated like the other five, and only its comparisons of
the vote-only gate with the direct prompt and the unguarded agent stay outside the Holm family of ten
(`primary_test.csv`) and are reported unadjusted (`primary_comparisons_descriptive.csv`).

## Supplementary analyses

```
python src/supplementary_analyses.py    # optional: --raw DIR [--raw DIR ...] --out DIR --boot 2000 --seed 20261115 --skip-selection
```

This reproduces Supplementary Tables S20-S27. It needs the built items (`data/processed/household.jsonl`,
see "Rebuild the benchmark") and the stored answers in `results/answers/`; no model is queried. The
module re-implements no decision: CATWAY, CATWAY-Lite and the vote-only gate are decided by
`method_study.run_variant` (fitted on the household calibration split and frozen), and the baselines by
the functions of `analyze.py`, with the paper's intervals (cluster bootstrap, 10,000 resamples, seed 0),
exact sign-flip tests and Holm adjustment. The analyses are exploratory and were added after the main
analyses; every row carries a `status` column saying so.

It writes `results/supplementary_analyses/`: CATWAY against the fitted gates and its identity with
`P7_learned`; the vote under other decision rules; comparisons per single new domain and on TRUST-NAV
under several Holm families; a bootstrap of the fitted pseudo-count; unsafe executions per configuration
with upper bounds; costs, comparison directions and re-selection of the gate under every alternative
loss matrix; a decomposition of the household-to-new-domain cost change into class mix and behaviour;
calibration metrics; secondary metrics (ask rate, share of clear instructions executed) of every gate
and baseline; and smaller checks. `key_numbers.csv` lists each quoted number with the file and column it
comes from, `validation.csv` compares the recomputed values with the stored tables of
`results/tables_test_six_configs/` and `results/tables_method_study/` (all checks must pass), and
`README.txt` describes every file. The run is deterministic; afterwards `git status` shows no change.

## Query a model

Set the API key of the provider in the environment (or in a `.env` file in the repository root, which
is ignored by git): `ANTHROPIC_API_KEY` (claude-haiku, claude-haiku-nothink), `OPENAI_API_KEY`
(gpt-luna, gpt-luna-think) or `OPENROUTER_API_KEY` (gemma, gemma-think). Model settings are in
`config/models.yaml`.

```
python src/elicit.py --models claude-haiku --sets production_cell production_cell_r2 --budget 5
python src/elicit.py --mock --sets production_cell     # synthetic answers, no API call
```

Answers are appended to `results/raw/<config>.jsonl`; a restarted run asks only what is missing. Every
record stores the SHA-256 of its prompt and request settings, and the analysis ignores answers whose
hash differs from the prompt the current code renders. Analyse with `--raw results/raw`.

## Names in the paper and in the code

| Paper | Code and tables |
|---|---|
| CATWAY | `F_vote_scores\|R_myopic\|A_none` (method study, gate `PROPOSED`); `P7_learned` in analyze.py (same decisions) |
| CATWAY-Lite | `F_verbal_scores\|R_myopic\|A_none` (method study, gate `LOW-COST`) |
| vote-only gate | `E_vote\|R_myopic\|A_none`; `P5_three_way_calibrated` in analyze.py |
| stated-only, token-only gate | `E_verbal\|R_myopic\|A_none`, `E_logit\|R_myopic\|A_none`; `P5_three_way_calibrated[p_verbal]`, `[p_logit]` |
| direct prompt | `P1_direct_prompt` |
| cost-aware prompt | `P1c_direct_prompt_with_costs` |
| unguarded agent | `P8_unguarded_agent` |
| agent with checker | `P9_agent_plus_checker` |
| rule cascade | `P2_cascade` |
| TRUST-NAV gate | `P3r_trustnav_runner_config` (defaults of the released decision module: `P3_trustnav_thresholds`) |
| conformal set rule | `P4_conformal` |
| top-probability threshold | `P4b_chow_threshold` |
| fuzzy rule base | `P6_fuzzy` |
| always execute | `P0_always_execute` |
| sequential rule with the vote | `P5s_sequential`; `R_sequential` in the method study |
| vote-only gate with the pseudo-count 1/2 (not fitted); refitted on the target (references) | `P5_uncalibrated`; `P5_oracle_refit`, `P5_fewlabel16_refit` |
| decision rules, adaptations | `R_myopic` (minimum expected loss), `R_sequential`, `R_conformal`, `R_riskcap`; `A_none`, `A_em`, `A_agreement` |
| vote, stated probabilities, token probabilities | `p_sample` (probe `cls_sample`), `p_verbal` (`verbal`), `p_logit` (`cls_logprob`) |
| graded scores, yes/no questions | probes `scores`; `bin_unsafe`, `bin_infeasible`, `bin_ambiguous` |
| Haiku-R, Haiku-N, Luna-R, Luna-N, Gemma-N, Gemma-R | `claude-haiku`, `claude-haiku-nothink`, `gpt-luna-think`, `gpt-luna`, `gemma`, `gemma-think` |
| household calibration split (cross-validated), household test split | `household_calibration_cv`, `household_test` |
| pooled new domains | `new_domains_pooled` (`production_cell` + `mobile_robot`) |
| TRUST-NAV with the completeness sentence | `trustnav_complete` |
| items used while the probe formats were developed (excluded) | pilot-seen items (`config/pilot_seen.yaml`; `*_incl_pilot_seen` rows) |

## Data sources and licences

| Source | Used for | Licence | In this repository |
|---|---|---|---|
| AmbiK (Ivanova et al., ACL 2025), github.com/cog-model/AmbiK-dataset | household clear and ambiguous items; kitchen scenes | none stated | item identifiers and our verdicts (`data/review/ambik_review.csv`); infeasible items adapted from AmbiK tasks stored as edits; text fetched by `download_sources.py` |
| SafeAgentBench (Yin et al., arXiv:2412.13178), huggingface.co/datasets/safeagentbench/SafeAgentBench | household clear and unsafe items | CC BY 4.0 | instructions with our labels in `data/review/household_unsafe_review.csv` (two texts patched, `data/text_patches.yaml`) |
| EmbodyGuard (Son et al., EMNLP 2025), github.com/Yonsei-MIR/EAI-safety | household unsafe items | MIT (`licenses/EmbodyGuard_MIT.txt`) | instructions with our labels in `data/review/embodyguard_review.csv` (texts normalised) |
| TRUST-NAV (Sobczak et al., arXiv:2609.37554), github.com/iitis/Risk-Aware-Semantic-Grounding | external test set; TRUST-NAV gate parameters | MIT (`licenses/TRUST-NAV_MIT.txt`) | built items `data/processed/trustnav*.jsonl` (labels mapped to our conditions) |

If you use these items, cite the original datasets as well.

## Licence

Code (`src/`, `paper/`, `tests/`): MIT, see `LICENSE`. Data and results created for this work (domain
specifications, authored items, labels and review decisions, configuration, model answers, result
tables): CC BY 4.0, see `LICENSE-DATA`, which lists the files covered. Third-party data stay under their
own licences (previous section).

## Citation

```
@unpublished{derner_catway,
  author = {Erik Derner},
  title  = {{CATWAY}: Cost-sensitive agentic three-way robot decisions with calibrated language-model confidence},
  note   = {Manuscript under review},
  year   = {2026}
}
```
