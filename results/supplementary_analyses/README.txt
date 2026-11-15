SUPPLEMENTARY ANALYSES (Supplementary Tables S20-S27; exploratory) - tables written by src/supplementary_analyses.py
Produced by: python src/supplementary_analyses.py --raw results/answers --boot 2000 --seed 20261115
Deterministic: cluster bootstrap analyze.N_BOOT=10000 seed 0 for every paired interval; exact sign-flip p; pseudo-count bootstrap B=2000, numpy default_rng(20261115).
Validation against the stored pipeline tables: 134 of 134 checks pass (validation.csv).


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

