METHOD STUDY - tables written by src/method_study.py
Produced by: python src/method_study.py --raw results/answers
Configurations: claude-haiku (results/answers/claude-haiku.jsonl.gz), claude-haiku-nothink (results/answers/claude-haiku-nothink.jsonl.gz), gemma (results/answers/gemma.jsonl.gz), gemma-think (results/answers/gemma-think.jsonl.gz), gpt-luna (results/answers/gpt-luna.jsonl.gz), gpt-luna-think (results/answers/gpt-luna-think.jsonl.gz)
Deterministic (fixed seeds: CV folds seed 0, inner selection seed 1, bootstrap analyze.N_BOOT=10000 seed 0; sign-flip p exact).

SELECTION RULE (applied on calibration cross-validation only):
  (applied by select_gates):
    1. Eligible variants: those defined for EVERY configuration analysed (so token-probability-only variants,
       which exist for Gemma alone, are reported but not eligible).
    2. Score = mean over configurations of the cross-validated single-decision cost under the primary loss
       matrix (household calibration round-one items). Its standard error is computed from cluster totals of
       the per-item cost averaged over configurations (household meta.group clusters).
    3. PROPOSED gate: among eligible variants with score <= best score + SE(best) (one-standard-error rule),
       the one with the fewest generations per decision (max over configurations); ties by lower score, then
       by name.
    4. LOW-COST gate: the same rule restricted to eligible variants with at most 2 generations per decision.

PROPOSED gate: F_vote_scores|R_myopic|A_none
LOW-COST gate: F_verbal_scores|R_myopic|A_none
Lowest CV cost (before the 1-SE rule): F_all_actions|R_myopic|A_none

PROPOSED = CATWAY, LOW-COST = CATWAY-Lite, E_vote|R_myopic|A_none = vote-only gate.

TABLES (every row has `status`):
  variants.csv: variant, estimator, rule, adaptation, configurations where defined, eligible_all_configs,
      generations_per_decision (max over configurations), generations_min.
  cv_by_config.csv: per variant and configuration, calibration 5-fold CV: n, cost, execute/ask/refuse rates,
      clear_executed, unsafe_executed, episode_cost, episode_n, nll, brier, n_missing_posterior, fold_info.
  selection_on_calibration.csv: per variant: cv_cost_mean (over configurations), cv_cost_se, cv_cost_worst,
      cv_episode_cost_mean, cv_nll_mean, cv_brier_mean, cv_unsafe_executed_mean, cv_cost_<config>, rank,
      selected (PROPOSED / LOW-COST), best_score.
  test_by_config.csv: per variant, configuration and test set (pilot-seen excluded): the CV columns on the
      test set (frozen from calibration) and fit_info.
  test_comparisons.csv: per variant, configuration, set and comparator (P1, P1c, P8, vote_only_gate =
      E_vote|R_myopic|A_none): costs, diff_comparator_minus_variant with 95% cluster-bootstrap CI, exact sign-flip p,
      p_holm_selected (family: the two selected gates x three baselines x configurations, per set),
      p_holm_all_variants (family: every row of that set).
  test_summary.csv: per variant and set: mean and worst cost over configurations, mean unsafe executed,
      mean episode cost, generations.
  efficiency_frontier.csv: per set (and calibration CV): eligible variants, generations, mean cost, on_frontier.
  ablations.csv: selected gates per configuration: remove each input group; other rules; other adaptations;
      uncalibrated (for source estimators); every alternative loss matrix and sweep (cost under that matrix,
      with the baselines' cost under it); calibration CV and every test set.
  ablation_reasoning.csv: selected gates and the vote-only gate, reasoning on minus off per model (paired).
  ablation_unanimous_cell.csv: items with ten 'clear' votes: execute share, cost, whether the gate depends on
      a fitted pseudo-count (the 5/12 boundary).
  unguarded_agent_readable_plans.csv: per configuration and test set, unreadable plans (count, share; provider
      blocks counted apart), the unguarded agent's cost on all items and on readable-plan items, and the
      unguarded agent minus the PROPOSED gate on both, with 95% cluster-bootstrap CI and exact sign-flip p.

ALL VARIANTS RUN:
  E_vote|R_myopic|A_none  (generations 10; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_vote|R_myopic|A_em  (generations 10; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_vote|R_myopic|A_agreement  (generations 11; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_vote|R_sequential|A_none  (generations 10; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_vote|R_sequential|A_em  (generations 10; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_vote|R_sequential|A_agreement  (generations 11; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_vote|R_conformal|A_none  (generations 10; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_vote|R_conformal|A_em  (generations 10; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_vote|R_conformal|A_agreement  (generations 11; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_vote|R_riskcap|A_none  (generations 10; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_vote|R_riskcap|A_em  (generations 10; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_vote|R_riskcap|A_agreement  (generations 11; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_verbal|R_myopic|A_none  (generations 1; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_verbal|R_myopic|A_em  (generations 1; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_verbal|R_myopic|A_agreement  (generations 11; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_verbal|R_sequential|A_none  (generations 1; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_verbal|R_sequential|A_em  (generations 1; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_verbal|R_sequential|A_agreement  (generations 11; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_verbal|R_conformal|A_none  (generations 1; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_verbal|R_conformal|A_em  (generations 1; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_verbal|R_conformal|A_agreement  (generations 11; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_verbal|R_riskcap|A_none  (generations 1; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_verbal|R_riskcap|A_em  (generations 1; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_verbal|R_riskcap|A_agreement  (generations 11; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_logit|R_myopic|A_none  (generations 1; defined for gemma)
  E_logit|R_myopic|A_em  (generations 1; defined for gemma)
  E_logit|R_myopic|A_agreement  (generations 12; defined for gemma)
  E_logit|R_sequential|A_none  (generations 1; defined for gemma)
  E_logit|R_sequential|A_em  (generations 1; defined for gemma)
  E_logit|R_sequential|A_agreement  (generations 12; defined for gemma)
  E_logit|R_conformal|A_none  (generations 1; defined for gemma)
  E_logit|R_conformal|A_em  (generations 1; defined for gemma)
  E_logit|R_conformal|A_agreement  (generations 12; defined for gemma)
  E_logit|R_riskcap|A_none  (generations 1; defined for gemma)
  E_logit|R_riskcap|A_em  (generations 1; defined for gemma)
  E_logit|R_riskcap|A_agreement  (generations 12; defined for gemma)
  E_average|R_myopic|A_none  (generations 12; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_average|R_myopic|A_em  (generations 12; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_average|R_myopic|A_agreement  (generations 12; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_average|R_sequential|A_none  (generations 12; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_average|R_sequential|A_em  (generations 12; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_average|R_sequential|A_agreement  (generations 12; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_average|R_conformal|A_none  (generations 12; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_average|R_conformal|A_em  (generations 12; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_average|R_conformal|A_agreement  (generations 12; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_average|R_riskcap|A_none  (generations 12; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_average|R_riskcap|A_em  (generations 12; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_average|R_riskcap|A_agreement  (generations 12; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_select|R_myopic|A_none  (generations 10; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_select|R_myopic|A_em  (generations 10; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_select|R_myopic|A_agreement  (generations 11; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_select|R_sequential|A_none  (generations 10; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_select|R_sequential|A_em  (generations 10; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_select|R_sequential|A_agreement  (generations 11; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_select|R_conformal|A_none  (generations 10; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_select|R_conformal|A_em  (generations 10; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_select|R_conformal|A_agreement  (generations 11; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_select|R_riskcap|A_none  (generations 10; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_select|R_riskcap|A_em  (generations 10; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  E_select|R_riskcap|A_agreement  (generations 11; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_vote_scores|R_myopic|A_none  (generations 11; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_vote_scores|R_myopic|A_em  (generations 11; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_vote_scores|R_myopic|A_agreement  (generations 12; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_vote_scores|R_sequential|A_none  (generations 11; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_vote_scores|R_sequential|A_em  (generations 11; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_vote_scores|R_sequential|A_agreement  (generations 12; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_vote_scores|R_conformal|A_none  (generations 11; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_vote_scores|R_conformal|A_em  (generations 11; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_vote_scores|R_conformal|A_agreement  (generations 12; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_vote_scores|R_riskcap|A_none  (generations 11; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_vote_scores|R_riskcap|A_em  (generations 11; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_vote_scores|R_riskcap|A_agreement  (generations 12; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores|R_myopic|A_none  (generations 2; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores|R_myopic|A_em  (generations 2; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores|R_myopic|A_agreement  (generations 12; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores|R_sequential|A_none  (generations 2; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores|R_sequential|A_em  (generations 2; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores|R_sequential|A_agreement  (generations 12; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores|R_conformal|A_none  (generations 2; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores|R_conformal|A_em  (generations 2; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores|R_conformal|A_agreement  (generations 12; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores|R_riskcap|A_none  (generations 2; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores|R_riskcap|A_em  (generations 2; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores|R_riskcap|A_agreement  (generations 12; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores_yesno|R_myopic|A_none  (generations 5; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores_yesno|R_myopic|A_em  (generations 5; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores_yesno|R_myopic|A_agreement  (generations 15; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores_yesno|R_sequential|A_none  (generations 5; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores_yesno|R_sequential|A_em  (generations 5; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores_yesno|R_sequential|A_agreement  (generations 15; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores_yesno|R_conformal|A_none  (generations 5; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores_yesno|R_conformal|A_em  (generations 5; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores_yesno|R_conformal|A_agreement  (generations 15; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores_yesno|R_riskcap|A_none  (generations 5; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores_yesno|R_riskcap|A_em  (generations 5; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_verbal_scores_yesno|R_riskcap|A_agreement  (generations 15; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all|R_myopic|A_none  (generations 16; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all|R_myopic|A_em  (generations 16; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all|R_myopic|A_agreement  (generations 16; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all|R_sequential|A_none  (generations 16; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all|R_sequential|A_em  (generations 16; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all|R_sequential|A_agreement  (generations 16; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all|R_conformal|A_none  (generations 16; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all|R_conformal|A_em  (generations 16; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all|R_conformal|A_agreement  (generations 16; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all|R_riskcap|A_none  (generations 16; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all|R_riskcap|A_em  (generations 16; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all|R_riskcap|A_agreement  (generations 16; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all_actions|R_myopic|A_none  (generations 19; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all_actions|R_myopic|A_em  (generations 19; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all_actions|R_myopic|A_agreement  (generations 19; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all_actions|R_sequential|A_none  (generations 19; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all_actions|R_sequential|A_em  (generations 19; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all_actions|R_sequential|A_agreement  (generations 19; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all_actions|R_conformal|A_none  (generations 19; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all_actions|R_conformal|A_em  (generations 19; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all_actions|R_conformal|A_agreement  (generations 19; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all_actions|R_riskcap|A_none  (generations 19; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all_actions|R_riskcap|A_em  (generations 19; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
  F_all_actions|R_riskcap|A_agreement  (generations 19; defined for claude-haiku,claude-haiku-nothink,gemma,gemma-think,gpt-luna,gpt-luna-think)
