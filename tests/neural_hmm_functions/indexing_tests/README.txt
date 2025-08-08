Status:
=======
PASS on 8/8/25



Summary: 
=========
Test scoring with neural tfk functions (akin to large indexing test)

All functions tested: 
----------------------
  - models.neural_hmm_predict.scoring_fns.score_indels
  - models.neural_hmm_predict.scoring_fns.score_f81_substitutions_marg_over_times
  - models.neural_hmm_predict.scoring_fns.score_f81_substitutions_t_per_samp
  - models.neural_hmm_predict.scoring_fns.score_gtr_substitutions
  - models.neural_hmm_predict.scoring_fns.score_transitions



All tests (4):
===============

test_score_indels
------------------------
CALCULATION TESTED: P(x_l)

ABOUT: calculate emission from indel sites, when this score is sample- and position-specific

functions tested:
  - models.neural_hmm_predict.scoring_fns.score_indels


test_score_subs_f81
------------------------
CALCULATION TESTED: P(y_l, align_l | x_l, t) with F81 model

ABOUT: calculate emission from match sites, when this score is sample- and position-specific; use F81

functions tested:
  - models.neural_hmm_predict.scoring_fns.score_f81_substitutions_marg_over_times
  - models.neural_hmm_predict.scoring_fns.score_f81_substitutions_t_per_samp


test_score_subs_gtr
------------------------
CALCULATION TESTED: P(y_l, align_l | x_l, t) with GTR model

ABOUT: calculate emission from match sites, when this score is sample- and position-specific; use GTR

functions tested:
  - models.neural_hmm_predict.scoring_fns.score_gtr_substitutions


test_score_transitions
------------------------
CALCULATION TESTED: P(tau_l | tau_{l-1}, t)

ABOUT: calculate transition probability, when this score is sample- and position-specific

functions tested:
  - models.neural_hmm_predict.scoring_fns.score_transitions
