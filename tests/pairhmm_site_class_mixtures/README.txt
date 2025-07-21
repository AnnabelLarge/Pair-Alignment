Status:
=======
7/21/25: all pass


Summary: 
=========
Test functions related to independent sites mixtures (pairhmm models)

All functions tested: 
----------------------
  - lse_over_match_logprobs_per_mixture
  - joint_prob_from_counts
  - anc_marginal_probs_from_counts
  - cond_prob_from_counts

All flax modules tested:
-------------------------
  - simple_site_class_predict.IndpSites


All tests:
===========

test_score_alignment
---------------------
CALCULATION TESTED: \sum_c \sum_k P(c,k) P(x,y|t,c,k)

ABOUT: test scoring function by comparing against hand-done calculation; uses geometric sequence length and a fake emissions matrix; really, this is a test of indexing and summation over correct axes

functions tested:
  - lse_over_match_logprobs_per_mixture
  - joint_prob_from_counts


test_mixture_model_degeneracy
------------------------------
CALCULATION TESTED: \sum_c \sum_k 1/(C*K) P(x,y|t,c,k,\theta) = P(x,y|t,\theta)

ABOUT: make sure an equal mixture of the same model results in the same likelihood as using the original model

functions tested:
  - lse_over_match_logprobs_per_mixture
  - joint_prob_from_counts


test_joint_cond_marg_with_scoring_funcs
----------------------------------------
CALCULATION TESTED: P(x,y|t) = P(y|x,t) * P(x)

ABOUT: make sure that you recover the conditional log-probability from the ancestor and joint log-probabilities (from functions alone)

functions tested:
  - joint_prob_from_counts
  - anc_marginal_probs_from_counts
  - cond_prob_from_counts



test_indp_site_classes_loglikes
----------------------------------------
CALCULATION TESTED: P(x,y|t), P(y|x,t), P(x), and P(y)

ABOUT: using the full IndpSites flax module, calculate all four probabilities of interest, and compare against hand-done calculations

flax modules tested:
  - IndpSites.__call__
  - IndpSites._get_scoring_matrices
  - IndpSites.calculate_all_loglikes





TODO:
=====
not that important, but for completion: some explicit test of cond_prob_from_counts and anc_marginal_probs_from_counts, desc_marginal_probs_from_counts against hand-done calculations... it's implied that these work though
