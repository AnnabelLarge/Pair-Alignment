Status:
=======


Summary: 
=========
Test functions related to independent sites mixtures (pairhmm models)

All functions tested: 
----------------------

All flax modules tested:
-------------------------


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

ABOUT: make sure that you recover the conditional log-probability from the ancestor and joint log-probabilities

functions tested:
  - joint_prob_from_counts
  - anc_marginal_probs_from_counts
  - cond_prob_from_counts








TODO:
=====
INPROGRESS <- work on this one next; going to assert that likelihood with IndpSites is the same as hand-done calculations
  > not that important, but for completion: some explicit test of cond_prob_from_counts and anc_marginal_probs_from_counts, desc_marginal_probs_from_counts against hand-done calculations... it's implied that these work though, since test_joint_cond_marg_with_scoring_funcs test passes (and hopefully, this INPROGRESS script will pass)

test_time_marginalization
test_full_model_validity




template:
=========

CALCULATION TESTED:

ABOUT: 

functions tested:

flax models tested:
