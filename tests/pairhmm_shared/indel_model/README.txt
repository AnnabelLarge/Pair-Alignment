Status:
=======
PASS on 8/6/25


Summary: 
=========
Test every aspect of calculating transition probabilities with TKF91, TKF92 indel models

All functions tested: 
----------------------
  - switch_tkf
  - regular_tkf
  - approx_tkf
  - get_tkf91_single_seq_marginal_transition_logprobs
  - get_tkf92_single_seq_marginal_transition_logprobs
  - get_cond_transition_logprobs

All flax modules tested:
-------------------------
- TKF91TransitionLogprobs
- TKF92TransitionLogprobs


All tests:
===========

test_original_tkf_param_fns
---------------------------------
CALCULATION TESTED: TKF alpha, beta, gamma

ABOUT: compare my implementation of TKF calculations against Ian's

functions tested:
  - switch_tkf
  - regular_tkf
  - approx_tkf


test_tkf91_joint_cond_marg
---------------------------------
CALCULATION TESTED: TKF91 solution

ABOUT: compare my implementation of TKF91 against hand-done calculations
  1.) calculate joint TKF91
  2.) calculate marginal TKF91
  3.) compose joint with marginal, to get conditional TKF91

functions tested:
  - get_tkf91_single_seq_marginal_transition_logprobs
  - get_cond_transition_logprobs

flax modules tested:
  - transition_models.TKF91TransitionLogprobs.fill_joint_tkf91


test_tkf92_joint_cond_marg
---------------------------------
CALCULATION TESTED: TKF92 solution

ABOUT: compare my implementation of TKF92 against hand-done calculations

functions tested:
  - get_tkf92_single_seq_marginal_transition_logprobs
  - get_cond_transition_logprobs

flax modules tested:
  - transition_models.TKF91TransitionLogprobs.fill_joint_tkf92
  

test_tkf92_frag_mix_joint_cond_marg
-------------------------------------
CALCULATION TESTED: TKF92 solution, but with different fragment mixtures

ABOUT: compare my implementation of mixture of TKF92 fragments against hand-done calculations

functions tested:
  - get_tkf92_single_seq_marginal_transition_logprobs
  - get_cond_transition_logprobs

flax modules tested:
  - transition_models.TKF91TransitionLogprobs.fill_joint_tkf92


test_tkf92_reduction_to_tkf91
---------------------------------
CALCULATION TESTED: TKF92 model reduction (with different domain and fragment mixtures, but all mixtures should be TKF91)

ABOUT: make sure that TKF92 model reduces to TKF91 solution when r=0

functions tested:

flax modules tested:
- TKF91TransitionLogprobs.return_all_matrices
- TKF92TransitionLogprobs.return_all_matrices

