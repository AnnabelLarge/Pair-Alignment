Status:
=======
7/21/25: all pass


Summary: 
=========
Test functions related to fragment and emission site class mixtures (pairhmm models)

All functions tested: 
----------------------
- joint_only_forward
- all_loglikes_forward

All flax modules tested:
-------------------------
- FragAndSiteClasses._get_scoring_matrices (particularly the marginalization over k classes)


All tests:
===========
test_marg_over_k_rate_mults
---------------------------
CALCULATION TESTED: \sum_k P(k|c) P(x,y|k,c,t)

ABOUT: have to marginalize over k rate multipliers when generating emission probability matrices; make sure this is happening correclty

functions tested:
- FragAndSiteClasses._get_scoring_matrices (particularly the marginalization over k classes)


test_joint_only_forward, test_joint_only_forward_uniq_branch_len
-------------------------------------------------------------------
CALCULATION TESTED: \prod_{l=1}^{|align|} P( c_l | c_{l-1} ) P( x_l, y_l | c_l, tau_l, t ) P(tau_l, c_l | tau_{l-1}, c_{l-1} )

ABOUT: compare forward algorithm implementation against hand-done python loops (where you manually enumerate all possible latent site class paths, score each, and sum) i.e. a 1D forward algorithm

functions tested:
- joint_only_forward


test_all_loglikes_forward_uniq_branch_len, test_all_loglikes_forward
---------------------------------------------------------------------
ABOUT: same as above, but also test conditional and single-sequence marginals

functions tested:
- joint_only_forward
- all_loglikes_forward




TODO:
=====
- INPROGRESS.py: unit test for full forward-backwards
- for completeness, should also have some version of "compare against hand-done calculations" using full model (but full model is basically a wrapper for all these function calls, so it probably works)
