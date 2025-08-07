Status:
=======
8/6/25: all pass


Summary: 
=========
Test functions related to fragment and emission site class mixtures (pairhmm models)

All functions tested: 
----------------------
- joint_only_forward
- all_loglikes_forward

All flax modules tested:
-------------------------
- FragAndSiteClasses._get_scoring_matrices (particularly the marginalization over site mixtures and rate mixtures)
- FragAndSiteClasses.__call__
- FragAndSiteClasses.calculate_all_loglikes


All tests:
===========
test_marg_over_site_mixes
---------------------------
CALCULATION TESTED: \sum_k P(k|c) P(x,y|k,c,t)

ABOUT: have to marginalize over k rate multipliers when generating emission probability matrices; make sure this is happening correctly. Technically, this is already tested in tests/pairhmm_site_class_mixtures/test_indp_site_classes_loglikes.py. This just makes sure the same calculation works when done with FragAndSiteClasses model.

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


frag_mix_reduction_test
------------------------
ABOUT: with one fragment mixture, FragAndSiteClasses strictly reduces to IndpSites

functions tested:
  - FragAndSiteClasses.calculate_all_loglikes




TODO:
=====
- INPROGRESS.py: unit test for full forward-backwards
