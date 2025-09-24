Status:
=======
8/27/25: all pass


Summary: 
=========
Test functions related to nested TKF


All flax modules tested:
-------------------------
- NestedTKF



All tests:
===========

test_joint_top_level_matrix_construction
-----------------------------------------------
CALCULATION TESTED: T = U_{MIDS, MIDE} + U_{MIDS, AB} * (I - U_{AB,AB})^{-1} * U_{AB,MIDE}

ABOUT: Have function that eliminates null cycles in top-level TKF91 model. Test this against hand-done calculation (done in prob-space, by following Ian's write-up)

functions tested:
- NestedTKF._get_joint_domain_transit_matrix_without_null_cycles


test_joint_transit_matrix_entries
-----------------------------------------------
CALCULATION TESTED: UX_{lf} -> VY_{mg}

ABOUT: Have function that fills in one of 16 types of joint transitions (i.e. in top-level and fragment-level model). Test this against hand-done calculation (done in prob-space, by following the main table in Ian's write-up)

functions tested:
- NestedTKF._retrieve_joint_transition_entries


test_final_joint_transit_matrix
-----------------------------------------------
CALCULATION TESTED: final transition matrix

ABOUT: Compare final (T, C_dom*C_frag, C_dom*C_frag, 4, 4) transition matrix to hand-done calculation in probability space

functions tested:
- NestedTKF._get_transition_scoring_matrices


test_marginal_top_level_matrix_construction
-----------------------------------------------
ABOUT: repeat test_joint_top_level_matrix_construction, but get the top-level SINGLE-SEQUENCE MARGINAL transition matrix

functions tested:
- NestedTKF._get_marginal_domain_transit_matrix_without_null_cycles


test_final_marginal_transit_matrix
-----------------------------------------------
ABOUT: repeat test_final_joint_transit_matrix, but get the final SINGLE-SEQUENCE MARGINAL transition matrix

functions tested:
- NestedTKF._get_transition_scoring_matrices (the other matrix)


test_nested_tkf_transit_mat_reduction
-----------------------------------------------
CALCULATION TESTED: NestedTKF_transits = FragMix_transits

ABOUT: make sure that the JOINT, MARGINAL, and CONDITIONAL transition matrices for the nested TKF model can reduce to fragment mixture model when C_dom=1

functions tested:
- NestedTKF._get_transition_scoring_matrices


test_nested_tkf_score_reduction
-----------------------------------------------
CALCULATION TESTED: NestedTKF_transits = FragMix_transits

ABOUT: given the same samples, make sure that the nested TKF model evaluates the same LOG-LIKELIHOOD as the fragment mixture model, when C_dom=1

functions tested:
- NestedTKF.calculate_all_loglikes




