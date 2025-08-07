Status:
=======
PASS on 8/6/25

Summary: 
=========
Test every aspect of calculating emission probabilities at match sites, including mixtures over rate classes and latent site class labels.

All functions tested: 
----------------------
  - upper_tri_vector_to_sym_matrix
  - rate_matrix_from_exch_equl
  - cond_logprob_emit_at_match_per_mixture
  - joint_logprob_emit_at_match_per_mixture
  - fill_f81_logprob_matrix (in F81Logprobs)

All flax modules tested:
-------------------------
  - emission_models.GTRLogprobs
  - emission_models.F81Logprobs


All tests (4):
===============

test_subs_rate_matrix_construction
-----------------------------------
CALCULATION TESTED: substitution rate matrix construction

ABOUT: confirm substitution rate matrix is constructed correctly, with GTR model
  > test the function that fills a symmetric matrix from a vector of upper triangular values (needed to fill the exchangeabilities)
  > compare rate matrix normalization against python loops (enforcing -\sum_i \pi_i q_ii = 1)
  > compare final values from functions against hand-done calculation (normed and non-normed rate matrices)
  > compare final values from GTRLogprobsFromFile flax model against LG08 rate matrix (file came from cherryML; authors did same test)
    > from vector of upper triangular values, and from full symmetric rate matrix

functions tested:
  - upper_tri_vector_to_sym_matrix
  - rate_matrix_from_exch_equl

flax modules tested:
  - emission_models.GTRLogprobsFromFile (specifically the part that goes into making the rate matrix)


test_conditional_prob_subs
---------------------------
CALCULATION TESTED: conditional emission probability, P(desc | anc, align=Match)

ABOUT: confirm conditional probability P(desc | anc, align=Match) is calculated correctly, by comparing against cherryML implementation

functions tested:
  - cond_logprob_emit_at_match_per_mixture

flax modules tested:
  - emission_models.GTRLogprobs


test_joint_prob_subs
---------------------
CALCULATION TESTED: joint emission probability, P(anc, desc | align=Match)

ABOUT: confirm joint probability P(anc, desc | align=Match) is calculated correctly, by comparing against a python loop

functions tested:
  - joint_logprob_emit_at_match_per_mixture

flax modules tested:
  - emission_models.GTRLogprobs


test_f81
---------
CALCULATION TESTED: f81 model solutions

ABOUT: Check the F81 solutions two ways: by hand, and by making sure GTR can reduce to F81 as expected

functions tested:
  - fill_f81_logprob_matrix (contained in emission_models.F81Logprobs.__call__)

flax modules tested:
  - emission_models.F81Logprobs


TODO:
=====
- could have an extra test here for joint = cond * anc_marg, but that inevitably gets tested later on, so... not really that important for now

