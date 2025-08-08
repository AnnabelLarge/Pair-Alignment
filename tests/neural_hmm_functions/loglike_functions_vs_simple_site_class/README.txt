Status:
=======
PASS on 8/8/25



Summary: 
=========
Test neural TKF models vs their implementation in the pairHMM code. Test both "global" model (where there's one set of parameters for all samples) and "local" model (sample- and column-specific evolutionary model parameters)

All functions tested: 
----------------------
- models.neural_hmm_predict.model_functions.logprob_f81
- models.neural_hmm_predict.model_functions.logprob_gtr
- models.neural_hmm_predict.model_functions.regular_tkf 
- models.neural_hmm_predict.model_functions.approx_tkf
- models.neural_hmm_predict.model_functions.logprob_tkf91
- models.neural_hmm_predict.model_functions.logprob_tkf92

All flax modules tested:
----------------------
- NeuralCondTKF (particularly neg_loglike_in_scan_fn and evaluate_loss_after_scan methods)


All tests (6):
===============

test_f81
------------------------
CALCULATION TESTED: F81 emission probability matrix

functions tested:
 - models.neural_hmm_predict.model_functions.logprob_f81


test_gtr
------------------------
CALCULATION TESTED: FTR emission probability matrix

functions tested:
- models.neural_hmm_predict.model_functions.logprob_gtr


test_tkf_funcs
------------------------
CALCULATION TESTED: TKF alpha, beta, gamma (two implementations)

functions tested:
- models.neural_hmm_predict.model_functions.regular_tkf 
- models.neural_hmm_predict.model_functions.approx_tkf


test_tkf91
------------------------
CALCULATION TESTED: TKF91 transition probabilities

functions tested:
- models.neural_hmm_predict.model_functions.logprob_tkf91


test_tkf92
------------------------
CALCULATION TESTED: TKF92 transition probabilities

functions tested:
- models.neural_hmm_predict.model_functions.logprob_tkf92




test_NeuralCondTKF_loglike
------------------------
CALCULATION TESTED: P(desc, align | t)

ABOUT: test alignment scoring for combinations of the following conditions:
  > F81, GTR
  > TKF91, TKF92
  > marginalization over time grid, one unique branch length per sample
  > jit-compiled vs not 

flax modules tested:
- NeuralCondTKF (particularly neg_loglike_in_scan_fn and evaluate_loss_after_scan methods)
