CONTENTS
8/8/25


FUNCTION/MODULE LEVEL:
=======================
these test specific functions/modules


data_dataloaders
---------------------------------------------
- validate data formatting and preprocessing
- validate dataloders


neural_components
---------------------------------------------
- validate selective concatenation function


neural_hmm_functions
---------------------
** for neural TKF **
- check indexing used for sample- and position-specific scoring
- make sure models in neural TKF codebase match models in pairHMM codebase


pairhmm_fragment_and_site_class_mixtures
---------------------------------------------
** for pairHMMs with mixture of fragments **

- tests marginalization over emission mixtures and dynamic programming algos


pairhmm_shared
---------------------------------------------
** for any pairHMM  ***

- tests of emission models and transition models
- check GTR reduces to F81
- check TKF92 reduces to TKF91
- includes direct comparisons to cherryML functions, as well as implementations from Ian


pairhmm_site_class_mixtures
---------------------------------------------
** for pairHMMs with mixture of site classes and rate multipliers **

- tests scoring and model degeneracy
- validating probabilistic factorization




USE WHOLE PIPELINE:
====================
these require running the whole codebase from CLI, using a JSON config file


feedforward_predict_recover_freqs
---------------------------------------------
** for neural seq2seq model **

- neural seq2seq should be able to recover frequency matrices:
  > P(desc, align | anc)
  > P(desc, align)


feedforward_overtrain
-----------------------
** for neural seq2seq model **

- check that model can overtrain on one sample


pairhmm_model_reduction
---------------------------------------------
** for mixture pairHMMs **

- check GTR reduces to F81
- check TKF92 reduces to TKF91
- assert code that evaluates mixture of fragments (which uses 1D forward algo over full alignment paths) can recover the same likelihood as the code that evaluates mixtures of sites+rate multipliers (which uses summary statistics)


pairhmm_vs_xrate
---------------------------------------------
** for basic pairHMM **

- check my GTR implementation matches XRATE implementation


neural_hmm_model_reduction
---------------------------------------------
** for neural TKF model **

- check neural F81+TKF92 reduces to pairHMM, when syncing parameters across all valid alignment positions
- check neural F81+TKF92 reduces to pairHMM, when forcing global set of parameters




OTHER:
=======
- data_processing.py: some useful functions for running unit tests
