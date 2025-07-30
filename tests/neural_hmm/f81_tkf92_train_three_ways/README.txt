7/8/25:
========
There's three ways to generate likelihoods from a simple F81+TKF92 model:
1.) with IndpSites model

2.) with CondNeuralTKF model, way 1: 
    > using no sequence embeddings (in fact, ignoring them)
    > setting all parameters to "global"

3.) with CondNeuralTKF model, way 2: 
    > using binary indicator all emissions (i.e. 1 if residue is present, 0 otherwise)
    > setting all parameters to "local"
    > for emissions, force rate multiplier to be 1 for all positions (this naturally happens when )
    > this technically fits evolutionary parameters per sample, per location, but since this uses binary indicators as features, all "local" parameters are actually the same


Test 1: training reaching same optimum
========================================
PASS: Methods (1) and (2) reach the same optimum with same likelihoods for all samples

WEIRD: Method (3) reaches a slightly better optimum; maybe because of a different initialization? A more optimum learning rate? Idk???


Test 2: Loading parameters from neuralTKF 
results in same likelihood as pairHMM code
============================================
Since Method (3) is reaching a slightly better optimum, I extracted the parameters from Method (3), and loaded them into pairHMM IndpSites model.

PASS: I get the same likelihood


Conclusion:
============
Not sure why Method (3) is able to fit better parameters, but since the likelihood calculation matches reference implementation, don't worry about it (or, at the very least, come back to this later)

