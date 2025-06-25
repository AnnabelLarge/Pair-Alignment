# PairHMM / Neural TKF Evaluation

## Total Conditions Tested: 8
Each condition corresponds to a unique config file testing combinations of:

- **Branch Lengths**:
  - Marginalizing over a grid of times
  - One unique branch length per sample

- **Substitution Models**:
  - GTR
  - F81

- **Transition Models**:
  - TKF91
  - TKF92


## Instructions
For each condition, run:

```python Pair_Alignment.py -task train -configs [config file]```

For each config file, follow these steps:

1. **Train with PairHMM Reference Implementation**  
   - Model: `simple_site_class_predict.IndpSites`  
   - Config location: `pairhmm_reference/configs/`

2. **Load Parameters from Reference PairHMM and Run Inference with Neural TKF Code**  
   - Config location: `neuralTKF_load_all/configs/`

3. **Train with Global Parameters Using Neural TKF**  
   - Model: `neural_hmm_predict.NeuralCondTKF` (functionally equivalent to `IndpSites`)  
   - Config location: `neuralTKF_train/configs/`


## Notes
- As of **6/25/25**, both scripts pass for all tested conditions.
- **TODO**: Build a cleaner unit testing interface to run and evaluate all conditions automatically.
