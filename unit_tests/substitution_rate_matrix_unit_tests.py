#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 19:23:54 2025

@author: annabel_large


ABOUT:
======
Confirm that rate matrix is constructed correctly
    1. by hand
    2. compared to LG08 rate matrix

LG08_rate_matrix.txt is from cherryML repo
"""
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from models.simple_site_class_predict.emission_models import (_upper_tri_vector_to_sym_matrix,
                                                              _rate_matrix_from_exch_equl,
                                                              EqulDistLogprobsFromFile,
                                                              GTRRateMatFromFile)

#####################################################
### check function that fills in symmetric matrix   #
#####################################################
vec = np.array([1, 2, 3, 4, 5, 6])
true_sym_matrix = np.array([[0, 1, 2, 3],
                            [1, 0, 4, 5],
                            [2, 4, 0, 6],
                            [3, 5, 6, 0]])
assert np.allclose( _upper_tri_vector_to_sym_matrix(vec), true_sym_matrix )

print('[PASS] _upper_tri_vector_to_sym_matrix can fill in correct symmetric matrix')

del vec, true_sym_matrix


                   
######################################################
### check function that constructs the rate matrix   #
######################################################
exchangeabilities = np.array([[0, 1, 2, 3],
                              [1, 0, 4, 5],
                              [2, 4, 0, 6],
                              [3, 5, 6, 0]])

equilibrium_distribution_1 = np.array([0.1, 0.2, 0.3, 0.4])
equilibrium_distribution_2 = np.array([0.4, 0.3, 0.2, 0.1])
equilibrium_distributions = np.stack([equilibrium_distribution_1,
                                      equilibrium_distribution_2])
del equilibrium_distribution_1, equilibrium_distribution_2


### construct true matrix by hand calculation
# q_ij = chi_ij * pi_j
true_1 = np.array([[-sum([1*0.2, 2*0.3, 3*0.4]), 1*0.2, 2*0.3, 3*0.4],
                   [1*0.1, -sum([1*0.1,4*0.3, 5*0.4]), 4*0.3, 5*0.4],
                   [2*0.1, 4*0.2, -sum([2*0.1, 4*0.2,6*0.4]), 6*0.4],
                   [3*0.1, 5*0.2, 6*0.3, -sum([3*0.1, 5*0.2, 6*0.3])]])
norm_factor_1 = -(-sum([1*0.2, 2*0.3, 3*0.4]) * 0.1 +
                  -sum([1*0.1,4*0.3, 5*0.4]) * 0.2 +
                  -sum([2*0.1, 4*0.2,6*0.4]) * 0.3 +
                  -sum([3*0.1, 5*0.2, 6*0.3]) * 0.4)
true_1_normed = true_1 / norm_factor_1
del norm_factor_1

true_2 = np.array([[-sum([1*0.3, 2*0.2, 3*0.1]), 1*0.3, 2*0.2, 3*0.1],
                   [1*0.4, -sum([1*0.4,4*0.2, 5*0.1]), 4*0.2, 5*0.1],
                   [2*0.4, 4*0.3, -sum([2*0.4, 4*0.3,6*0.1]), 6*0.1],
                   [3*0.4, 5*0.3, 6*0.2, -sum([3*0.4, 5*0.3, 6*0.2])]])
norm_factor_2 = -(-sum([1*0.3, 2*0.2, 3*0.1]) * 0.4 +
                  -sum([1*0.4,4*0.2, 5*0.1]) * 0.3 +
                  -sum([2*0.4, 4*0.3,6*0.1]) * 0.2 +
                  -sum([3*0.4, 5*0.3, 6*0.2]) * 0.1)
true_2_normed = true_2 / norm_factor_2
del norm_factor_2

true = np.stack([true_1, true_2])
true_normed = np.stack([true_1_normed, true_2_normed])
del true_1_normed, true_2_normed
del true_1, true_2


### check against my function
pred = _rate_matrix_from_exch_equl(exchangeabilities,
                                   equilibrium_distributions,
                                   norm=False)
assert np.allclose( true, pred )

pred_normed = _rate_matrix_from_exch_equl(exchangeabilities,
                                          equilibrium_distributions,
                                          norm=True)
assert np.allclose( true_normed, pred_normed )


print('[PASS] rate matrix function produces same result as hand calculation')
del true, true_normed


### test row sums and normalization
for c in range(pred.shape[0]):
    assert (np.abs(pred[c,...].sum(axis=-1)) <= 1e-6).all()
    assert (np.abs(pred_normed[c,...].sum(axis=-1)) <= 1e-6).all()
    
    checksum = 0
    for i in range(pred_normed[c,...].shape[0]):
        checksum += pred_normed[c,i,i] * equilibrium_distributions[c,i]
    assert np.allclose( -checksum, 1 )


print('[PASS] rate matrix rows sum to zero, normalization worked')
del c, i, checksum, pred, pred_normed, exchangeabilities, equilibrium_distributions
    


#######################################################################
### calculate rate matrix with the full module, starting from upper   # 
### triangular vector                                                 #
#######################################################################
# read the LG rate matrix (true values)
df = pd.read_csv('req_files/unit_tests/LG08_rate_matrix.txt',sep=' ',header=None,index_col=0)
df.columns = df.index

# rearrange row/col order
new_order = list(df.columns)
new_order.sort()

true_lg_rate_matrix = np.zeros( (20,20) )

for j, col in enumerate(new_order):
    for i, row in enumerate(new_order):
        elem = df[col].loc[row]
        true_lg_rate_matrix[i,j] = elem

df_new_order = pd.DataFrame(true_lg_rate_matrix,
                            index=new_order,
                            columns=new_order)

for j, col in enumerate(new_order):
    for i, row in enumerate(new_order):
        pred = df_new_order[col].loc[row]
        true = df[col].loc[row]
        assert pred == true

del df, new_order, j, col, i, row, elem, df_new_order, pred, true

### use my functions to calculate the same
# LG equilibrium
config = {'filenames': {'equl_dist': 'req_files/unit_tests/LG08_equl_dist.npy'}}
my_equl_fn = EqulDistLogprobsFromFile(config=config,
                                      name = 'equl_fn')
logprob_equl = my_equl_fn.apply(variables = {})

assert logsumexp(logprob_equl) == 0

del config, my_equl_fn


# LG exchangeabilities
config = {'num_emit_site_classes': 1,
          'filenames': {'rate_mult': None,
                        'exch': 'req_files/unit_tests/LG08_exchangeability_vec.npy'}}
my_mod = GTRRateMatFromFile(config=config,
                            name='my_mod')

rate_mat = my_mod.apply(variables = {},
                        logprob_equl = logprob_equl)[0,...]

max_abs_row_sum = np.abs( rate_mat.sum(axis=-1).max() )
assert max_abs_row_sum <= 1e-6

assert np.allclose(rate_mat, true_lg_rate_matrix)
print( ('[PASS] correcty calculate rate matrix from exchangeability vector and'+
        ' equilibrium distribution') )

del config, my_mod, rate_mat, max_abs_row_sum


####################################################################
### repeat, this time by reading the full exchangeability matrix   #
####################################################################
config = {'num_emit_site_classes': 1,
          'filenames': {'rate_mult': None,
                        'exch': 'req_files/unit_tests/LG08_exchangeability_r.npy'}}
my_mod = GTRRateMatFromFile(config=config,
                            name='my_mod')

rate_mat = my_mod.apply(variables = {},
                        logprob_equl = logprob_equl)[0,...]

max_abs_row_sum = np.abs( rate_mat.sum(axis=-1).max() )
assert max_abs_row_sum <= 1e-6

assert np.allclose(rate_mat, true_lg_rate_matrix)

print( ('[PASS] correcty calculate rate matrix from exchangeability MATRIX and'+
        ' equilibrium distribution') )

