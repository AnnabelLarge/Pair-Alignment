#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 19:23:54 2025

@author: annabel_large


ABOUT:
======
1st test for substitution models

Confirm that rate matrix is constructed correctly
    1. by hand
    2. compared to LG08 rate matrix

LG08_rate_matrix.txt is from cherryML repo
"""
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
from scipy.special import logsumexp

import numpy.testing as npt
import unittest

from models.simple_site_class_predict.emission_models import (EqulDistLogprobsFromFile,
                                                              GTRRateMatFromFile)
from models.simple_site_class_predict.model_functions import (upper_tri_vector_to_sym_matrix, 
                                                              rate_matrix_from_exch_equl)

THRESHOLD = 1e-5

###############################################################################
### helpers   #################################################################
###############################################################################
def construct_fake_rate_matrix():
    """
    PURPOSE: construct fake data, and hand-calculate expected values
        
    C: hidden site classes
    A: alphabet

    """
    exchangeabilities = np.array([[0, 1, 2, 3],
                                  [1, 0, 4, 5],
                                  [2, 4, 0, 6],
                                  [3, 5, 6, 0]])#(A,A)
    
    equilibrium_distribution_1 = np.array([0.1, 0.2, 0.3, 0.4])
    equilibrium_distribution_2 = np.array([0.4, 0.3, 0.2, 0.1])
    equilibrium_distributions = np.stack([equilibrium_distribution_1,
                                          equilibrium_distribution_2]) #(C,A)
    del equilibrium_distribution_1, equilibrium_distribution_2
    
    
    ### construct true matrix by hand calculation
    # q_ij = chi_ij * pi_j
    true_1 = np.array([[-sum([1*0.2, 2*0.3, 3*0.4]), 1*0.2, 2*0.3, 3*0.4],
                       [1*0.1, -sum([1*0.1,4*0.3, 5*0.4]), 4*0.3, 5*0.4],
                       [2*0.1, 4*0.2, -sum([2*0.1, 4*0.2,6*0.4]), 6*0.4],
                       [3*0.1, 5*0.2, 6*0.3, -sum([3*0.1, 5*0.2, 6*0.3])]]) #(A,A)
    norm_factor_1 = -(-sum([1*0.2, 2*0.3, 3*0.4]) * 0.1 +
                      -sum([1*0.1,4*0.3, 5*0.4]) * 0.2 +
                      -sum([2*0.1, 4*0.2,6*0.4]) * 0.3 +
                      -sum([3*0.1, 5*0.2, 6*0.3]) * 0.4)
    true_1_normed = true_1 / norm_factor_1 #(A,A)
    del norm_factor_1
    
    true_2 = np.array([[-sum([1*0.3, 2*0.2, 3*0.1]), 1*0.3, 2*0.2, 3*0.1],
                       [1*0.4, -sum([1*0.4,4*0.2, 5*0.1]), 4*0.2, 5*0.1],
                       [2*0.4, 4*0.3, -sum([2*0.4, 4*0.3,6*0.1]), 6*0.1],
                       [3*0.4, 5*0.3, 6*0.2, -sum([3*0.4, 5*0.3, 6*0.2])]]) #(A,A)
    norm_factor_2 = -(-sum([1*0.3, 2*0.2, 3*0.1]) * 0.4 +
                      -sum([1*0.4,4*0.2, 5*0.1]) * 0.3 +
                      -sum([2*0.4, 4*0.3,6*0.1]) * 0.2 +
                      -sum([3*0.4, 5*0.3, 6*0.2]) * 0.1)
    true_2_normed = true_2 / norm_factor_2 #(A,A)
    del norm_factor_2
    
    true = np.stack([true_1, true_2])  #(C,A,A)
    true_normed = np.stack([true_1_normed, true_2_normed])  #(C,A,A)
    del true_1_normed, true_2_normed
    del true_1, true_2
    
    return {'exchangeabilities': exchangeabilities, #(A,A)
            'equilibrium_distributions': equilibrium_distributions, #(C,A)
            'true': true, #(C,A,A)
            'true_normed': true_normed} #(C,A,A)

def construct_lg_rate_matrix():
    """
    PURPOSE: load the true LG08 rate matrix
    
    C: hidden site classes
    A: alphabet

    """
    # read the LG rate matrix (true values)
    df = pd.read_csv('tests/substitution_model_tests/req_files/LG08_rate_matrix.txt',sep=' ',header=None,index_col=0)
    df.columns = df.index

    # rearrange row/col order
    new_order = list(df.columns)
    new_order.sort()

    true_lg_rate_matrix = np.zeros( (20,20) ) # (A,A)

    for j, col in enumerate(new_order):
        for i, row in enumerate(new_order):
            elem = df[col].loc[row]
            true_lg_rate_matrix[i,j] = elem

    df_new_order = pd.DataFrame(true_lg_rate_matrix,
                                index=new_order,
                                columns=new_order)

    # check the re-arrangement
    for j, col in enumerate(new_order):
        for i, row in enumerate(new_order):
            pred = df_new_order[col].loc[row]
            true = df[col].loc[row]
            assert pred == true
            
    return true_lg_rate_matrix



###############################################################################
### start of unit tests   #####################################################
###############################################################################
class TestSubsRateMatrixConstruction(unittest.TestCase):
    """
    SUBSTITUTION PROCESS SCORING TEST 1
    
    
    C: hidden site classes
    A: alphabet
    
    About
    ------
    test functions involved with rate matrix construction:
        - upper_tri_vector_to_sym_matrix
        - rate_matrix_from_exch_equl
    
    also test the flax module that strings these functions together:
        - GTRRateMatFromFile
        
    """
    def testupper_tri_vector_to_sym_matrix(self):
        """
        PURPOSE: make sure upper_tri_vector_to_sym_matrix fills in
          a symmetric rate matrix correctly
        """
        vec = np.array([1, 2, 3, 4, 5, 6])
        true_sym_matrix = np.array([[0, 1, 2, 3],
                                    [1, 0, 4, 5],
                                    [2, 4, 0, 6],
                                    [3, 5, 6, 0]])
        pred_sym_matrix = upper_tri_vector_to_sym_matrix(vec)
    
        npt.assert_allclose(true_sym_matrix, pred_sym_matrix, atol=THRESHOLD)
    
    def testrate_matrix_from_exch_equl(self):
        """
        PURPOSE: compare output from rate_matrix_from_exch_equl to
          ground truth (calculated by hand)
        """
        out = construct_fake_rate_matrix()
        
        exchangeabilities = out['exchangeabilities'] #(A,A)
        equilibrium_distributions = out['equilibrium_distributions'] #(C,A)
        true = out['true'] #(C,A,A)
        
        pred = rate_matrix_from_exch_equl(exchangeabilities,
                                            equilibrium_distributions,
                                            norm=False) #(C,A,A)
        
        npt.assert_allclose(true, pred, atol=THRESHOLD)
    
    def testrate_matrix_from_exch_equl_normalized_matrix(self):
        """
        PURPOSE: compare output from rate_matrix_from_exch_equl to
          ground truth (calculated by hand), including rate matrix 
          normalization
        """
        out = construct_fake_rate_matrix()
        
        exchangeabilities = out['exchangeabilities'] #(A,A)
        equilibrium_distributions = out['equilibrium_distributions'] #(C,A)
        true_normed = out['true_normed'] #(C,A,A)
        
        pred_normed = rate_matrix_from_exch_equl(exchangeabilities,
                                                  equilibrium_distributions,
                                                  norm=True) #(C,A,A)
        
        npt.assert_allclose(true_normed, pred_normed, atol=THRESHOLD)
    
    def test_rate_matrix_row_sums(self):
        """
        PURPOSE: with rate_matrix_from_exch_equl, make sure rows sum to zero
        """
        out = construct_fake_rate_matrix()
        
        exchangeabilities = out['exchangeabilities'] #(A,A)
        equilibrium_distributions = out['equilibrium_distributions'] #(C,A)
        
        pred = rate_matrix_from_exch_equl(exchangeabilities,
                                            equilibrium_distributions,
                                            norm=False) #(C,A,A)
        
        for c in range(pred.shape[0]):
            v = pred[c, ...].sum(axis=-1) #(A,)
            npt.assert_allclose(v, 0, atol=THRESHOLD, 
                                err_msg=f"matrix {c}: Rowsum is {v}"
                                )
    
    def test_rate_matrix_row_sums_normalized_matrix(self):
        """
        PURPOSE: with rate_matrix_from_exch_equl AND after normalizing the 
            rate matrix, make sure rows still sum to zero
        """
        out = construct_fake_rate_matrix()
        
        exchangeabilities = out['exchangeabilities'] #(A,A)
        equilibrium_distributions = out['equilibrium_distributions'] #(C,A)
        
        pred_normed = rate_matrix_from_exch_equl(exchangeabilities,
                                                  equilibrium_distributions,
                                                  norm=True) #(C,A,A)
        
        for c in range(pred_normed.shape[0]):
            v = pred_normed[c, ...].sum(axis=-1) #(A,)
            npt.assert_allclose(v, 0, atol=THRESHOLD, 
                                err_msg=f"matrix {c}: Rowsum is {v}"
                                )
    
    def test_rate_matrix_normalization(self):
        """
        PURPOSE: with rate_matrix_from_exch_equl, make sure normalization 
            works as expected
        """
        out = construct_fake_rate_matrix()
        
        exchangeabilities = out['exchangeabilities'] #(A,A)
        equilibrium_distributions = out['equilibrium_distributions'] #(C,A)
        
        pred_normed = rate_matrix_from_exch_equl(exchangeabilities,
                                                  equilibrium_distributions,
                                                  norm=True) #(C,A,A)
        
        for c in range(pred_normed.shape[0]):
            v = pred_normed[c, ...] #(A,A)
            checksum = 0
            for i in range(v.shape[0]):
                checksum += v[i,i] * equilibrium_distributions[c,i]
            
            err = f'matrix {c}: matrix normalized to {-checksum}'
            with self.subTest(value=checksum):
                npt.assert_allclose(-checksum, 1, atol=THRESHOLD, 
                                    err_msg=f"matrix {c}: Checksum is {-checksum}"
                                    )
    
    def test_GTRRateMatFromFile_with_upper_triag_values(self):
        """
        PURPOSE: using the full rate matrix function, load upper triangular 
            values and calculate a rate matrix; compare to ground truth 
            (LG rate matrix, from cherryML repo)
        """
        true = construct_lg_rate_matrix() #(A,A)
        
        # equlibrium distribution
        config = {'filenames': {'equl_dist': 'tests/substitution_model_tests/req_files/LG08_equl_dist.npy'}}
        my_equl_fn = EqulDistLogprobsFromFile(config=config,
                                              name = 'equl_fn')
        logprob_equl = my_equl_fn.apply(variables = {}) #(A,)
    
        # LG exchangeabilities
        config = {'num_mixtures': 1,
                  'norm_rate_matrix': True,
                  'filenames': {'rate_mult': None,
                                'exch': 'tests/substitution_model_tests/req_files/LG08_exchangeability_vec.npy'}}
        my_mod = GTRRateMatFromFile(config=config,
                                    name='my_mod')

        pred = my_mod.apply(variables = {},
                            logprob_equl = logprob_equl)[0,...] #(A,A)
        
        npt.assert_allclose(true, pred, atol=THRESHOLD)
    
    def test_GTRRateMatFromFile_with_full_rate_mat(self):
        """
        PURPOSE: using the full rate matrix function, load full 
            exchangeability matrix and calculate a rate matrix; compare to 
            ground truth (LG rate matrix, from cherryML repo)
        """
        true = construct_lg_rate_matrix() #(A,A)
        
        # equlibrium distribution
        config = {'filenames': {'equl_dist': 'tests/substitution_model_tests/req_files/LG08_equl_dist.npy'}}
        my_equl_fn = EqulDistLogprobsFromFile(config=config,
                                              name = 'equl_fn')
        logprob_equl = my_equl_fn.apply(variables = {}) #(A,)
    
        # LG exchangeabilities
        config = {'num_mixtures': 1,
                  'norm_rate_matrix': True,
                  'filenames': {'rate_mult': None,
                                'exch': 'tests/substitution_model_tests/req_files/LG08_exchangeability_R.npy'}}
        my_mod = GTRRateMatFromFile(config=config,
                                    name='my_mod')

        pred = my_mod.apply(variables = {},
                            logprob_equl = logprob_equl)[0,...] #(A,A)
        
        npt.assert_allclose(true, pred, atol=THRESHOLD)
        
if __name__ == '__main__':
    unittest.main( verbosity=2 )
               