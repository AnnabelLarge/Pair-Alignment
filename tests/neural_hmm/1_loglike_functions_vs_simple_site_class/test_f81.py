#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 18:37:07 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
import numpy as np

import numpy.testing as npt
import unittest

from models.simple_site_class_predict.emission_models import F81Logprobs
from models.neural_hmm_predict.model_functions import logprob_f81 as neural_f81

THRESHOLD = 1e-6


class TestF81(unittest.TestCase):
    """
    F81 model in pairHMM code has already been tested
    
    See if F81 model in neural TKF code matches this reference implementation
    """
    def setUp(self):    
        self.t_array = np.array([0.1, 0.2, 0.3])
        self.B = self.t_array.shape[0]
        self.A = 4
        
        # reference implementation from pairHMM codebase
        self.ref = F81Logprobs(config={'num_site_mixtures': 1,
                                       "k_rate_mults": 1,
                                       'norm_rate_matrix': True,
                                       'norm_rate_mults': False},
                               name='ref')
        
        
    def test_one_f81_matrix(self):
        rate_multipliers = np.array( [[1.0]] ) #(C, K)
        equl = np.array([[0.1, 0.2, 0.3, 0.4]]) #(C, A)
        
        # reference implementation; return the conditional logprob
        pairhmm_mat, _ = self.ref.apply(variables = {},
                                        logprob_equl = np.log(equl),
                                        rate_multipliers = rate_multipliers, 
                                        t_array = self.t_array,
                                        return_cond=True,
                                        sow_intermediates=False) #(B, C, K, A, A)
        pairhmm_mat = pairhmm_mat[:,0,0,...] #(B, A, A)
        
        # function in neural TKF codebase
        # rate multiliers is fine; it can now be interpretted as (B, L)
        equl_exp = equl[None,...] #(B, L, A)
        
        neural_mat = neural_f81(equl = equl[None,:],
                              rate_multiplier = np.ones((1,1)),
                              t_array = self.t_array,
                              unique_time_per_sample = True) #(B, L, A, 2)
        neural_mat = neural_mat[:, 0,...] #(B, A, 2)
        
        # compare matches i == j
        true_f81_diags = jnp.diagonal(pairhmm_mat, axis1=1, axis2=2)
        npt.assert_allclose(true_f81_diags, neural_mat[...,0], atol=THRESHOLD) 
        
        # compare mis-matches i!=j
        for b in range(self.B):
            for i in range(self.A):
                for j in range(self.A):
                    # don't do anything if i == j
                    if i == j:
                        continue

                    true_val = pairhmm_mat[b, i, j]
                    pred_val = neural_mat[b, j, 1]
                    
                    npt.assert_allclose(true_val, pred_val, atol=THRESHOLD) 

    def test_multi_f81_matrix(self):
        """
        unqiue equilibrium distribution for every sample in B, every 
          alignment column in L
        """
        B = self.B
        L = 5
        A = self.A
        
        ### generate a unique equilibrium distribution and rate multiplier 
        ###   for every B and L
        local_equl = np.zeros( (B, L, A) )
        local_rate_mult = np.zeros( (B, L) )
        true_f81 = np.zeros( (B, L, A, A) )
        
        # reference implementation
        for b in range(B):
            for l in range(L):
                # random state
                rngkey = jax.random.key( b*100+l )
                eq_key, rate_key = jax.random.split(rngkey, 2)
                
                # random equilibrium distribution
                logits = jax.random.normal( eq_key, (A,) )
                logits_squared = logits**2
                probs = logits_squared / logits_squared.sum()
                local_equl[b, l, :] = probs
                del logits, logits_squared
                
                # random rate
                logits = jax.random.normal( eq_key, () )
                rate = logits**2
                local_rate_mult[b, l] = rate
                del logits
                
                # expand params
                probs = probs[None,...] #(C, A)
                rate = rate[None, None] #(C, K)
                
                # true value, from pairHMM results
                f81_at_b_l, _ = self.ref.apply(variables = {},
                                               logprob_equl = np.log(probs),
                                               rate_multipliers = rate, 
                                               t_array = self.t_array[b][None],
                                               return_cond=True,
                                               sow_intermediates=False) #(B, C, K, A, A)
                f81_at_b_l = f81_at_b_l[0,0,0,...] #(A, A)
                
                true_f81[b, l, ...] = f81_at_b_l
        
        # function in neural TKF codebase
        pred_f81 = neural_f81(equl = local_equl,
                              rate_multiplier = local_rate_mult,
                              t_array = self.t_array,
                              unique_time_per_sample = True) #(B, L, A, 2)
        
        # compare
        for b in range(self.B):
            for l in range(L):
                for i in range(self.A):
                    for j in range(self.A):
                        true_val = true_f81[b, l, i, j]
                        
                        if i == j:
                            pred_val = pred_f81[b, l, j, 0]
                        
                        elif i != j:
                            pred_val = pred_f81[b, l, j, 1]
                        
                        npt.assert_allclose(true_val, pred_val, atol=THRESHOLD) 
                
if __name__ == '__main__':
    unittest.main()
