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
        self.ref = F81Logprobs(config={'num_mixtures': 1,
                                       'norm_rate_matrix': True,
                                       'norm_rate_mults': False},
                               name='ref')
        
        
    def test_one_f81_matrix(self):
        equl = np.array([0, 0.3, 0.3, 0.4])
        
        # reference implementation; return the conditional logprob
        true_f81 = self.ref.apply( variables={},
                                   equl=equl[None,...],
                                   rate_multiplier=jnp.ones((1,)),
                                   t_array = self.t_array,
                                   return_cond = True,
                                   method='_fill_f81' )[:,0,...] #(B, A, A)
        
        # function in neural TKF codebase
        pred_f81 = neural_f81(equl = equl[None,None,:],
                              rate_multiplier = np.ones((1,1)),
                              t_array = self.t_array,
                              unique_time_per_sample = True)[:,0,...] #(B, A, 2)
        
        # compare matches i == j
        true_f81_diags = jnp.diagonal(true_f81, axis1=1, axis2=2)
        npt.assert_allclose(true_f81_diags, pred_f81[...,0], atol=THRESHOLD) 
        
        # compare mis-matches i!=j
        for b in range(self.B):
            for i in range(self.A):
                for j in range(self.A):
                    # don't do anything if i == j
                    if i == j:
                        continue

                    true_val = true_f81[b, i, j]
                    pred_val = pred_f81[b, j, 1]
                    
                    npt.assert_allclose(true_val, pred_val, atol=THRESHOLD) 

    def test_multi_f81_matrix(self):
        """
        unqiue equilibrium distribution for every sample in B, every 
          alignment column in L
        """
        L = 5
        A = self.A
        
        ### generate a unique equilibrium distribution for every B and L
        local_equl = np.zeros( (self.B, L, A) )
        true_f81 = np.zeros( (self.B, L, A, A) )
        
        # reference implementation
        for b in range(self.B):
            for l in range(L):
                # random equilibrium distribution
                rngkey = jax.random.key( b*100+l )
                logits = jax.random.normal( rngkey, (A,) )
                logits_squared = logits**2
                probs = logits_squared / logits_squared.sum()
                local_equl[b, l, :] = probs
                
                # true value
                f81_at_b_l = self.ref.apply( variables={},
                                             equl=probs[None,...],
                                             rate_multiplier=jnp.ones((1,)),
                                             t_array = self.t_array[b][None],
                                             return_cond = True,
                                             method='_fill_f81' )[0,0,...] #(A,A)
                
                true_f81[b, l, ...] = f81_at_b_l
        
        # function in neural TKF codebase
        pred_f81 = neural_f81(equl = local_equl,
                              rate_multiplier = np.ones((1,1)),
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
