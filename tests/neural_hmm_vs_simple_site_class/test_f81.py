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

from models.simple_site_class_predict.emission_models import F81LogProbs
from models.neural_hmm_predict.model_functions import logprob_f81 as neural_f81

THRESHOLD = 1e-6


class TestF81(unittest.TestCase):
    """
    F81 model in pairHMM code has already been tested
    
    See if F81 model in neural TKF code matches this reference implementation
    """
    def setUp(self):    
        self.equl = np.array([0.1, 0.2, 0.3, 0.4])
        normalizing_factor = np.array( [1 / ( 1 - np.square(self.equl).sum() )] )
        self.t_array = np.array([0.1, 0.2, 0.3])
        
        # reference implementation; return the conditional logprob
        my_model = F81LogProbs(config={'num_mixtures': 1},
                               name='ref')
        self.true_f81 = my_model.apply( variables={},
                                        equl=self.equl[None,...],
                                        rate_multiplier=normalizing_factor,
                                        t_array = self.t_array,
                                        return_cond = True,
                                        method='_fill_f81' )[:,0,...] #(T, A, A)
    
    def test_one_f81_matrix(self):
        pred_f81 = neural_f81(equl = self.equl[None,None,:],
                              rate_multiplier = np.ones((1,1)),
                              t_array = self.t_array,
                              unique_time_per_sample = True)[:,0,...] #(T, A, 2)
        
        # compare matches
        true_f81_diags = jnp.diagonal(self.true_f81, axis1=1, axis2=2)
        npt.assert_allclose(true_f81_diags, pred_f81[...,0], atol=THRESHOLD) 
        
        # compare mis-matches; do this by indexing individually and avoiding the
        # diagonals
        npt.assert_allclose(self.true_f81[:,1,0], pred_f81[...,0,1], atol=THRESHOLD) 
        npt.assert_allclose(self.true_f81[:,0,1], pred_f81[...,1,1], atol=THRESHOLD) 
        npt.assert_allclose(self.true_f81[:,1,2], pred_f81[...,2,1], atol=THRESHOLD) 
        npt.assert_allclose(self.true_f81[:,1,3], pred_f81[...,3,1], atol=THRESHOLD) 
    
    def test_multi_f81_matrix(self):
        B = 3
        L = 5
        A = self.equl.shape[-1]
        equl_exp = jnp.broadcast_to( self.equl[None,None,:], (B, L, A) )
        pred_f81 = neural_f81(equl = equl_exp,
                              rate_multiplier = jnp.ones((B,L)),
                              t_array = self.t_array,
                              unique_time_per_sample = True) #(T, B, A, 2)
        
        for b in range(B):
            pred_at_b = pred_f81[:,b,...]
            
            # compare matches
            true_f81_diags = jnp.diagonal(self.true_f81, axis1=1, axis2=2)
            npt.assert_allclose(true_f81_diags, pred_at_b[...,0], atol=THRESHOLD) 
        
            # compare mis-matches; do this by indexing individually and avoiding the
            # diagonals
            npt.assert_allclose(self.true_f81[:,1,0], pred_at_b[...,0,1], atol=THRESHOLD) 
            npt.assert_allclose(self.true_f81[:,0,1], pred_at_b[...,1,1], atol=THRESHOLD) 
            npt.assert_allclose(self.true_f81[:,1,2], pred_at_b[...,2,1], atol=THRESHOLD) 
            npt.assert_allclose(self.true_f81[:,1,3], pred_at_b[...,3,1], atol=THRESHOLD)

if __name__ == '__main__':
    unittest.main()
