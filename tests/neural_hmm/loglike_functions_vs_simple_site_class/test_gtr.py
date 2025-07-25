#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 19:34:11 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
import numpy as np

import numpy.testing as npt
import unittest

from models.simple_site_class_predict.model_functions import (upper_tri_vector_to_sym_matrix,
                                                              rate_matrix_from_exch_equl,
                                                              get_cond_logprob_emit_at_match_per_class)
from models.simple_site_class_predict.emission_models import GTRRateMat
from models.neural_hmm_predict.model_functions import logprob_gtr as neural_gtr

THRESHOLD = 1e-6


class TestGTR(unittest.TestCase):
    """
    gtr model in pairHMM code has already been tested
    
    See if gtr in neural TKF code matches this implementation
    """
    def setUp(self):
        equl = np.array([0.1, 0.2, 0.3, 0.4])
        exch_vec = np.array([1,2,3,4,5,6])
        t_array = np.array([0.1, 0.2, 0.3])
    
        self.equl = equl
        self.exch_vec = exch_vec
        self.t_array = t_array
        
        # reference implementation; return the conditional logprob
        exch = upper_tri_vector_to_sym_matrix(exch_vec)
        my_model = GTRRateMat(config={'num_mixtures': 1,
                                      'emission_alphabet_size':4},
                              name='ref')
        init_vars = my_model.init( rngs=jax.random.key(0),
                                   logprob_equl = jnp.zeros((1,4)),
                                   log_class_probs = jnp.zeros((1,)),
                                   sow_intermediates = False )
                                  
        rate_mat = my_model.apply( variables=init_vars,
                                   exchangeabilities = exch,
                                   equilibrium_distributions = equl[None,...],
                                   rate_multiplier=jnp.ones((1,)),
                                   norm = True,
                                   method='_prepare_rate_matrix' ) #(1, A, A)
        
        true_cond_gtr,_ = get_cond_logprob_emit_at_match_per_class(t_array = t_array,
                                                                 scaled_rate_mat_per_class = rate_mat)
        
        self.true_cond_gtr = true_cond_gtr[:,0,...] #(T, A, A)
    
    def test_one_gtr_matrix(self):
        pred_gtr = neural_gtr( exch_upper_triag_values = self.exch_vec[None,None,...],
                               equilibrium_distributions = self.equl[None,None,...],
                               rate_multiplier = jnp.ones((1,1)),
                               t_array = self.t_array,
                               unique_time_per_sample = True )[:,0,0,...] #(T, A, A)
        npt.assert_allclose(pred_gtr, self.true_cond_gtr)
    
    def test_multi_gtr_matrix(self):
        B = 6
        L = 5
        A = self.equl.shape[-1]
        N = self.exch_vec.shape[-1]
        equl_exp = jnp.broadcast_to( self.equl[None,None,:], (B, L, A) )
        exch_exp = jnp.broadcast_to( self.exch_vec[None,None,:], (B, L, N) )
        
        pred_gtr = neural_gtr( exch_upper_triag_values = exch_exp,
                               equilibrium_distributions = equl_exp,
                               rate_multiplier = jnp.ones((1,1)),
                               t_array = self.t_array,
                               unique_time_per_sample = True )
        
        for b in range(B):
            for l in range(L):
                pred = pred_gtr[:,b,l,...]
                npt.assert_allclose(pred, self.true_cond_gtr, atol=THRESHOLD)
        
if __name__ == '__main__':
    unittest.main()
