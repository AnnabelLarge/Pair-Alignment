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

from models.simple_site_class_predict.model_functions import upper_tri_vector_to_sym_matrix
from models.simple_site_class_predict.emission_models import GTRLogprobs
from models.neural_hmm_predict.model_functions import logprob_gtr as neural_gtr

jax.config.update("jax_enable_x64", True)
THRESHOLD = 1e-6

class GTRLogprobsForDebug(GTRLogprobs):
    def _get_square_exchangeabilities_matrix(self,*args,**kwargs):
        return self.config['exchangeabilities_mat']

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
        
        # expand dims
        equl = equl[None,...] #(C, A)
        
        # reference implementation; return the conditional logprob
        exch = upper_tri_vector_to_sym_matrix(exch_vec)
        my_model = GTRLogprobsForDebug(config={'num_mixtures': 1,
                                               "k_rate_mults": 1,
                                               "norm_rate_matrix": True,
                                               'norm_rate_mults': False,
                                               'emission_alphabet_size': 4,
                                               'exchangeabilities_mat': exch},
                                       name='ref')
        
        init_params = my_model.init(rngs = jax.random.key(0),
                                    logprob_equl = np.log(equl),
                                    rate_multipliers = np.ones((1,1)),
                                    t_array = t_array,
                                    sow_intermediates=False,
                                    return_cond=True,
                                    return_intermeds=False)
        
        log_true_cond_gtr,_ = my_model.apply(variables = init_params,
                                  logprob_equl = np.log(equl),
                                  rate_multipliers = np.ones((1,1)),
                                  t_array = t_array,
                                  sow_intermediates=False,
                                  return_cond=True,
                                  return_intermeds=False)
        
        log_true_cond_gtr = log_true_cond_gtr[:,0,0,...] #(B, A, A)
        self.log_true_cond_gtr = log_true_cond_gtr #(B, A, A)
    
    def test_one_gtr_matrix(self):
        pred_gtr = neural_gtr( exch_upper_triag_values = self.exch_vec[None,None,...],
                               equilibrium_distributions = self.equl[None,None,...],
                               rate_multiplier = jnp.ones((1,1)),
                               t_array = self.t_array,
                               unique_time_per_sample = True ) #(B, L, A, A)
        pred_gtr = pred_gtr[:,0,...] #(B, A, A)
        
        npt.assert_allclose(pred_gtr, self.log_true_cond_gtr)
    
    def test_multi_gtr_matrix(self):
        B = self.t_array.shape[0]
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
                pred = pred_gtr[b,l,...] #(A, A)
                true = self.log_true_cond_gtr[b,...] #(A, A)
                npt.assert_allclose(pred, true, atol=THRESHOLD)
        
if __name__ == '__main__':
    unittest.main()
