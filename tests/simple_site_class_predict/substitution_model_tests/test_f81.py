#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 21:00:45 2025

@author: annabel_large


About:
======
Check the F81 matrix two ways:
    1.) by hand
    2.) by making sure that GTR reduces to F81 (GTR already validated)

Test in both single-class and multiclass setting

"""
import jax
from jax import numpy as jnp
import numpy as np

import numpy.testing as npt
import unittest

from models.simple_site_class_predict.emission_models import F81Logprobs
from models.simple_site_class_predict.emission_models import GTRLogprobs

from models.simple_site_class_predict.model_functions import (upper_tri_vector_to_sym_matrix,
                                                              rate_matrix_from_exch_equl,
                                                              cond_logprob_emit_at_match_per_mixture)

THRESHOLD = 1e-6

class GTRLogprobsForDebug(GTRLogprobs):
    def _get_square_exchangeabilities_matrix(self,*args,**kwargs):
        return self.config['exchangeabilities_mat']

class TestF81(unittest.TestCase):
    def setUp(self):
        self.equl = np.array([0.1, 0.2, 0.3, 0.4])[None,:] #(C=1,A=4)
        self.t_array = np.array([0.1, 0.3, 0.5, 0.7, 0.9]) #(T=5)
        
        self.rate_mult_multiclass = np.array([[1, 2, 3],
                                              [4, 5, 6]]).T #(C=3, K=2)
        self.equl_multiclass = np.array([[0.1, 0.2, 0.3, 0.4],
                                         [0.2, 0.3, 0.4, 0.1],
                                         [0.7, 0.1, 0.1, 0.1]]) #(C, A)
        
    
    def test_one_f81_hand_calc(self,):
        """
        because there's only one class: normalize to 1 substitution per site
          by default
        """
        T = self.t_array.shape[0]
        A = self.equl.shape[-1]
        C = 1
        K = 1
        
        ### true
        rate_multipliers = np.array( [[1.0]] )
        normalization_factor = 1 / ( 1 - np.square(self.equl).sum(axis=-1) )
        cond_prob_true = np.zeros( (T, C, K, A, A) )
        for t_idx, t in enumerate(self.t_array):
            mat = np.zeros((A,A))
            for i in range(A):
                for j in range(A):
                    pi_j = self.equl[0,j]
                    
                    if i == j:
                        val = pi_j + (1-pi_j) * np.exp(-normalization_factor * rate_multipliers * t)
                    
                    elif i != j:
                        val = pi_j * (1 - np.exp(-normalization_factor * rate_multipliers * t) )
                    
                    mat[i,j] = val
            
            cond_prob_true[t_idx, 0, 0, ...] = mat
        
        ### by my formula
        my_model = F81Logprobs(config={'num_mixtures': 1,
                                        'norm_rate_matrix': True},
                                name='mymod')
        cond_prob_pred,_ = my_model.apply(variables = {},
                                        logprob_equl = np.log(self.equl),
                                        rate_multipliers = rate_multipliers, 
                                        t_array = self.t_array,
                                        return_cond=True,
                                        sow_intermediates=False)
        cond_prob_pred = np.exp(cond_prob_pred)
        assert np.allclose( cond_prob_pred.sum(axis=-1),
                            np.ones( cond_prob_pred.sum(axis=-1).shape ) )
        
        npt.assert_allclose(cond_prob_true, cond_prob_pred, atol=THRESHOLD)
        

    def test_one_f81_against_gtr(self):
        """
        because there's only one class: normalize to 1 substitution per site
          by default
        """
        T = self.t_array.shape[0]
        A = self.equl.shape[-1]
        C = 1
        rate_multipliers = np.array( [[1]] )
        
        # by f81
        my_model = F81Logprobs(config={'num_mixtures': 1,
                                        'norm_rate_matrix': True},
                                name='mymod')
        cond_logprob_f81,_ = my_model.apply(variables = {},
                                        logprob_equl = np.log(self.equl),
                                        rate_multipliers = rate_multipliers, 
                                        t_array = self.t_array,
                                        return_cond=True,
                                        sow_intermediates=False)
        del my_model
        
        # by gtr, with uniform exchangeabilities
        exch_mat = upper_tri_vector_to_sym_matrix( jnp.ones((6,)) )
        
        my_gtr_model = GTRLogprobsForDebug(config={'emission_alphabet_size': A,
                                               'exchangeabilities_mat': exch_mat,
                                               'norm_rate_matrix': True})
        
        init_params = my_gtr_model.init(rngs = jax.random.key(0),
                                    logprob_equl = np.log(self.equl),
                                    rate_multipliers = rate_multipliers,
                                    t_array = self.t_array,
                                    sow_intermediates=False,
                                    return_cond=True,
                                    return_intermeds=False)
        
        cond_logprob_gtr,_ = my_gtr_model.apply(variables = init_params,
                                  logprob_equl = np.log(self.equl),
                                  rate_multipliers = rate_multipliers,
                                  t_array = self.t_array,
                                  sow_intermediates=False,
                                  return_cond=True,
                                  return_intermeds=False)
        
        npt.assert_allclose(cond_logprob_f81, cond_logprob_gtr, atol=THRESHOLD)
    
    
    def _multiclass_f81_hand_calc(self, norm):
        T = self.t_array.shape[0]
        A = self.equl_multiclass.shape[1]
        C = self.equl_multiclass.shape[0]
        K = self.rate_mult_multiclass.shape[1]
        
        ### true
        if norm:
            normalization_factor = 1 / ( 1 - np.square(self.equl_multiclass).sum(axis=-1) ) #(C,)
        else:
            normalization_factor = jnp.ones( (C,) )  #(C,)
        
        cond_prob_true = np.zeros( (T, C, K, A, A) )
        for t_idx, t in enumerate(self.t_array):
            for c in range(C):
                for k in range(K):
                    mat = np.zeros((A,A))
                    for i in range(A):
                        for j in range(A):
                            pi_j = self.equl_multiclass[c,j]
                            
                            if i == j:
                                val = pi_j + (1-pi_j) * np.exp(-self.rate_mult_multiclass[c,k] * normalization_factor[c] * t)
                            
                            elif i != j:
                                val = pi_j * (1 - np.exp(-self.rate_mult_multiclass[c,k] * normalization_factor[c] * t) )
                            
                            mat[i,j] = val
                
                    cond_prob_true[t_idx, c, k, ...] = mat
        
        ### by my formula
        my_model = F81Logprobs(config={'num_mixtures': C,
                                        'norm_rate_matrix': norm,
                                        'norm_rate_mults': False},
                                name='mymod')
        
        dummy_variables = my_model.init( rngs = jax.random.key(0),
                                          logprob_equl = jnp.zeros( (C,A) ),
                                          log_class_probs = jnp.zeros( (C,) ),
                                          rate_multipliers = self.rate_mult_multiclass, 
                                          t_array = self.t_array,
                                          return_cond=True, 
                                          sow_intermediates = False)
        cond_prob_pred,_ = my_model.apply(variables = dummy_variables,
                                        logprob_equl = np.log(self.equl_multiclass),
                                        rate_multipliers = self.rate_mult_multiclass, 
                                        t_array = self.t_array,
                                        return_cond=True,
                                        sow_intermediates=False)
        
        cond_prob_pred = np.exp(cond_prob_pred)
        
        assert np.allclose( cond_prob_pred.sum(axis=-1),
                            np.ones( cond_prob_pred.sum(axis=-1).shape ) )
        
        npt.assert_allclose(cond_prob_true, cond_prob_pred, atol=THRESHOLD)
    
    def test_multiclass_f81_hand_calc_norm(self):
        self._multiclass_f81_hand_calc(norm=True)
        
    def test_multiclass_f81_hand_calc_not_normed(self):
        self._multiclass_f81_hand_calc(norm=False)
    
    def _multiclass_f81_against_gtr(self, norm):
        T = self.t_array.shape[0]
        A = self.equl_multiclass.shape[1]
        C = self.equl_multiclass.shape[0]
        
        if norm:
            rate = 1 / ( 1 - np.square(self.equl_multiclass).sum(axis=(-1)) )
        elif not norm:
            rate = self.rate_mult_multiclass
        
        # by f81
        my_model = F81Logprobs(config={'num_mixtures': C,
                                        'norm_rate_matrix': norm,
                                        'norm_rate_mults': False},
                                name='mymod')
        
        dummy_variables = my_model.init( rngs = jax.random.key(0),
                                          logprob_equl = jnp.zeros( (C,A) ),
                                          log_class_probs = jnp.zeros( (C,) ),
                                          rate_multipliers = self.rate_mult_multiclass, 
                                          t_array = self.t_array,
                                          return_cond=True, 
                                          sow_intermediates = False)
        cond_logprob_f81,_ = my_model.apply(variables = dummy_variables,
                                        logprob_equl = np.log(self.equl_multiclass),
                                        rate_multipliers = self.rate_mult_multiclass, 
                                        t_array = self.t_array,
                                        return_cond=True,
                                        sow_intermediates=False)
        
        cond_prob_f81 = np.exp(cond_logprob_f81)  #(T, C, K, A, A)
        assert np.allclose( cond_prob_f81.sum(axis=-1),
                            np.ones( cond_prob_f81.sum(axis=-1).shape ) )
        
        del my_model, dummy_variables, cond_logprob_f81
        
        # by gtr, with uniform exchangeabilities
        exch_mat = upper_tri_vector_to_sym_matrix( jnp.ones((6,)) )
        
        my_gtr_model = GTRLogprobsForDebug(config={'emission_alphabet_size': A,
                                               'exchangeabilities_mat': exch_mat,
                                               'norm_rate_matrix': norm})
        
        init_params = my_gtr_model.init(rngs = jax.random.key(0),
                                    logprob_equl = np.log(self.equl_multiclass),
                                    rate_multipliers = self.rate_mult_multiclass,
                                    t_array = self.t_array,
                                    sow_intermediates=False,
                                    return_cond=True,
                                    return_intermeds=False)
        
        cond_logprob_gtr,_ = my_gtr_model.apply(variables = init_params,
                                  logprob_equl = np.log(self.equl_multiclass),
                                  rate_multipliers = self.rate_mult_multiclass,
                                  t_array = self.t_array,
                                  sow_intermediates=False,
                                  return_cond=True,
                                  return_intermeds=False)
        
        cond_prob_gtr = jnp.exp(cond_logprob_gtr) #(T, C, A, A)
        
        npt.assert_allclose(cond_prob_f81, cond_prob_gtr, atol=THRESHOLD)
    
    def test_multiclass_f81_against_gtr_normed(self):
        self._multiclass_f81_against_gtr(norm=True)
        
    def test_multiclass_f81_against_gtr_not_normed(self):
        self._multiclass_f81_against_gtr(norm=False)
        
    
if __name__ == '__main__':
    unittest.main()