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
from scipy.special import softmax

import numpy.testing as npt
import unittest

from models.simple_site_class_predict.emission_models import F81Logprobs,GTRLogprobs
from models.simple_site_class_predict.model_functions import upper_tri_vector_to_sym_matrix

THRESHOLD = 1e-6

class GTRLogprobsForDebug(GTRLogprobs):
    def _get_square_exchangeabilities_matrix(self,*args,**kwargs):
        return self.config['exchangeabilities_mat']

def check_rows_sum_to_one(mat):
    npt.assert_allclose( mat.sum(axis=-1),
                         np.ones( mat.sum(axis=-1).shape ), 
                         atol=THRESHOLD )
    
def check_final_mat_sums_to_one(mat):
    assert mat.shape[-2] == mat.shape[-1]
    matsum = mat.sum( axis=(-2, -1) )
    npt.assert_allclose( matsum,
                         np.ones( matsum.shape ), 
                         atol=THRESHOLD )

class TestF81(unittest.TestCase):
    def setUp(self):
        self.t_array = np.array([0.1, 0.3, 0.5, 0.7, 0.9]) #(T=5)
        T = self.t_array.shape[0]

        # single class
        self.equl = np.array([0.1, 0.2, 0.3, 0.4])[None,None,:] #(C_tr=1, C_sites=1, A=4)
        A = self.equl.shape[-1]

        # mixtures of transition classes (domain * fragments), mixtures
        #   of site classes, mixture of rates
        C_tr = 2
        C_sites = 3 
        K = 6
        
        logits = np.random.rand( C_tr, C_sites, K ) 
        self.rate_mult_multiclass = softmax(logits, axis=-1) #(C_tr=3, C_sites=6, K=2)
        del logits
        
        logits = np.random.rand( C_tr, C_sites, A ) 
        self.equl_multiclass = softmax(logits, axis=-1) #(C_tr, C_sites, A)
        del logits
        
        self.A = A
        self.T = T
        self.C_tr = C_tr
        self.C_sites = C_sites
        self.K = K
    
    def test_one_f81_hand_calc(self,):
        """
        because there's only one class: normalize to 1 substitution per site
          by default
        """
        T = self.T
        A = self.A
        C_tr = 1
        C_sites = 1
        K = 1
        
        ### true
        rate_multipliers = np.ones( (C_tr, C_sites, K) )
        normalization_factor = 1 / ( 1 - np.square(self.equl).sum(axis=-1) )
        cond_prob_true = np.zeros( (T, C_tr, C_sites, K, A, A) )
        for t_idx, t in enumerate(self.t_array):
            mat = np.zeros((A,A))
            for i in range(A):
                for j in range(A):
                    pi_j = self.equl[0,0,j]
                    
                    if i == j:
                        val = pi_j + (1-pi_j) * np.exp(-normalization_factor * rate_multipliers * t)
                    
                    elif i != j:
                        val = pi_j * (1 - np.exp(-normalization_factor * rate_multipliers * t) )
                    
                    mat[i,j] = val
            
            cond_prob_true[t_idx, 0, 0, 0, ...] = mat
        
        ### by my formula
        my_model = F81Logprobs(config={'num_domain_mixtures': 1,
                                       'num_fragment_mixtures': 1,
                                       'num_site_mixtures': 1,
                                       'k_rate_mults': 1,
                                       'norm_rate_matrix': True},
                                name='mymod')
        cond_prob_pred,_ = my_model.apply(variables = {},
                                        log_equl_dist = np.log(self.equl),
                                        rate_multipliers = rate_multipliers, 
                                        t_array = self.t_array,
                                        return_cond=True,
                                        sow_intermediates=False)
        cond_prob_pred = np.exp(cond_prob_pred)
        
        # check shape
        npt.assert_allclose( cond_prob_pred.shape, (T, C_tr, C_sites, K, A, A) )
        
        # make sure rows sum to one
        check_rows_sum_to_one( cond_prob_pred )
        
        # check values
        npt.assert_allclose(cond_prob_true, cond_prob_pred, atol=THRESHOLD)
        

    def test_one_f81_against_gtr(self):
        """
        because there's only one class: normalize to 1 substitution per site
          by default
        """
        T = self.T
        A = self.A
        C_tr = 1
        C_sites = 1
        K = 1
        
        rate_multipliers = np.ones( (C_tr, C_sites, K) )
        
        ### by f81
        my_model = F81Logprobs(config={'num_domain_mixtures': 1,
                                       'num_fragment_mixtures': 1,
                                       'num_site_mixtures': 1,
                                       'k_rate_mults': 1,
                                       'norm_rate_matrix': True},
                                name='mymod')
        cond_logprob_f81,_ = my_model.apply(variables = {},
                                        log_equl_dist = np.log(self.equl),
                                        rate_multipliers = rate_multipliers, 
                                        t_array = self.t_array,
                                        return_cond=True,
                                        sow_intermediates=False)
        del my_model
        
        # check shape
        npt.assert_allclose( cond_logprob_f81.shape, (T, C_tr, C_sites, K, A, A) )
        
        # make sure rows sum to one
        check_rows_sum_to_one( np.exp(cond_logprob_f81) )
        
        
        ### by gtr, with uniform exchangeabilities
        exch_mat = upper_tri_vector_to_sym_matrix( jnp.ones((6,)) )
        
        my_gtr_model = GTRLogprobsForDebug(config={'num_domain_mixtures': 1,
                                                   'num_fragment_mixtures': 1,
                                                   'num_site_mixtures': 1,
                                                   'k_rate_mults': 1,
                                                   'emission_alphabet_size': A,
                                                   'exchangeabilities_mat': exch_mat,
                                                   'norm_rate_matrix': True})
        
        init_params = my_gtr_model.init(rngs = jax.random.key(0),
                                    log_equl_dist = np.log(self.equl),
                                    rate_multipliers = rate_multipliers,
                                    t_array = self.t_array,
                                    sow_intermediates=False,
                                    return_cond=True,
                                    return_intermeds=False)
        
        cond_logprob_gtr,_ = my_gtr_model.apply(variables = init_params,
                                  log_equl_dist = np.log(self.equl),
                                  rate_multipliers = rate_multipliers,
                                  t_array = self.t_array,
                                  sow_intermediates=False,
                                  return_cond=True,
                                  return_intermeds=False)
        
        # check shape
        npt.assert_allclose( cond_logprob_gtr.shape, (T, C_tr, C_sites, K, A, A) )
        
        # make sure rows sum to one
        check_rows_sum_to_one( np.exp(cond_logprob_gtr) )
        
        
        ### check values
        npt.assert_allclose(cond_logprob_f81, cond_logprob_gtr, atol=THRESHOLD)
    
    
    def _multiclass_f81_hand_calc(self, norm):
        T = self.T
        A = self.A
        C_tr = self.C_tr
        C_sites = self.C_sites
        K = self.K
        
        ### true
        if norm:
            normalization_factor = 1 / ( 1 - np.square(self.equl_multiclass).sum(axis=-1) ) #(C_tr, C_sites)
        else:
            normalization_factor = jnp.ones( (C_tr, C_sites) ) #(C_tr, C_sites)
        
        cond_prob_true = np.zeros( (T, C_tr, C_sites, K, A, A) )
        for t_idx, t in enumerate(self.t_array):
            for c_tr in range(C_tr):
                for c_s in range(C_sites):
                    for k in range(K):
                        mat = np.zeros((A,A))
                        for i in range(A):
                            for j in range(A):
                                pi_j = self.equl_multiclass[c_tr, c_s, j]
                                operand = self.rate_mult_multiclass[c_tr, c_s, k] * normalization_factor[c_tr, c_s] * t
                                
                                if i == j:
                                    val = pi_j + (1-pi_j) * np.exp(-operand)
                                
                                elif i != j:
                                    val = pi_j * (1 - np.exp(-operand) )
                                
                                mat[i,j] = val
                    
                        cond_prob_true[t_idx, c_tr, c_s, k, ...] = mat
        
        # check shape
        npt.assert_allclose( cond_prob_true.shape, (T, C_tr, C_sites, K, A, A) )
        
        # make sure rows sum to one
        check_rows_sum_to_one( cond_prob_true ) 
        
        
        ### by my formula
        my_model = F81Logprobs(config={'num_domain_mixtures': 1,
                                       'num_fragment_mixtures': C_tr,
                                       'num_site_mixtures': C_sites,
                                       'k_rate_mults': K,
                                       'norm_rate_matrix': norm,
                                       'norm_rate_mults': False},
                                name='mymod')
        
        dummy_variables = my_model.init( rngs = jax.random.key(0),
                                          log_equl_dist = jnp.zeros( (C_tr, C_sites, A) ),
                                          rate_multipliers = self.rate_mult_multiclass, 
                                          t_array = self.t_array,
                                          return_cond=True, 
                                          sow_intermediates = False)
        
        cond_logprob_pred,_ = my_model.apply(variables = dummy_variables,
                                        log_equl_dist = np.log(self.equl_multiclass),
                                        rate_multipliers = self.rate_mult_multiclass, 
                                        t_array = self.t_array,
                                        return_cond=True,
                                        sow_intermediates=False)
        
        cond_prob_pred = np.exp(cond_logprob_pred)
        
        # check shape
        npt.assert_allclose( cond_prob_pred.shape, (T, C_tr, C_sites, K, A, A) )
        
        # make sure rows sum to one
        check_rows_sum_to_one( cond_prob_pred )
        
        # check values
        npt.assert_allclose(cond_prob_true, cond_prob_pred, atol=THRESHOLD)
    
    def test_multiclass_f81_hand_calc_norm(self):
        self._multiclass_f81_hand_calc(norm=True)
        
    def test_multiclass_f81_hand_calc_not_normed(self):
        self._multiclass_f81_hand_calc(norm=False)
    
    def _multiclass_f81_against_gtr(self, norm):
        T = self.T
        A = self.A
        C_tr = self.C_tr
        C_sites = self.C_sites
        K = self.K
        
        if norm:
            rate = 1 / ( 1 - np.square(self.equl_multiclass).sum(axis=(-1)) )
        elif not norm:
            rate = self.rate_mult_multiclass
        
        ### by f81
        my_model = F81Logprobs(config={'num_domain_mixtures': 1,
                                       'num_fragment_mixtures': C_tr,
                                       'num_site_mixtures': C_sites,
                                       'k_rate_mults': K,
                                       'norm_rate_matrix': norm,
                                       'norm_rate_mults': False},
                                name='mymod')
        
        dummy_variables = my_model.init( rngs = jax.random.key(0),
                                          log_equl_dist = jnp.zeros( (C_tr, C_sites, A) ),
                                          rate_multipliers = self.rate_mult_multiclass, 
                                          t_array = self.t_array,
                                          return_cond=True, 
                                          sow_intermediates = False)
        
        cond_logprob_f81,_ = my_model.apply(variables = dummy_variables,
                                        log_equl_dist = np.log(self.equl_multiclass),
                                        rate_multipliers = self.rate_mult_multiclass, 
                                        t_array = self.t_array,
                                        return_cond=True,
                                        sow_intermediates=False)
        
        cond_prob_f81 = np.exp(cond_logprob_f81)
        
        # check shape
        npt.assert_allclose( cond_prob_f81.shape, (T, C_tr, C_sites, K, A, A) )
        
        # make sure rows sum to one
        check_rows_sum_to_one( cond_prob_f81 )
        
        del my_model, dummy_variables, cond_logprob_f81
        
        
        ### by gtr, with uniform exchangeabilities
        exch_mat = upper_tri_vector_to_sym_matrix( jnp.ones((6,)) )
        
        my_gtr_model = GTRLogprobsForDebug(config={'num_domain_mixtures': 1,
                                                   'num_fragment_mixtures': C_tr,
                                                   'num_site_mixtures': C_sites,
                                                   'k_rate_mults': K,
                                                   'emission_alphabet_size': A,
                                                   'exchangeabilities_mat': exch_mat,
                                                   'norm_rate_matrix': norm})
        
        init_params = my_gtr_model.init(rngs = jax.random.key(0),
                                    log_equl_dist = np.log(self.equl_multiclass),
                                    rate_multipliers = self.rate_mult_multiclass,
                                    t_array = self.t_array,
                                    sow_intermediates=False,
                                    return_cond=True,
                                    return_intermeds=False)
        
        cond_logprob_gtr,_ = my_gtr_model.apply(variables = init_params,
                                  log_equl_dist = np.log(self.equl_multiclass),
                                  rate_multipliers = self.rate_mult_multiclass,
                                  t_array = self.t_array,
                                  sow_intermediates=False,
                                  return_cond=True,
                                  return_intermeds=False)
        
        cond_prob_gtr = jnp.exp(cond_logprob_gtr)
        
        # check shape
        npt.assert_allclose( cond_prob_gtr.shape, (T, C_tr, C_sites, K, A, A) )
        
        # make sure rows sum to one
        check_rows_sum_to_one( cond_prob_gtr )
        
        
        ### check values
        npt.assert_allclose(cond_prob_f81, cond_prob_gtr, atol=THRESHOLD)
    
    def test_multiclass_f81_against_gtr_normed(self):
        self._multiclass_f81_against_gtr(norm=True)
        
    def test_multiclass_f81_against_gtr_not_normed(self):
        self._multiclass_f81_against_gtr(norm=False)
        
    
if __name__ == '__main__':
    unittest.main()