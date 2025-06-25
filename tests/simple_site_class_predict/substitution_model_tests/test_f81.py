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

from models.simple_site_class_predict.model_functions import (upper_tri_vector_to_sym_matrix,
                                                              rate_matrix_from_exch_equl,
                                                              get_cond_logprob_emit_at_match_per_class)

THRESHOLD = 1e-6


class TestF81(unittest.TestCase):
    def setUp(self):
        self.equl = np.array([0.1, 0.2, 0.3, 0.4])[None,:] #(A=4)
        self.t_array = np.array([0.1, 0.3, 0.5, 0.7, 0.9]) #(T=5)
        
        self.rate_mult_multiclass = np.array([1, 2, 3]) #(C=3)
        self.equl_multiclass = np.array([[0.1, 0.2, 0.3, 0.4],
                                         [0.2, 0.3, 0.4, 0.1],
                                         [0.4, 0.3, 0.2, 0.1]]) #(C, A)
        
    
    def test_one_f81_hand_calc(self):
        """
        one class, normalized rate matrix
        """
        T = self.t_array.shape[0]
        A = self.equl.shape[-1]
        C = self.equl.shape[0]
        
        ### true
        normed_rate = 1 / ( 1 - np.square(self.equl).sum(axis=-1) )
        cond_prob_true = np.zeros( (T, C, A, A) )
        for t_idx, t in enumerate(self.t_array):
            mat = np.zeros((A,A))
            for i in range(A):
                for j in range(A):
                    pi_j = self.equl[0,j]
                    
                    if i == j:
                        val = pi_j + (1-pi_j) * np.exp(-normed_rate * t)
                    
                    elif i != j:
                        val = pi_j * (1 - np.exp(-normed_rate*t) )
                    
                    mat[i,j] = val
            
            cond_prob_true[t_idx, 0, ...] = mat
        
        ### by my formula
        # unit_norm_rate_multiplier = 1 / ( 1 - np.square(self.equl).sum(axis=(-1)) )
        my_model = F81Logprobs(config={'num_mixtures': 1},
                               name='mymod')
        cond_prob_pred = my_model._fill_f81(equl = self.equl,
                                            rate_multiplier = normed_rate, 
                                            t_array = self.t_array,
                                            return_cond=True)
        cond_prob_pred = np.exp(cond_prob_pred)
        assert np.allclose( cond_prob_pred.sum(axis=-1),
                            np.ones( cond_prob_pred.sum(axis=-1).shape ) )
        
        npt.assert_allclose(cond_prob_true, cond_prob_pred, atol=THRESHOLD)
        

    def test_one_f81_against_gtr(self):
        """
        one class, normalized rate matrix
        """
        T = self.t_array.shape[0]
        A = self.equl.shape[-1]
        C = 1
        unit_norm_rate_multiplier = 1 / ( 1 - np.square(self.equl).sum(axis=(-1)) )
        
        # by f81
        my_model = F81Logprobs(config={'num_mixtures': 1},
                               name='mymod')
        cond_prob_f81 = my_model._fill_f81(equl = self.equl,
                                           rate_multiplier = unit_norm_rate_multiplier, 
                                           t_array = self.t_array,
                                           return_cond=True)
        del my_model
        
        # by gtr, with uniform exchangeabilities
        exch_mat = upper_tri_vector_to_sym_matrix( jnp.ones((6,)) )
        rate_mat = rate_matrix_from_exch_equl(exchangeabilities = exch_mat,
                                              equilibrium_distributions = self.equl,
                                              norm = True)
        
        cond_prob_gtr,_ = get_cond_logprob_emit_at_match_per_class(t_array = self.t_array,
                                                                  scaled_rate_mat_per_class = rate_mat)
        
        npt.assert_allclose(cond_prob_f81, cond_prob_gtr, atol=THRESHOLD)
    
    
    def _multiclass_f81_hand_calc(self, norm):
        T = self.t_array.shape[0]
        A = self.equl_multiclass.shape[-1]
        C = self.equl_multiclass.shape[0]
        
        ### true
        if norm:
            rate = 1 / ( 1 - np.square(self.equl_multiclass).sum(axis=-1) ) #(C,)
        else:
            rate = self.rate_mult_multiclass #(C,)
        
        cond_prob_true = np.zeros( (T, C, A, A) )
        for t_idx, t in enumerate(self.t_array):
            for c in range(C):
                mat = np.zeros((A,A))
                for i in range(A):
                    for j in range(A):
                        pi_j = self.equl_multiclass[c,j]
                        
                        if i == j:
                            val = pi_j + (1-pi_j) * np.exp(-rate[c] * t)
                        
                        elif i != j:
                            val = pi_j * (1 - np.exp(-rate[c]*t) )
                        
                        mat[i,j] = val
            
                cond_prob_true[t_idx, c, ...] = mat
        
        ### by my formula
        my_model = F81Logprobs(config={'num_mixtures': C},
                               name='mymod')
        cond_prob_pred = my_model._fill_f81(equl = self.equl_multiclass,
                                            rate_multiplier = rate, 
                                            t_array = self.t_array,
                                            return_cond=True)
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
        A = self.equl_multiclass.shape[-1]
        C = self.equl_multiclass.shape[0]
        
        if norm:
            rate = 1 / ( 1 - np.square(self.equl_multiclass).sum(axis=(-1)) )
        elif not norm:
            rate = self.rate_mult_multiclass
        
        # by f81
        my_model = F81Logprobs(config={'num_mixtures': C},
                                name='mymod')
        cond_prob_f81 = my_model._fill_f81(equl = self.equl_multiclass,
                                            rate_multiplier = rate, 
                                            t_array = self.t_array,
                                            return_cond=True)
        del my_model
        
        # by gtr, with uniform exchangeabilities
        exch_mat = upper_tri_vector_to_sym_matrix( jnp.ones((6,)) )
        
        if norm:
            rate_mat = rate_matrix_from_exch_equl(exchangeabilities = exch_mat,
                                                  equilibrium_distributions = self.equl_multiclass,
                                                  norm = norm)
            
        if not norm:
            exch_mat = jnp.einsum('ij,c->cij', exch_mat, rate)
            rate_mat = np.zeros((C,A,A))
            for c in range(C):
                r = rate_matrix_from_exch_equl(exchangeabilities = exch_mat[c,...],
                                               equilibrium_distributions = self.equl_multiclass[c,...][None,...],
                                               norm = norm)
                rate_mat[c,...] = r
                
        cond_prob_gtr,_ = get_cond_logprob_emit_at_match_per_class(t_array = self.t_array,
                                                                  scaled_rate_mat_per_class = rate_mat)
        npt.assert_allclose(cond_prob_f81, cond_prob_gtr, atol=THRESHOLD)
    
    
    def test_multiclass_f81_against_gtr_normed(self):
        self._multiclass_f81_against_gtr(norm=True)
        
    def test_multiclass_f81_against_gtr_not_normed(self):
        self._multiclass_f81_against_gtr(norm=False)
        
    
if __name__ == '__main__':
    unittest.main()