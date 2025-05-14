#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 20:47:38 2025

@author: annabel_large


About:
======
3rd test for substitution models

Confirm that joint probability of emission is calculated as expected, vs 
  a hand-done loop

"""
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

import numpy.testing as npt
import unittest

from models.simple_site_class_predict.model_functions import (rate_matrix_from_exch_equl,
                                                              get_cond_logprob_emit_at_match_per_class,
                                                              get_joint_logprob_emit_at_match_per_class)

THRESHOLD = 1e-6


class TestJointProbSubs(unittest.TestCase):
    """
    SUBSTITUTION PROCESS SCORING TEST 3
    
    
    C: hidden site classes
    T: branch lengths (time)
    A: alphabet
    
    About
    ------
    test get_joint_logprob_emit_at_match_per_class, which gets the joint
        probability of substitution x->y for every match site
    
    this is really just testing the underlying einsum recipe...
      
    """
    def test_get_joint_logprob_emit_at_match_per_class(self):
        """
        PURPOSE: test get_joint_logprob_emit_at_match_per_class, which gets 
            the joint probability of substitution x->y for every match site
        """
        ### params to work with
        exchangeabilities = np.array([[0, 1, 2, 3],
                                      [1, 0, 4, 5],
                                      [2, 4, 0, 6],
                                      [3, 5, 6, 0]]) #(A,A)
        
        equilibrium_distribution_1 = np.array([0.1, 0.2, 0.3, 0.4])
        equilibrium_distribution_2 = np.array([0.4, 0.3, 0.2, 0.1])
        equilibrium_distributions = np.stack([equilibrium_distribution_1,
                                              equilibrium_distribution_2]) #(C,A)
        del equilibrium_distribution_1, equilibrium_distribution_2
        
        Q = rate_matrix_from_exch_equl(exchangeabilities,
                                        equilibrium_distributions,
                                        norm=True) #(C,A,A)
        
        t_array = np.array( [0.3, 1, 1.5] ) #(T,)
        
        C = Q.shape[0]
        T = t_array.shape[0]
        A = Q.shape[1]
        
        log_cond,_ = get_cond_logprob_emit_at_match_per_class(t_array = t_array,
                                                              scaled_rate_mat_per_class = Q) #(T,C,A,A)
        del Q, exchangeabilities
        
        
        ### manually calculate the joint log probability
        true_joint_logprob = np.zeros(log_cond.shape) #(T,C,A,A)
        for t in range(T):
            for c in range(C):
                for i in range(A):
                    for j in range(A):
                        true_joint_logprob[t,c,i,j] = np.log(equilibrium_distributions[c,i]) + log_cond[t,c,i,j]
        
        pred_joint_logprob = get_joint_logprob_emit_at_match_per_class(cond_logprob_emit_at_match_per_class = log_cond,
                                                                       log_equl_dist_per_class = np.log(equilibrium_distributions))  #(T,C,A,A)
        
        npt.assert_allclose(true_joint_logprob, pred_joint_logprob, atol=THRESHOLD)

if __name__ == '__main__':
    unittest.main()