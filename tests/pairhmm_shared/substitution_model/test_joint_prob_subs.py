#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 20:47:38 2025

@author: annabel_large


About:
======
Confirm that joint probability of emission is calculated as expected, vs 
  a hand-done loop

"""
import jax
import numpy as np

import numpy.testing as npt
import unittest

from models.latent_class_mixtures.model_functions import (rate_matrix_from_exch_equl,
                                                              cond_logprob_emit_at_match_per_mixture,
                                                              joint_logprob_emit_at_match_per_mixture)
from models.latent_class_mixtures.emission_models import GTRLogprobs

THRESHOLD = 1e-6

class GTRLogprobsForDebug(GTRLogprobs):
    def _get_square_exchangeabilities_matrix(self,*args,**kwargs):
        return self.config['exchangeabilities_mat']

def check_final_mat_sums_to_one(mat):
    assert mat.shape[-2] == mat.shape[-1]
    matsum = mat.sum( axis=(-2, -1) )
    npt.assert_allclose( matsum,
                         np.ones( matsum.shape ), 
                         atol=THRESHOLD )

class TestJointProbSubs(unittest.TestCase):
    """
    C: hidden site classes
    K: rate multipliers
    T: branch lengths (time)
    A: alphabet
    
    About
    ------
    test joint_logprob_emit_at_match_per_mixture, which gets the joint
        probability of substitution x->y for every match site
    """
    def setUp(self):
        jax.config.update("jax_enable_x64", True)
        
        ### params to work with
        exchangeabilities = np.array([[0, 1, 2, 3],
                                      [1, 0, 4, 5],
                                      [2, 4, 0, 6],
                                      [3, 5, 6, 0]]) #(A,A)
        A = exchangeabilities.shape[0]
        
        equilibrium_distribution_1 = np.array([0.1, 0.2, 0.3, 0.4])
        equilibrium_distribution_2 = np.array([0.4, 0.3, 0.2, 0.1])
        equilibrium_distributions = np.stack([equilibrium_distribution_1,
                                              equilibrium_distribution_2])[None,...] #(1,C,A)
        C = equilibrium_distributions.shape[1]
        raw_Q = rate_matrix_from_exch_equl(exchangeabilities,
                                            equilibrium_distributions,
                                          norm=True) #(1,C,A,A)
        
        rate_mults = np.array([[   1,   2,   3,   4,   5],
                                [0.1, 0.2, 0.3, 0.4, 0.5]])[None,...] #(1,C,K)
        K = rate_mults.shape[2]
        
        Q = np.multiply( raw_Q[:,:,None,:,:], rate_mults[:,:,:,None,None] ) #(1,C, K, A, A)
        t_array = np.array( [0.3, 1, 1.5] ) #(T,)
        T = t_array.shape[0]
        
        log_cond = cond_logprob_emit_at_match_per_mixture(t_array = t_array,
                                                          scaled_rate_mat_per_mixture = Q) #(T,1,C,K,A,A)
        assert log_cond.shape == (T, 1, C, K, A, A)
        
        
        ### manually calculate the joint log probability
        true_joint_logprob = np.zeros(log_cond.shape) #(T,1,C,A,A)
        for t in range(T):
            for c in range(C):
                for k in range(K):
                    for i in range(A):
                        for j in range(A):
                            # P(x|c,k) * P(y|x,t,c,k) = P(x,y|t,c,k)
                            # logP(x|c,k) + logP(y|x,t,c,k) = logP(x,y|t,c,k)
                            true_joint_logprob[t,0,c,k,i,j] = np.log(equilibrium_distributions[0,c,i]) + log_cond[t,0,c,k,i,j] 
        check_final_mat_sums_to_one( np.exp( true_joint_logprob ) )
        
        ### final attributes
        self.exchangeabilities = exchangeabilities
        self.equilibrium_distributions = equilibrium_distributions
        self.rate_mults = rate_mults
        self.Q = Q
        self.t_array = t_array
        self.C = C
        self.K = K
        self.T = T
        self.A = A
        self.log_cond = log_cond
        self.true_joint_logprob = true_joint_logprob
    
    
    def test_class_and_rate_mult_mixtures(self):
        """
        PURPOSE: test joint_logprob_emit_at_match_per_mixture, which gets 
            the joint probability of substitution x->y for every match site
        """
        pred_joint_logprob = joint_logprob_emit_at_match_per_mixture(cond_logprob_emit_at_match_per_mixture = self.log_cond,
                                           log_equl_dist_per_mixture = np.log(self.equilibrium_distributions))  #(T,1,C,A,A)
        
        # check shape
        npt.assert_allclose( pred_joint_logprob.shape, (self.T, 1, self.C, self.K, self.A, self.A) )
        
        # check that joint prob sums to 1
        check_final_mat_sums_to_one( np.exp( pred_joint_logprob ) )
        
        # check value
        true = np.reshape(self.true_joint_logprob, pred_joint_logprob.shape)
        npt.assert_allclose(true, pred_joint_logprob, atol=THRESHOLD)
    
    
    def test_GTRLogprobs_fw(self):
        """
        repeat, but with full GTRLogProbs
        """
        my_model = GTRLogprobsForDebug(config={'num_domain_mixtures': 1,
                                               'num_fragment_mixtures': 1,
                                               'num_site_mixtures': self.C,
                                               'k_rate_mults': self.K,
                                               'emission_alphabet_size': self.A,
                                               'exchangeabilities_mat': self.exchangeabilities})
        
        init_params = my_model.init(rngs = jax.random.key(0),
                                    log_equl_dist = np.log(self.equilibrium_distributions),
                                    rate_multipliers = self.rate_mults,
                                    t_array = self.t_array,
                                    sow_intermediates=False,
                                    return_cond=False,
                                    return_intermeds=False)
        
        pred_joint_logprob,_ = my_model.apply(variables = init_params,
                                  log_equl_dist = np.log(self.equilibrium_distributions),
                                  rate_multipliers = self.rate_mults,
                                  t_array = self.t_array,
                                  sow_intermediates=False,
                                  return_cond=False,
                                  return_intermeds=False)
        
        # check shape
        npt.assert_allclose( pred_joint_logprob.shape, (self.T, 1, self.C, self.K, self.A, self.A) )
        
        # check that joint prob sums to 1
        check_final_mat_sums_to_one( np.exp( pred_joint_logprob ) )
        
        # check value
        true = np.reshape(self.true_joint_logprob, pred_joint_logprob.shape)
        npt.assert_allclose(true, pred_joint_logprob, atol=THRESHOLD)
    

if __name__ == '__main__':
    unittest.main()