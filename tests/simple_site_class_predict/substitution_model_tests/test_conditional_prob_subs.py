#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 20:28:18 2025

@author: annabel_large


About:
======
2nd test for substitution models

Confirm that conditional probability of emission is calculated correclty,
  using the corresponding function from cherryML
"""
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import numpy.testing as npt
import unittest

from models.simple_site_class_predict.model_functions import (rate_matrix_from_exch_equl,
                                                              cond_logprob_emit_at_match_per_mixture)

from models.simple_site_class_predict.emission_models import GTRLogprobs


THRESHOLD = 1e-6

###############################################################################
### helpers   #################################################################
###############################################################################
class GTRLogprobsForDebug(GTRLogprobs):
    def _get_square_exchangeabilities_matrix(self,*args,**kwargs):
        return self.config['exchangeabilities_mat']

def cherryml_matrix_exp(Q, exponents):
    """
    This is copied verbatim from cherryML repo
    
    matrix_exp(t * Q) = P2 @ U @ np.diag(np.exp(length * D)) @ U_t @ P1
    """
    
    ### factorize Q rate matrix
    eigvals, eigvecs = np.linalg.eig(Q.transpose())
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    eigvals = np.abs(eigvals)
    index = np.argmin(eigvals)
    stationary_dist = eigvecs[:, index]
    stationary_dist = stationary_dist / sum(stationary_dist)

    P1 = np.diag(np.sqrt(stationary_dist))
    P2 = np.diag(np.sqrt(1 / stationary_dist))

    S = P1 @ Q @ P2

    D, U = np.linalg.eigh(S)
    
    P2 = P2
    U = U
    D = D
    U_t = U.transpose()
    P1 = P1
    
    
    ### compute matexp(Qt)
    num_states = len(D)
    batch_size = len(exponents)
    expTD = np.zeros(
        shape=(
            batch_size,
            num_states,
            num_states,
        )
    )
    
    for i in range(batch_size):
        expTD[i, :, :] = np.diag(np.exp(exponents[i] * D))
        expTQ = (P2[None, :, :] @ U[None, :, :]) @ (
            expTD @ (U_t[None, :, :] @ P1[None, :, :])
        )
    return expTQ


class TestConditionalProbSubs(unittest.TestCase):
    """
    SUBSTITUTION PROCESS SCORING TEST 2
    
    C: hidden site classes
    K: rate multipliers
    T: branch lengths (time)
    A: alphabet
    
    About
    ------
    test cond_logprob_emit_at_match_per_mixture, which gets the conditional
        probability of substitution x->y for every match site; compare against
        implementation from cherryML
      
    """
    def test_class_and_rate_mult_mixtures(self):
        """
        PURPOSE: cond_logprob_emit_at_match_per_mixture, which gets the 
            conditional probability of substitution x->y for every match site;
            use cherryML function as reference implementation
            
            
        C = 2
        K = 4
        """
        # params to work with
        exchangeabilities = np.array([[0, 1, 2, 3],
                                      [1, 0, 4, 5],
                                      [2, 4, 0, 6],
                                      [3, 5, 6, 0]]) #(A,A)
        
        equilibrium_distribution_1 = np.array([0.1, 0.2, 0.3, 0.4])
        equilibrium_distribution_2 = np.array([0.4, 0.3, 0.2, 0.1])
        equilibrium_distributions = np.stack([equilibrium_distribution_1,
                                              equilibrium_distribution_2]) #(C,A)
        del equilibrium_distribution_1, equilibrium_distribution_2
        
        raw_Q = rate_matrix_from_exch_equl(exchangeabilities,
                                            equilibrium_distributions,
                                          norm=True) #(C,A,A)
        del exchangeabilities, equilibrium_distributions
        
        rate_mults = np.array([[  1,   2,   3,   4],
                                [0.1, 0.2, 0.3, 0.4]]) #(C,K)
        Q = np.multiply( raw_Q[:,None,:,:], rate_mults[...,None,None] ) #(C, K, A, A)
        
        t_array = np.array( [0.3, 1, 1.5] ) #(T,)
       
        C = Q.shape[0]
        K = rate_mults.shape[1]
        T = t_array.shape[0]
        A = Q.shape[1]

        # with cherryML function
        cherryml_results = np.zeros( (T,C,K,A,A) )
        for c in range(C):
            for k in range(K):
                out = cherryml_matrix_exp(Q=Q[c,k,...],
                                          exponents=t_array) #(A, A)
                for t in range(T):
                    cherryml_results[t,c,k,...] = out[t,...]
        
        # with my functions
        log_pred = cond_logprob_emit_at_match_per_mixture(t_array = t_array,
                                                          scaled_rate_mat_per_mixture = Q) #(T,C,K,A,A)
        pred = np.exp(log_pred) #(T,C,K,A,A)
        
        npt.assert_allclose(cherryml_results, pred, atol=THRESHOLD)
    
    
    def test_GTRLogprobs_fw(self):
        """
        repeat, but use full GTRLogprobs function
            
        C = 2
        K = 4
        """
        # params to work with
        exchangeabilities = np.array([[0, 1, 2, 3],
                                      [1, 0, 4, 5],
                                      [2, 4, 0, 6],
                                      [3, 5, 6, 0]]) #(A,A)
        
        equilibrium_distribution_1 = np.array([0.1, 0.2, 0.3, 0.4])
        equilibrium_distribution_2 = np.array([0.4, 0.3, 0.2, 0.1])
        equilibrium_distributions = np.stack([equilibrium_distribution_1,
                                              equilibrium_distribution_2]) #(C,A)
        del equilibrium_distribution_1, equilibrium_distribution_2
        
        raw_Q = rate_matrix_from_exch_equl(exchangeabilities,
                                           equilibrium_distributions,
                                          norm=True) #(C,A,A)
        
        rate_mults = np.array([[  1,   2,   3,   4],
                               [0.1, 0.2, 0.3, 0.4]]) #(C,K)
        Q = np.multiply( raw_Q[:,None,:,:], rate_mults[...,None,None] ) #(C, K, A, A)
        
        t_array = np.array( [0.3, 1, 1.5] ) #(T,)
       
        C = Q.shape[0]
        K = rate_mults.shape[1]
        T = t_array.shape[0]
        A = Q.shape[1]

        # with cherryML function
        cherryml_results = np.zeros( (T,C,K,A,A) )
        for c in range(C):
            for k in range(K):
                out = cherryml_matrix_exp(Q=Q[c,k,...],
                                          exponents=t_array) #(A, A)
                for t in range(T):
                    cherryml_results[t,c,k,...] = out[t,...]
        
        # with my model
        my_model = GTRLogprobsForDebug(config={'emission_alphabet_size': A,
                                               'exchangeabilities_mat': exchangeabilities})
        
        init_params = my_model.init(rngs = jax.random.key(0),
                                    logprob_equl = np.log(equilibrium_distributions),
                                    rate_multipliers = rate_mults,
                                    t_array = t_array,
                                    sow_intermediates=False,
                                    return_cond=True,
                                    return_intermeds=False)
        
        log_pred,_ = my_model.apply(variables = init_params,
                                  logprob_equl = np.log(equilibrium_distributions),
                                  rate_multipliers = rate_mults,
                                  t_array = t_array,
                                  sow_intermediates=False,
                                  return_cond=True,
                                  return_intermeds=False)
        pred = np.exp(log_pred) #(T,C,K,A,A)
        
        npt.assert_allclose(cherryml_results, pred, atol=THRESHOLD)
        
    

if __name__ == '__main__':
    unittest.main()