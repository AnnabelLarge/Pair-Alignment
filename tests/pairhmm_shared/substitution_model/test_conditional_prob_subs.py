#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 20:28:18 2025

@author: annabel_large


About:
======
Confirm that conditional probability of emission is calculated correclty,
  using the corresponding function from cherryML
"""
import jax
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

def check_rows_sum_to_one(mat):
    npt.assert_allclose( mat.sum(axis=-1),
                         np.ones( mat.sum(axis=-1).shape ), 
                         atol=THRESHOLD )

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

        ### cherryML implementation, per mixture
        cherryml_results = np.zeros( (T,C,K,A,A) )
        for c in range(C):
            for k in range(K):
                out = cherryml_matrix_exp(Q=Q[0,c,k,...],
                                          exponents=t_array) #(A, A)
                for t in range(T):
                    cherryml_results[t,c,k,...] = out[t,...]
                    
        # make attributes for later
        self.exchangeabilities = exchangeabilities
        self.equilibrium_distributions = equilibrium_distributions
        self.rate_mults = rate_mults
        self.Q = Q
        self.t_array = t_array
        self.cherryml_results = cherryml_results
        self.C = C
        self.K = K
        self.T = T
        self.A = A
        
        
    def test_class_and_rate_mult_mixtures(self):
        """
        PURPOSE: cond_logprob_emit_at_match_per_mixture, which gets the 
            conditional probability of substitution x->y for every match site;
            use cherryML function as reference implementation
        """
        log_pred = cond_logprob_emit_at_match_per_mixture(t_array = self.t_array,
                                                          scaled_rate_mat_per_mixture = self.Q) #(T,1,C,K,A,A)
        
        # check shape
        npt.assert_allclose( log_pred.shape, (self.T, 1, self.C, self.K, self.A, self.A) )
        
        # make sure rows sum to one
        pred = np.exp(log_pred)
        check_rows_sum_to_one(pred)
        
        # check value
        pred = np.exp(log_pred) #(T,1,C,K,A,A)
        true = np.reshape( self.cherryml_results, pred.shape )
        npt.assert_allclose(true, pred, atol=THRESHOLD)
    
    
    def test_GTRLogprobs_fw(self):
        """
        repeat, but use full GTRLogprobs function
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
                                    return_cond=True,
                                    return_intermeds=False)
        
        log_pred,_ = my_model.apply(variables = init_params,
                                  log_equl_dist = np.log(self.equilibrium_distributions),
                                  rate_multipliers = self.rate_mults,
                                  t_array = self.t_array,
                                  sow_intermediates=False,
                                  return_cond=True,
                                  return_intermeds=False)
        pred = np.exp(log_pred) #(T,1,C,K,A,A)
        
        # check shape
        npt.assert_allclose( pred.shape, (self.T, 1, self.C, self.K, self.A, self.A) )
        
        # make sure rows sum to one
        pred = np.exp(log_pred)
        check_rows_sum_to_one(pred)
        
        # check value
        true = np.reshape( self.cherryml_results, pred.shape )
        npt.assert_allclose(true, pred, atol=THRESHOLD)
        

if __name__ == '__main__':
    unittest.main()