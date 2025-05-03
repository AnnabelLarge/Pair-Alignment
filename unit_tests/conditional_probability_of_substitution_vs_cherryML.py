#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 20:28:18 2025

@author: annabel_large


About:
======
Confirm that conditional probability of emission is calculated correclty,
  using the corresponding function from cherryML


run this first:
===============
unit_tests/substitution_rate_matrix_unit_tests.py
"""
import numpy as np
from models.simple_site_class_predict.emission_models import (_rate_matrix_from_exch_equl,
                                                              get_cond_logprob_emit_at_match_per_class)


### params to work with
exchangeabilities = np.array([[0, 1, 2, 3],
                              [1, 0, 4, 5],
                              [2, 4, 0, 6],
                              [3, 5, 6, 0]])

equilibrium_distribution_1 = np.array([0.1, 0.2, 0.3, 0.4])
equilibrium_distribution_2 = np.array([0.4, 0.3, 0.2, 0.1])
equilibrium_distributions = np.stack([equilibrium_distribution_1,
                                      equilibrium_distribution_2])
del equilibrium_distribution_1, equilibrium_distribution_2

Q = _rate_matrix_from_exch_equl(exchangeabilities,
                                equilibrium_distributions,
                                norm=True)
del exchangeabilities, equilibrium_distributions

t_array = np.array( [0.3, 1, 1.5] )

C = Q.shape[0]
T = t_array.shape[0]
A = Q.shape[1]


###############################
### functions from cherryML   #
###############################
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


cherryml_results = np.zeros( (T,C,A,A) )
for c in range(C):
    out = cherryml_matrix_exp(Q=Q[c,...],
                              exponents=t_array)
    for t in range(T):
        cherryml_results[t,c,...] = out[t,...]



####################
### my functions   #
####################
log_pred,_ = get_cond_logprob_emit_at_match_per_class(t_array = t_array,
                                                      scaled_rate_mat_per_class = Q)

assert np.allclose( np.exp(log_pred), cherryml_results )

print('[PASS] conditional probability of substitutions matches cherryML implementation')
