#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 20:47:38 2025

@author: annabel_large


About:
======
Confirm that joint probability of emission is calculated as expected, vs 
  a hand-done loop


run these first:
=================
unit_tests/conditional_probability_of_substitution_vs_cherryML.py
unit_tests/substitution_rate_matrix_unit_tests.py

"""
import numpy as np
from models.simple_site_class_predict.emission_models import (_rate_matrix_from_exch_equl,
                                                              get_cond_logprob_emit_at_match_per_class,
                                                              get_joint_logprob_emit_at_match_per_class)


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

t_array = np.array( [0.3, 1, 1.5] )

C = Q.shape[0]
T = t_array.shape[0]
A = Q.shape[1]

log_cond,_ = get_cond_logprob_emit_at_match_per_class(t_array = t_array,
                                                      scaled_rate_mat_per_class = Q)
del Q, exchangeabilities


### manually calculate the joint log probability
true_joint_logprob = np.zeros(log_cond.shape)
for t in range(T):
    for c in range(C):
        for i in range(A):
            for j in range(A):
                true_joint_logprob[t,c,i,j] = np.log(equilibrium_distributions[c,i]) + log_cond[t,c,i,j]

pred_joint_logprob = get_joint_logprob_emit_at_match_per_class(cond_logprob_emit_at_match_per_class = log_cond,
                                                               log_equl_dist_per_class = np.log(equilibrium_distributions))

assert np.allclose(true_joint_logprob, pred_joint_logprob)

print('[PASS] Joint probability of substitutions matches manual python loop')

        
