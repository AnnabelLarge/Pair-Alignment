#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 21:00:45 2025

@author: annabel_large


About:
======
Check the log-probability of some fake alignments by hand loops, and with
  my functions


run these first:
=================
unit_tests/check_joint_probability_of_substitution_calculation.py
unit_tests/conditional_probability_of_substitution_vs_cherryML.py
unit_tests/substitution_rate_matrix_unit_tests.py

"""
import jax
from jax import numpy as jnp
import numpy as np
from unit_tests.req_files.data_processing import (str_aligns_to_tensor,
                                                  summarize_alignment)
from models.simple_site_class_predict.emission_models import (_rate_matrix_from_exch_equl,
                                                              get_cond_logprob_emit_at_match_per_class,
                                                              get_joint_logprob_emit_at_match_per_class)
from models.simple_site_class_predict.PairHMM_indp_sites import (_score_alignment)


### generate fake alignments
fake_aligns = [ ('AC-A','D-ED'),
                ('D-ED','AC-A'),
                ('ECDAD','-C-A-'),
                ('-C-A-','ECDAD') ]

fake_aligns =  str_aligns_to_tensor(fake_aligns)
    
vmapped_summarize_alignment = jax.vmap(summarize_alignment, 
                                       in_axes=0, 
                                       out_axes=0)
counts =  vmapped_summarize_alignment( fake_aligns )
counts['emit_counts'] = counts['emit_counts'].sum(axis=0)


### params to work with
exchangeabilities = np.array([[0, 1, 2, 3],
                              [1, 0, 4, 5],
                              [2, 4, 0, 6],
                              [3, 5, 6, 0]])

equilibrium_distributions = np.array([0.1, 0.2, 0.3, 0.4])[None,:]

Q = _rate_matrix_from_exch_equl(exchangeabilities,
                                equilibrium_distributions,
                                norm=True)

t_array = np.array( [0.3, 1, 1.5] )

B = fake_aligns.shape[0]
L = fake_aligns.shape[1]
C = Q.shape[0]
T = t_array.shape[0]
A = Q.shape[1]

log_cond,_ = get_cond_logprob_emit_at_match_per_class(t_array = t_array,
                                                      scaled_rate_mat_per_class = Q)
del Q, exchangeabilities

log_joint = get_joint_logprob_emit_at_match_per_class(cond_logprob_emit_at_match_per_class = log_cond,
                                                      log_equl_dist_per_class = np.log(equilibrium_distributions))


### calculate by loops
true_scores = np.zeros( (T,B) )
for t in range(T):
    for b in range(B):
        for l in range(L):
            anc_tok, desc_tok, alignment_tok = fake_aligns[b, l, :]
            if alignment_tok == 1:
                logprob_of_this_column = log_joint[t,0,anc_tok-3, desc_tok-3]
                true_scores[t,b] += logprob_of_this_column


### calculate with my function
match_counts = counts['match_counts'][:, :4, :4]
pred_scores = _score_alignment(subCounts = match_counts,
                      insCounts = np.zeros((B,A)),
                      delCounts = np.zeros((B,A)),
                      transCounts = np.zeros((B, 4, 4)),
                      logprob_emit_at_match = log_joint[:,0,...],
                      logprob_emit_at_indel = np.zeros((A)),
                      transit_mat = np.zeros((T,4,4)))

assert np.allclose(pred_scores, true_scores)

print('[PASS] hand loop matches implementation used in pairhmm models')



### TODO: calculate with full GTR model, to prove an extra, extra point




