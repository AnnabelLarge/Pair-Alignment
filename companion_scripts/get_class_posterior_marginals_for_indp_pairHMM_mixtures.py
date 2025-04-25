#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 16:04:59 2025

@author: annabel_large

P(class=c|emissions) = P(emissions | class=c) P(class=c) / P( emissions )

"""
import numpy as np
import pickle
from scipy.linalg import expm
from scipy.special import logsumexp

in_folder = 'RESULTS_2indp-classes_TKF92_bound-sigmoid'
model_ckpts_folder = f'{in_folder}/model_ckpts'
out_arrs_folder = f'{in_folder}/out_arrs'


def safe_log(x):
    return np.where(x > 0,
                    np.log(x),
                    np.finfo('float32').smallest_normal )



######################
### build matrices   #
######################
### mixture params
with open(f'{out_arrs_folder}/PARAMS_class_probs.txt','r') as f:
    class_probs = np.array([float(l.strip()) for l in f])
log_class_probs = safe_log(class_probs)
del class_probs, f


### emissions
# at indels
with open(f'{out_arrs_folder}/PARAMS-ARR_equilibriums.npy', 'rb') as g:
    equl_dist_per_class = np.load(g)
log_equl_dist_per_class = safe_log(equl_dist_per_class)
weighted_logprob_emit_at_indel = log_equl_dist_per_class + log_class_probs[:, None]
del equl_dist_per_class, g

# joint logprob of emissions at match sites, per class
# to_exp is: rate_mult * ( X * diag(pi) ) * t
with open(f'{out_arrs_folder}/to_expm.npy','rb') as g:
    to_expm = np.load(g)

cond_prob_emit_at_match_per_class = expm(to_expm)
cond_logprob_emit_at_match_per_class = safe_log( cond_prob_emit_at_match_per_class )
joint_logprob_emit_at_match_per_class = ( cond_logprob_emit_at_match_per_class + 
                                          log_equl_dist_per_class[None,:,:,None] )
weighted_joint_logprob_emit_at_match_per_class = ( log_class_probs[None,:,None,None] + 
                                                   joint_logprob_emit_at_match_per_class )
del cond_prob_emit_at_match_per_class, log_class_probs
del cond_logprob_emit_at_match_per_class, joint_logprob_emit_at_match_per_class
del log_equl_dist_per_class, to_expm, g


### class marginals
# at match sites
match_denom = logsumexp(weighted_joint_logprob_emit_at_match_per_class,
                        axis=1)
match_site_class_log_marginals = (weighted_joint_logprob_emit_at_match_per_class -
                              match_denom[:,None,:,:])
match_site_class_marginals = np.exp(match_site_class_log_marginals)
see = match_site_class_marginals[0,...]

# at insert and delete sites
indel_denom = logsumexp(weighted_logprob_emit_at_indel,
                        axis=0)
indel_site_class_log_marginals = (weighted_logprob_emit_at_indel -
                              indel_denom[None,:])
indel_site_class_marginals = np.exp( indel_site_class_log_marginals )



