#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 10:54:39 2025

@author: annabel

Make sure marginalization over k classes works as expected
"""
import jax
from jax import numpy as jnp
import numpy as np
import pickle
from functools import partial
jax.config.update("jax_enable_x64", True)

import numpy.testing as npt
import unittest

from tests.data_processing import (str_aligns_to_tensor,
                                   summarize_alignment)
from models.simple_site_class_predict.FragAndSiteClasses import FragAndSiteClasses

THRESHOLD = 1e-6


class TestMargOverKRateMults(unittest.TestCase):
    """
    About
    ------
    make sure marginalization over k possible rate multipliers works as expected
    """
    def test_rate_mult_calc(self):
        ###############################
        ### generate fake alignments  #
        ###############################
        fake_aligns = [ ('ECDADD','-C-D-A'),
                        ('-C-D-A','ECDADD') ]
        t_array = jnp.array([0.5, 1.0, 1.5]) #(T,)
        fake_aligns_pairhmm =  str_aligns_to_tensor(fake_aligns) #(B, L, 3)
        del fake_aligns
        
        # dims
        T = 3
        B = 2
        A = 20
        C = 2
        K = 4
        S = 4 #four types of transitions: M, I, D, start/end
        
        # put together a batch for pairHMM
        vmapped_summarize_alignment = jax.vmap(summarize_alignment, 
                                               in_axes=0, 
                                               out_axes=0)
        counts =  vmapped_summarize_alignment( fake_aligns_pairhmm )
        
        pairhmm_batch = (fake_aligns_pairhmm[...,[0,1,]],
                         None)
        training_dset_emit_counts = counts['emit_counts'].sum(axis=0)
        
        # template config
        pairhmm_config_template = {'num_mixtures': C,
                                    'k_rate_mults': K,
                                    'num_tkf_fragment_classes': C,
                                    'indp_rate_mults': False,
                                    'norm_rate_mults': True,
                                    'norm_rate_matrix': True,
                                    'times_from': 't_array_from_file',
                                    'exponential_dist_param': 1.1,
                                    'training_dset_emit_counts': training_dset_emit_counts,
                                    'emission_alphabet_size': A,
                                    'tkf_function_name': 'regular',
                                    'norm_reported_loss_by': 'desc_len',
                                    'subst_model_type': 'f81',
                                    'indel_model_type': 'tkf92'}
        
        pairhmm = FragAndSiteClasses(config=pairhmm_config_template,
                            name='pairhmm')
        exponential_dist_param = pairhmm_config_template['exponential_dist_param']
        
        # generate scoring parameters
        init_params = pairhmm.init( rngs = jax.random.key(42),
                                    batch = pairhmm_batch,
                                    t_array = t_array,
                                    sow_intermediates = False )
        
        scoring_mat_dict = pairhmm.apply( variables=init_params,
                                          t_array=t_array,
                                          sow_intermediates=False,
                                          return_intermeds=True,
                                          method = '_get_scoring_matrices')
        
        scoring_mat_dict.keys()
        
        
        # calculate substitution probability matrix by hand too
        true_cond_mat = np.zeros( (T, C, K, A, A) )
        for t_idx in range( T ):
            for c_idx in range(C):
                for k_idx in range(K):
                    mat_at_t = np.zeros( (A, A) )
                    
                    equl_dist_for_class_c = np.exp(scoring_mat_dict['logprob_emit_at_indel'])[c_idx,...] #(A,)
                    norm_factor_for_class_c = 1 / ( 1 - jnp.square(equl_dist_for_class_c).sum() )
                    
                    for anc_i in range( A ):
                        for desc_j in range( A ):
                            rate = scoring_mat_dict['rate_multipliers'][c_idx, k_idx]
                            pi_j = equl_dist_for_class_c[desc_j]
                            time = t_array[t_idx]
                            
                            exp_oper = np.exp( -rate * norm_factor_for_class_c * time )
                            
                            # for i != j: pi_j * ( 1 - exp(-rate*t) )
                            if anc_i != desc_j:
                                prob = pi_j * (1 - exp_oper)
                            
                            # for i == j: pi_j + (1-pi_j) * exp(-rate*t)
                            elif anc_i == desc_j:
                                prob = pi_j + (1 - pi_j) * exp_oper
                            
                            mat_at_t[anc_i, desc_j] = prob
                    
                    pred_mat_at_t = scoring_mat_dict['cond_subst_logprobs_per_mixture'][t_idx,c_idx,k_idx,...]
                    npt.assert_allclose( np.log(mat_at_t), pred_mat_at_t )
                    
                    true_cond_mat[ t_idx,c_idx,k_idx,... ] = mat_at_t
        
        # do marginalization over k in probability space; compare to predicted value
        true_joint_prob = np.zeros( (T, C, A, A) )
        
        for t_idx in range( T ):
            for c_idx in range(C):
                
                # marginalize over k rate multiplier mixtures
                # this is P(y | x, c, t)
                cond_mat_at_t_and_class_c = np.zeros( (A, A) )
                
                # this does \sum_k P(y, k | x, c, t) = P(y | x, c, t)
                for k_idx in range(K):
                    # P(k|c)
                    prob_rate_mult_given_class = np.exp( scoring_mat_dict['log_rate_mult_probs'][0, c_idx, k_idx, 0, 0] )
                    
                    # P(y|x,c,t,k) * P(k|c) = P(y, k | x, c, t)
                    mat_times_mix_weight = prob_rate_mult_given_class * true_cond_mat[t_idx, c_idx, k_idx]
                    
                    cond_mat_at_t_and_class_c += mat_times_mix_weight
                
                # get joint logprob
                equl_dist_for_class_c = np.exp(scoring_mat_dict['logprob_emit_at_indel'])[c_idx,...] #(A,)
                
                for anc_i in range(A):
                    for desc_j in range(A):
                        # P(x,y|c,t) = P(x|c) * P(y | x, c, t)
                        true_joint_prob[t_idx, c_idx, anc_i, desc_j] = equl_dist_for_class_c[anc_i] * cond_mat_at_t_and_class_c[anc_i, desc_j]
                
          
        npt.assert_allclose( np.log(true_joint_prob), 
                            scoring_mat_dict['joint_logprob_emit_at_match'] )
            
if __name__ == '__main__':
    unittest.main()