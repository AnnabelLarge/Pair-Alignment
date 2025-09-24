#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 10:54:39 2025

@author: annabel

Make sure marginalization works as expected
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
from models.latent_class_mixtures.FragAndSiteClasses import FragAndSiteClasses

THRESHOLD = 1e-6


class TestMargOverSiteMixes(unittest.TestCase):
    """
    About
    ------
    make sure marginalization works as expected
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
        C_dom = 1
        C_frag = 2
        C_sites = 3
        K = 5
        S = 4 #four types of transitions: M, I, D, start/end
        
        # put together a batch for pairHMM
        vmapped_summarize_alignment = jax.vmap(summarize_alignment, 
                                               in_axes=0, 
                                               out_axes=0)
        counts =  vmapped_summarize_alignment( fake_aligns_pairhmm )
        
        pairhmm_batch = (fake_aligns_pairhmm, None)
        training_dset_emit_counts = counts['emit_counts'].sum(axis=0)
        
        # template config
        pairhmm_config_template = { 'num_domain_mixtures': C_dom,
                                    'num_fragment_mixtures': C_frag,
                                    'num_site_mixtures': C_sites,
                                    'k_rate_mults': K,
                                    'indp_rate_mults': False,
                                    'norm_rate_mults': True,
                                    'norm_rate_matrix': True,
                                    'times_from': 't_array_from_file',
                                    'exponential_dist_param': 1.1,
                                    'training_dset_emit_counts': training_dset_emit_counts,
                                    'emission_alphabet_size': A,
                                    'tkf_function': 'regular_tkf',
                                    'norm_reported_loss_by': 'desc_len',
                                    'subst_model_type': 'f81',
                                    'indel_model_type': 'tkf92' }
        
        pairhmm = FragAndSiteClasses(config=pairhmm_config_template,
                            name='pairhmm')
        exponential_dist_param = pairhmm_config_template['exponential_dist_param']
        

        ###################################
        ### generate scoring parameters   #
        ###################################
        init_params = pairhmm.init( rngs = jax.random.key(42),
                                    batch = pairhmm_batch,
                                    t_array = t_array,
                                    sow_intermediates = False)
        
        scoring_mat_dict = pairhmm.apply( variables=init_params,
                                          t_array=t_array,
                                          sow_intermediates=False,
                                          return_intermeds=True,
                                          return_all_matrices = True, 
                                          method = '_get_scoring_matrices')
        
        
        ##################################################################
        ### calculate F81 conditional probability per mixture, by hand   #
        ##################################################################
        true_cond_mat_all_mixes = np.zeros( (T, C_frag, C_sites, K, A, A) )
        for t_idx in range( T ):
            for c_fr in range(C_frag):
                for c_s in range(C_sites):
                    for k_idx in range(K):
                        mat_at_t = np.zeros( (A, A) )
                        equl_dist_for_class_c = np.exp(scoring_mat_dict['log_equl_dist_per_mixture'])[c_fr, c_s, ...] #(A,)
                        norm_factor_for_class_c = 1 / ( 1 - jnp.square(equl_dist_for_class_c).sum() )
                        
                        for anc_i in range( A ):
                            for desc_j in range( A ):
                                rate = scoring_mat_dict['rate_multipliers'][c_fr, c_s, k_idx]
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
                        
                        true_cond_mat_all_mixes[ t_idx,c_fr, c_s,k_idx,... ] = mat_at_t
        
        npt.assert_allclose( np.log(true_cond_mat_all_mixes), scoring_mat_dict['cond_subst_logprobs_per_mixture'] )
        
        
        ###########################################################################
        ### calculate conditional probability by hand (including marginalizing)   #
        ###########################################################################
        true_cond_prob = np.zeros( (T, C_frag, A, A) )
        
        for t_idx in range( T ):
            for c_fr in range(C_frag):
                # this does \sum_{c_sites} \sum_k P(c_sites, k | c_frag) * P(y | x, c_sites, k, c_frag, t) = P(y | x, c_frag, t)
                for c_s in range(C_sites):
                    for k_idx in range(K):
                        cond_prob = true_cond_mat_all_mixes[t_idx, c_fr, c_s, k_idx, ...] # (A_from, A_to)
                        
                        # P(c_site, k | c_frag) = P(c_site | c_frag) * P(k | c_site, c_frag)
                        prob_site_class = np.exp( scoring_mat_dict['log_site_class_probs'] )[c_fr, c_s]
                        prob_rate_mult_given_site_class = np.exp( scoring_mat_dict['log_rate_mult_probs'] )[c_fr, c_s, k_idx]
                        joint_mixture_weight = prob_site_class * prob_rate_mult_given_site_class
                        
                        # P(y, c_sites, k | x, c_frag, t) = P(y | x, c_frag, c_sites, k, t) * P(c_sites, k | c_frag)
                        joint_mat_times_mix_weight = joint_mixture_weight * cond_prob
                        true_cond_prob[t_idx, c_fr, ...] += joint_mat_times_mix_weight
                        
        npt.assert_allclose( np.log(true_cond_prob), scoring_mat_dict['cond_logprob_emit_at_match'] )
        
        
        #####################################################################
        ### calculate joint probability by hand (including marginalizing)   #
        #####################################################################
        true_joint_prob = np.zeros( (T, C_frag, A, A) )
        
        for t_idx in range( T ):
            for c_fr in range(C_frag):
                # this does \sum_{c_sites} \sum_k P(c_sites, k | c_frag) * P(x, y | c_sites, k, c_frag, t) = P(x, y | c_frag, t)
                for c_s in range(C_sites):
                    for k_idx in range(K):
                        # P(x, y | c_frag, c_sites, k, t) = P(x | c_frag, c_sites) * P(y | x, c_frag, c_sites, k, t)
                        cond_prob = true_cond_mat_all_mixes[t_idx, c_fr, c_s, k_idx, ...] # (A_from, A_to
                        anc_marg = np.exp(scoring_mat_dict['log_equl_dist_per_mixture'])[c_fr, c_s, ...] #(A,)
                        joint_prob = np.zeros( cond_prob.shape )
                        for i in range(A):
                            for j in range(A):
                                joint_prob[i,j] = anc_marg[i] * cond_prob[i,j]
                        
                        # P(c_site, k | c_frag) = P(c_site | c_frag) * P(k | c_site, c_frag)
                        prob_site_class = np.exp( scoring_mat_dict['log_site_class_probs'] )[c_fr, c_s]
                        prob_rate_mult_given_site_class = np.exp( scoring_mat_dict['log_rate_mult_probs'] )[c_fr, c_s, k_idx]
                        joint_mixture_weight = prob_site_class * prob_rate_mult_given_site_class
                        
                        # P(x, y, c_sites, k | c_frag, t) = P(x, y | c_frag, c_sites, k, t) * P(c_sites, k | c_frag)
                        joint_mat_times_mix_weight = joint_mixture_weight * joint_prob
                        true_joint_prob[t_idx, c_fr, ...] += joint_mat_times_mix_weight
                        
        npt.assert_allclose( np.log(true_joint_prob), scoring_mat_dict['joint_logprob_emit_at_match'] )
        
        
        ########################################################################
        ### calculate ancestor probability by hand (including marginalizing)   #
        ########################################################################
        true_equl_dist = np.zeros( (C_frag, A) )
        
        for c_fr in range(C_frag):
            # this loop does \sum_{c_sites} P(x | c_sites, c_frag) * P(c_sites | c_frag) = P(x | c_frag)
            for c_s in range(C_sites):
                # P(x | c_sites, c_frag)
                equl_dist_for_mixture = np.exp(scoring_mat_dict['log_equl_dist_per_mixture'])[c_fr, c_s, ...] #(A,)
                
                # P(c_sites | c_frag)
                prob_site_class = np.exp( scoring_mat_dict['log_site_class_probs'] )[c_fr, c_s]
                
                # P(x, c_sites | c_frag) = P(x | c_sites, c_frag) * P(c_sites | c_frag)
                weighted_equl_mix = equl_dist_for_mixture * prob_site_class
                
                true_equl_dist[c_fr, :] += weighted_equl_mix
        
        npt.assert_allclose( np.log(true_equl_dist), scoring_mat_dict['logprob_emit_at_indel'] )
        
        
        
if __name__ == '__main__':
    unittest.main()