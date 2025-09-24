#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 14:41:49 2025

@author: annabel_large
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
from models.latent_class_mixtures.IndpSites import IndpSites

THRESHOLD = 1e-6


class TestIndpSiteClassesLoglikes(unittest.TestCase):
    """
    make sure that likelihood calculation is the same as hand-done calculations
    
    for this test, C refers to C_sites
    
    C_frag = 1
    C_dom = 1
    """
    def setUp(self):
        self.path = f'./tests/simple_site_class_predict/full_model_tests/req_files'
        self.req_files_path = f'{self.path}/req_files'
        
        
        ###############################
        ### generate fake alignments  #
        ###############################
        fake_aligns = [ ('ECDADD','-C-D-A'),
                        ('-C-D-A','ECDADD') ]
        self.t_array = jnp.array([0.5, 1.0, 1.5]) #(T,)
        fake_aligns_pairhmm =  str_aligns_to_tensor(fake_aligns) #(B, L, 3)
        del fake_aligns

        # dims
        self.T = 3
        self.B = 2
        self.A = 20
        self.C = 2
        self.K = 4
        self.S = 4 #four types of transitions: M, I, D, start/end
        
        # put together a batch for pairHMM
        vmapped_summarize_alignment = jax.vmap(summarize_alignment, 
                                               in_axes=0, 
                                               out_axes=0)
        counts =  vmapped_summarize_alignment( fake_aligns_pairhmm )
        
        self.pairhmm_batch = (counts['match_counts'],
                              counts['ins_counts'],
                              counts['del_counts'],
                              counts['transit_counts'],
                              None,
                              None,
                              )
        training_dset_emit_counts = counts['emit_counts'].sum(axis=0)
        
        # template config
        self.pairhmm_config_template = {'num_domain_mixtures': 1,
                                        'num_fragment_mixtures': 1,
                                        'num_site_mixtures': self.C,
                                        'k_rate_mults': self.K,
                                        'indp_rate_mults': False,
                                        'times_from': 't_array_from_file',
                                        'exponential_dist_param': 1.1,
                                        'training_dset_emit_counts': training_dset_emit_counts,
                                        'emission_alphabet_size': self.A,
                                        'tkf_function': 'regular_tkf',
                                        'norm_reported_loss_by': 'desc_len'}
        
    
    def _run_test(self,
                  subst_model_type,
                  indel_model_type):
        ### setup
        # init model
        to_add = {'subst_model_type': subst_model_type,
                  'indel_model_type': indel_model_type}
        config = {**self.pairhmm_config_template, **to_add}
        
        pairhmm = IndpSites(config=config,
                            name='pairhmm')
        exponential_dist_param = config['exponential_dist_param']
        
        # generate scoring parameters
        init_params = pairhmm.init( rngs = jax.random.key(42),
                                    batch = self.pairhmm_batch,
                                    t_array = self.t_array,
                                    sow_intermediates = False )
        
        scoring_mat_dict = pairhmm.apply( variables=init_params,
                                          t_array=self.t_array,
                                          sow_intermediates=False,
                                          return_intermeds=True,
                                          return_all_matrices=True,
                                          method = '_get_scoring_matrices')
        
        
        ### score sequences with IndpSites
        pred_scores = pairhmm.apply( variables=init_params,
                                     batch = self.pairhmm_batch,
                                     t_array = self.t_array,
                                     return_intermeds = True,
                                     method = 'calculate_all_loglikes')
        
        
        ### score by hand from intermediates
        # unpack
        match_counts = self.pairhmm_batch[0]
        ins_counts = self.pairhmm_batch[1]
        del_counts = self.pairhmm_batch[2]
        transit_counts = self.pairhmm_batch[3]
        
        # mixture params
        log_class_probs = scoring_mat_dict['log_site_class_probs'] #(C)
        log_rate_mult_probs = scoring_mat_dict['log_rate_mult_probs'] #(C,K)
        
        # emission scoring matrices
        log_equl_dist_per_mixture = scoring_mat_dict['log_equl_dist_per_mixture'] #(C, A)
        joint_subst_logprobs_per_mixture = scoring_mat_dict['joint_subst_logprobs_per_mixture'] #(T, C, K, A, A)
        
        # transition scoring matrices: no mixtures
        joint_transit_logprob = scoring_mat_dict['all_transit_matrices']['joint'] #(T, S, S)
        marginal_transit_logprob = scoring_mat_dict['all_transit_matrices']['marginal'] #(S, S)
        
        
        ### score by hand: joint
        emit_joint_score = np.zeros( (self.T, self.B) )
        transit_joint_score = np.zeros( (self.T, self.B) )
        for t in range( self.T ):
            for b in range( self.B ):
                # emissions from match sites
                for anc_i in range( self.A ):
                    for desc_j in range( self.A ):
                        prob_for_site = 0
                        for c in range(self.C):
                            for k in range(self.K):
                                # P(x,y|c,k,t)
                                prob_for_mix = np.exp(joint_subst_logprobs_per_mixture[t, c, k, anc_i, desc_j])
                                
                                # P(c,k)
                                mixture_weight = np.exp(log_rate_mult_probs[c,k] + log_class_probs[c])
                                
                                # P(x,y|c,k,t) * P(c,k) = P(x,y,c,k|t)
                                prob_for_site += prob_for_mix * mixture_weight
                        
                        logprob_for_site = np.log(prob_for_site)
                        count = match_counts[b, anc_i, desc_j]
                        emit_joint_score[t,b] += count * logprob_for_site
                        
                        del prob_for_site, c, k, prob_for_mix, mixture_weight, 
                        del logprob_for_site, count
                        
                # score by hand: single-sequence emissions
                for aa_idx in range( self.A ): 
                    anc_prob_for_site = 0
                    desc_prob_for_site = 0
                    for c in range(self.C):
                        # P(x|c), P(y|c)
                        anc_prob_for_mixture = np.exp( log_equl_dist_per_mixture[c,aa_idx] )
                        desc_prob_for_mixture = np.exp( log_equl_dist_per_mixture[c,aa_idx] )
                        
                        # P(c)
                        mixture_weight = np.exp( log_class_probs[c] )
                        
                        # P(x|c) * P(c) = P(x,c), same for y
                        anc_prob_for_site += mixture_weight * anc_prob_for_mixture
                        desc_prob_for_site += mixture_weight * desc_prob_for_mixture
                
                    anc_logprob_for_site = np.log( anc_prob_for_site )
                    desc_logprob_for_site = np.log( desc_prob_for_site )
                    
                    joint_ins_count = ins_counts[b, aa_idx] 
                    joint_del_count = del_counts[b, aa_idx]
                    emit_joint_score[t,b] += (joint_del_count * anc_logprob_for_site + 
                                              joint_ins_count * desc_logprob_for_site)
                    
                del anc_prob_for_site, desc_prob_for_site, anc_prob_for_mixture
                del desc_prob_for_mixture, mixture_weight, anc_logprob_for_site
                del desc_logprob_for_site, joint_ins_count, joint_del_count
                        
                
                ### score by hand: transitions
                # joint
                for transit_from in range(self.S):
                    for transit_to in range(self.S):
                        logprob = joint_transit_logprob[t, transit_from, transit_to]
                        count = transit_counts[b, transit_from, transit_to]
                        transit_joint_score[t, b] += logprob * count
                        
                del transit_from, transit_to, logprob, count
        
        
        ### score by hand: single-sequence marginals
        emit_anc_marg_score = np.zeros( (self.B) )
        emit_desc_marg_score = np.zeros( (self.B) )
        transit_anc_marg_score = np.zeros( (self.B) )
        transit_desc_marg_score = np.zeros( (self.B) )
        for b in range( self.B ):
            anc_len = 0
            desc_len = 0
               
            # emissions
            for aa_idx in range( self.A ): 
                anc_prob_for_site = 0
                desc_prob_for_site = 0
                for c in range(self.C):
                    # P(x|c), P(y|c)
                    anc_prob_for_mixture = np.exp( log_equl_dist_per_mixture[c,aa_idx] )
                    desc_prob_for_mixture = np.exp( log_equl_dist_per_mixture[c,aa_idx] )
                    
                    # P(c)
                    mixture_weight = np.exp( log_class_probs[c] )
                    
                    # P(x|c) * P(c) = P(x,c), same for y
                    anc_prob_for_site += mixture_weight * anc_prob_for_mixture
                    desc_prob_for_site += mixture_weight * desc_prob_for_mixture
            
                anc_logprob_for_site = np.log( anc_prob_for_site )
                desc_logprob_for_site = np.log( desc_prob_for_site )
                
                anc_count = match_counts[b, aa_idx, :].sum() + del_counts[b, aa_idx]
                desc_count = match_counts[b, :, aa_idx].sum() + ins_counts[b, aa_idx]
                
                emit_anc_marg_score[b] += anc_count * anc_logprob_for_site
                emit_desc_marg_score[b] += desc_count * desc_logprob_for_site
                
                anc_len += anc_count
                desc_len += desc_count
                
            del anc_prob_for_site, desc_prob_for_site, anc_prob_for_mixture
            del desc_prob_for_mixture, mixture_weight, anc_logprob_for_site
            del desc_logprob_for_site, anc_count, desc_count
            
            # geometric sequence length term
            start_to_emit = marginal_transit_logprob[1,0]
            emit_to_emit = marginal_transit_logprob[0,0]
            emit_to_end = marginal_transit_logprob[0,1]
            
            transit_anc_marg_score[b] = start_to_emit + (anc_len-1)*emit_to_emit + emit_to_end
            transit_desc_marg_score[b] = start_to_emit + (desc_len-1)*emit_to_emit + emit_to_end
            
            del start_to_emit, emit_to_emit, emit_to_end, anc_len, desc_len
        
        
        ### sum all scores
        log_true_joint_perTime = transit_joint_score + emit_joint_score #(T,B)
        log_true_anc = transit_anc_marg_score + emit_anc_marg_score  #(B,)
        log_true_desc = transit_desc_marg_score + emit_desc_marg_score  #(B,)
        
        del transit_joint_score, transit_anc_marg_score, transit_desc_marg_score
        del emit_joint_score, emit_anc_marg_score, emit_desc_marg_score
        
        
        ### postproc joint
        # P(t) = \lambda exp( -\lambda * t )
        # logP(t) = log(\lambda) - ( \lambda * t )
        logP_time = ( jnp.log(exponential_dist_param) - 
                      (exponential_dist_param * self.t_array) ) #(T,)
        
        # dt, log(dt)
        log_t_grid = jnp.log( self.t_array[1:] - self.t_array[:-1] ) #(T-1,)
        log_t_grid = jnp.concatenate( [log_t_grid, log_t_grid[-1][None] ], axis=0) #(T,)
        const = logP_time + log_t_grid # (T,)
        
        # add constant in log space, sum probabilities in probability space
        true_joint_perTime = np.exp( log_true_joint_perTime + const[:, None] ) #(T, B)
        true_joint = true_joint_perTime.sum(axis=0) #(B,)
                                   
        # cond = joint / anc marginal
        true_cond = true_joint / np.exp( log_true_anc ) #(B,)
        
        npt.assert_allclose( np.log( true_joint ), -pred_scores['joint_neg_logP'] ) 
        npt.assert_allclose( log_true_anc, -pred_scores['anc_neg_logP'] ) 
        npt.assert_allclose( log_true_desc, -pred_scores['desc_neg_logP'] ) 
        npt.assert_allclose( np.log( true_cond ), -pred_scores['cond_neg_logP'] ) 


    def test_tkf91_f81(self):
        self._run_test(indel_model_type = 'tkf91',
                        subst_model_type = 'f81')
    
    def test_tkf91_gtr(self):
        self._run_test(indel_model_type = 'tkf91',
                        subst_model_type = 'gtr')
        
    def test_tkf92_f81(self):
        self._run_test(indel_model_type = 'tkf92',
                        subst_model_type = 'f81')
    
    def test_tkf92_gtr(self):
        self._run_test(indel_model_type = 'tkf92',
                        subst_model_type = 'gtr')
        
   
if __name__ == '__main__':
    unittest.main()