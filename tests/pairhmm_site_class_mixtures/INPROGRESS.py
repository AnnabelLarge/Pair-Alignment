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
from models.simple_site_class_predict.IndpSites import IndpSites

THRESHOLD = 1e-6


class TestIndpSitesClass(unittest.TestCase):
    """
    make sure that likelihood calculation is the same as hand-done calculations
    """
    def setUp(self):
        self.path = f'./tests/simple_site_class_predict/full_model_tests/req_files'
        self.req_files_path = f'{self.path}/req_files'
        
        
        ###############################
        ### generate fake alignments  #
        ###############################
        fake_aligns = [ ('ECDADD','-C-D-A'),
                        ('-C-D-A','ECDADD') ]
        self.t_array = jnp.array([1.0, 0.5, 1.5]) #(T,)
        fake_aligns_pairhmm =  str_aligns_to_tensor(fake_aligns) #(B, L, 3)
        del fake_aligns

        # dims
        self.T = 3
        self.B = 2
        self.A = 20
        self.C = 2
        self.K = 4
        
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
        self.pairhmm_config_template = {'num_mixtures': self.C,
                                        'k_rate_mults': self.K,
                                        'num_tkf_fragment_classes': 1,
                                        'indp_rate_mults': False,
                                        'times_from': 't_array_from_file',
                                        'exponential_dist_param': 1.1,
                                        'training_dset_emit_counts': training_dset_emit_counts,
                                        'emission_alphabet_size': self.A,
                                        'tkf_function_name': 'regular',
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
        
        # generate scoring parameters
        init_params = pairhmm.init( rngs = jax.random.key(42),
                                    batch = self.pairhmm_batch,
                                    t_array = self.t_array,
                                    sow_intermediates = False,
                                    whole_dset_grad_desc = False )
        
        scoring_mat_dict = pairhmm.apply( variables=init_params,
                                          t_array=self.t_array,
                                          sow_intermediates=False,
                                          return_intermeds=True,
                                          method = '_get_scoring_matrices')
        
        
        ### score sequences with IndpSites
        pred_scores = pairhmm.apply( variables=init_params,
                                     batch = self.pairhmm_batch,
                                     t_array = self.t_array,
                                     method = 'calculate_all_loglikes')
        
        
        ### score by hand
        # unpack
        match_counts = self.pairhmm_batch[0]
        ins_counts = self.pairhmm_batch[1]
        del_counts = self.pairhmm_batch[2]
        transit_counts = self.pairhmm_batch[3]
        
        # mixture params
        log_class_probs = scoring_mat_dict['log_class_probs']
        log_rate_mult_probs = scoring_mat_dict['log_rate_mult_probs']
        
        # emission scoring matrices
        log_equl_dist_per_mixture = scoring_mat_dict['log_equl_dist_per_mixture']
        joint_subst_logprobs_per_mixture = scoring_mat_dict['joint_subst_logprobs_per_mixture']
        
        # transition scoring matrices
        joint_transit_logprob = scoring_mat_dict['all_transit_matrices']['joint']
        marginal_transit_logprob = scoring_mat_dict['all_transit_matrices']['marginal']
        
        
        ### score by hand
        emit_joint_score = np.zeros( (self.T, self.B) )
        emit_anc_marg_score = np.zeros( (self.B) )
        emit_desc_marg_score = np.zeros( (self.B) )
        for t in range( self.T ):
            for b in range( self.B ):
                anc_len = 0
                desc_len = 0
                align_len = 0
                
                ### score by hand: emissions
                for anc_i in range( self.A ):
                    for desc_j in range( self.A ):
                        # emissions from match sites (joint)
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
                        
                        
                        # single-sequence emissions (joint, marginals)
                        anc_prob_for_site = 0
                        desc_prob_for_site = 0
                        for c in range(self.C):
                            # P(x|c), P(y|c)
                            anc_prob_for_mixture = np.exp( log_equl_dist_per_mixture[c,anc_i] )
                            desc_prob_for_mixture = np.exp( log_equl_dist_per_mixture[c,desc_j] )
                            
                            # P(c)
                            mixture_weight = np.exp( log_class_probs[c] )
                            
                            # P(x|c) * P(c) = P(x,c), same for y
                            anc_prob_for_site += mixture_weight * anc_prob_for_mixture
                            desc_prob_for_site += mixture_weight * desc_prob_for_mixture
                        
                        anc_logprob_for_site = np.log( anc_prob_for_site )
                        desc_logprob_for_site = np.log( desc_prob_for_site )
                        
                        joint_ins_count = ins_counts[b, desc_j] 
                        joint_del_count = del_counts[b, anc_i]
                        anc_count = match_counts[b, anc_i, :].sum() + del_counts[b, anc_i]
                        desc_count = match_counts[b, :, desc_j].sum() + ins_counts[b, desc_j]
                        
                        emit_joint_score[t,b] += (joint_del_count * anc_logprob_for_site + 
                                             joint_ins_count * desc_logprob_for_site)
                        emit_anc_marg_score[b] += anc_count * anc_logprob_for_site
                        emit_desc_marg_score[b] += desc_count * desc_logprob_for_site
                        
                        anc_len += anc_count
                        desc_len += desc_count
                        align_len += transit_counts[b, :-1, :-1].sum() + 1
                        
                        del anc_prob_for_site, desc_prob_for_site, anc_prob_for_mixture
                        del desc_prob_for_mixture, mixture_weight, anc_logprob_for_site
                        del desc_logprob_for_site, joint_ins_count, joint_del_count
                        del anc_count, desc_count
                        
                
                # ### score by hand: transitions
                # # geometric length
                # if indel_model_type is None:
                #     logprob_emit, logprob_end = joint_transit_logprob
                #     emit_joint_score[t,b] += logprob_emit * align_len + logprob_end
                #     emit_anc_marg_score[b] += logprob_emit * anc_len + logprob_end
                #     emit_desc_marg_score[b] += logprob_emit * desc_len + logprob_end
                    
                #     breakpoint()
                    
        
        

    def test_geom_len_f81(self):
        self._run_test(indel_model_type = None,
                                subst_model_type = 'f81')
   
if __name__ == '__main__':
    unittest.main()