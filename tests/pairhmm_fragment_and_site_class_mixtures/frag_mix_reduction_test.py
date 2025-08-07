#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 22:36:43 2025

@author: annabel
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
from models.simple_site_class_predict.FragAndSiteClasses import FragAndSiteClasses

THRESHOLD = 1e-6


class FragMixReductionTest(unittest.TestCase):
    """
    make sure FragAndSiteClasses reduces to IndpSites
    """
    def setUp(self):
        ###############################
        ### generate fake alignments  #
        ###############################
        fake_aligns = [ ('ECDADD','-C-D-A'),
                        ('-C-D-A','ECDADD') ]
        self.t_array = jnp.array([0.5, 1.0, 1.5]) #(T,)
        fake_aligns_pairhmm =  str_aligns_to_tensor(fake_aligns) #(B, L, 3)
        del fake_aligns

        # dims
        self.T = self.t_array.shape[0] #3
        self.B = 2
        self.A = 20
        self.C_frag = 1
        self.C_sites = 5
        self.K = 6
        self.S = 4 #four types of transitions: M, I, D, start/end
        
        # put together a batch for pairHMM
        vmapped_summarize_alignment = jax.vmap(summarize_alignment, 
                                               in_axes=0, 
                                               out_axes=0)
        counts =  vmapped_summarize_alignment( fake_aligns_pairhmm )
        
        self.indp_sites_batch = (counts['match_counts'],
                                 counts['ins_counts'],
                                 counts['del_counts'],
                                 counts['transit_counts'],
                                 None,
                                 None )
        training_dset_emit_counts = counts['emit_counts'].sum(axis=0)
        
        
        self.frag_and_sites_batch = (fake_aligns_pairhmm, None)

        
        #######################
        ### template config   #
        #######################
        self.pairhmm_config_template = {'num_domain_mixtures': 1,
                                        'num_fragment_mixtures': self.C_frag,
                                        'num_site_mixtures': self.C_sites,
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
        #############
        ### setup   #
        #############
        # init model
        to_add = {'subst_model_type': subst_model_type,
                  'indel_model_type': indel_model_type}
        config = {**self.pairhmm_config_template, **to_add}
        
        pairhmm = IndpSites(config=config,
                            name='pairhmm')
        exponential_dist_param = config['exponential_dist_param']
        
        # generate scoring parameters
        init_params = pairhmm.init( rngs = jax.random.key(42),
                                    batch = self.indp_sites_batch,
                                    t_array = self.t_array,
                                    sow_intermediates = False )
        
        scoring_mat_dict = pairhmm.apply( variables=init_params,
                                          t_array=self.t_array,
                                          sow_intermediates=False,
                                          return_intermeds=False,
                                          return_all_matrices=True,
                                          method = '_get_scoring_matrices')
        
        
        ######################################
        ### score sequences with IndpSites   #
        ######################################
        indp_sites_scores = pairhmm.apply( variables=init_params,
                                           batch = self.indp_sites_batch,
                                           t_array = self.t_array,
                                           return_intermeds = True,
                                           method = 'calculate_all_loglikes')
        
        del pairhmm
    
        
        ###############################################
        ### score sequences with FragAndSiteClasses   #
        ###############################################
        pairhmm = FragAndSiteClasses(config=config,
                                     name='pairhmm')
        exponential_dist_param = config['exponential_dist_param']
        
        init_params = pairhmm.init( rngs = jax.random.key(42),
                                    batch = self.frag_and_sites_batch,
                                    t_array = self.t_array,
                                    sow_intermediates = False )
        
        frag_mix_scores = pairhmm.apply( variables=init_params,
                                            batch = self.frag_and_sites_batch,
                                            t_array = self.t_array,
                                            method = 'calculate_all_loglikes')
        del pairhmm
        
        keys = ['joint_neg_logP', 
                'joint_neg_logP_length_normed',
                'anc_neg_logP', 
                'anc_neg_logP_length_normed', 
                'desc_neg_logP', 
                'desc_neg_logP_length_normed', 
                'cond_neg_logP', 
                'cond_neg_logP_length_normed' ]
        for k in keys:
            npt.assert_allclose( indp_sites_scores[k], frag_mix_scores[k] )
    
    def test_tkf92_f81(self):
        self._run_test(indel_model_type = 'tkf92',
                        subst_model_type = 'f81')
    
    def test_tkf92_gtr(self):
        self._run_test(indel_model_type = 'tkf92',
                        subst_model_type = 'gtr')
   
    
if __name__ == '__main__':
    unittest.main()