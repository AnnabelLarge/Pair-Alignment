#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 13:34:13 2025

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

from models.simple_site_class_predict.NestedTKF import NestedTKF
from models.simple_site_class_predict.FragAndSiteClasses import FragAndSiteClasses

THRESHOLD = 1e-6


class TestNestedTKFScoreReduction(unittest.TestCase):
    """
    make sure NestedTKF reduces to mixture of fragment and site classes
    """
    def test_this(self):
        ###############################
        ### generate fake alignments  #
        ###############################
        fake_aligns = [ ('ECDADD','-C-D-A'),
                        ('-C-D-A','ECDADD') ]
        t_array = jnp.array([0.5, 1.0]) #(T,)
        fake_aligns_pairhmm =  str_aligns_to_tensor(fake_aligns) #(B, L, 3)
        del fake_aligns

        # dims
        T = t_array.shape[0] #3
        B = T # 2
        A = 20
        C_dom = 1
        C_frag = 3
        C_sites = 5
        K = 6
        S = 4 #four types of transitions: M, I, D, start/end
        
        # put together a batch 
        batch = (fake_aligns_pairhmm, t_array, None)
        
        # config for both
        config = {'num_domain_mixtures': C_dom,
                  'num_fragment_mixtures': C_frag,
                  'num_site_mixtures': C_sites,
                  'k_rate_mults': K,
                  'subst_model_type': 'f81',
                  'tkf_function': 'regular_tkf',
                  'indp_rate_mults': False,
                  'norm_reported_loss_by': 'desc_len',
                  'exponential_dist_param': 1.1,
                  'emission_alphabet_size': 20,
                  'times_from': 't_per_sample' }
        
        ####################
        ### setup models   #
        ####################
        # nested TKF
        nested_tkf = NestedTKF(config = config,
                               name = 'nested_tkf')
        
        nested_tkf_params = nested_tkf.init( rngs = jax.random.key(0),
                                             batch = batch,
                                             t_array = None,
                                             sow_intermediates = False )
        
        # Frag mix
        frag_mix_tkf = FragAndSiteClasses(config = config,
                                          name = 'frag_mix')
        
        frag_mix_params = {'params': {}}
        frag_mix_params['params']['get equilibrium'] = nested_tkf_params['params']['get equilibrium']
        frag_mix_params['params']['get rate multipliers'] = nested_tkf_params['params']['get rate multipliers']
        frag_mix_params['params']['tkf92 indel model'] = nested_tkf_params['params']['tkf92 frag indel model']
        
        
        #######################
        ### score sequences   #
        #######################
        nested_tkf_scores = nested_tkf.apply( variables = nested_tkf_params,
                                              batch = batch,
                                              t_array = t_array,
                                              method = 'calculate_all_loglikes' )
        del nested_tkf

        frag_mix_scores = frag_mix_tkf.apply( variables = frag_mix_params,
                                              batch = batch,
                                              t_array = t_array,
                                              method = 'calculate_all_loglikes' )
        del frag_mix_tkf
        
        # check every key
        keys = ['joint_neg_logP', 
                'joint_neg_logP_length_normed',
                'anc_neg_logP', 
                'anc_neg_logP_length_normed', 
                'desc_neg_logP', 
                'desc_neg_logP_length_normed', 
                'cond_neg_logP', 
                'cond_neg_logP_length_normed' ]
        for k in keys:
            npt.assert_allclose( nested_tkf_scores[k], frag_mix_scores[k] )

if __name__ == '__main__':
    unittest.main()
