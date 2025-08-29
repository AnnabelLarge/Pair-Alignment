#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 19:48:04 2025

@author: annabel
"""
import jax
from jax import numpy as jnp

import numpy as np
import numpy.testing as npt
import unittest

from models.simple_site_class_predict.NestedTKF import NestedTKF
from models.simple_site_class_predict.FragAndSiteClasses import FragAndSiteClasses

THRESHOLD = 1e-6

class TestNestedTKFTransitMatReduction(unittest.TestCase):
    """
    About
    ------
    with some strong-arming, the nested TKF92 model should reduce to a mixture
      of fragments model
    """
    def setUp(self):
        ############
        ### init   #
        ############
        # dims
        C_dom = 1
        C_frag = 5
        C_sites = 7
        K = 8
        T = 6 #when one time per sample, this is also B
        L = 3
        
        # time
        t_array = jax.random.uniform(key = jax.random.key(0), 
                                     shape=(T,), 
                                     minval=0.01, 
                                     maxval=1.0) #(T,)
        
        # fake batch (for init)
        fake_align = jnp.zeros( (T,L,3),dtype=int )
        fake_batch = [fake_align, t_array, None]
        
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
        
        ##############################################
        ### Get the Nested TKF transition matrices   #
        ##############################################
        # model
        nested_tkf = NestedTKF(config = config,
                               name = 'nested_tkf')
        
        init_params = nested_tkf.init( rngs = jax.random.key(0),
                                  batch = fake_batch,
                                  t_array = None,
                                  sow_intermediates = False )
        
        # get matrices
        self.nested_tkf_matrix_dict = nested_tkf.apply( variables = init_params,
                           t_array = t_array,
                           sow_intermediates = False,
                           return_all_matrices = True,
                           method = '_get_transition_scoring_matrices' )
        del nested_tkf
        
        
        #################################################
        ### Get the Frag mixtures transition matrices   #
        #################################################
        # model
        frag_mix_tkf = FragAndSiteClasses(config = config,
                                          name = 'frag_mix')
        
        # make param set
        frag_mix_params = {'params': {}}
        frag_mix_params['params']['get equilibrium'] = init_params['params']['get equilibrium']
        frag_mix_params['params']['get rate multipliers'] = init_params['params']['get rate multipliers']
        frag_mix_params['params']['tkf92 indel model'] = init_params['params']['tkf92 frag indel model']
        del init_params
        
        # get matrices
        self.frag_mix_matrix_dict = frag_mix_tkf.apply( variables = frag_mix_params,
                           t_array = t_array,
                           sow_intermediates = False,
                           return_intermeds = False,
                           return_all_matrices = True,
                           method = '_get_scoring_matrices' )
        del frag_mix_tkf
        
        
    def test_joint(self):
        nested_joint = self.nested_tkf_matrix_dict[2]['joint'] #(T, C_dom_from*C_frag_from, C_dom_to*C_frag_to, S_from, S_to)
        frag_mix_joint = self.frag_mix_matrix_dict['all_transit_matrices']['joint'] #(T, C_frag_from, C_frag_to, S_from, S_to)

        npt.assert_allclose(nested_joint, 
                            frag_mix_joint,
                            rtol = THRESHOLD,
                            atol = 0)
        
    def test_marginal(self):
        nested_marg = self.nested_tkf_matrix_dict[2]['marginal'] #(C_dom_from*C_frag_from, C_dom_to*C_frag_to, 2, 2)
        frag_mix_marg = self.frag_mix_matrix_dict['all_transit_matrices']['marginal'] #(C_frag_from, C_frag_to, 2, 2)

        npt.assert_allclose(nested_marg, 
                            frag_mix_marg,
                            rtol = THRESHOLD,
                            atol = 0)
        
    def test_conditional(self):
        nested_cond = self.nested_tkf_matrix_dict[2]['conditional'] #(T, C_dom_from*C_frag_from, C_dom_to*C_frag_to, S_from, S_to)
        frag_mix_cond = self.frag_mix_matrix_dict['all_transit_matrices']['conditional'] #(T, C_frag_from, C_frag_to, S_from, S_to)

        npt.assert_allclose(nested_cond, 
                            frag_mix_cond,
                            rtol = THRESHOLD,
                            atol = 0)


if __name__ == '__main__':
    unittest.main()
    