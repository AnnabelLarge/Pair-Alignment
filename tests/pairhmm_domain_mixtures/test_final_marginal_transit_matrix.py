#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 13:27:12 2025

@author: annabel

"""
import jax
from jax import numpy as jnp

import numpy as np
import numpy.testing as npt
import unittest

from models.latent_class_mixtures.NestedTKF import NestedTKF
                                                              

THRESHOLD = 1e-6

class TestFinalMarginalTransitMatrix(unittest.TestCase):
    """
    About
    ------
    marginal transition matrix for nested TKF92 model is made by combining
      intermediates; check these intermediates against hand-done calculations
    """
    def test_final_marginal_transit_matrix(self):
        ############
        ### init   #
        ############
        # dims
        C_dom = 2
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
        
        # model
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
            
        mymod = NestedTKF(config = config,
                          name = 'mymod')
        
        init_params = mymod.init( rngs = jax.random.key(0),
                                  batch = fake_batch,
                                  t_array = None,
                                  sow_intermediates = False )
        
        
        ########################################
        ### get scoring matrices, indel params #
        ########################################
        indel_param_dict = mymod.apply( variables = init_params,
                                        t_array = t_array,
                                        sow_intermediates = False,
                                        method = '_retrieve_both_indel_matrices' )
        
        # domain-level
        log_domain_class_probs = indel_param_dict['log_domain_class_probs']
        dom_marginal_transit_mat = indel_param_dict['dom_marginal_transit_mat'] #(2, 2)
        lam_dom = indel_param_dict['lam_dom'] #float
        mu_dom = indel_param_dict['mu_dom'] #float
        offset_dom = indel_param_dict['offset_dom'] #float
        
        # fragment-level
        log_frag_class_probs = indel_param_dict['log_frag_class_probs']
        frag_tkf_params_dict = indel_param_dict['frag_tkf_params_dict']
        frag_marginal_transit_mat = indel_param_dict['frag_marginal_transit_mat'] #(C_dom, C_frag_to, C_frag_from, 2, 2)
        lam_frag = indel_param_dict['lam_frag'] #(C_dom,)
        mu_frag = indel_param_dict['mu_frag'] #(C_dom,)
        offset_frag = indel_param_dict['offset_frag'] #(C_dom,)
        r_frag = indel_param_dict['r_frag'] #(C_dom, C_frag)
        
        
        #########################################
        ### Predicted joint transition matrix   #
        #########################################
        out = mymod.apply( variables = init_params,
                           t_array = t_array,
                           sow_intermediates = False,
                           return_all_matrices = True,
                           method = '_get_transition_scoring_matrices' )
        pred_marginal_transit = out[2]['marginal'] #(C_dom*C_frag, C_dom*C_frag, S, S)
        pred_marginal_transit = jnp.reshape(pred_marginal_transit, (C_dom, C_frag, C_dom, C_frag, 2, 2))  #(C_dom, C_frag, C_dom, C_frag, S, S)
        
        
        ##################################
        ### check against true entries   #
        ##################################
        log_T_mat = mymod.apply( variables = init_params,
                                 log_domain_class_probs = log_domain_class_probs,
                                 frag_tkf_params_dict = frag_tkf_params_dict,
                                 dom_marginal_transit_mat = dom_marginal_transit_mat,
                                 method = '_get_marginal_domain_transit_matrix_without_null_cycles' ) #(2, 2)
        T_mat = np.exp(log_T_mat) #(2, 2)
        
        
        ### em -> em
        for c_dom_from in range(C_dom):
            for c_dom_to in range(C_dom):
                for c_frag_from in range(C_frag):
                    for c_frag_to in range(C_frag):
                        true = ( ( 1 - r_frag[c_dom_from, c_frag_from] ) *
                                 ( 1 - (lam_frag[c_dom_from]/mu_frag[c_dom_from]) ) *
                                 T_mat[0,0] *
                                 np.exp(log_domain_class_probs)[c_dom_to] *
                                 lam_frag[c_dom_to]/mu_frag[c_dom_to] *
                                 np.exp(log_frag_class_probs)[c_dom_to, c_frag_to] )
                        
                        if c_dom_from == c_dom_to:
                            true += np.exp(frag_marginal_transit_mat)[c_dom_to, c_frag_from, c_frag_to, 0, 0]
                            
                        pred = pred_marginal_transit[c_dom_from, c_frag_from, c_dom_to, c_frag_to, 0, 0]
                        npt.assert_allclose( np.log(true), pred, rtol=THRESHOLD, atol=0 )
        
        del c_dom_from, c_dom_to, c_frag_from, c_frag_to, true, pred
        
        
        ### em -> E
        for c_dom_from in range(C_dom):
            for c_dom_to in range(C_dom):
                for c_frag_from in range(C_frag):
                    for c_frag_to in range(C_frag):
                        true = ( ( 1 - r_frag[c_dom_from, c_frag_from] ) *
                                 ( 1 - (lam_frag[c_dom_from]/mu_frag[c_dom_from]) ) *
                                 T_mat[0,1] )
                        
                        pred = pred_marginal_transit[c_dom_from, c_frag_from, c_dom_to, c_frag_to, 0, 1]
                        npt.assert_allclose( np.log(true), pred, rtol=THRESHOLD, atol=0 )
        
        del c_dom_from, c_dom_to, c_frag_from, c_frag_to, true, pred
        
        
        ### S -> em
        for c_dom_from in range(C_dom):
            for c_dom_to in range(C_dom):
                for c_frag_from in range(C_frag):
                    for c_frag_to in range(C_frag):
                        true = ( T_mat[1,0] *
                                 np.exp(log_domain_class_probs)[c_dom_to] *
                                 lam_frag[c_dom_to]/mu_frag[c_dom_to] *
                                 np.exp(log_frag_class_probs)[c_dom_to, c_frag_to] )
                        
                        pred = pred_marginal_transit[c_dom_from, c_frag_from, c_dom_to, c_frag_to, 1, 0]
                        npt.assert_allclose( np.log(true), pred, rtol=THRESHOLD, atol=0 )
                
        del c_dom_from, c_dom_to, c_frag_from, c_frag_to, true, pred
        
        
        ### S -> E
        for c_dom_from in range(C_dom):
            for c_dom_to in range(C_dom):
                for c_frag_from in range(C_frag):
                    for c_frag_to in range(C_frag):
                        pred = pred_marginal_transit[c_dom_from, c_frag_from, c_dom_to, c_frag_to, 1, 1]
                        npt.assert_allclose( np.log(T_mat[1,1]), pred, rtol=THRESHOLD, atol=0 )
                        
if __name__ == '__main__':
    unittest.main()
    