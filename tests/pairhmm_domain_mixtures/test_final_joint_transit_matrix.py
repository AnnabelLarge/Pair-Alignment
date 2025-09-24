#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 19:27:42 2025

@author: annabel
"""
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

import numpy as np
import numpy.testing as npt
import unittest

from models.latent_class_mixtures.NestedTKF import NestedTKF

THRESHOLD = 1e-6

class TestFinalJointTransitMatrix(unittest.TestCase):
    """
    About
    ------
    test the final joint transition matrix (which includes top-level TKF91
        AND fragment-level TKF92) against hand-done calculation
    """
    def test_final_joint_transit_matrix(self):
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
        dom_joint_transit_mat = indel_param_dict['dom_joint_transit_mat'] #(T, S_from, S_to)
        dom_marginal_transit_mat = indel_param_dict['dom_marginal_transit_mat'] #(2, 2)
        lam_dom = indel_param_dict['lam_dom'] #float
        mu_dom = indel_param_dict['mu_dom'] #float
        offset_dom = indel_param_dict['offset_dom'] #float
        
        # fragment-level
        log_frag_class_probs = indel_param_dict['log_frag_class_probs']
        frag_tkf_params_dict = indel_param_dict['frag_tkf_params_dict']
        frag_joint_transit_mat = indel_param_dict['frag_joint_transit_mat'] #(T, C_dom, C_frag_to, C_frag_from, S_from, S_to)
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
                           return_all_matrices = False,
                           method = '_get_transition_scoring_matrices' )
        pred_joint_transit = out[2]['joint'] #(T, C_dom*C_frag, C_dom*C_frag, S, S)
        pred_joint_transit = jnp.reshape(pred_joint_transit, (T, C_dom, C_frag, C_dom, C_frag, 4, 4))  #(T, C_dom, C_frag, C_dom, C_frag, S, S)
        
        
        #################################################
        ### Piece together the true transition matrix   #
        #################################################
        log_T_mat = mymod.apply( variables = init_params,
                                 log_domain_class_probs = log_domain_class_probs,
                                 frag_tkf_params_dict = frag_tkf_params_dict,
                                 dom_joint_transit_mat = dom_joint_transit_mat,
                                 method = '_get_joint_domain_transit_matrix_without_null_cycles' ) #(T, S, S)
        
        log_true_joint_entries = mymod.apply( variables = init_params,
                                          log_domain_class_probs = log_domain_class_probs,
                                          log_frag_class_probs = log_frag_class_probs,
                                          frag_tkf_params_dict = frag_tkf_params_dict,
                                          frag_joint_transit_mat = frag_joint_transit_mat,
                                          frag_marginal_transit_mat = frag_marginal_transit_mat,
                                          r_frag = r_frag,
                                          log_T_mat = log_T_mat,
                                          method = '_retrieve_joint_transition_entries' )
        
        true_joint_entries = {k: np.exp(v) for k,v in log_true_joint_entries.items()}
        
        true = np.zeros( pred_joint_transit.shape )
        for t in range(T):
            for c_dom_from in range(C_dom):
                for c_frag_from in range(C_frag):
                    for c_dom_to in range(C_dom):
                        for c_frag_to in range(C_frag):
                            ### M -> any
                            # M -> M
                            val = true_joint_entries['mx_to_my'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 0, 0]
                            true[t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 0, 0] = val
                            
                            # M -> I
                            val = ( true_joint_entries['mx_to_my'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 0, 1] +
                                    true_joint_entries['mx_to_ii'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 0] )
                            true[t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 0, 1] = val
                            
                            # M -> D
                            val = ( true_joint_entries['mx_to_my'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 0, 2] +
                                    true_joint_entries['mx_to_dd'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 0] )
                            true[t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 0, 2] = val
                            
                            # M -> E
                            val = true_joint_entries['mx_to_ee'][t, c_dom_from, c_frag_from, 0]
                            true[t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 0, 3] = val
                            
                            
                            ### I -> any
                            # I -> M
                            val = ( true_joint_entries['mx_to_my'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 1, 0] +
                                    true_joint_entries['ii_to_my'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 0] )
                            true[t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 1, 0] = val
                            
                            # I -> I
                            val = ( true_joint_entries['mx_to_my'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 1, 1] +
                                   true_joint_entries['mx_to_ii'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 1] +
                                   true_joint_entries['ii_to_my'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 1] +
                                    true_joint_entries['ii_to_ii'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to] )
                            true[t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 1, 1] = val
                            
                            # I -> D
                            val = ( true_joint_entries['mx_to_my'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 1, 2] +
                                   true_joint_entries['mx_to_dd'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 1] +
                                   true_joint_entries['ii_to_my'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 2] +
                                    true_joint_entries['ii_to_dd'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to] )
                            true[t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 1, 2] = val
                            
                            # I -> E
                            val = ( true_joint_entries['mx_to_ee'][t, c_dom_from, c_frag_from, 1] +
                                    true_joint_entries['ii_to_ee'][t, c_dom_from, c_frag_from] )
                            true[t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 1, 3] = val
                            
                            
                            ### D -> any
                            # D -> M
                            val = ( true_joint_entries['mx_to_my'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 2, 0] +
                                    true_joint_entries['dd_to_my'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 0] )
                            true[t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 2, 0] = val
                            
                            # D -> I
                            val = ( true_joint_entries['mx_to_my'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 2, 1] +
                                   true_joint_entries['mx_to_ii'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 2] +
                                   true_joint_entries['dd_to_my'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 1] +
                                    true_joint_entries['dd_to_ii'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to] )
                            true[t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 2, 1] = val
                            
                            # D -> D
                            val = ( true_joint_entries['mx_to_my'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 2, 2] +
                                   true_joint_entries['mx_to_dd'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 2] +
                                   true_joint_entries['dd_to_my'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 2] +
                                    true_joint_entries['dd_to_dd'][t, c_dom_from, c_frag_from, c_dom_to, c_frag_to] )
                            true[t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 2, 2] = val
                            
                            # D -> E
                            val = ( true_joint_entries['mx_to_ee'][t, c_dom_from, c_frag_from, 2] +
                                    true_joint_entries['dd_to_ee'][t, c_dom_from, c_frag_from] )
                            true[t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 2, 3] = val
                            
                            
                            ### S -> any
                            # S -> M
                            val = true_joint_entries['ss_to_my'][t, c_dom_to, c_frag_to, 0]
                            true[t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 3, 0] = val
                            
                            # S -> I
                            val = ( true_joint_entries['ss_to_my'][t, c_dom_to, c_frag_to, 1] +
                                    true_joint_entries['ss_to_ii'][t, c_dom_to, c_frag_to] )
                            true[t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 3, 1] = val
                            
                            # S -> D
                            val = ( true_joint_entries['ss_to_my'][t, c_dom_to, c_frag_to, 2] +
                                    true_joint_entries['ss_to_dd'][t, c_dom_to, c_frag_to] )
                            true[t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 3, 2] = val
                            
                            # S -> E
                            val = true_joint_entries['ss_to_ee'][t]
                            true[t, c_dom_from, c_frag_from, c_dom_to, c_frag_to, 3, 3] = val
        
        
        npt.assert_allclose( np.log(true), pred_joint_transit,
                             rtol = THRESHOLD,
                             atol = 0 )

if __name__ == '__main__':
    unittest.main()