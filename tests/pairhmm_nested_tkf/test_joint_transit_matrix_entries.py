#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 13:27:12 2025

@author: annabel

"""
import jax
from jax import numpy as jnp
import numpy.testing as npt
import unittest

from models.simple_site_class_predict.NestedTKF import NestedTKF

THRESHOLD = 1e-6

class TestJointTransitMatixEntries(unittest.TestCase):
    """
    About
    ------
    joint transition matrix for nested TKF92 model is made by combining
      intermediates; check these intermediates against hand-done calculations
    """
    def test_joint_transit_matrix_entries(self):
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
        
        
        ###############################
        ### Predicted joint entries   #
        ###############################
        log_T_mat = mymod.apply( variables = init_params,
                                 log_domain_class_probs = log_domain_class_probs,
                                 frag_tkf_params_dict = frag_tkf_params_dict,
                                 dom_joint_transit_mat = dom_joint_transit_mat,
                                 method = '_get_joint_domain_transit_matrix_without_null_cycles' ) #(T, S, S)
        
        pred_joint_entries = mymod.apply( variables = init_params,
                                          log_domain_class_probs = log_domain_class_probs,
                                          log_frag_class_probs = log_frag_class_probs,
                                          frag_tkf_params_dict = frag_tkf_params_dict,
                                          frag_joint_transit_mat = frag_joint_transit_mat,
                                          frag_marginal_transit_mat = frag_marginal_transit_mat,
                                          r_frag = r_frag,
                                          log_T_mat = log_T_mat,
                                          method = '_retrieve_joint_transition_entries' )
        
        T_mat = np.exp(log_T_mat)
        
        
        ##################################
        ### check against true entries   #
        ##################################
        ### SS -> any
        for t in range(T):
            for c_dom in range(C_dom):
                for c_frag_to in range(C_frag):
                    # SS -> MY
                    for j in range(3):
                        true = ( T_mat[t, 3, 0] * 
                                  np.exp(log_domain_class_probs)[c_dom] * 
                                  np.exp(frag_joint_transit_mat)[t, c_dom, 0, c_frag_to, 3, j] ) # this contains w_{m,g}
                        
                        npt.assert_allclose( np.log(true), 
                                            pred_joint_entries['ss_to_my'][t, c_dom, c_frag_to, j],
                                            rtol = THRESHOLD,
                                            atol = 0 )
                        
                    # SS -> II
                    true = ( T_mat[t, 3, 1] * 
                             np.exp(log_domain_class_probs)[c_dom] * 
                             (lam_frag[c_dom] / mu_frag[c_dom]) *
                             np.exp(log_frag_class_probs)[c_dom, c_frag_to] )
                    npt.assert_allclose( np.log(true), pred_joint_entries['ss_to_ii'][t, c_dom, c_frag_to],
                                         rtol = THRESHOLD,
                                         atol = 0  )
                    
                    # SS -> DD
                    true = ( T_mat[t, 3, 2] * 
                             np.exp(log_domain_class_probs)[c_dom] * 
                             (lam_frag[c_dom] / mu_frag[c_dom]) *
                             np.exp(log_frag_class_probs)[c_dom, c_frag_to] )
                    npt.assert_allclose( np.log(true), pred_joint_entries['ss_to_dd'][t, c_dom, c_frag_to],
                                         rtol = THRESHOLD,
                                         atol = 0  )
                    
                    # SS -> EE
                    true = T_mat[t, 3, 3] 
                    npt.assert_allclose( np.log(true), pred_joint_entries['ss_to_ee'][t],
                                         rtol = THRESHOLD,
                                         atol = 0  )
             
        del t, c_dom, c_frag_to, j, true
        
                    
        ### MX -> any
        for t in range(T):
            for c_dom_from in range(C_dom):
                for c_dom_to in range(C_dom):
                    for c_frag_from in range(C_frag):
                        for c_frag_to in range(C_frag):
                            for i in range(3):
                                # MX -> II
                                true = ( np.exp(frag_joint_transit_mat)[t, c_dom_from, c_frag_from, c_frag_to, i, 3] *  # this contains (1 - r_f)
                                         T_mat[t, 0, 1] * 
                                         np.exp(log_domain_class_probs)[c_dom_to] * 
                                         (lam_frag[c_dom_to] / mu_frag[c_dom_to]) *
                                         np.exp(log_frag_class_probs)[c_dom_to, c_frag_to] )
                                
                                pred = pred_joint_entries['mx_to_ii'][t, c_dom_from, c_dom_to, c_frag_from, c_frag_to, i] 
                                npt.assert_allclose( np.log(true), pred,
                                                     rtol = THRESHOLD,
                                                     atol = 0 )
                                
                                # MX -> DD
                                true = ( np.exp(frag_joint_transit_mat)[t, c_dom_from, c_frag_from, c_frag_to, i, 3] *  # this contains (1 - r_f)
                                         T_mat[t, 0, 2] * 
                                         np.exp(log_domain_class_probs)[c_dom_to] * 
                                         (lam_frag[c_dom_to] / mu_frag[c_dom_to]) *
                                         np.exp(log_frag_class_probs)[c_dom_to, c_frag_to] )
                                
                                pred = pred_joint_entries['mx_to_dd'][t, c_dom_from, c_dom_to, c_frag_from, c_frag_to, i] 
                                npt.assert_allclose( np.log(true), pred,
                                                     rtol = THRESHOLD,
                                                     atol = 0 )
                                
                                # MX -> EE
                                true = ( np.exp(frag_joint_transit_mat)[t, c_dom_from, c_frag_from, c_frag_to, i, 3] *  # this contains (1 - r_f)
                                         T_mat[t, 0, 3] )
                                
                                pred = pred_joint_entries['mx_to_ee'][t, c_dom_from, c_frag_from, i] 
                                npt.assert_allclose( np.log(true), pred,
                                                     rtol = THRESHOLD,
                                                     atol = 0 )
                                
                                # MX -> MY
                                for j in range(3):
                                    true = ( np.exp(frag_joint_transit_mat)[t, c_dom_from, c_frag_from, c_frag_to, i, 3] *  # this contains (1 - r_f)
                                             T_mat[t, 0, 0] * 
                                             np.exp(log_domain_class_probs)[c_dom_to] * 
                                             np.exp(frag_joint_transit_mat)[t, c_dom_to, c_frag_from, c_frag_to, 3, j] ) # this contains w_{m,g}
                                    
                                    pred = pred_joint_entries['mx_to_my'][t, c_dom_from, c_dom_to, c_frag_from, c_frag_to, i, j] 
                                    
                                    if c_dom_from == c_dom_to:
                                        true += np.exp(frag_joint_transit_mat)[t, c_dom_to, c_frag_from, c_frag_to, i, j]
                                    
                                    npt.assert_allclose( np.log(true), pred,
                                                         rtol = THRESHOLD,
                                                         atol = 0 )
        
        del t, c_dom_from, c_dom_to, c_frag_from, c_frag_to, i, j, true, pred
        
                                    
        ### II -> any
        for t in range(T):
            for c_dom_from in range(C_dom):
                for c_dom_to in range(C_dom):
                    for c_frag_from in range(C_frag):
                        for c_frag_to in range(C_frag):
                            # II -> II
                            true = ( (1 - r_frag[c_dom_from, c_frag_from]) *
                                     (1 - (lam_frag[c_dom_from] / mu_frag[c_dom_from]) ) *
                                     T_mat[t, 1, 1] * 
                                     np.exp(log_domain_class_probs)[c_dom_to] * 
                                     (lam_frag[c_dom_to] / mu_frag[c_dom_to]) *
                                     np.exp(log_frag_class_probs)[c_dom_to, c_frag_to] )
                            
                            if c_dom_from == c_dom_to:
                                to_add = ( (1 - r_frag[c_dom_from, c_frag_from]) * 
                                            (lam_frag[c_dom_to] / mu_frag[c_dom_to]) *
                                            np.exp(log_frag_class_probs)[c_dom_to, c_frag_to] )
                                
                                if c_frag_from == c_frag_to:
                                    to_add += r_frag[c_dom_from, c_frag_to]
                                
                                true += to_add
                                del to_add
                            
                            pred = pred_joint_entries['ii_to_ii'][t, c_dom_from, c_dom_to, c_frag_from, c_frag_to] 
                            npt.assert_allclose( np.log(true), pred,
                                                 rtol = THRESHOLD,
                                                 atol = 0 )
                                
                            # II -> DD
                            true = ( (1 - r_frag[c_dom_from, c_frag_from]) *
                                      (1 - (lam_frag[c_dom_from] / mu_frag[c_dom_from]) ) *
                                      T_mat[t, 1, 2] * 
                                      np.exp(log_domain_class_probs)[c_dom_to] * 
                                      (lam_frag[c_dom_to] / mu_frag[c_dom_to]) *
                                      np.exp(log_frag_class_probs)[c_dom_to, c_frag_to] )
                            
                            pred = pred_joint_entries['ii_to_dd'][t, c_dom_from, c_dom_to, c_frag_from, c_frag_to] 
                            npt.assert_allclose( np.log(true), pred,
                                                 rtol = THRESHOLD,
                                                 atol = 0 )
                                
                            # II -> EE
                            true = ( (1 - r_frag[c_dom_from, c_frag_from]) *
                                      (1 - (lam_frag[c_dom_from] / mu_frag[c_dom_from]) ) *  
                                      T_mat[t, 1, 3] )
                            
                            pred = pred_joint_entries['ii_to_ee'][t, c_dom_from, c_frag_from] 
                            npt.assert_allclose( np.log(true), pred,
                                                 rtol = THRESHOLD,
                                                 atol = 0 )
                                
                            # II -> MY
                            for j in range(3):
                                true = ( (1 - r_frag[c_dom_from, c_frag_from]) *
                                          (1 - (lam_frag[c_dom_from] / mu_frag[c_dom_from]) ) * 
                                          T_mat[t, 1, 0] * 
                                          np.exp(log_domain_class_probs)[c_dom_to] * 
                                          np.exp(frag_joint_transit_mat)[t, c_dom_to, c_frag_from, c_frag_to, 3, j] ) # this contains w_{m,g}
                                
                                pred = pred_joint_entries['ii_to_my'][t, c_dom_from, c_dom_to, c_frag_from, c_frag_to, j] 
                                
                                npt.assert_allclose( np.log(true), pred,
                                                     rtol = THRESHOLD,
                                                     atol = 0 )
        
        del t, c_dom_from, c_dom_to, c_frag_from, c_frag_to, j, true, pred
        
        
        ### DD -> any
        for t in range(T):
            for c_dom_from in range(C_dom):
                for c_dom_to in range(C_dom):
                    for c_frag_from in range(C_frag):
                        for c_frag_to in range(C_frag):
                            # DD -> DD
                            true = ( (1 - r_frag[c_dom_from, c_frag_from]) *
                                     (1 - (lam_frag[c_dom_from] / mu_frag[c_dom_from]) ) *
                                     T_mat[t, 2, 2] * 
                                     np.exp(log_domain_class_probs)[c_dom_to] * 
                                     (lam_frag[c_dom_to] / mu_frag[c_dom_to]) *
                                     np.exp(log_frag_class_probs)[c_dom_to, c_frag_to] )
                            
                            if c_dom_from == c_dom_to:
                                to_add = ( (1 - r_frag[c_dom_from, c_frag_from]) * 
                                            (lam_frag[c_dom_to] / mu_frag[c_dom_to]) *
                                            np.exp(log_frag_class_probs)[c_dom_to, c_frag_to] )
                                
                                if c_frag_from == c_frag_to:
                                    to_add += r_frag[c_dom_from, c_frag_to]
                                
                                true += to_add
                                del to_add
                            
                            pred = pred_joint_entries['dd_to_dd'][t, c_dom_from, c_dom_to, c_frag_from, c_frag_to] 
                            npt.assert_allclose( np.log(true), pred,
                                                 rtol = THRESHOLD,
                                                 atol = 0 )
                                
                            # DD -> II
                            true = ( (1 - r_frag[c_dom_from, c_frag_from]) *
                                      (1 - (lam_frag[c_dom_from] / mu_frag[c_dom_from]) ) *
                                      T_mat[t, 2, 1] * 
                                      np.exp(log_domain_class_probs)[c_dom_to] * 
                                      (lam_frag[c_dom_to] / mu_frag[c_dom_to]) *
                                      np.exp(log_frag_class_probs)[c_dom_to, c_frag_to] )
                            
                            pred = pred_joint_entries['dd_to_ii'][t, c_dom_from, c_dom_to, c_frag_from, c_frag_to] 
                            npt.assert_allclose( np.log(true), pred,
                                                 rtol = THRESHOLD,
                                                 atol = 0 )
                                
                            # DD -> EE
                            true = ( (1 - r_frag[c_dom_from, c_frag_from]) *
                                      (1 - (lam_frag[c_dom_from] / mu_frag[c_dom_from]) ) *  
                                      T_mat[t, 2, 3] )
                            
                            pred = pred_joint_entries['dd_to_ee'][t, c_dom_from, c_frag_from] 
                            npt.assert_allclose( np.log(true), pred,
                                                 rtol = THRESHOLD,
                                                 atol = 0 )
                                
                            # DD -> MY
                            for j in range(3):
                                true = ( (1 - r_frag[c_dom_from, c_frag_from]) *
                                          (1 - (lam_frag[c_dom_from] / mu_frag[c_dom_from]) ) * 
                                          T_mat[t, 2, 0] * 
                                          np.exp(log_domain_class_probs)[c_dom_to] * 
                                          np.exp(frag_joint_transit_mat)[t, c_dom_to, c_frag_from, c_frag_to, 3, j] ) # this contains w_{m,g}
                                
                                pred = pred_joint_entries['dd_to_my'][t, c_dom_from, c_dom_to, c_frag_from, c_frag_to, j] 
                                
                                npt.assert_allclose( np.log(true), pred,
                                                     rtol = THRESHOLD,
                                                     atol = 0 )
    
if __name__ == '__main__':
    unittest.main()
    