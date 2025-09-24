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
from models.latent_class_mixtures.transition_models import TKF91DomainTransitionLogprobs
from models.latent_class_mixtures.model_functions import regular_tkf

THRESHOLD = 1e-6

class TestJointTopLevelMatrixConstruction(unittest.TestCase):
    """
    About
    ------
    test the top-level TKF91 indel model against hand calculation
    
    """
    def test_joint_top_level_matrix_construction(self):
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
        
        
        ###################################
        ### check T matrix construction   #
        ###################################
        ############
        ### Pred   #
        ############
        pred_T_mat = mymod.apply( variables = init_params,
                                  log_domain_class_probs = log_domain_class_probs,
                                  frag_tkf_params_dict = frag_tkf_params_dict,
                                  dom_joint_transit_mat = dom_joint_transit_mat,
                                  method = '_get_joint_domain_transit_matrix_without_null_cycles' ) #(T, S, S)
        
        
        ############
        ### True   #
        ############
        # tkf91 base matrix
        tkf91_true_mod = TKF91DomainTransitionLogprobs(config = config,
                                                       name = 'true tkf91')
        dummy_params = tkf91_true_mod.init( rngs = jax.random.key(0),
                                            t_array = t_array,
                                            return_all_matrices=False,
                                            sow_intermediates = False )
        dom_tkf_param_dict, _ = regular_tkf( mu = mu_dom[None],
                                          offset = offset_dom[None],
                                          t_array = t_array )
        dom_tkf_param_dict['log_offset'] = jnp.log( offset_dom[None] )
        dom_tkf_param_dict['log_one_minus_offset'] = jnp.log( 1-offset_dom[None] )
        true_tkf91_mat = tkf91_true_mod.apply( variables = dummy_params,
                                                tkf_param_dict = dom_tkf_param_dict,
                                                method = 'fill_joint_tkf91')[:,0,...] #(T, S, S)
        true_tkf91_mat = np.exp(true_tkf91_mat) #(T, S, S)
        
        npt.assert_allclose( np.log(true_tkf91_mat),  dom_joint_transit_mat,
                             rtol = THRESHOLD,
                             atol = 0 )
        
        
        # z_t, z_0
        z_t = np.zeros( (T,) )
        z_0 = 0
        
        for n in range(C_dom):
            kappa = lam_frag[n] / mu_frag[n]
            z_0 += np.exp(log_domain_class_probs)[n] * (1-kappa) 
            
            for t_idx in range(T):
                beta_n = np.exp( frag_tkf_params_dict['log_beta'] )[t_idx, n]
                z_t[t_idx] += np.exp(log_domain_class_probs)[n] * (1-kappa) * (1-beta_n)
                
                
        # build up true 6x6 matrix from if conditionals
        # M: 0
        # I: 1
        # D: 2
        # S/E: 3
        # A: 4
        # B: 5
        v = np.zeros( (T, 6, 6) ) #(T, S+2, S+2)
        for t_idx in range(T):
            z_t_here = z_t[t_idx]
            alpha_here = np.exp( dom_tkf_param_dict['log_alpha'] )[t_idx, 0]
            kappa = lam_dom / mu_dom
            
            for i in range(6):
                # i in {Del, B}
                if i in [2, 5]:
                    beta_star = np.exp( dom_tkf_param_dict['log_gamma'] )[t_idx, 0]
                
                # i not in {Del, B}
                else:
                    beta_star = np.exp( dom_tkf_param_dict['log_beta'] )[t_idx, 0]
                
                
                for j in range(6):
                    # j = Match
                    if j == 0:
                        entry = (1 - z_t_here) * (1 - beta_star) * kappa * alpha_here
                    
                    # j = Ins
                    elif j == 1:
                        entry = (1 - z_0) * beta_star
                    
                    # j = Del
                    elif j == 2:
                        entry = (1 - z_0) * (1 - beta_star) * kappa * (1 - alpha_here)
                    
                    # j = End
                    elif j == 3:
                        entry = (1 - beta_star) * (1 - kappa)
                    
                    # j = A (M/I, but don't emit a sequence)
                    elif j == 4:
                        entry = ( z_t_here * (1 - beta_star) * kappa * alpha_here ) + ( z_0 * beta_star )
                    
                    # j = B (D, but don't emit a sequence)
                    elif j == 5:
                        entry = z_0 * (1 - beta_star) * kappa * (1 - alpha_here)
                    
                    v[t_idx, i, j] = entry
                    
        del t_idx, i, j, entry, alpha_here, beta_star, kappa, z_t_here
        
        # T = U_{MIDS, MIDE} + U_{MIDS, AB} * (I - U_{AB,AB})^-1 * U_{AB, MIDE} )
        I = np.eye( 2 )[None, :, :] #(1, 2, 2)
        I = np.repeat(I, repeats = T, axis=0) #(T, 2, 2)
        modifier = v[:, 0:4, 4:] @ np.linalg.inv( I - v[:, 4:, 4:] ) @ v[:, 4:, 0:4] #(T, 4, 4)
        true_T_mat = v[:, 0:4, 0:4] + modifier #(T, 4, 4)
        
        npt.assert_allclose( np.log(true_T_mat),  pred_T_mat,
                             rtol = THRESHOLD,
                             atol = 0 )


if __name__ == '__main__':
    unittest.main()
    