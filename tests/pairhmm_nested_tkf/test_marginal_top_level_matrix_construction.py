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
from models.simple_site_class_predict.transition_models import TKF91DomainTransitionLogprobs
from models.simple_site_class_predict.model_functions import (regular_tkf,
                                                              get_tkf91_single_seq_marginal_transition_logprobs)
                                                              

THRESHOLD = 1e-6

class TestMarginalTopLevelMatrixConstruction(unittest.TestCase):
    """
    About
    ------
    test the top-level TKF91 indel model against hand calculation 
      (the single-sequence marginal transition matrix)
    """
    def test_marginal_top_level_matrix_construction(self):
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
                                 dom_marginal_transit_mat = dom_marginal_transit_mat,
                                 method = '_get_marginal_domain_transit_matrix_without_null_cycles' ) #(2, 2)
        
        
        ############
        ### True   #
        ############
        # tkf91 single-sequence base matrix
        log_true_tkf91_mat = get_tkf91_single_seq_marginal_transition_logprobs(offset_dom) #(2, 2)
        true_tkf91_mat = np.exp(log_true_tkf91_mat) #(2, 2)
        
        # z_t, z_0
        z_0 = 0
        
        for n in range(C_dom):
            kappa = lam_frag[n] / mu_frag[n]
            z_0 += np.exp(log_domain_class_probs)[n] * (1-kappa) 
                
                
        # build up true 3x3 matrix from if conditionals
        # em: 0
        # S/E: 1
        # C: 2
        kappa = lam_dom / mu_dom
        
        cell1 = (1 - z_0) * (lam_dom/mu_dom)
        cell2 = 1 - (lam_dom/mu_dom)
        cell3 = z_0 * (lam_dom/mu_dom)
        row = np.stack([cell1, cell2, cell3])
        v = np.repeat( row[None, :], 3, axis=0 )
            
        # redistribute null state, C
        modifier = np.outer( v[0:2, 2], v[2, 0:2] ) * ( 1 / (1 - v[2,2]) )
        true_T_mat = v[0:2, 0:2] + modifier
        
        npt.assert_allclose( np.log(true_T_mat),  
                             pred_T_mat,
                             rtol = THRESHOLD,
                             atol = 0 )

if __name__ == '__main__':
    unittest.main()
    