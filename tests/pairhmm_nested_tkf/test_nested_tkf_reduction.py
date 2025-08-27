#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 19:48:04 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
import numpy.testing as npt
import unittest

from models.simple_site_class_predict.NestedTKF import NestedTKF
from models.simple_site_class_predict.FragAndSiteClasses import FragAndSiteClasses

THRESHOLD = 1e-6

class TestNestedTKFReduction(unittest.TestCase):
    """
    About
    ------
    with some strong-arming, the nested TKF92 model should reduce to a mixture
      of fragments model
    """
    def test_nested_tkf_reduction(self):
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
        
        ###############################################################
        ### Get the Nested TKF joint transition matrix, intermediates #
        ###############################################################
        ### init model
        nested_tkf = NestedTKF(config = config,
                               name = 'nested_tkf')
        
        init_params = nested_tkf.init( rngs = jax.random.key(0),
                                  batch = fake_batch,
                                  t_array = None,
                                  sow_intermediates = False )
        
        ### intermediates
        indel_param_dict = nested_tkf.apply( variables = init_params,
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
        
        
        ### entries for the joint matrix
        log_T_mat = jnp.ones( (T, 4, 4) ) * jnp.finfo(jnp.float32).min # (T, S_from, S_to)
        log_T_mat = log_T_mat.at[:, 3, 0].set(0.0)
        log_T_mat = log_T_mat.at[:, 0, 3].set(0.0)
        
        nested_tkf_joint_entries = nested_tkf.apply( variables = init_params,
                                          log_domain_class_probs = log_domain_class_probs,
                                          log_frag_class_probs = log_frag_class_probs,
                                          frag_tkf_params_dict = frag_tkf_params_dict,
                                          frag_joint_transit_mat = frag_joint_transit_mat,
                                          frag_marginal_transit_mat = frag_marginal_transit_mat,
                                          r_frag = r_frag,
                                          log_T_mat = log_T_mat,
                                          method = '_retrieve_joint_transition_entries' )
        del log_T_mat, log_domain_class_probs, dom_joint_transit_mat
        del dom_marginal_transit_mat, lam_dom, mu_dom, offset_dom
        del log_frag_class_probs, frag_tkf_params_dict
        del frag_marginal_transit_mat, lam_frag, mu_frag, offset_frag, r_frag
        
        
        ### final mat
        out = nested_tkf.apply( variables = init_params,
                           t_array = t_array,
                           sow_intermediates = False,
                           return_all_matrices = False,
                           method = '_get_transition_scoring_matrices' )
        nested_joint_transit = out[2]['joint'] #(T, C_dom_from*C_frag_from, C_dom_to*C_frag_to, S_from, S_to)
        del nested_tkf, out
        
        
        ###################################################
        ### Get the Frag mixtures joint transition matrix #
        ###################################################
        frag_mix_tkf = FragAndSiteClasses(config = config,
                               name = 'frag_mix')
        
        # make param set
        frag_mix_params = {'params': {}}
        frag_mix_params['params']['get equilibrium'] = init_params['params']['get equilibrium']
        frag_mix_params['params']['get rate multipliers'] = init_params['params']['get rate multipliers']
        frag_mix_params['params']['tkf92 indel model'] = init_params['params']['tkf92 frag indel model']
        del init_params
        
        matrix_dict = frag_mix_tkf.apply( variables = frag_mix_params,
                           t_array = t_array,
                           sow_intermediates = False,
                           return_intermeds = False,
                           return_all_matrices = False,
                           method = '_get_scoring_matrices' )
        frag_mix_joint_transit = matrix_dict['all_transit_matrices']['joint'] #(T, C_frag_from, C_frag_to, S_from, S_to)
        
        ### check values against each other
        npt.assert_allclose(frag_mix_joint_transit, nested_joint_transit,
                            rtol = THRESHOLD,
                            atol = 0)


if __name__ == '__main__':
    unittest.main()













### Old code that checked intermediates
# # check that transition matrix from fragment mixture model matches the 
# #   fragment-level transition matrix in the nested TKF model
# npt.assert_allclose( frag_mix_joint_transit,  frag_joint_transit_mat[:,0,...])

# del frag_mix_tkf, matrix_dict



# ##########################################################
# ### check entries in joint matrix for nested TKF model   #
# ##########################################################
# # SS -> MY should be equal to fragment model
# ss_to_my = nested_tkf_joint_entries['ss_to_my'][:, 0, ...] #(T, C_frag_to, (S_to \in MID) )
# checkmat = frag_mix_joint_transit[:, 0, :, 3, 0:3] #(T, C_frag_to, (S_to \in MID) )
# npt.assert_allclose(ss_to_my, checkmat)
# del ss_to_my, checkmat

# # MX -> MY should be equal to fragment model
# mx_to_my = nested_tkf_joint_entries['mx_to_my'][:,0,0,...] #(T, C_frag_from, C_frag_to, (S_from \in MID), (S_to \in MID) )
# checkmat = frag_mix_joint_transit[..., 0:3, 0:3] #(T, C_frag_from, C_frag_to, (S_from \in MID), (S_to \in MID) )
# npt.assert_allclose(mx_to_my, checkmat)
# del mx_to_my, checkmat

# # MX -> EE should be equal to fragment model
# mx_to_ee = nested_tkf_joint_entries['mx_to_ee'][:, 0, ...] #(T, C_frag_from, (S_from \in MID) )
# checkmat = frag_mix_joint_transit[:, :, -1, 0:3, 3] #(T, C_frag_from, (S_from \in MID) )
# npt.assert_allclose(mx_to_ee, checkmat)

# # all other entries should be neg inf, or zeros in prob space
# npt.assert_allclose( np.exp( nested_tkf_joint_entries['mx_to_ii'] ),
#                     np.zeros( nested_tkf_joint_entries['mx_to_ii'].shape ) )

# npt.assert_allclose( np.exp( nested_tkf_joint_entries['mx_to_dd'] ),
#                     np.zeros( nested_tkf_joint_entries['mx_to_dd'].shape ) )


# npt.assert_allclose( np.exp( nested_tkf_joint_entries['ii_to_my'] ),
#                     np.zeros( nested_tkf_joint_entries['ii_to_my'].shape ) )

# npt.assert_allclose( np.exp( nested_tkf_joint_entries['ii_to_ii'] ),
#                     np.zeros( nested_tkf_joint_entries['ii_to_ii'].shape ) )

# npt.assert_allclose( np.exp( nested_tkf_joint_entries['ii_to_dd'] ),
#                     np.zeros( nested_tkf_joint_entries['ii_to_dd'].shape ) )

# npt.assert_allclose( np.exp( nested_tkf_joint_entries['ii_to_ee'] ),
#                     np.zeros( nested_tkf_joint_entries['ii_to_ee'].shape ) )


# npt.assert_allclose( np.exp( nested_tkf_joint_entries['dd_to_my'] ),
#                     np.zeros( nested_tkf_joint_entries['dd_to_my'].shape ) )

# npt.assert_allclose( np.exp( nested_tkf_joint_entries['dd_to_ii'] ),
#                     np.zeros( nested_tkf_joint_entries['dd_to_ii'].shape ) )

# npt.assert_allclose( np.exp( nested_tkf_joint_entries['dd_to_dd'] ),
#                     np.zeros( nested_tkf_joint_entries['dd_to_dd'].shape ) )

# npt.assert_allclose( np.exp( nested_tkf_joint_entries['dd_to_ee'] ),
#                     np.zeros( nested_tkf_joint_entries['dd_to_ee'].shape ) )


# npt.assert_allclose( np.exp( nested_tkf_joint_entries['ss_to_ii'] ),
#                     np.zeros( nested_tkf_joint_entries['ss_to_ii'].shape ) )

# npt.assert_allclose( np.exp( nested_tkf_joint_entries['ss_to_dd'] ),
#                     np.zeros( nested_tkf_joint_entries['ss_to_dd'].shape ) )

# npt.assert_allclose( np.exp( nested_tkf_joint_entries['ss_to_ee'] ),
#                     np.zeros( nested_tkf_joint_entries['ss_to_ee'].shape ) )


