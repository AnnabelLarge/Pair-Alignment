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

from models.latent_class_mixtures.FragAndSiteClasses import FragAndSiteClasses
from models.latent_class_mixtures.transition_models import TKF91TransitionLogprobs

THRESHOLD = 1e-6

class TestJointTransitMatrixFromModelObj(unittest.TestCase):
    """
    About
    ------
    test the final joint transition matrix against hand-done calculation
    
    technically, this is the same thing as 
    test_tkf92_domain_and_frag_mix_joint_cond_marg.py, but now get the
    transition matrix from FragAndSiteClasses object itself
    """
    def test_joint_transit_matrix_from_model_obj(self):
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
            
        mymod = FragAndSiteClasses(config = config,
                          name = 'mymod')
        
        init_params = mymod.init( rngs = jax.random.key(0),
                                  batch = fake_batch,
                                  t_array = t_array,
                                  sow_intermediates = False )
        
        altered_config = config.copy()
        altered_config['num_fragment_mixtures'] = 1
        tkf91_mod = TKF91TransitionLogprobs(config = altered_config,
                                            name = 'tkf91_mod')
        
        dummy_tkf91_params = tkf91_mod.init( rngs = jax.random.key(0),
                                             t_array = t_array,
                                             return_all_matrices = False,
                                             sow_intermediates = False )
        
        
        ########################################
        ### get scoring matrices, indel params #
        ########################################
        indel_param_dict = mymod.apply( variables = init_params,
                                        t_array = t_array,
                                        sow_intermediates = False,
                                        return_all_matrices = False,
                                        return_intermeds = True,
                                        method = '_get_scoring_matrices' )
        
        log_pred_joint_transit = indel_param_dict['all_transit_matrices']['joint'] #(T, C_frag, C_frag, S, S)
        frag_class_probs = jnp.exp( indel_param_dict['log_frag_class_probs'] ) #(C_frag,)
        lam = indel_param_dict['all_transit_matrices']['lam'] #float
        mu = indel_param_dict['all_transit_matrices']['mu'] #float
        r = indel_param_dict['all_transit_matrices']['r_extend'] #(1, C_frag)
        r = r[0,:] #(C_frag)
        tkf_param_dict = indel_param_dict['tkf_param_dict']
        
        # this has already been prove to be true
        log_tkf91_true_joint_mat = tkf91_mod.apply( variables = dummy_tkf91_params,
                                                tkf_param_dict = tkf_param_dict,
                                                method = 'fill_joint_tkf91' ) #(T, 1, S, S)
        tkf91_true_joint_mat = jnp.exp(log_tkf91_true_joint_mat[:,0,...]) #(T, S, S)
        
        # check shapes
        npt.assert_allclose( log_pred_joint_transit.shape, (T, C_frag, C_frag, 4, 4) )
        npt.assert_allclose( frag_class_probs.shape, (C_frag,) )
        npt.assert_allclose( lam.shape, (1,) )
        npt.assert_allclose( mu.shape, (1,) )
        npt.assert_allclose( r.shape, (C_frag) )
        npt.assert_allclose( tkf91_true_joint_mat.shape, (T, 4, 4) )
        
        
        #################################################
        ### Piece together the true transition matrix   #
        #################################################
        # true = np.zeros( log_pred_joint_transit.shape ) #(T, C_frag, C_frag, S, S)
        for t in range(T):
            for c_frag_from in range(C_frag):
                for c_frag_to in range(C_frag):
                    ### M -> any
                    # M -> M
                    true = ( (1 - r[c_frag_from]) * 
                            tkf91_true_joint_mat[t,0,0] * 
                            frag_class_probs[c_frag_to] )
                    
                    if c_frag_from == c_frag_to:
                        true = true + r[c_frag_from]
                    
                    pred = log_pred_joint_transit[t,c_frag_from,c_frag_to,0,0]
                    npt.assert_allclose( np.log(true), pred,
                                         rtol = THRESHOLD,
                                         atol = 0 )
                    del true, pred
                    
                    # M -> I
                    true = ( (1 - r[c_frag_from]) * 
                            tkf91_true_joint_mat[t,0,1] * 
                            frag_class_probs[c_frag_to] )
                    
                    pred = log_pred_joint_transit[t,c_frag_from,c_frag_to,0,1]
                    npt.assert_allclose( np.log(true), pred,
                                         rtol = THRESHOLD,
                                         atol = 0 )
                    del true, pred
                    
                    # M -> D
                    true = ( (1 - r[c_frag_from]) * 
                            tkf91_true_joint_mat[t,0,2] * 
                            frag_class_probs[c_frag_to] )
                    
                    pred = log_pred_joint_transit[t,c_frag_from,c_frag_to,0,2]
                    npt.assert_allclose( np.log(true), pred,
                                         rtol = THRESHOLD,
                                         atol = 0 )
                    del true, pred
                    
                    # M -> E
                    true = ( (1 - r[c_frag_from]) * 
                            tkf91_true_joint_mat[t,0,3] )
                    
                    pred = log_pred_joint_transit[t,c_frag_from,c_frag_to,0,3]
                    npt.assert_allclose( np.log(true), pred,
                                         rtol = THRESHOLD,
                                         atol = 0 )
                    del true, pred
                    
                    
                    ### I -> any
                    # I -> M
                    true = ( (1 - r[c_frag_from]) * 
                            tkf91_true_joint_mat[t,1,0] * 
                            frag_class_probs[c_frag_to] )
                    
                    pred = log_pred_joint_transit[t,c_frag_from,c_frag_to,1,0]
                    npt.assert_allclose( np.log(true), pred,
                                         rtol = THRESHOLD,
                                         atol = 0 )
                    del true, pred
                    
                    # I -> I
                    true = ( (1 - r[c_frag_from]) * 
                            tkf91_true_joint_mat[t,1,1] * 
                            frag_class_probs[c_frag_to] )
                    
                    if c_frag_from == c_frag_to:
                        true = true + r[c_frag_from]
                    
                    pred = log_pred_joint_transit[t,c_frag_from,c_frag_to,1,1]
                    npt.assert_allclose( np.log(true), pred,
                                         rtol = THRESHOLD,
                                         atol = 0 )
                    del true, pred
                    
                    # I -> D
                    true = ( (1 - r[c_frag_from]) * 
                            tkf91_true_joint_mat[t,1,2] * 
                            frag_class_probs[c_frag_to] )
                    
                    pred = log_pred_joint_transit[t,c_frag_from,c_frag_to,1,2]
                    npt.assert_allclose( np.log(true), pred,
                                         rtol = THRESHOLD,
                                         atol = 0 )
                    del true, pred
                    
                    # I -> E
                    true = ( (1 - r[c_frag_from]) * 
                            tkf91_true_joint_mat[t,1,3] )
                    
                    pred = log_pred_joint_transit[t,c_frag_from,c_frag_to,1,3]
                    npt.assert_allclose( np.log(true), pred,
                                         rtol = THRESHOLD,
                                         atol = 0 )
                    del true, pred
                    
                    
                    ### D -> any
                    # D -> M
                    true = ( (1 - r[c_frag_from]) * 
                            tkf91_true_joint_mat[t,2,0] * 
                            frag_class_probs[c_frag_to] )
                    
                    pred = log_pred_joint_transit[t,c_frag_from,c_frag_to,2,0]
                    npt.assert_allclose( np.log(true), pred,
                                         rtol = THRESHOLD,
                                         atol = 0 )
                    del true, pred
                    
                    # D -> I
                    true = ( (1 - r[c_frag_from]) * 
                            tkf91_true_joint_mat[t,2,1] * 
                            frag_class_probs[c_frag_to] )
                    
                    pred = log_pred_joint_transit[t,c_frag_from,c_frag_to,2,1]
                    npt.assert_allclose( np.log(true), pred,
                                         rtol = THRESHOLD,
                                         atol = 0 )
                    del true, pred
                    
                    # D -> D
                    true = ( (1 - r[c_frag_from]) * 
                            tkf91_true_joint_mat[t,2,2] * 
                            frag_class_probs[c_frag_to] )
                    
                    if c_frag_from == c_frag_to:
                        true = true + r[c_frag_from]
                    
                    pred = log_pred_joint_transit[t,c_frag_from,c_frag_to,2,2]
                    npt.assert_allclose( np.log(true), pred,
                                         rtol = THRESHOLD,
                                         atol = 0 )
                    del true, pred
                    
                    # D -> E
                    true = ( (1 - r[c_frag_from]) * 
                            tkf91_true_joint_mat[t,2,3] )
                    
                    pred = log_pred_joint_transit[t,c_frag_from,c_frag_to,2,3]
                    npt.assert_allclose( np.log(true), pred,
                                         rtol = THRESHOLD,
                                         atol = 0 )
                    del true, pred
                    
                    ### S -> any
                    # S -> M
                    true = ( tkf91_true_joint_mat[t,3,0] *
                             frag_class_probs[c_frag_to] )
                    
                    pred = log_pred_joint_transit[t,c_frag_from,c_frag_to,3,0]
                    npt.assert_allclose( np.log(true), pred,
                                         rtol = THRESHOLD,
                                         atol = 0 )
                    del true, pred
                    
                    # S -> I
                    true = ( tkf91_true_joint_mat[t,3,1] *
                            frag_class_probs[c_frag_to] )
                    
                    pred = log_pred_joint_transit[t,c_frag_from,c_frag_to,3,1]
                    npt.assert_allclose( np.log(true), pred,
                                         rtol = THRESHOLD,
                                         atol = 0 )
                    del true, pred
                    
                    # S -> D
                    true = ( tkf91_true_joint_mat[t,3,2] *
                             frag_class_probs[c_frag_to] )
                    
                    pred = log_pred_joint_transit[t,c_frag_from,c_frag_to,3,2]
                    npt.assert_allclose( np.log(true), pred,
                                         rtol = THRESHOLD,
                                         atol = 0 )
                    del true, pred
                    
                    # S -> E
                    true = tkf91_true_joint_mat[t,3,3]
                    
                    pred = log_pred_joint_transit[t,c_frag_from,c_frag_to,3,3]
                    npt.assert_allclose( np.log(true), pred,
                                         rtol = THRESHOLD,
                                         atol = 0 )
                    del true, pred
        

if __name__ == '__main__':
    unittest.main()
