#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 11:48:02 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
import numpy as np

import numpy.testing as npt
import unittest

from models.simple_site_class_predict.model_functions import regular_tkf as pairhmm_regular_tkf
from models.simple_site_class_predict.transition_models import TKF91TransitionLogprobs

from models.neural_hmm_predict.model_functions import regular_tkf as neural_regular_tkf
from models.neural_hmm_predict.model_functions import logprob_tkf91

THRESHOLD = 1e-6


class TestTKF91(unittest.TestCase):
    def setUp(self):
        self.t_array = jnp.array([0.1, 0.2, 0.3])
        
        # for global models
        self.mu_glob = jnp.array(0.06)
        self.offset_glob = jnp.array(0.01)
        
        # for local models
        self.mu_loc = jnp.array([[0.06, 0.05, 0.04],
                                 [ 0.6,  0.5,  0.4]])
        self.offset_loc = jnp.array( [[0.01, 0.02, 0.03],
                                      [0.03, 0.02, 0.01]] )
    
    def test_global_tkf91(self):
        mu = self.mu_glob
        offset = self.offset_glob
        
        # neural function
        pred_tkf_params,_ = neural_regular_tkf( mu = mu[None,None], 
                                                offset = offset[None,None], 
                                                t_array = self.t_array,
                                                unique_time_per_sample = False )
        
        pred_cond = logprob_tkf91( tkf_params_dict = pred_tkf_params,
                                   offset = offset[None,None],
                                   unique_time_per_sample = False )[:,0,0,...] #(T,4,4)
    
        # reference implementation
        true_tkf_params,_ = pairhmm_regular_tkf( mu = mu,
                                                 offset = offset,
                                                 t_array = self.t_array )
        true_tkf_params['log_one_minus_offset'] = jnp.log( 1-offset )
        true_tkf_params['log_offset'] = jnp.log( offset )
        
        my_model = TKF91TransitionLogprobs(config={'num_tkf_fragment_classes': 1},
                                           name='my_model')
        init_vars = my_model.init( rngs=jax.random.key(0),
                                   t_array=self.t_array,
                                   sow_intermediates=False)
        
        true_joint_mat = my_model.apply( variables=init_vars,
                                         out_dict=true_tkf_params,
                                         method='fill_joint_tkf91'
                                         )
        
        true_all_mats = my_model.apply( variables=init_vars,
                                         offset=offset,
                                         joint_matrix=true_joint_mat,
                                         method='return_all_matrices'
                                         )
        true_cond = true_all_mats['conditional'] #(T, 4, 4)
        
        npt.assert_allclose(pred_cond, true_cond, atol=THRESHOLD)
        
        
    def test_local_tkf91(self):
        mu = self.mu_loc
        offset = self.offset_loc

        B = mu.shape[0]
        L = mu.shape[1]
        
        # neural function
        pred_tkf_params,_ = neural_regular_tkf( mu = mu, 
                                                offset = offset, 
                                                t_array = self.t_array,
                                                unique_time_per_sample = False )
        
        pred_cond = logprob_tkf91( tkf_params_dict = pred_tkf_params,
                                   offset = offset,
                                   unique_time_per_sample = False ) #(T, B, L, 4, 4)
        
        # reference implementation
        for b in range(B):
            for l in range(L):
                pred = pred_cond[:,b,l,...] #(T,4,4)
                
                true_tkf_params,_ = pairhmm_regular_tkf( mu = mu[b,l],
                                                         offset = offset[b,l],
                                                         t_array = self.t_array )
                true_tkf_params['log_one_minus_offset'] = jnp.log( 1-offset[b,l] )[None]
                true_tkf_params['log_offset'] = jnp.log( offset[b,l] )[None]
                
                my_model = TKF91TransitionLogprobs(config={'num_tkf_fragment_classes': 1},
                                                   name='my_model')
                init_vars = my_model.init( rngs=jax.random.key(0),
                                           t_array=self.t_array,
                                           sow_intermediates=False)
                
                true_joint_mat = my_model.apply( variables=init_vars,
                                                 out_dict=true_tkf_params,
                                                 method='fill_joint_tkf91'
                                                 )
                
                true_all_mats = my_model.apply( variables=init_vars,
                                                 offset=offset[b,l],
                                                 joint_matrix=true_joint_mat,
                                                 method='return_all_matrices'
                                                 )
                true_cond = true_all_mats['conditional'] #(T, 4, 4)
                del my_model, init_vars, true_joint_mat, true_all_mats, true_tkf_params
                
                npt.assert_allclose(pred, true_cond, atol=THRESHOLD)


if __name__ == '__main__':
    unittest.main()
    