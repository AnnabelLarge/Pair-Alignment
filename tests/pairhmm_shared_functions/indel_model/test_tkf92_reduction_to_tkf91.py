#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 18:40:55 2025

@author: annabel
"""
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

import numpy as np

import numpy.testing as npt
import unittest

from models.simple_site_class_predict.transition_models import (TKF91TransitionLogprobs,
                                                                TKF92TransitionLogprobs)
from models.simple_site_class_predict.model_functions import (switch_tkf,
                                                              regular_tkf,
                                                              approx_tkf)

THRESHOLD = 1e-6


class TestTKF92ReductionToTKF91(unittest.TestCase):
    """
    About
    ------
    tkf92 should always reduce to tkf91
    
    """
    def setUp(self):
        # fake params
        self.lam = jnp.array(0.3)
        self.mu = jnp.array(0.5)
        self.offset = 1 - (self.lam/self.mu)
        self.r_mix = jnp.array([0., 0., 0.])
        self.log_class_probs = jnp.log( jnp.array([0.2, 0.3, 0.5]) )
        self.C = self.r_mix.shape[0]
        
        self.tkf91_mod = TKF91TransitionLogprobs(config={}, 
                                                 name='tkf91')
        
        self.tkf92_mod = TKF92TransitionLogprobs(config={'num_tkf_fragment_classes': self.C}, 
                                                 name='tkf92')
    
    def _get_tkf_param_dict(self, 
                            tkf_function,
                            t_array):
        tkf_param_dict, _ = tkf_function(mu = self.mu, 
                                         offset = self.offset,
                                         t_array = t_array)
        tkf_param_dict['log_one_minus_offset'] = jnp.log1p(-self.offset)
        tkf_param_dict['log_offset'] = jnp.log(self.offset)
        return tkf_param_dict
        
        
    #############
    ### joint   #
    #############
    def _run_joint_reduction_test(self,
                                  tkf_function,
                                  t_array):
        tkf_param_dict = self._get_tkf_param_dict(tkf_function, t_array)
        
        # tkf91
        fake_tkf91_params = self.tkf91_mod.init(rngs=jax.random.key(0),
                                                t_array = t_array,
                                                sow_intermediates = False)
        
        log_joint_tkf91 = self.tkf91_mod.apply(variables = fake_tkf91_params,
                                            out_dict = tkf_param_dict,
                                            method = 'fill_joint_tkf91') #(T, 4, 4)
        
        # tkf92, in parts
        fake_tkf92_params = self.tkf92_mod.init(rngs=jax.random.key(0),
                                                t_array = t_array,
                                                log_class_probs = self.log_class_probs,
                                                sow_intermediates = False)
        
        log_joint_tkf92_in_parts =  self.tkf92_mod.apply(variables = fake_tkf92_params,
                                                      out_dict = tkf_param_dict,
                                                      r_extend = self.r_mix,
                                                      class_probs = np.exp( self.log_class_probs ),
                                                      method = 'fill_joint_tkf92') #(T, C, C, 4, 4)
        joint_tkf92_in_parts = np.exp(log_joint_tkf92_in_parts) #(T, C, C, 4, 4)
        
        for c in range( self.C ):
            # all transitions except any->End are weighted by P(d); sum over all P(d)
            joint_tkf92 = np.zeros( log_joint_tkf91.shape ) 
            
            for d in range( self.C ):
                weighted_mat_c_d = joint_tkf92_in_parts[:, c, d, :, :-1] #(T, 4, 3)
                joint_tkf92[..., :-1] += weighted_mat_c_d
            
            # add end probability, which is not weighted
            joint_tkf92[..., -1] = joint_tkf92_in_parts[:, c, d, :, -1]
            npt.assert_allclose( np.exp(log_joint_tkf91) , joint_tkf92, atol=THRESHOLD)
        
    def test_switch_tkf_joint(self):
        t_array = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._run_joint_reduction_test( tkf_function = switch_tkf,
                                        t_array = t_array ) 
    
    def test_regular_tkf_joint(self):
        t_array = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._run_joint_reduction_test( tkf_function = regular_tkf,
                                        t_array = t_array )
    
    def test_approx_tkf_joint(self):
        """
        run this at small times only
        """
        t_array = jnp.array([0.0003, 0.0005, 0.0009])
        self._run_joint_reduction_test( tkf_function = approx_tkf,
                                        t_array = t_array )
    
    ################
    ### marginal   #
    ################
    def test_marginal_reduction(self):
        ### to make this run, technically need to get the joint matrix
        ###   however, it's not really used in this calculation
        dummy_tkf_param_dict = self._get_tkf_param_dict(switch_tkf, np.ones(1,))
        dummy_time = np.ones( (1,) )
        offset = 1 - (self.lam / self.mu)
        
        # tkf91
        fake_tkf91_params = self.tkf91_mod.init(rngs=jax.random.key(0),
                                                t_array = dummy_time,
                                                sow_intermediates = False)
        
        log_joint_tkf91 = self.tkf91_mod.apply(variables = fake_tkf91_params,
                                            out_dict = dummy_tkf_param_dict,
                                            method = 'fill_joint_tkf91')
        
        all_tkf91 = self.tkf91_mod.apply(variables = fake_tkf91_params,
                                          offset = offset,
                                          joint_matrix = log_joint_tkf91,
                                          method = 'return_all_matrices' )
        
        # tkf92, in parts
        fake_tkf92_params = self.tkf92_mod.init(rngs=jax.random.key(0),
                                                t_array = dummy_time,
                                                log_class_probs = self.log_class_probs,
                                                sow_intermediates = False)
        
        log_joint_tkf92 =  self.tkf92_mod.apply(variables = fake_tkf92_params,
                                                out_dict = dummy_tkf_param_dict,
                                                r_extend = self.r_mix,
                                                class_probs = np.exp( self.log_class_probs ),
                                                method = 'fill_joint_tkf92') 
        
        all_tkf92 = self.tkf92_mod.apply(variables = fake_tkf92_params,
                                          offset = offset,
                                          joint_matrix = log_joint_tkf92,
                                          r_ext_prob = self.r_mix,
                                          class_probs = np.exp( self.log_class_probs ),
                                          method = 'return_all_matrices' )
        
        ### test marginals
        log_marg_tkf91 = all_tkf91['marginal']
        marg_tkf92_in_parts = np.exp( all_tkf92['marginal'] )
        
        for c in range( self.C ):
            marg_tkf92 = np.zeros( log_marg_tkf91.shape )
            
            # start -> emit and emit -> emit are weighted by P(d)
            start_to_emit = 0
            emit_to_emit = 0
            
            for d in range( self.C ):
                emit_to_emit +=  marg_tkf92_in_parts[c, d, 0, 0]
                start_to_emit += marg_tkf92_in_parts[c, d, 1, 0]
            
            # others are not weighted; fill the final matrix
            marg_tkf92[0,0] = emit_to_emit
            marg_tkf92[1,0] = start_to_emit
            marg_tkf92[0,1] = marg_tkf92_in_parts[c, d, 0, 1]
            marg_tkf92[1,1] = marg_tkf92_in_parts[c, d, 1, 1]
                
            npt.assert_allclose( np.exp(log_marg_tkf91) , marg_tkf92, atol=THRESHOLD)
    
    
    ###################
    ### conditional   #
    ###################
    def _run_cond_reduction_test(self,
                                  tkf_function,
                                  t_array):
        tkf_param_dict = self._get_tkf_param_dict(tkf_function, t_array)
        offset = 1 - (self.lam / self.mu)
        
        ### generate all matrices
        # tkf91
        fake_tkf91_params = self.tkf91_mod.init(rngs=jax.random.key(0),
                                               t_array = t_array,
                                               sow_intermediates = False)
        
        log_joint_tkf91 = self.tkf91_mod.apply(variables = fake_tkf91_params,
                                           out_dict = tkf_param_dict,
                                           method = 'fill_joint_tkf91')
        
        all_tkf91 = self.tkf91_mod.apply(variables = fake_tkf91_params,
                                         offset = offset,
                                         joint_matrix = log_joint_tkf91,
                                         method = 'return_all_matrices' )
        del log_joint_tkf91
        
        log_cond_tkf91 = all_tkf91['conditional']
        
        # tkf92, in parts
        fake_tkf92_params = self.tkf92_mod.init(rngs=jax.random.key(0),
                                               t_array = t_array,
                                               log_class_probs = self.log_class_probs,
                                               sow_intermediates = False)
        
        log_joint_tkf92 =  self.tkf92_mod.apply(variables = fake_tkf92_params,
                                                out_dict = tkf_param_dict,
                                                r_extend = self.r_mix,
                                                class_probs = np.exp( self.log_class_probs ),
                                                method = 'fill_joint_tkf92') 
        
        all_tkf92 = self.tkf92_mod.apply(variables = fake_tkf92_params,
                                         offset = offset,
                                         joint_matrix = log_joint_tkf92,
                                         r_ext_prob = self.r_mix,
                                         class_probs = np.exp( self.log_class_probs ),
                                         method = 'return_all_matrices' )
        
        del log_joint_tkf92
        
        cond_tkf92_in_parts = np.exp( all_tkf92['conditional'] )
        
        
        for c in range( self.C ):
            cond_tkf92 = np.zeros( log_cond_tkf91.shape ) 
            
            # after simplification, only any->Ins transitions are weighted by 
            #   P(d); sum over all P(d)
            for d in range( self.C ):
                weighted_mat_c_d = cond_tkf92_in_parts[:, c, d, :, 1] #(T, 4)
                cond_tkf92[..., 1] += weighted_mat_c_d
            
            # all others aren't weighted; use as-is
            cond_tkf92[..., 0] = cond_tkf92_in_parts[:, c, d, :, 0]
            cond_tkf92[..., 2:] = cond_tkf92_in_parts[:, c, d, :, 2:]
            npt.assert_allclose( np.exp(log_cond_tkf91) , cond_tkf92, atol=THRESHOLD)
        
    def test_switch_tkf_cond(self):
        t_array = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._run_cond_reduction_test( tkf_function = switch_tkf,
                                        t_array = t_array ) 
    
    def test_regular_tkf_cond(self):
        t_array = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._run_cond_reduction_test( tkf_function = regular_tkf,
                                        t_array = t_array )
    
    def test_approx_tkf_cond(self):
        """
        run this at small times only
        """
        t_array = jnp.array([0.0003, 0.0005, 0.0009])
        self._run_cond_reduction_test( tkf_function = approx_tkf,
                                        t_array = t_array )
    
if __name__ == '__main__':
    unittest.main()

