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
                                                              approx_tkf,
                                                              MargTKF91TransitionLogprobs,
                                                              MargTKF92TransitionLogprobs,
                                                              CondTransitionLogprobs)

THRESHOLD = 1e-6

class TestTKF92ReductionToTKF91(unittest.TestCase):
    """
    INDEL PROCESS SCORING TEST 4
    
    
    About
    ------
    tkf92 should always reduce to tkf91
    
    """
    def setUp(self):
        # fake params
        self.lam = jnp.array(0.3)
        self.mu = jnp.array(0.5)
        self.offset = 1 - (self.lam/self.mu)
        self.r = jnp.array([0.0])
        
        # tkf91
        self.tkf91_mod = TKF91TransitionLogprobs(config={}, 
                                                 name='tkf91')
        self.fake_tkf91_params = self.tkf91_mod.init(rngs=jax.random.key(0),
                                               t_array = jnp.zeros((1,)),
                                               sow_intermediates = False)
        
        # tkf92
        self.tkf92_mod = TKF92TransitionLogprobs(config={'num_tkf_fragment_classes': 1}, 
                                                 name='tkf92')
        self.fake_tkf92_params = self.tkf92_mod.init(rngs=jax.random.key(0),
                                               t_array = jnp.zeros((1,)),
                                               log_class_probs = jnp.array([0]),
                                               sow_intermediates = False)
    
    def _get_tkf_param_dict(self, 
                            tkf_function,
                            t_array):
        # alpha, beta, gamma, yadda yadda
        tkf_param_dict, _ = tkf_function(mu = self.mu, 
                                         offset = self.offset,
                                         t_array = t_array)
        tkf_param_dict['log_one_minus_offset'] = jnp.log1p(-self.offset)
        tkf_param_dict['log_offset'] = jnp.log(self.offset)
        return tkf_param_dict
        
        
    def _run_joint_reduction_test(self,
                                  tkf_function,
                                  t_array):
        tkf_param_dict = self._get_tkf_param_dict(tkf_function, t_array)
        joint_tkf91 = self.tkf91_mod.apply(variables = self.fake_tkf91_params,
                                           out_dict = tkf_param_dict,
                                           method = 'fill_joint_tkf91') #(T, 4, 4)
        
        joint_tkf92 =  self.tkf92_mod.apply(variables = self.fake_tkf92_params,
                                            out_dict = tkf_param_dict,
                                            r_extend = self.r,
                                            class_probs = jnp.array([1]),
                                            method = 'fill_joint_tkf92') #(T, 1, 1, 4, 4)
        joint_tkf92 = jnp.squeeze(joint_tkf92)
        
        npt.assert_allclose(joint_tkf91, joint_tkf92, atol=THRESHOLD)
    
    def _run_conditional_reduction_test(self,
                                        tkf_function,
                                        t_array):
        tkf_param_dict = self._get_tkf_param_dict(tkf_function, t_array)
        
        # tkf91
        joint_tkf91 = self.tkf91_mod.apply(variables = self.fake_tkf91_params,
                                           out_dict = tkf_param_dict,
                                           method = 'fill_joint_tkf91') #(T, 4, 4)
        
        marg_tkf91 = MargTKF91TransitionLogprobs( offset = self.offset ) #(2,2)
        
        cond_tkf91 = CondTransitionLogprobs( marg_tkf91, joint_tkf91 ) #(T, 4, 4)
        
        # tkf92
        joint_tkf92 =  self.tkf92_mod.apply(variables = self.fake_tkf92_params,
                                            out_dict = tkf_param_dict,
                                            r_extend = self.r,
                                            class_probs = jnp.array([1]),
                                            method = 'fill_joint_tkf92') #(T, 1, 1, 4, 4)
        
        marg_tkf92 = MargTKF92TransitionLogprobs(offset = self.offset,
                                                 class_probs = jnp.array([1]),
                                                 r_ext_prob = self.r)#(1, 1, 2, 2)
        
        cond_tkf92 = CondTransitionLogprobs( marg_tkf92, joint_tkf92 ) #(T, 1, 1, 4, 4)
        cond_tkf92 = jnp.squeeze(cond_tkf92)
        
        npt.assert_allclose(cond_tkf91, cond_tkf92, atol=THRESHOLD)
        
        
        
    def test_marginal_reduction(self):
        marg_tkf91 = MargTKF91TransitionLogprobs( offset = self.offset ) #(2,2)
        
        marg_tkf92 = MargTKF92TransitionLogprobs(offset = self.offset,
                                                 class_probs = jnp.array([1]),
                                                 r_ext_prob = self.r)#(1, 1, 2, 2)
        marg_tkf92 = jnp.squeeze(marg_tkf92)
        
        npt.assert_allclose(marg_tkf91, marg_tkf92, atol=THRESHOLD)
        
        
        
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
    
    
    
    def test_switch_tkf_conditional(self):
        t_array = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._run_conditional_reduction_test( tkf_function = switch_tkf,
                                              t_array = t_array ) 
    
    def test_regular_tkf_conditional(self):
        t_array = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._run_conditional_reduction_test( tkf_function = regular_tkf,
                                              t_array = t_array )
    
    def test_approx_tkf_conditional(self):
        """
        run this at small times only
        """
        t_array = jnp.array([0.0003, 0.0005, 0.0009])
        self._run_conditional_reduction_test( tkf_function = approx_tkf,
                                              t_array = t_array )
    
    
        
        
if __name__ == '__main__':
    unittest.main()

