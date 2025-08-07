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

from models.simple_site_class_predict.transition_models import TKF91TransitionLogprobs
from models.simple_site_class_predict.model_functions import (switch_tkf,
                                                              regular_tkf,
                                                              approx_tkf,
                                                              get_tkf91_single_seq_marginal_transition_logprobs,
                                                              get_cond_transition_logprobs)

THRESHOLD = 1e-6

### these are from ian
def TKF_coeffs (lam, mu, t):
    alpha = jnp.exp(-mu*t)
    beta = (lam*(jnp.exp(-lam*t)-jnp.exp(-mu*t))) / (mu*jnp.exp(-lam*t)-lam*jnp.exp(-mu*t))
    gamma = 1 - ( (mu * beta) / ( lam * (1-alpha) ) )
    return alpha, beta, gamma

### this is from overleaf
def true_conditional_tkf91 (lam, mu, t):
    alpha, beta, gamma = TKF_coeffs (lam, mu, t)
    return jnp.array ([[(1-beta)*alpha,  beta,  (1-beta)*(1-alpha),  1 - beta],
                       [(1-beta)*alpha,  beta,  (1-beta)*(1-alpha),  1 - beta],
                       [(1-gamma)*alpha, gamma, (1-gamma)*(1-alpha), 1 - gamma],
                       [(1-beta)*alpha,  beta,  (1-beta)*(1-alpha),  1 - beta]])

def true_joint_tkf91 (lam, mu, t):
    alpha, beta, gamma = TKF_coeffs (lam, mu, t)
    p_emit = (lam/mu)
    p_end = 1 - (lam/mu)
    return jnp.array ([[(1-beta)*alpha*p_emit,  beta,  (1-beta)*(1-alpha)*p_emit,  (1 - beta)*p_end],
                       [(1-beta)*alpha*p_emit,  beta,  (1-beta)*(1-alpha)*p_emit,  (1 - beta)*p_end],
                       [(1-gamma)*alpha*p_emit, gamma, (1-gamma)*(1-alpha)*p_emit, (1 - gamma)*p_end],
                       [(1-beta)*alpha*p_emit,  beta,  (1-beta)*(1-alpha)*p_emit,  (1 - beta)*p_end]])

def true_marg_tkf91 (lam, mu):
    p_emit = (lam/mu)
    p_end = 1 - (lam/mu)
    return jnp.array( [[p_emit, p_end],
                       [p_emit, p_end]] )



class TestTKF91JointCondMarg(unittest.TestCase):
    """
    About
    ------
    Test joint, conditonal, and single-sequence transition marginals
    
    """
    def setUp(self):
        # fake params
        self.lam = jnp.array(0.3)
        self.mu = jnp.array(0.5)
        self.offset = 1 - (self.lam/self.mu)
        
        
    #############
    ### joint   #
    #############
    def _check_joint_tkf91_calc(self,
                                tkf_function,
                                t_array):
        # by my function (in a flax module)
        my_tkf_params, _ = tkf_function(mu = self.mu, 
                                        offset = self.offset,
                                        t_array = t_array)
        my_tkf_params['log_offset'] = jnp.log(self.offset)
        my_tkf_params['log_one_minus_offset'] = jnp.log1p(-self.offset)
        
        my_model = TKF91TransitionLogprobs(config={'tkf_function': 'regular_tkf'}, 
                                            name='tkf91')
        fake_params = my_model.init(rngs=jax.random.key(0),
                                    t_array = t_array,
                                    return_all_matrices = False,
                                    sow_intermediates = False)
        
        log_pred =  my_model.apply(variables = fake_params,
                                   out_dict = my_tkf_params,
                                   method = 'fill_joint_tkf91') #(T, 4, 4)
        
        # get true values
        true_tkf91 = []
        for i,t in enumerate(t_array):
            true_tkf91.append( true_joint_tkf91 (self.lam, self.mu, t) )
        true_tkf91 = jnp.stack(true_tkf91)
        
        npt.assert_allclose(true_tkf91, jnp.exp(log_pred), atol=THRESHOLD)
    
    def test_joint_tkf91_with_switch_tkf(self):
        times = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._check_joint_tkf91_calc( tkf_function = switch_tkf,
                                      t_array = times )
    
    def test_joint_tkf91_with_regular_tkf(self):
        times = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._check_joint_tkf91_calc( tkf_function = regular_tkf,
                                      t_array = times )
    
    def test_joint_tkf91_with_approx_tkf(self):
        """
        run this at small times only
        """
        times = jnp.array([0.0003, 0.0005, 0.0009])
        self._check_joint_tkf91_calc( tkf_function = approx_tkf,
                                      t_array = times )
    
    ################################
    ### single-sequence marginal   #
    ################################
    def test_marg_tkf91_calc(self):
        log_pred = get_tkf91_single_seq_marginal_transition_logprobs(self.offset) #(2, 2)
        true = true_marg_tkf91 (lam = self.lam, 
                                 mu = self.mu)
        npt.assert_allclose(true, jnp.exp(log_pred), atol=THRESHOLD)
    
    
    ###################
    ### conditional   #
    ###################
    def _check_cond_tkf91_calc(self,
                               tkf_function,
                               t_array):
        # my function comes packaged in a flax module
        my_tkf_params, _ = tkf_function(mu = self.mu, 
                                        offset = self.offset,
                                        t_array = t_array)
        my_tkf_params['log_offset'] = jnp.log(self.offset)
        my_tkf_params['log_one_minus_offset'] = jnp.log1p(-self.offset)
        
        my_model = TKF91TransitionLogprobs(config={'num_domain_mixtures': 1,
                                                   'num_fragment_mixtures': 1,
                                                   'num_site_mixtures': 1,
                                                   'k_rate_mults': 1,
                                                   'tkf_function': 'regular_tkf'}, 
                                            name='tkf91')
        fake_params = my_model.init(rngs=jax.random.key(0),
                                    t_array = t_array,
                                    return_all_matrices = False,
                                    sow_intermediates = False)
        
        log_joint_tkf91 =  my_model.apply(variables = fake_params,
                                      out_dict = my_tkf_params,
                                      method = 'fill_joint_tkf91') #(T, 4, 4)
        
        log_marg_tkf91 = get_tkf91_single_seq_marginal_transition_logprobs(self.offset) #(2, 2)
        
        log_cond_tkf91 = get_cond_transition_logprobs( log_marg_tkf91, log_joint_tkf91 ) #(T, 4, 4)
        
        # get true values
        true_tkf91 = []
        for i,t in enumerate(t_array):
            true_tkf91.append( true_conditional_tkf91 (self.lam, self.mu, t) )
        true_tkf91 = jnp.stack(true_tkf91)
        
        npt.assert_allclose(true_tkf91, jnp.exp(log_cond_tkf91), atol=THRESHOLD)
    
    def test_cond_tkf91_with_switch_tkf(self):
        times = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._check_cond_tkf91_calc( tkf_function = switch_tkf,
                                     t_array = times )
    
    def test_cond_tkf91_with_regular_tkf(self):
        times = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._check_cond_tkf91_calc( tkf_function = regular_tkf,
                                     t_array = times )
    
    def test_cond_tkf91_with_approx_tkf(self):
        """
        run this at small times only
        """
        times = jnp.array([0.0003, 0.0005, 0.0009])
        self._check_cond_tkf91_calc( tkf_function = approx_tkf,
                                     t_array = times )
        


if __name__ == '__main__':
    unittest.main()

