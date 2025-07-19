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

from models.simple_site_class_predict.transition_models import TKF92TransitionLogprobs
from models.simple_site_class_predict.model_functions import (switch_tkf,
                                                              regular_tkf,
                                                              approx_tkf,
                                                              get_tkf92_single_seq_marginal_transition_logprobs,
                                                              get_cond_transition_logprobs)

THRESHOLD = 1e-6

def TKF_coeffs (lam, mu, t):
    alpha = jnp.exp(-mu*t)
    beta = (lam*(jnp.exp(-lam*t)-jnp.exp(-mu*t))) / (mu*jnp.exp(-lam*t)-lam*jnp.exp(-mu*t))
    gamma = 1 - ( (mu * beta) / ( lam * (1-alpha) ) )
    return alpha, beta, gamma

def true_joint_tkf92 (lam, mu, r, t):
    alpha, beta, gamma = TKF_coeffs (lam, mu, t)
    kappa = lam / mu
    
    # M -> any
    m_to_m = r + ( (1-r) * (1-beta) * kappa * alpha ) 
    m_to_i = (1-r) * beta
    mi_to_d = (1-r) * (1-beta) * kappa * (1-alpha)
    mi_to_e = (1-r) * (1-beta) * (1 - kappa)
    match_row = jnp.array([m_to_m, m_to_i, mi_to_d, mi_to_e])
    
    # I -> any
    i_to_m = (1-r) * (1-beta) * kappa * alpha
    i_to_i = r + ( (1-r) * beta )
    # i -> d is same as m -> d
    # i -> end is same as m -> end
    ins_row = jnp.array([i_to_m, i_to_i, mi_to_d, mi_to_e])
    
    # D -> any
    d_to_m = (1-r) * (1-gamma) * kappa * alpha 
    d_to_i = (1-r) * gamma
    d_to_d = r + ( (1-r) * (1-gamma) * kappa * (1-alpha) )
    d_to_e = (1-r) * (1-gamma) * (1 - kappa)
    del_row = jnp.array([d_to_m, d_to_i, d_to_d, d_to_e])
    
    # start -> any
    s_to_m = (1 - beta) * kappa * alpha
    s_to_i = beta
    s_to_d = (1 - beta) * kappa * (1 - alpha)
    s_to_e = (1 - beta) * (1 - kappa)
    start_row = jnp.array([s_to_m, s_to_i, s_to_d, s_to_e])
    
    return jnp.stack([match_row, ins_row, del_row, start_row])

def true_cond_tkf92 (lam, mu, r, t):
    alpha, beta, gamma = TKF_coeffs (lam, mu, t)
    kappa = lam / mu
    nu = r + (1-r) * (lam / mu)
    
    # M -> any
    m_to_m = (1/nu) * (r + ( (1-r) * (1-beta) * kappa * alpha ) )
    m_to_i = (1-r) * beta
    mi_to_d = (1/nu) * (1-r) * (1-beta) * kappa * (1-alpha)
    mi_to_e = 1 - beta
    match_row = jnp.array([m_to_m, m_to_i, mi_to_d, mi_to_e])
    
    # I -> any
    i_to_m = (1/nu) * (1-r) * (1-beta) * kappa * alpha
    i_to_i = r + ( (1-r) * beta )
    # i -> d is same as m -> d
    # i -> end is same as m -> end
    ins_row = jnp.array([i_to_m, i_to_i, mi_to_d, mi_to_e])
    
    # D -> any
    d_to_m = (1/nu) * (1-r) * (1-gamma) * kappa * alpha 
    d_to_i = (1-r) * gamma
    d_to_d = (1/nu) * (r + ( (1-r) * (1-gamma) * kappa * (1-alpha) ) )
    d_to_e = 1 - gamma
    del_row = jnp.array([d_to_m, d_to_i, d_to_d, d_to_e])
    
    # start -> any
    s_to_m = (1 - beta) * alpha
    s_to_i = beta
    s_to_d = (1 - beta) * (1 - alpha)
    s_to_e = 1 - beta
    start_row = jnp.array([s_to_m, s_to_i, s_to_d, s_to_e])
    
    return jnp.stack([match_row, ins_row, del_row, start_row])

def true_marg_tkf92 (lam, mu, r):
    emit_to_emit = r + ( (1-r) * (lam/mu) )
    emit_to_end = (1 - r) * ( 1 - (lam/mu) )
    start_to_emit = lam/mu
    start_to_end =  1 - (lam/mu)
    
    return jnp.array( [[emit_to_emit,  emit_to_end],
                       [start_to_emit, start_to_end]] )




class TestTKF92JointCondMarg(unittest.TestCase):
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
        self.r = jnp.array([0.1])
        
        
    ##############################
    ### joint: one TKF92 model   #
    ##############################
    def _check_joint_tkf92_calc(self,
                                tkf_function,
                                t_array):
        # by my function (in a flax module)
        my_tkf_params, _ = tkf_function(mu = self.mu, 
                                        offset = self.offset,
                                        t_array = t_array)
        my_tkf_params['log_offset'] = jnp.log(self.offset)
        my_tkf_params['log_one_minus_offset'] = jnp.log1p(-self.offset)
        
        my_model = TKF92TransitionLogprobs(config={'num_tkf_fragment_classes': 1}, 
                                            name='tkf92')
        fake_params = my_model.init(rngs=jax.random.key(0),
                                    t_array = t_array,
                                    log_class_probs = jnp.array([0]),
                                    sow_intermediates = False)
        
        log_pred =  my_model.apply(variables = fake_params,
                                    out_dict = my_tkf_params,
                                    r_extend = self.r,
                                    class_probs = jnp.array([1]),
                                    method = 'fill_joint_tkf92') #(T, 1, 1, 4, 4)
        log_pred = log_pred[:,0,0,...] # (T, 4, 4)
        
        # get true values
        true_tkf92 = []
        for i,t in enumerate(t_array):
            true_tkf92.append( true_joint_tkf92 (self.lam, self.mu, self.r.item(), t) )
        true_tkf92 = jnp.stack(true_tkf92)
        
        npt.assert_allclose(true_tkf92, jnp.exp(log_pred), atol=THRESHOLD)
    
    def test_joint_tkf92_with_switch_tkf(self):
        times = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._check_joint_tkf92_calc( tkf_function = switch_tkf,
                                      t_array = times )
    
    def test_joint_tkf92_with_regular_tkf(self):
        times = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._check_joint_tkf92_calc( tkf_function = regular_tkf,
                                      t_array = times )
    
    def test_joint_tkf92_with_approx_tkf(self):
        """
        run this at small times only
        """
        times = jnp.array([0.0003, 0.0005, 0.0009])
        self._check_joint_tkf92_calc( tkf_function = approx_tkf,
                                      t_array = times )
    
    #################################################
    ### single-sequence marginal: one TKF92 model   #
    #################################################
    def test_marg_tkf92_calc(self):
        log_pred = get_tkf92_single_seq_marginal_transition_logprobs(offset = self.offset,
                                                                      class_probs = jnp.array([1]),
                                                                      r_ext_prob = self.r ) #(1, 1, 2, 2)
        log_pred = log_pred[0,0,...]
        
        true = true_marg_tkf92 (lam = self.lam, 
                                mu = self.mu,
                                r = self.r.item())
        
        npt.assert_allclose(true, jnp.exp(log_pred), atol=THRESHOLD)
    
    
    ####################################
    ### conditional: one TKF92 model   #
    ####################################
    def _check_cond_tkf92_calc(self,
                                tkf_function,
                                t_array):
        # by my function (in a flax module)
        my_tkf_params, _ = tkf_function(mu = self.mu, 
                                        offset = self.offset,
                                        t_array = t_array)
        my_tkf_params['log_offset'] = jnp.log(self.offset)
        my_tkf_params['log_one_minus_offset'] = jnp.log1p(-self.offset)
        
        my_model = TKF92TransitionLogprobs(config={'num_tkf_fragment_classes': 1}, 
                                            name='tkf92')
        fake_params = my_model.init(rngs=jax.random.key(0),
                                    t_array = t_array,
                                    log_class_probs = jnp.array([0]),
                                    sow_intermediates = False)
        
        log_joint_tkf92 =  my_model.apply(variables = fake_params,
                                    out_dict = my_tkf_params,
                                    r_extend = self.r,
                                    class_probs = jnp.array([1]),
                                    method = 'fill_joint_tkf92') #(T, 1, 1, 4, 4)
        
        log_marg_tkf92 = get_tkf92_single_seq_marginal_transition_logprobs(offset = self.offset,
                                                                          class_probs = jnp.array([1]),
                                                                          r_ext_prob = self.r ) #(1, 1, 2, 2)
        
        log_cond_tkf92 = get_cond_transition_logprobs( log_marg_tkf92, 
                                                       log_joint_tkf92 ) #(T, 1, 1, 4, 4)
        log_cond_tkf92 = log_cond_tkf92[:,0,0,...]
        
        
        # get true values
        true_tkf92 = []
        for i,t in enumerate(t_array):
            true_tkf92.append( true_cond_tkf92 (self.lam, self.mu, self.r.item(), t) )
        true_tkf92 = jnp.stack(true_tkf92)
        
        npt.assert_allclose(true_tkf92, jnp.exp(log_cond_tkf92), atol=THRESHOLD)
    
    def test_cond_tkf92_with_switch_tkf(self):
        times = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._check_cond_tkf92_calc( tkf_function = switch_tkf,
                                      t_array = times )
    
    def test_cond_tkf92_with_regular_tkf(self):
        times = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._check_cond_tkf92_calc( tkf_function = regular_tkf,
                                      t_array = times )
    
    def test_cond_tkf92_with_approx_tkf(self):
        """
        run this at small times only
        """
        times = jnp.array([0.0003, 0.0005, 0.0009])
        self._check_cond_tkf92_calc( tkf_function = approx_tkf,
                                      t_array = times )


if __name__ == '__main__':
    unittest.main()