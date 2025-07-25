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

def true_conditional_tkf91 (lam, mu, t):
    alpha, beta, gamma = TKF_coeffs (lam, mu, t)
    return jnp.array ([[(1-beta)*alpha,  beta,  (1-beta)*(1-alpha),  1 - beta],
                       [(1-beta)*alpha,  beta,  (1-beta)*(1-alpha),  1 - beta],
                       [(1-gamma)*alpha, gamma, (1-gamma)*(1-alpha), 1 - gamma],
                       [(1-beta)*alpha,  beta,  (1-beta)*(1-alpha),  1 - beta]])

def true_joint_tkf92_scaled (lam, mu, r_c, t, c, d, prob_d):
    # M = 0, I = 1, D = 2, S/E = 3
    U = true_conditional_tkf91 (lam, mu, t) #(S_from, S_to)
    match_idx = 0
    ins_idx = 1
    del_idx = 2
    se_idx = 3
    
    ### matrix where c = d
    if c == d:
        # M -> any
        m_to_m = r_c + ( (1-r_c) * (lam/mu) * U[match_idx,match_idx] * prob_d )
        m_to_i = (1-r_c) * U[match_idx, ins_idx] * prob_d
        m_to_d = (1-r_c) * (lam/mu) * U[match_idx, del_idx] * prob_d
        m_to_e = (1-r_c) * (1 - (lam/mu)) * U[match_idx, se_idx]
        match_row = jnp.array([m_to_m, m_to_i, m_to_d, m_to_e])
        
        # I -> any
        i_to_m = (1-r_c) * (lam/mu) * U[ins_idx, match_idx] * prob_d
        i_to_i = r_c + ( (1-r_c) * U[ins_idx, ins_idx] * prob_d )
        i_to_d = (1-r_c) * (lam/mu) * U[ins_idx, del_idx] * prob_d
        i_to_e = (1-r_c) * (1 - (lam/mu)) * U[ins_idx, se_idx]
        ins_row = jnp.array([i_to_m, i_to_i, i_to_d, i_to_e])
        
        # D -> any
        d_to_m = (1-r_c) * (lam/mu) * U[del_idx, match_idx] * prob_d
        d_to_i = (1-r_c) * U[del_idx, ins_idx] * prob_d
        d_to_d = r_c + ( (1-r_c) * (lam/mu) * U[del_idx, del_idx] * prob_d )
        d_to_e = (1-r_c) * (1 - (lam/mu)) * U[del_idx, se_idx] 
        del_row = jnp.array([d_to_m, d_to_i, d_to_d, d_to_e])
        
        # start -> any
        s_to_m = (lam/mu) * U[se_idx, match_idx] * prob_d
        s_to_i =  U[se_idx, ins_idx] * prob_d
        s_to_d = (lam/mu)  * U[se_idx, del_idx] * prob_d
        s_to_e = (1 - (lam/mu)) * U[se_idx, se_idx]
        start_row = jnp.array([s_to_m, s_to_i, s_to_d, s_to_e])
        
        
    ### matrix where c != d
    elif c != d:
        # M -> any
        m_to_m = ( (1-r_c) * (lam/mu) * U[match_idx,match_idx] * prob_d )
        m_to_i = (1-r_c) * U[match_idx, ins_idx] * prob_d
        m_to_d = (1-r_c) * (lam/mu) * U[match_idx, del_idx] * prob_d
        m_to_e = (1-r_c) * (1 - (lam/mu)) * U[match_idx, se_idx]
        match_row = jnp.array([m_to_m, m_to_i, m_to_d, m_to_e])
        
        # I -> any
        i_to_m = (1-r_c) * (lam/mu) * U[ins_idx, match_idx] * prob_d
        i_to_i = ( (1-r_c) * U[ins_idx, ins_idx] * prob_d )
        i_to_d = (1-r_c) * (lam/mu) * U[ins_idx, del_idx] * prob_d
        i_to_e = (1-r_c) * (1 - (lam/mu)) * U[ins_idx, se_idx]
        ins_row = jnp.array([i_to_m, i_to_i, i_to_d, i_to_e])
        
        # D -> any
        d_to_m = (1-r_c) * (lam/mu) * U[del_idx, match_idx] * prob_d
        d_to_i = (1-r_c) * U[del_idx, ins_idx] * prob_d
        d_to_d = ( (1-r_c) * (lam/mu) * U[del_idx, del_idx] * prob_d )
        d_to_e = (1-r_c) * (1 - (lam/mu)) * U[del_idx, se_idx] 
        del_row = jnp.array([d_to_m, d_to_i, d_to_d, d_to_e])
        
        # start -> any
        s_to_m = (lam/mu) * U[se_idx, match_idx] * prob_d
        s_to_i =  U[se_idx, ins_idx] * prob_d
        s_to_d = (lam/mu)  * U[se_idx, del_idx] * prob_d
        s_to_e = (1 - (lam/mu)) * U[se_idx, se_idx]
        start_row = jnp.array([s_to_m, s_to_i, s_to_d, s_to_e])
    
    return jnp.stack([match_row, ins_row, del_row, start_row])

def true_cond_tkf92_scaled (lam, mu, r_c, t, c, d, prob_d):
    alpha, beta, gamma = TKF_coeffs (lam, mu, t)
    denom = r_c + (1-r_c) * (lam / mu) * prob_d
    
    # M = 0, I = 1, D = 2, S/E = 3
    U = true_conditional_tkf91 (lam, mu, t) #(S_from, S_to)
    match_idx = 0
    ins_idx = 1
    del_idx = 2
    se_idx = 3
    
    ### matrix where c = d
    if c == d:
        # M -> any
        m_to_m = (1/denom) * ( r_c + ( (1-r_c) * (lam/mu) * U[match_idx,match_idx] * prob_d ) )
        m_to_i = (1-r_c) * U[match_idx, ins_idx] * prob_d
        m_to_d = (1/denom) * ( (1-r_c) * (lam/mu) * U[match_idx, del_idx] * prob_d )
        m_to_e = U[match_idx, se_idx]
        match_row = jnp.array([m_to_m, m_to_i, m_to_d, m_to_e])
        
        # I -> any
        i_to_m = (1/denom) * ( (1-r_c) * (lam/mu) * U[ins_idx, match_idx] * prob_d )
        i_to_i = r_c + ( (1-r_c) * U[ins_idx, ins_idx] * prob_d )
        i_to_d = (1/denom) * ( (1-r_c) * (lam/mu) * U[ins_idx, del_idx] * prob_d )
        i_to_e = U[ins_idx, se_idx]
        ins_row = jnp.array([i_to_m, i_to_i, i_to_d, i_to_e])
        
        # D -> any
        d_to_m = (1/denom) * ( (1-r_c) * (lam/mu) * U[del_idx, match_idx] * prob_d )
        d_to_i = (1-r_c) * U[del_idx, ins_idx] * prob_d
        d_to_d = (1/denom) * ( r_c + ( (1-r_c) * (lam/mu) * U[del_idx, del_idx] * prob_d ) )
        d_to_e = U[del_idx, se_idx] 
        del_row = jnp.array([d_to_m, d_to_i, d_to_d, d_to_e])
        
        # start -> any
        s_to_m = U[se_idx, match_idx]
        s_to_i =  U[se_idx, ins_idx] * prob_d
        s_to_d = U[se_idx, del_idx]
        s_to_e = U[se_idx, se_idx]
        start_row = jnp.array([s_to_m, s_to_i, s_to_d, s_to_e])
        
        
    ### matrix where c != d
    elif c != d:
        # M -> any
        m_to_m = U[match_idx,match_idx]                         
        m_to_i = (1- r_c) * U[match_idx, ins_idx] * prob_d
        m_to_d = U[match_idx, del_idx]
        m_to_e = U[match_idx, se_idx]
        match_row = jnp.array([m_to_m, m_to_i, m_to_d, m_to_e])
        
        # I -> any
        i_to_m = U[ins_idx, match_idx]
        i_to_i = ( (1-r_c) * U[ins_idx, ins_idx] * prob_d )
        i_to_d = U[ins_idx, del_idx] 
        i_to_e = U[ins_idx, se_idx]
        ins_row = jnp.array([i_to_m, i_to_i, i_to_d, i_to_e])
        
        # D -> any
        d_to_m = U[del_idx, match_idx]
        d_to_i = (1-r_c) * U[del_idx, ins_idx] * prob_d
        d_to_d = U[del_idx, del_idx] 
        d_to_e = U[del_idx, se_idx] 
        del_row = jnp.array([d_to_m, d_to_i, d_to_d, d_to_e])
        
        # start -> any
        s_to_m = U[se_idx, match_idx]
        s_to_i =  U[se_idx, ins_idx] * prob_d
        s_to_d = U[se_idx, del_idx]
        s_to_e = U[se_idx, se_idx]
        start_row = jnp.array([s_to_m, s_to_i, s_to_d, s_to_e])
    
    return jnp.stack([match_row, ins_row, del_row, start_row])

def true_marg_tkf92_scaled (lam, mu, r_c, c, d, prob_d):
    if c == d:
        emit_to_emit = r_c + ( (1-r_c) * (lam/mu) * prob_d )
        
    elif c != d:
        emit_to_emit = (1-r_c) * (lam/mu) * prob_d
    
    emit_to_end = (1 - r_c) * (1 - (lam/mu))
    start_to_emit = (lam/mu) * prob_d
    start_to_end = 1 - (lam/mu)
    
    return jnp.array( [[emit_to_emit,  emit_to_end],
                        [start_to_emit, start_to_end]] )




class TestTKF92FragMixJointCondMarg(unittest.TestCase):
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
        self.r_mix = jnp.array([0.1, 0.5, 0.9])
        self.log_class_probs = jnp.log( jnp.array([0.2, 0.3, 0.5]) )
        self.C = self.r_mix.shape[0]
        
        
    ##################################
    ### joint: mix of TKF92 models   #
    ##################################
    def _check_joint_tkf92_calc(self,
                                tkf_function,
                                t_array):
        # by my function (in a flax module)
        my_tkf_params, _ = tkf_function(mu = self.mu, 
                                        offset = self.offset,
                                        t_array = t_array)
        my_tkf_params['log_offset'] = jnp.log(self.offset)
        my_tkf_params['log_one_minus_offset'] = jnp.log1p(-self.offset)
        
        my_model = TKF92TransitionLogprobs(config={'num_tkf_fragment_classes': self.C}, 
                                            name='tkf92')
        fake_params = my_model.init(rngs=jax.random.key(0),
                                    t_array = t_array,
                                    log_class_probs = self.log_class_probs,
                                    sow_intermediates = False)
        
        log_pred =  my_model.apply(variables = fake_params,
                                    out_dict = my_tkf_params,
                                    r_extend = self.r_mix,
                                    class_probs = jnp.exp( self.log_class_probs ),
                                    method = 'fill_joint_tkf92') #(T, C, C, 4, 4)
        
        # get true values
        true_tkf92 = np.zeros( log_pred.shape )
        for c in range(self.C):
            r_c = self.r_mix[c].item()
            
            for d in range(self.C):
                prob_d = np.exp( self.log_class_probs[d].item() )
                
                for t_idx, t in enumerate(t_array):
                    out = true_joint_tkf92_scaled (lam = self.lam, 
                                                   mu = self.mu, 
                                                   r_c = r_c, 
                                                   t = t, 
                                                   c = c, 
                                                   d = d, 
                                                   prob_d = prob_d) #(S, S)
                    true_tkf92[t_idx, c, d, :, :] = out
                    
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
    
    #####################################################
    ### single-sequence marginal: mix of TKF92 models   #
    #####################################################
    def test_marg_tkf92_calc(self):
        # by my function
        log_pred = get_tkf92_single_seq_marginal_transition_logprobs(offset = self.offset,
                                                                      class_probs = jnp.exp( self.log_class_probs ),
                                                                      r_ext_prob = self.r_mix ) #(C, C, 2, 2)
        
        # get true values
        true = np.zeros( log_pred.shape )
        for c in range(self.C):
            r_c = self.r_mix[c].item()
            
            for d in range(self.C):
                prob_d = np.exp( self.log_class_probs[d].item() )
                
                out = true_marg_tkf92_scaled (lam = self.lam, 
                                       mu = self.mu, 
                                       r_c = r_c, 
                                       c = c, 
                                       d = d, 
                                       prob_d = prob_d) #(S, S)
                true[c, d, :, :] = out
        
        npt.assert_allclose(true, jnp.exp(log_pred), atol=THRESHOLD)
    
    
    ########################################
    ### conditional: mix of TKF92 models   #
    ########################################
    def _check_cond_tkf92_calc(self,
                                tkf_function,
                                t_array):
        # by my function (in a flax module)
        my_tkf_params, _ = tkf_function(mu = self.mu, 
                                        offset = self.offset,
                                        t_array = t_array)
        my_tkf_params['log_offset'] = jnp.log(self.offset)
        my_tkf_params['log_one_minus_offset'] = jnp.log1p(-self.offset)
        
        my_model = TKF92TransitionLogprobs(config={'num_tkf_fragment_classes': self.C}, 
                                            name='tkf92')
        fake_params = my_model.init(rngs=jax.random.key(0),
                                    t_array = t_array,
                                    log_class_probs = self.log_class_probs,
                                    sow_intermediates = False)
        
        log_joint_tkf92 =  my_model.apply(variables = fake_params,
                                    out_dict = my_tkf_params,
                                    r_extend = self.r_mix,
                                    class_probs = jnp.exp( self.log_class_probs ),
                                    method = 'fill_joint_tkf92') #(T, C, C, 4, 4)
        
        log_marg_tkf92 = get_tkf92_single_seq_marginal_transition_logprobs(offset = self.offset,
                                                                          class_probs = jnp.exp( self.log_class_probs ),
                                                                          r_ext_prob = self.r_mix ) #(C, C, 2, 2)
        
        log_cond_tkf92 = get_cond_transition_logprobs( log_marg_tkf92, 
                                                        log_joint_tkf92 ) #(T, C, C, 4, 4)
        
        
        # get true values
        true_tkf92 = np.zeros( log_cond_tkf92.shape )
        for c in range(self.C):
            r_c = self.r_mix[c].item()
            
            for d in range(self.C):
                prob_d = np.exp( self.log_class_probs[d].item() )
                
                for t_idx, t in enumerate(t_array):
                    out = true_cond_tkf92_scaled (lam = self.lam, 
                                           mu = self.mu, 
                                           r_c = r_c, 
                                           t = t, 
                                           c = c, 
                                           d = d, 
                                           prob_d = prob_d) #(S, S)
                    
                    true_tkf92[t_idx, c, d, :, :] = out
                    
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