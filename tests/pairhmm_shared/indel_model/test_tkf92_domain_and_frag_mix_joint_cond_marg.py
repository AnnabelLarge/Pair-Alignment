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
from scipy.special import softmax

import numpy.testing as npt
import unittest

from models.simple_site_class_predict.transition_models import TKF92TransitionLogprobs
from models.simple_site_class_predict.model_functions import (switch_tkf,
                                                              regular_tkf,
                                                              get_tkf92_single_seq_marginal_transition_logprobs,
                                                              get_cond_transition_logprobs)

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




class TestTKF92DomainAndFragMixJointCondMarg(unittest.TestCase):
    """
    About
    ------
    Test joint, conditonal, and single-sequence transition marginals
    """
    def setUp(self):
        # fake params
        # lam = jnp.array(0.3)
        # mu = jnp.array(0.5)
        # offset = 1 - (lam/mu)
        
        C_dom = 3
        C_frag = 2
        
        logits = np.random.rand( C_dom, C_frag )
        self.r_mix = np.abs(logits)
        del logits
        
        self.domain_class_probs = softmax( np.random.rand( (C_dom) ) )
        
        logits = np.random.rand( C_dom, C_frag )
        self.fragment_class_probs = softmax( logits, axis=-1 )
        del logits
        
        self.C_dom = C_dom
        self.C_frag = C_frag
        
        
    #############
    ### joint   #
    #############
    def _check_joint_tkf92_calc(self,
                                lam,
                                mu,
                                tkf_function,
                                t_array,
                                rtol):
        T = t_array.shape[0]
        C_dom = self.C_dom
        C_frag = self.C_frag
        S = 4
        
        # check shapes
        assert lam.shape == (C_dom,)
        assert mu.shape == (C_dom,)
        
        # get offset
        offset = 1 - lam/mu
        
        
        ### get true values
        true = np.zeros( (T, C_dom, C_frag, C_frag, S, S) )
        for c_dom in range(C_dom):
            for c in range(C_frag):                
                r_c = self.r_mix[c_dom, c].item()
                
                for d in range(C_frag):
                    prob_d = self.fragment_class_probs[c_dom, d].item()
                    
                    for t_idx, t in enumerate(t_array):
                        out = true_joint_tkf92_scaled (lam = lam[c_dom], 
                                                       mu = mu[c_dom], 
                                                       r_c = r_c, 
                                                       t = t, 
                                                       c = c, 
                                                       d = d, 
                                                       prob_d = prob_d) #(S, S)
                        
                        true[t_idx, c_dom, c, d, :, :] = out
        
        # check shape
        npt.assert_allclose( true.shape, (T, C_dom, C_frag, C_frag, S, S) )
        
        
        ### by my function (in a flax module)
        my_tkf_params, _ = tkf_function(mu = mu, 
                                        offset = offset,
                                        t_array = t_array)
        my_tkf_params['log_offset'] = jnp.log(offset)
        my_tkf_params['log_one_minus_offset'] = jnp.log1p(-offset)
        
        #init params with regular_tkf, but don't use it
        my_model = TKF92TransitionLogprobs(config={'num_domain_mixtures': C_dom,
                                                   'num_fragment_mixtures': C_frag,
                                                   'num_site_mixtures': 1,
                                                   'k_rate_mults': 1,
                                                   'tkf_function': 'regular_tkf'}, 
                                            name='tkf92')
        fake_params = my_model.init(rngs=jax.random.key(0),
                                    t_array = t_array,
                                    return_all_matrices = False,
                                    sow_intermediates = False)
        
        log_pred =  my_model.apply(variables = fake_params,
                                    out_dict = my_tkf_params,
                                    r_extend = self.r_mix,
                                    frag_class_probs = self.fragment_class_probs,
                                    method = 'fill_joint_tkf92') #(T, C_dom, C_frag, C_frag, 4, 4)
        
        # check shape
        npt.assert_allclose( log_pred.shape, (T, C_dom, C_frag, C_frag, S, S), rtol=rtol )
        
        
        ### check values
        npt.assert_allclose(jnp.log(true), log_pred, rtol=rtol)
    
    def test_joint_tkf92_with_regular_tkf(self):
        """
        C_dom = 3
        C_frag = 2
        """
        lam = jnp.array([0.3, 0.4, 0.5])
        mu = lam + 0.2
        rtol = 1e-6
        times = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._check_joint_tkf92_calc( lam = lam,
                                      mu = mu,
                                      tkf_function = regular_tkf,
                                      t_array = times,
                                      rtol=rtol)

    def test_joint_tkf92_with_switch_tkf(self):
        lam = jnp.array([0.3, 0.4, 0.5])
        mu = jnp.array([0.30001, 0.40001, 0.7])
        rtol = 1e-4
        times = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._check_joint_tkf92_calc( lam = lam,
                                      mu = mu,
                                      tkf_function = switch_tkf,
                                      t_array = times,
                                      rtol=rtol)
    
    #####################################################
    ### single-sequence marginal: mix of TKF92 models   #
    #####################################################
    def test_marg_tkf92_calc(self):
        C_dom = self.C_dom
        C_frag = self.C_frag
        S = 2
        
        lam = jnp.array([0.3, 0.4, 0.5])
        mu = lam + 0.2
        offset = 1 - lam/mu
        
        ### get true values
        true = np.zeros( (C_dom, C_frag, C_frag, S, S) )
        
        for c_dom in range(C_dom):
            for c in range(C_frag):                
                r_c = self.r_mix[c_dom, c].item()
                
                for d in range(C_frag):
                    prob_d = self.fragment_class_probs[c_dom, d].item()
                    
                    out = true_marg_tkf92_scaled (lam = lam[c_dom], 
                                            mu = mu[c_dom], 
                                            r_c = r_c, 
                                            c = c, 
                                            d = d, 
                                            prob_d = prob_d) #(S, S)
                    true[c_dom,c, d, :, :] = out
                
        # check shape
        npt.assert_allclose( true.shape, (C_dom, C_frag, C_frag, S, S) )
                
        
        ### by my function
        log_pred = get_tkf92_single_seq_marginal_transition_logprobs(offset = offset,
                                                        frag_class_probs = self.fragment_class_probs,
                                                        r_ext_prob = self.r_mix ) #(C_dom, C_frag, C_frag, 2, 2)
        
        # check shape
        npt.assert_allclose( log_pred.shape, (C_dom, C_frag, C_frag, S, S) )
        
        
        ### check values
        true = np.reshape(true, log_pred.shape)
        npt.assert_allclose( jnp.log(true), log_pred, rtol=1e-6)
    
    
    ###################
    ### conditional   #
    ###################
    def _check_cond_tkf92_calc(self,
                                lam,
                                mu,
                                tkf_function,
                                t_array,
                                rtol):
        ### True
        T = t_array.shape[0]
        C_dom = self.C_dom
        C_frag = self.C_frag
        S = 4
        
        # check shapes
        assert lam.shape == (C_dom,)
        assert mu.shape == (C_dom,)
        
        # get offset
        offset = 1 - lam/mu
        
        ### get true values
        true = np.zeros( (T, C_dom, C_frag, C_frag, S, S) )
        for c_dom in range(C_dom):
            for c in range(C_frag):                
                r_c = self.r_mix[c_dom, c].item()
                
                for d in range(C_frag):
                    prob_d = self.fragment_class_probs[c_dom, d].item()
                    
                    for t_idx, t in enumerate(t_array):
                        out = true_cond_tkf92_scaled (lam = lam[c_dom], 
                                                        mu = mu[c_dom], 
                                                        r_c = r_c, 
                                                        t = t, 
                                                        c = c, 
                                                        d = d, 
                                                        prob_d = prob_d) #(S, S)
                        true[t_idx, c_dom, c, d, :, :] = out
        
        # check shape
        npt.assert_allclose( true.shape, (T, C_dom, C_frag, C_frag, S, S) )
        
        
        ### by my function (in a flax module)
        my_tkf_params, _ = tkf_function(mu = mu, 
                                        offset = offset,
                                        t_array = t_array)
        my_tkf_params['log_offset'] = jnp.log(offset)
        my_tkf_params['log_one_minus_offset'] = jnp.log1p(-offset)
        
        # init with regular_tkf, but don't actually use it
        my_model = TKF92TransitionLogprobs(config={'num_domain_mixtures': C_dom,
                                                    'num_fragment_mixtures': C_frag,
                                                    'num_site_mixtures': 1,
                                                    'k_rate_mults': 1,
                                                    'tkf_function': 'regular_tkf'}, 
                                            name='tkf92')
        fake_params = my_model.init(rngs=jax.random.key(0),
                                    t_array = t_array,
                                    return_all_matrices = False,
                                    sow_intermediates = False)
        
        log_joint_tkf92 =  my_model.apply(variables = fake_params,
                                    out_dict = my_tkf_params,
                                    r_extend = self.r_mix,
                                    frag_class_probs = self.fragment_class_probs,
                                    method = 'fill_joint_tkf92') #(T, C_dom, C_frag, C_frag, 4, 4)
        
        log_marg_tkf92 = get_tkf92_single_seq_marginal_transition_logprobs(offset = offset,
                                                        frag_class_probs = self.fragment_class_probs,
                                                        r_ext_prob = self.r_mix ) #(C_dom, C_frag, C_frag, 2, 2)
        
        log_cond_tkf92 = get_cond_transition_logprobs( log_marg_tkf92, log_joint_tkf92 ) #(T, C_dom, C_frag, C_frag, 4, 4)
        
        # check shape
        npt.assert_allclose( log_cond_tkf92.shape, (T, C_dom, C_frag, C_frag, S, S) )
        
        
        ### check values
        true = np.reshape(true, log_cond_tkf92.shape)
        npt.assert_allclose(jnp.log(true), log_cond_tkf92, rtol=rtol)
    
    def test_cond_tkf92_with_regular_tkf(self):
        lam = jnp.array([0.3, 0.4, 0.5])
        mu = lam + 0.2
        rtol = 1e-6
        times = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._check_cond_tkf92_calc( lam = lam,
                                      mu = mu,
                                      tkf_function = regular_tkf,
                                      t_array = times,
                                      rtol=rtol)
        
    def test_cond_tkf92_with_switch_tkf(self):
        lam = jnp.array([0.3, 0.4, 0.5])
        mu = jnp.array([0.30001, 0.40001, 0.7])
        rtol = 5e-3
        times = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._check_cond_tkf92_calc( lam = lam,
                                      mu = mu,
                                      tkf_function = switch_tkf,
                                      t_array = times,
                                      rtol=rtol)
    

if __name__ == '__main__':
    unittest.main()