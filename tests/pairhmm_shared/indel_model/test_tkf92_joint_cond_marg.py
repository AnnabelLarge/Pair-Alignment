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
                                                              get_tkf92_single_seq_marginal_transition_logprobs,
                                                              get_cond_transition_logprobs)


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
    ###########
    ### joint #
    ###########
    def _check_joint_tkf92_calc(self,
                                lam,
                                mu,
                                tkf_function,
                                t_array,
                                rtol):
        T = t_array.shape[0]
        C_dom = lam.shape[0]
        C_frag = 1
        S = 4
        offset = 1 - lam/mu
        
        # generate random r_extend for each domain
        r = jax.random.uniform(key = jax.random.key(0), 
                               shape=(C_dom, C_frag), 
                               minval=1e-4, 
                               maxval=0.9999)
        
        ### true values
        true = np.zeros( (T, C_dom, C_frag, C_frag, S, S) )
        for t_idx in range(T):
            for c_dom in range(C_dom):
                for c_fr_from in range(C_frag):
                    for c_fr_to in range(C_frag):
                        t = t_array[t_idx]
                        out = true_joint_tkf92 (lam[c_dom], mu[c_dom], r[c_dom, c_fr_from], t)
                        true[t_idx, c_dom, c_fr_from, c_fr_to, :, :] = out
                        del out
                
        # check shape
        npt.assert_allclose( true.shape, (T, C_dom, C_frag, C_frag, S, S) )
        
        # check rowsums
        rowsums = true.sum(axis=-1)
        npt.assert_allclose( rowsums, np.ones( rowsums.shape ) )
        del rowsums
        
        
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
                                    r_extend = r,
                                    frag_class_probs = jnp.ones( (C_dom, C_frag) ),
                                    method = 'fill_joint_tkf92') #(T, C_dom, 1, 1, 4, 4)
        
        # check shape
        npt.assert_allclose( log_pred.shape, (T, C_dom, C_frag, C_frag, S, S) )
        
        # check row sums
        rowsums = np.exp(log_pred).sum(axis=-1)
        npt.assert_allclose( rowsums, np.ones( rowsums.shape ) )
        
        
        ### check values in log space
        npt.assert_allclose(jnp.log(true), log_pred, rtol=rtol)
    
    def test_joint_tkf92_with_regular_tkf(self):
        lam = jnp.array([0.3])
        mu = jnp.array([0.5])
        rtol = 1e-6
        times = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._check_joint_tkf92_calc( lam = lam,
                                      mu = mu,
                                      tkf_function = regular_tkf,
                                      t_array = times,
                                      rtol=rtol)

    def test_joint_tkf92_with_switch_tkf(self):
        lam = jnp.array([0.3, 0.3])
        mu = jnp.array([0.5, 0.30001])
        rtol = 1e-4
        times = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._check_joint_tkf92_calc( lam = lam,
                                      mu = mu,
                                      tkf_function = switch_tkf,
                                      t_array = times,
                                      rtol=rtol)
        
        
    #################################################
    ### single-sequence marginal: one TKF92 model   #
    #################################################
    def test_marg_tkf92_calc(self):
        lam = jnp.array([0.3])
        mu = jnp.array([0.5])
        r = jnp.array([0.02])[None]
        offset = 1 - lam/mu
        
        C_dom = 1
        C_frag = 1
        S = 2

        ### true values
        true = np.zeros( (C_dom, C_frag, C_frag, S, S) )
        for c_dom in range(C_dom):
            for c_fr_from in range(C_frag):
                for c_fr_to in range(C_frag):
                    out =  true_marg_tkf92 (lam = lam[c_dom], 
                                            mu = mu[c_dom],
                                            r = r[c_dom, c_fr_from].item() )
                    true[c_dom, c_fr_from, c_fr_to, ...] = out
                    
        # check shape
        npt.assert_allclose( true.shape, (C_dom, C_frag, C_frag, S, S) )
        
        # check rowsums
        rowsums = true.sum(axis=-1)
        npt.assert_allclose( rowsums, np.ones( rowsums.shape ) )
        del rowsums
        
        
        ### values by my function
        log_pred = get_tkf92_single_seq_marginal_transition_logprobs(offset = offset,
                                                                      frag_class_probs = jnp.ones( (C_dom, C_frag) ),
                                                                      r_ext_prob = r ) #(1, 1, 1, 2, 2)
        
        # check shape
        npt.assert_allclose( log_pred.shape, (C_dom, C_frag, C_frag, S, S) )
        
        # check rowsums
        rowsums = np.exp(log_pred).sum(axis=-1)
        npt.assert_allclose( rowsums, np.ones( rowsums.shape ) )
        
        
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
        T = t_array.shape[0]
        C_dom = lam.shape[0]
        C_frag = 1
        S = 4
        offset = 1 - lam/mu
        
        # generate random r_extend for each domain
        r = jax.random.uniform(key = jax.random.key(0), 
                               shape=(C_dom, C_frag), 
                               minval=1e-4, 
                               maxval=0.9999)
        
        
        ### true values
        true = np.zeros( (T, C_dom, C_frag, C_frag, S, S) )
        for t_idx in range(T):
            for c_dom in range(C_dom):
                for c_fr_from in range(C_frag):
                    for c_fr_to in range(C_frag):
                        t = t_array[t_idx]
                        out = true_cond_tkf92 (lam[c_dom], mu[c_dom], r[c_dom, c_fr_from], t)
                        true[t_idx, c_dom, c_fr_from, c_fr_to, :, :] = out
                        del out
                        
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
        
        log_joint_tkf92 =  my_model.apply(variables = fake_params,
                                    out_dict = my_tkf_params,
                                    r_extend = r,
                                    frag_class_probs = jnp.ones( (C_dom, C_frag) ),
                                    method = 'fill_joint_tkf92') #(T, C_dom, 1, 1, 4, 4)
        
        log_marg_tkf92 = get_tkf92_single_seq_marginal_transition_logprobs(offset = offset,
                                                                          frag_class_probs = jnp.ones( (C_dom, C_frag) ),
                                                                          r_ext_prob = r ) #(1, 1, 1, 2, 2)
        
        log_cond_tkf92 = get_cond_transition_logprobs( log_marg_tkf92, log_joint_tkf92 ) #(T, 1, 1, 1, 4, 4)
        
        # check shape
        npt.assert_allclose( log_cond_tkf92.shape, (T, C_dom, C_frag, C_frag, S, S) )
        
        
        ### check values in log space
        npt.assert_allclose(jnp.log(true), log_cond_tkf92, rtol=rtol)
    
    def test_cond_tkf92_with_switch_tkf(self):
        lam = jnp.array([0.3])
        mu = jnp.array([0.5])
        rtol = 1e-6
        times = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._check_cond_tkf92_calc( lam = lam,
                                      mu = mu,
                                      tkf_function = regular_tkf,
                                      t_array = times,
                                      rtol=rtol)
        
    def test_cond_tkf92_with_regular_tkf(self):
        lam = jnp.array([0.3, 0.3])
        mu = jnp.array([0.5, 0.30001])
        rtol = 1e-3
        times = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._check_cond_tkf92_calc( lam = lam,
                                      mu = mu,
                                      tkf_function = switch_tkf,
                                      t_array = times,
                                      rtol=rtol)


if __name__ == '__main__':
    unittest.main()