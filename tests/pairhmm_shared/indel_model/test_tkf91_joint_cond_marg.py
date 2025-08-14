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
from models.simple_site_class_predict.model_functions import (regular_tkf,
                                                              get_tkf91_single_seq_marginal_transition_logprobs,
                                                              get_cond_transition_logprobs,
                                                              switch_tkf)


### these are from ian (in numpy, so always float64)
def TKF_coeffs (lam, mu, t):
    alpha = np.exp(-mu*t)
    beta = (lam*(np.exp(-lam*t)-np.exp(-mu*t))) / (mu*np.exp(-lam*t)-lam*np.exp(-mu*t))
    gamma = 1 - ( (mu * beta) / ( lam * (1-alpha) ) )
    return alpha, beta, gamma

### this is from overleaf
def true_conditional_tkf91 (lam, mu, t):
    alpha, beta, gamma = TKF_coeffs (lam, mu, t)
    return np.array ([[(1-beta)*alpha,  beta,  (1-beta)*(1-alpha),  1 - beta],
                       [(1-beta)*alpha,  beta,  (1-beta)*(1-alpha),  1 - beta],
                       [(1-gamma)*alpha, gamma, (1-gamma)*(1-alpha), 1 - gamma],
                       [(1-beta)*alpha,  beta,  (1-beta)*(1-alpha),  1 - beta]])

def true_joint_tkf91 (lam, mu, t):
    alpha, beta, gamma = TKF_coeffs (lam, mu, t)
    p_emit = (lam/mu)
    p_end = 1 - (lam/mu)
    return np.array ([[(1-beta)*alpha*p_emit,  beta,  (1-beta)*(1-alpha)*p_emit,  (1 - beta)*p_end],
                       [(1-beta)*alpha*p_emit,  beta,  (1-beta)*(1-alpha)*p_emit,  (1 - beta)*p_end],
                       [(1-gamma)*alpha*p_emit, gamma, (1-gamma)*(1-alpha)*p_emit, (1 - gamma)*p_end],
                       [(1-beta)*alpha*p_emit,  beta,  (1-beta)*(1-alpha)*p_emit,  (1 - beta)*p_end]])

def true_marg_tkf91 (lam, mu):
    p_emit = (lam/mu)
    p_end = 1 - (lam/mu)
    return np.array( [[p_emit, p_end],
                       [p_emit, p_end]] )



class TestTKF91JointCondMarg(unittest.TestCase):
    """
    About
    ------
    Test joint, conditonal, and single-sequence transition marginals
    
    """
    #############
    ### joint   #
    #############
    def _check_joint_tkf91_calc(self,
                                lam,
                                mu,
                                tkf_function,
                                t_array,
                                rtol):
        T = t_array.shape[0]
        C_dom = lam.shape[0]
        offset = 1 - lam/mu
        
        ### by my function (in a flax module)
        my_tkf_params, _ = tkf_function(mu = mu, 
                                        offset = offset,
                                        t_array = t_array)
        
        my_tkf_params['log_offset'] = jnp.log(offset)
        my_tkf_params['log_one_minus_offset'] = jnp.log1p(-offset)
        
        my_model = TKF91TransitionLogprobs(config={'num_domain_mixtures': C_dom,
                                                    'num_fragment_mixtures': 1,
                                                    'num_site_mixtures': 1,
                                                    'k_rate_mults': 1,
                                                    'tkf_function': 'regular_tkf'}, 
                                            name='tkf91')
        fake_params = my_model.init(rngs=jax.random.key(0),
                                    t_array = t_array,
                                    return_all_matrices = False,
                                    sow_intermediates = False)
        
        log_pred =  my_model.apply(variables = fake_params,
                                   out_dict = my_tkf_params,
                                   method = 'fill_joint_tkf91') #(T, 1, 4, 4)
        
        # check shape
        assert log_pred.shape == (T, C_dom, 4, 4)
        
        
        ### get true values
        true_tkf91 = np.zeros((T,C_dom,4,4))
        for c in range(C_dom):
            for i,t in enumerate(t_array):
                true_tkf91[i,c,...] = true_joint_tkf91 (lam[c], mu[c], t)
                
        true_tkf91 = jnp.reshape(true_tkf91, log_pred.shape)
        
        # check values IN LOG SPACE
        npt.assert_allclose(jnp.log(true_tkf91), log_pred, rtol=rtol)
    
    def test_joint_tkf91_with_regular_tkf(self):
        lam = jnp.array([0.3])
        mu = jnp.array([0.5])
        rtol = 1e-6
        times = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._check_joint_tkf91_calc( lam = lam,
                                      mu = mu,
                                      tkf_function = regular_tkf,
                                      t_array = times,
                                      rtol = rtol )
    
    def test_joint_tkf91_with_switch_tkf(self):
        lam = jnp.array([0.3, 0.3])
        mu = jnp.array([0.5, 0.30001])
        rtol = 1e-4
        times = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._check_joint_tkf91_calc( lam = lam,
                                      mu = mu,
                                      tkf_function = switch_tkf,
                                      t_array = times,
                                      rtol = rtol )
    
    
    ###############################
    ## single-sequence marginal   #
    ###############################
    def test_marg_tkf91_calc(self):
        lam = jnp.array([0.3])
        mu = jnp.array([0.5])
        offset = 1 - lam/mu
        
        ### pred
        log_pred = get_tkf91_single_seq_marginal_transition_logprobs(offset) #(C_dom, 2, 2)
        
        # check shape
        assert log_pred.shape == (1, 2, 2)
        
        ## true
        true = true_marg_tkf91 (lam = lam.item(), 
                                mu = mu.item())
        true = np.reshape(true, log_pred.shape)
        
        
        ### compare values in LOG space
        npt.assert_allclose( np.log(true), log_pred, rtol = 1e-6)
    
    
    ###################
    ### conditional   #
    ###################
    def _check_cond_tkf91_calc(self,
                                lam,
                                mu,
                                tkf_function,
                                t_array,
                                rtol):
        T = t_array.shape[0]
        C_dom = lam.shape[0]
        offset = 1 - lam/mu
        
        ### my function comes packaged in a flax module
        my_tkf_params, approx_dict = tkf_function(mu = mu, 
                                                  offset = offset,
                                                  t_array = t_array)
        my_tkf_params['log_offset'] = jnp.log(offset)
        my_tkf_params['log_one_minus_offset'] = jnp.log1p(-offset)
        
        # uncomment to look at specific tkf parameters, but this was 
        #   technically covered already
        # true_gamma = []
        # for i,t in enumerate(t_array):
        #     a, b, g = TKF_coeffs (lam.item(), mu.item(), t)
        #     true_gamma.append( np.log(g) )
        # true_gamma = np.array(true_gamma)
        
        my_model = TKF91TransitionLogprobs(config={'num_domain_mixtures': C_dom,
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
                                      method = 'fill_joint_tkf91') #(T, 1, 4, 4)
        
        log_marg_tkf91 = get_tkf91_single_seq_marginal_transition_logprobs(offset) #(2, 2)
        
        log_cond_tkf91 = get_cond_transition_logprobs( log_marg_tkf91, log_joint_tkf91 ) #(T, 4, 4)

        # check shapes
        assert log_joint_tkf91.shape == (T, C_dom, 4, 4)
        assert log_marg_tkf91.shape == (C_dom, 2, 2)
        assert log_cond_tkf91.shape == (T, C_dom, 4, 4)
        
        
        ### get true values
        true_tkf91 = np.zeros((T,C_dom,4,4))
        for c in range(C_dom):
            for i,t in enumerate(t_array):
                true_tkf91[i,c,...] = true_conditional_tkf91 (lam[c], mu[c], t)
                
        true_tkf91 = jnp.reshape(true_tkf91, log_cond_tkf91.shape)
        
        # check values IN LOG SPACE
        npt.assert_allclose(jnp.log(true_tkf91), log_cond_tkf91, rtol=rtol)
        
    
    def test_cond_tkf91_with_regular_tkf(self):
        lam = jnp.array([0.3])
        mu = jnp.array([0.5])
        rtol = 1e-6
        times = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._check_cond_tkf91_calc( lam = lam,
                                     mu = mu,
                                     tkf_function = regular_tkf,
                                     t_array = times,
                                     rtol = rtol )

    def test_cond_tkf91_with_switch_tkf(self):
        lam = jnp.array([0.3, 0.3])
        mu = jnp.array([0.5, 0.30001])
        rtol = 1e-3
        times = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009])
        self._check_cond_tkf91_calc( lam = lam,
                                      mu = mu,
                                      tkf_function = switch_tkf,
                                      t_array = times,
                                      rtol = rtol )
    


if __name__ == '__main__':
    unittest.main()

