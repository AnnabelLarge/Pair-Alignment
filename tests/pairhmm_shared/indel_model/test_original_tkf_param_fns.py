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

from models.latent_class_mixtures.model_functions import (switch_tkf,
                                                              regular_tkf)


### function from Ian (in numpy, so always float64)
def TKF_coeffs (lam, mu, t):
    alpha = np.exp(-mu*t)
    beta = (lam*(np.exp(-lam*t)-np.exp(-mu*t))) / (mu*np.exp(-lam*t)-lam*np.exp(-mu*t))
    gamma = 1 - ( (mu * beta) / ( lam * (1-alpha) ) )
    return alpha, beta, gamma


class TestOriginalTKFParamFns(unittest.TestCase):
    """
    About
    ------
    make sure my method 'tkf_params' returns the same thing as Ian's 
        original function
    
    """
    def _run_test(self, 
                  lam,
                  mu,
                  t_array,
                  tkf_function,
                  rtol):
        C = lam.shape[0]
        T = t_array.shape[0]
        offset = 1 - lam/mu
        
        ### get true values
        true_log_alpha = np.zeros( (T,C) )
        true_log_beta = np.zeros( (T,C) )
        true_log_gamma = np.zeros( (T,C) )

        for c in range(C):
            for i,t in enumerate( t_array ):
                a,b,g = TKF_coeffs (lam[c], mu[c], t)
                true_log_alpha[i,c] = np.log( a )
                true_log_beta[i,c] = np.log( b )
                true_log_gamma[i,c] = np.log( g )
        
        
        ### my function comes packaged in a flax module
        my_tkf_params, approx_dict = tkf_function(mu = mu, 
                                        offset = offset,
                                        t_array = t_array)
        
        pred_log_alpha = my_tkf_params['log_alpha']
        pred_log_beta = my_tkf_params['log_beta']
        pred_log_gamma = my_tkf_params['log_gamma']
        
        
        # check shape
        assert pred_log_alpha.shape == (T, C)
        assert pred_log_beta.shape == (T, C)
        assert pred_log_gamma.shape == (T, C)
        
        # check values IN LOG SPACE
        npt.assert_allclose(true_log_alpha, pred_log_alpha, rtol=rtol, err_msg='log_alpha')
        npt.assert_allclose(true_log_beta, pred_log_beta, rtol=rtol, err_msg='log_beta')
        npt.assert_allclose(true_log_gamma, pred_log_gamma, rtol=rtol, err_msg='log_gamma')
        
        
    def test_regular_tkf(self):
        lam = jnp.array([0.3, 0.4, 0.5])
        mu = jnp.array([0.5, 0.6, 0.7])
        rtol = 1e-6
        self._run_test( lam = lam,
                        mu = mu,
                        tkf_function = regular_tkf,
                        t_array = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009]),
                        rtol = rtol )

    def test_switch_tkf(self):
        """
        needs a lower rtol, since approx formulas aren't exact
        """
        lam = jnp.array([0.3, 0.3])
        mu = jnp.array([0.5, 0.30001])
        rtol = 1e-4
        self._run_test( lam = lam,
                        mu = mu,
                        tkf_function = switch_tkf,
                        t_array = jnp.array([0.3, 0.5, 0.9, 0.0003, 0.0005, 0.0009]),
                        rtol = rtol )
    

if __name__ == '__main__':
    unittest.main()

