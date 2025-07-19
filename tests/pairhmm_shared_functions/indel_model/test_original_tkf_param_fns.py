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

from models.simple_site_class_predict.model_functions import (switch_tkf,
                                                              regular_tkf,
                                                              approx_tkf)


THRESHOLD = 1e-6

### function from Ian
def TKF_coeffs (lam, mu, t):
    alpha = jnp.exp(-mu*t)
    beta = (lam*(jnp.exp(-lam*t)-jnp.exp(-mu*t))) / (mu*jnp.exp(-lam*t)-lam*jnp.exp(-mu*t))
    gamma = 1 - ( (mu * beta) / ( lam * (1-alpha) ) )
    return alpha, beta, gamma


class TestOriginalTKFParamFns(unittest.TestCase):
    """
    About
    ------
    make sure my method 'tkf_params' returns the same thing as Ian's 
        original function
    
    """
    def setUp(self):
        self.lam = jnp.array(0.3)
        self.mu = jnp.array(0.5)
        self.offset = 1 - (self.lam/self.mu)
        
    def _run_test(self, 
                  t_array,
                  tkf_function):
        # get true values
        true_alpha = []
        true_beta = []
        true_gamma = []

        for t in t_array:
            out = TKF_coeffs (self.lam, self.mu, t)
            true_alpha.append(out[0])
            true_beta.append(out[1])
            true_gamma.append(out[2])
        
        true_alpha = np.array(true_alpha)
        true_beta = np.array(true_beta)
        true_gamma = np.array(true_gamma)
        
        # my function comes packaged in a flax module
        my_tkf_params, _ = tkf_function(mu = self.mu, 
                                        offset = self.offset,
                                        t_array = t_array)
        
        pred_alpha = jnp.exp(my_tkf_params['log_alpha'])
        pred_beta = jnp.exp(my_tkf_params['log_beta'])
        pred_gamma = jnp.exp(my_tkf_params['log_gamma'])
        
        npt.assert_allclose(true_alpha, pred_alpha, atol=THRESHOLD)
        npt.assert_allclose(true_beta, pred_beta, atol=THRESHOLD)
        npt.assert_allclose(true_gamma, pred_gamma, atol=THRESHOLD)
        
    def test_switch_tkf(self):
        self._run_test( tkf_function = switch_tkf,
                        t_array = jnp.array([0.3, 0.5, 0.9,
                                             0.0003, 0.0005, 0.0009]) )
    
    def test_regular_tkf(self):
        self._run_test( tkf_function = regular_tkf,
                        t_array = jnp.array([0.3, 0.5, 0.9,
                                             0.0003, 0.0005, 0.0009]) )
    
    def test_approx_tkf(self):
        """
        run this at small times only
        """
        self._run_test( tkf_function = approx_tkf,
                        t_array = jnp.array([0.0003, 0.0005, 0.0009]) )
    
    

if __name__ == '__main__':
    unittest.main()

