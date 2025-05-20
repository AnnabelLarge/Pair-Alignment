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

THRESHOLD = 1e-6

### function from Ian
def TKF_coeffs (lam, mu, t):
    alpha = jnp.exp(-mu*t)
    beta = (lam*(jnp.exp(-lam*t)-jnp.exp(-mu*t))) / (mu*jnp.exp(-lam*t)-lam*jnp.exp(-mu*t))
    gamma = 1 - ( (mu * beta) / ( lam * (1-alpha) ) )
    return alpha, beta, gamma


class TestOriginalTKFParamFns(unittest.TestCase):
    """
    INDEL PROCESS SCORING TEST 1
    
    
    About
    ------
    make sure my method 'tkf_params' returns the same thing as Ian's 
        original function
    
    """
    def test_tkf_param_no_approx(self):
        # fake params
        lam = jnp.array(0.3)
        mu = jnp.array(0.5)
        t_array = jnp.array([0.3, 0.5, 0.9])
        
        # my function comes packaged in a flax module
        config = {'tkf_err': 1e-4}
        my_model = TKF91TransitionLogprobs(config=config, name='tkf91')
        fake_params = my_model.init(rngs=jax.random.key(0),
                                    t_array = t_array,
                                    sow_intermediates = False)
        
        my_tkf_params = my_model.apply(variables = fake_params,
                                       lam = lam,
                                       mu = mu,
                                       t_array = t_array,
                                       method = 'tkf_params')
        
        pred_alpha = jnp.exp(my_tkf_params['log_alpha'])
        pred_beta = jnp.exp(my_tkf_params['log_beta'])
        pred_gamma = jnp.exp(my_tkf_params['log_gamma'])
        
        # get true values
        true_alpha = []
        true_beta = []
        true_gamma = []

        for t in t_array:
            out = TKF_coeffs (lam, mu, t)
            true_alpha.append(out[0])
            true_beta.append(out[1])
            true_gamma.append(out[2])
        
        true_alpha = np.array(true_alpha)
        true_beta = np.array(true_beta)
        true_gamma = np.array(true_gamma)
        
        npt.assert_allclose(true_alpha, pred_alpha, atol=THRESHOLD)
        npt.assert_allclose(true_beta, pred_beta, atol=THRESHOLD)
        npt.assert_allclose(true_gamma, pred_gamma, atol=THRESHOLD)

if __name__ == '__main__':
    unittest.main()

