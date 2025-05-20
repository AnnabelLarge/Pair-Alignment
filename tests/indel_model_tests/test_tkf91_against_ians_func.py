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
from models.simple_site_class_predict.model_functions import (MargTKF91TransitionLogprobs,
                                                              CondTransitionLogprobs)

THRESHOLD = 1e-6

### function from Ian
def TKF_coeffs (lam, mu, t):
    alpha = jnp.exp(-mu*t)
    beta = (lam*(jnp.exp(-lam*t)-jnp.exp(-mu*t))) / (mu*jnp.exp(-lam*t)-lam*jnp.exp(-mu*t))
    gamma = 1 - ( (mu * beta) / ( lam * (1-alpha) ) )
    return alpha, beta, gamma

def TKF91_Ftransitions (lam, mu, t):
    alpha, beta, gamma = TKF_coeffs (lam, mu, t)
    if gamma < 0:
        print(f'Gamma is negative: {gamma}')
    return jnp.array ([[(1-beta)*alpha, beta, (1-beta)*(1-alpha)],
                       [(1-beta)*alpha, beta, (1-beta)*(1-alpha)],
                       [(1-gamma)*alpha, gamma, (1-gamma)*(1-alpha)]])


class TestTKF91AgainstIansFunc(unittest.TestCase):
    """
    INDEL PROCESS SCORING TEST 2
    
    
    About
    ------
    make sure my conditional tkf91 matches ian's implementation
    
    """
    def test_tkf91_cond(self):
        # fake params
        lam = jnp.array(0.3)
        mu = jnp.array(0.5)
        t_array = jnp.array([0.3, 0.5, 0.9])
        
        
        ### my function comes packaged in a flax module
        config = {'tkf_err': 1e-4}
        my_model = TKF91TransitionLogprobs(config=config, name='tkf91')
        fake_params = my_model.init(rngs=jax.random.key(0),
                                    t_array = t_array,
                                    sow_intermediates = False)
        
        out_dict = my_model.apply(variables = fake_params,
                                       lam = lam,
                                       mu = mu,
                                       t_array = t_array,
                                       method = 'tkf_params')
        
        joint_tkf91 =  my_model.apply(variables = fake_params,
                                      out_dict = out_dict,
                                      method = 'fill_joint_tkf91') #(T, 4, 4)
        
        marg_tkf91 = MargTKF91TransitionLogprobs(lam, mu)
        cond_tkf91 = CondTransitionLogprobs( marg_tkf91, joint_tkf91 )
        
        # don't include sentinel tokens 
        pred = cond_tkf91[:, :3, :3] #(T, 3, 3)
        
        
        ### get true values
        true_tkf91 = []

        for i,t in enumerate(t_array):
            true_tkf91.append( TKF91_Ftransitions (lam, mu, t) )
        
        true_tkf91 = jnp.stack(true_tkf91)
        npt.assert_allclose(true_tkf91, jnp.exp(pred), atol=THRESHOLD)

if __name__ == '__main__':
    unittest.main()

