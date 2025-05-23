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
from models.simple_site_class_predict.model_functions import (MargTKF92TransitionLogprobs,
                                                              CondTransitionLogprobs)

THRESHOLD = 1e-6

### function from Ian; this is based on internal document
def TKF_coeffs (lam, mu, t):
    alpha = jnp.exp(-mu*t)
    beta = (lam*(jnp.exp(-lam*t)-jnp.exp(-mu*t))) / (mu*jnp.exp(-lam*t)-lam*jnp.exp(-mu*t))
    gamma = 1 - ( (mu * beta) / ( lam * (1-alpha) ) )
    return alpha, beta, gamma

def TKF92_Ftransitions (lam, mu, r, t):
    alpha, beta, gamma = TKF_coeffs (lam, mu, t)
    kappa = lam / mu
    nu = r + (1-r) * (lam / mu)
    
    # M -> any
    m_to_m = (1/nu) * (r + ( (1-r) * (1-beta) * kappa * alpha ) )
    m_to_i = (1-r) * beta
    m_to_d = (1/nu) * (1-r) * (1-beta) * kappa * (1-alpha)
    m_to_e = 1 - beta
    match_row = jnp.array([m_to_m, m_to_i, m_to_d, m_to_e])
    
    # I -> any
    i_to_m = (1/nu) * (1-r) * (1-beta) * kappa * alpha
    i_to_i = r + ( (1-r) * beta )
    i_to_d = (1/nu) * (1-r) * (1-beta) * kappa * (1-alpha)
    i_to_e = 1 - beta
    ins_row = jnp.array([i_to_m, i_to_i, i_to_d, i_to_e])
    
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

out = TKF92_Ftransitions (lam = 0.1, mu = 0.2, r=0.5, t=1)    


class TestTKF92AgainstIansFunc(unittest.TestCase):
    """
    INDEL PROCESS SCORING TEST 3
    
    
    About
    ------
    make sure my conditional tkf92 matches ian's implementation
    this tests the conditional formula, from internal documents
    
    """
    def test_tkf92_cond(self):
        # fake params
        lam = jnp.array([0.3])
        mu = jnp.array([0.5])
        r = jnp.array([0.1])
        t_array = jnp.array([0.3, 0.5, 0.9])
        
        ### my function comes packaged in a flax module
        config = {'tkf_err': 1e-4,
                  'num_tkf_site_classes': 1}
        my_model = TKF92TransitionLogprobs(config=config, name='tkf92')
        fake_params = my_model.init(rngs=jax.random.key(0),
                                    t_array = t_array,
                                    class_probs = jnp.array([1]),
                                    sow_intermediates = False)
        
        out_dict = my_model.apply(variables = fake_params,
                                       lam = jnp.squeeze(lam),
                                       mu = jnp.squeeze(mu),
                                       t_array = t_array,
                                       method = 'tkf_params')
        
        joint_tkf92 =  my_model.apply(variables = fake_params,
                                      out_dict = out_dict,
                                      r_extend = r,
                                      class_probs = jnp.array([1]),
                                      method = 'fill_joint_tkf92') #(T, 1, 1, 4, 4)
        
        
        marg_tkf92 = MargTKF92TransitionLogprobs(lam = lam, 
                                                 mu = mu,
                                                 class_probs = jnp.array([1]),
                                                 r_ext_prob = r)
        pred_cond_tkf92 = CondTransitionLogprobs( marg_tkf92, joint_tkf92 )
        
        
        ### get true values
        true_tkf92 = []

        for i,t in enumerate(t_array):
            true_tkf92.append( TKF92_Ftransitions (lam, mu, r, t) ) #(T, 4, 4, 1)
        
        
        ### reshaping to (T, 4, 4)
        true_tkf92 = jnp.stack(true_tkf92)[...,0]
        pred_cond_tkf92 = pred_cond_tkf92[:,0,0,...]
        
        
        npt.assert_allclose(true_tkf92, jnp.exp(pred_cond_tkf92), atol=THRESHOLD)

if __name__ == '__main__':
    unittest.main()

