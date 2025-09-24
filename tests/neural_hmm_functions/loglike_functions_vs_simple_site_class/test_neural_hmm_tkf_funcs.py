#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 11:48:02 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
import numpy as np

import numpy.testing as npt
import unittest

from models.latent_class_mixtures.model_functions import regular_tkf as pairhmm_regular_tkf
from models.latent_class_mixtures.model_functions import approx_tkf as pairhmm_approx_tkf

from models.neural_hmm_predict.model_functions import regular_tkf as neural_regular_tkf
from models.neural_hmm_predict.model_functions import approx_tkf as neural_approx_tkf

THRESHOLD = 1e-6


class TestTKFFuncs(unittest.TestCase):
    def setUp(self):
        self.t_array = jnp.array([0.1, 0.2, 0.3])
    
    def _single_model_test(self,
                           neural_fn, 
                           ref_fn,
                           mu,
                           offset):
        true_out,_ = ref_fn( mu = mu,
                             offset = offset,
                             t_array = self.t_array )
        
        pred_out,_ = neural_fn( mu = mu[None,None], 
                                offset = offset[None,None], 
                                t_array = self.t_array,
                                unique_time_per_sample = False )
        
        for key in pred_out.keys():
            pred = pred_out[key][:,0,0]
            npt.assert_allclose(pred, true_out[key])
    
    def test_one_regular_tkf(self):
        mu = jnp.array(0.06)
        offset = jnp.array(0.01)
        self._single_model_test( neural_fn = neural_regular_tkf,
                                 ref_fn = pairhmm_regular_tkf,
                                 mu=mu,
                                 offset=offset )
    
    def test_one_approx_tkf(self):
        mu = jnp.array(0.06)
        offset = jnp.array(0.01)
        self._single_model_test( neural_fn = neural_approx_tkf,
                                 ref_fn = pairhmm_approx_tkf,
                                 mu=mu,
                                 offset=offset )
    
    def _multi_model_test(self,
                          neural_fn, 
                          ref_fn,
                          mu,
                          offset):
        B = mu.shape[0]
        L = mu.shape[1]
        
        pred_out,_ = neural_fn( mu = mu, 
                                offset = offset, 
                                t_array = self.t_array,
                                unique_time_per_sample = False ) #(T, B, L)
        
        for b in range(B):
            for l in range(L):
                true_out,_ = ref_fn( mu = mu[b,l],
                                      offset = offset[b,l],
                                      t_array = self.t_array )
        
                for key in pred_out.keys():
                    pred = pred_out[key][:,b,l]
                    npt.assert_allclose(pred, true_out[key])
    
    def test_multi_regular_tkf(self):
        mu = jnp.array([[0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6]])
        offset = jnp.array([[0.01, 0.02, 0.03],
                            [0.04, 0.05, 0.06]])
        self._multi_model_test( neural_fn = neural_regular_tkf,
                                 ref_fn = pairhmm_regular_tkf,
                                 mu=mu,
                                 offset=offset )
    
    def test_multi_approx_tkf(self):
        mu = jnp.array([[0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6]])
        offset = jnp.array([[0.01, 0.02, 0.03],
                            [0.04, 0.05, 0.06]])
        self._multi_model_test( neural_fn = neural_approx_tkf,
                                  ref_fn = pairhmm_approx_tkf,
                                  mu=mu,
                                  offset=offset )

if __name__ == '__main__':
    unittest.main()