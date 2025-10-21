#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 11:37:13 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
from flax import linen as nn
import numpy as np

import numpy.testing as npt
import unittest
from models.latent_class_mixtures.one_dim_fwd_bkwd_helpers import (flip_backward_outputs_with_time_grid,
                                                                   flip_backward_outputs_with_len_per_samp)

class TestFlipBackwardOutputs(unittest.TestCase):
    def _add_time_dim(self, mat):
        mat = mat[:, None, :, :] # (L, 1, C, B) 
        new_time_dim1 = mat * 2 # (L, 1, C, B) 
        new_time_dim2 = mat * -1 # (L, 1, C, B) 
        new_time_dim3 = mat * -2 # (L, 1, C, B) 
        expanded_mat = jnp.concat([mat, 
                                   new_time_dim1,
                                   new_time_dim2,
                                   new_time_dim3],
                                  axis=1) #(L, T, C, B)
        return expanded_mat
    
    def setUp(self):
        ### example inputs
        # len = 4
        samp1 = jnp.array( [[1, 10, 100],
                            [2, 20, 200],
                            [3, 30, 300],
                            [4, 40, 400],
                            [0,  0,   0]] ) #(L, C)

        # len = 3
        samp2 = jnp.array( [[5, 50, 500],
                            [6, 60, 600],
                            [7, 70, 700],
                            [0,  0,   0],
                            [0,  0,   0]] ) #(L, C)
        
        inputs = jnp.stack([samp1, samp2], axis=0) #(B, L, C)
        bkw_out = jnp.stack([samp1[1:,:], samp2[1:,:]], axis=-1) #(L-1, C, B)
        # bkw_out = self._add_time_dim(bkw_out) #(L, T, C, B)
        del samp1, samp2
        
        
        ### expected results
        # len = 4
        samp1 = jnp.array( [[4, 40, 400],
                            [3, 30, 300],
                            [2, 20, 200],
                            [0,  0,   0]] ) #(L-1, C)

        # len = 3
        samp2 = jnp.array( [[7, 70, 700],
                            [6, 60, 600],
                            [0,  0,   0],
                            [0,  0,   0]] ) #(L-1, C)
        
        true = jnp.stack([samp1, samp2], axis=-1) #(L-1, C, B)
        # true = self._add_time_dim(true) #(L, T, C, B)
        del samp1, samp2
        
        self.inputs = inputs #(B, L, C)
        self.bkw_out = bkw_out #(L-1, C, B)
        self.true = true #(L-1, C, B)
        
    
    def test_with_time_grid(self):
        flip_outputs = flip_backward_outputs_with_time_grid
        
        inputs = self.inputs #(B, L, C)
        bkw_out = self._add_time_dim(self.bkw_out) #(L-1, T, C, B)
        true = self._add_time_dim(self.true) #(L-1, T, C, B)
        assert inputs.shape[1]-1 == true.shape[0]
        assert inputs.shape[1]-1 == bkw_out.shape[0]
        
        # compare pred vs true
        pred = flip_outputs(inputs, bkw_out)  #(L-1, T, C, B)
        assert inputs.shape[1]-1 == pred.shape[0]
        
        npt.assert_allclose( pred, true )
    
    
    def test_with_len_per_samp(self):
        flip_outputs = flip_backward_outputs_with_len_per_samp
        
        inputs = self.inputs #(B, L, C)
        bkw_out = self.bkw_out #(L-1, C, B)
        true = self.true #(L-1, C, B)
        assert inputs.shape[1]-1 == true.shape[0]
        assert inputs.shape[1]-1 == bkw_out.shape[0]
        
        # compare pred vs true
        pred = flip_outputs(inputs, bkw_out)  #(L-1, C, B)
        assert inputs.shape[1]-1 == pred.shape[0]
        
        npt.assert_allclose( pred, true )



if __name__ == '__main__':
    unittest.main()  
            
