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
from models.latent_class_mixtures.one_dim_fwd_bkwd_helpers import flip_alignments

class TestFlipAlignments(unittest.TestCase):
    def test_this(self):
        ### example inputs
        # len = 4
        all_match = jnp.array( [[1, 1, 4],
                                [3, 4, 1],
                                [4, 5, 1],
                                [2, 2, 5],
                                [0, 0, 0]] ) #(L, 3)

        # len = 3
        all_ins = jnp.array( [[ 1, 1, 4],
                              [43, 4, 2],
                              [ 2, 2, 5],
                              [ 0, 0, 0],
                              [ 0, 0, 0]] ) #(L, 3)

        # len = 5
        all_del = jnp.array( [[1,  1, 4],
                              [4, 43, 3],
                              [5, 43, 3],
                              [6, 43, 3],
                              [2,  2, 5]] ) #(L, 3)

        # len = 5
        mix = jnp.array( [[ 1,  1, 4],
                          [ 3,  4, 1],
                          [43,  4, 2],
                          [ 4, 43, 3],
                          [ 2,  2, 5]] ) #(L, 3)
        
        
        aligned_inputs = jnp.stack([all_match,
                                    all_ins,
                                    all_del,
                                    mix]) #(B, L, 3)
        
        B = aligned_inputs.shape[0]
        L = aligned_inputs.shape[1]
        
        
        ### expected results
        # len = 4
        rev_all_match = jnp.array( [
                                    [2, 2, 5],
                                    [4, 5, 1],
                                    [3, 4, 1],
                                    [1, 1, 4],
                                    [0, 0, 0]
                                    ] ) #(L, 3)

        # len = 3
        rev_all_ins = jnp.array( [
                                  [ 2, 2, 5],
                                  [43, 4, 2],
                                  [ 1, 1, 4],
                                  [ 0, 0, 0],
                                  [ 0, 0, 0]
                                  ] ) #(L, 3)

        # len = 5
        rev_all_del = jnp.array( [
                                  [2,  2, 5],
                                  [6, 43, 3],
                                  [5, 43, 3],
                                  [4, 43, 3],
                                  [1,  1, 4]
                                  ] ) #(L, 3)

        # len = 5
        rev_mix = jnp.array( [
                              [ 2,  2, 5],
                              [ 4, 43, 3],
                              [43,  4, 2],
                              [ 3,  4, 1],
                              [ 1,  1, 4]
                              ] ) #(L, 3)
        
        true = jnp.stack([rev_all_match,
                          rev_all_ins,
                          rev_all_del,
                          rev_mix]) #(B, L, 3)
        
        
        ### compare pred vs true
        pred = flip_alignments(aligned_inputs)  #(B, L, 3)
        npt.assert_allclose( pred, true )

if __name__ == '__main__':
    unittest.main()  
            
