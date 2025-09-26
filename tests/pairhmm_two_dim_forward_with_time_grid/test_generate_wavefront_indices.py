#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 20:26:36 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
import numpy as np

import numpy.testing as npt
import unittest

from models.latent_class_mixtures.forward_algo_helpers import generate_ij_coords_at_diagonal_k


class TestWavefrontIdxes(unittest.TestCase):
    def test_this(self):
        ####################
        ### true indices   #
        ####################
        def generate_mask(idxes):
            K = idxes.shape[0]
            W = idxes.shape[1]
            out = np.ones((K,W),dtype=bool)
            
            for k in range(K):
                for w in range(W):
                    i = idxes[k,w,0]
                    j = idxes[k,w,1]
                    
                    if (i==0 and j==0):
                        out[k,w] = False
            
            # first (0,0) should not be masked
            out[0,0] = True
            
            # make sure masked idxes is the same as the original input
            npt.assert_allclose(idxes, jnp.multiply(idxes, out[...,None]))
            
            return out
        
        # alignment grid 1
        # np.array( [ [[0,0], [0,1], [0,2], [0,3]],
        #             [[1,0], [1,1], [1,2], [1,3]],
        #             [[2,0], [2,1], [2,2], [2,3]],
        #             [[3,0], [3,1], [3,2], [3,3]],
        #             [[4,0], [4,1], [4,2], [4,3]]
        #            ] )
        true_idx1 = np.array( [ [[0,0], [0,0], [0,0], [0,0]],
                                  [[1,0], [0,1], [0,0], [0,0]],
                                  [[2,0], [1,1], [0,2], [0,0]],
                                  [[3,0], [2,1], [1,2], [0,3]],
                                  [[4,0], [3,1], [2,2], [1,3]],
                                  [[4,1], [3,2], [2,3], [0,0]],
                                  [[4,2], [3,3], [0,0], [0,0]],
                                  [[4,3], [0,0], [0,0], [0,0]]
                                  ] )
        true_mask1 = generate_mask(true_idx1)
        seqlens1 = np.array( [4, 3] )
        
        # alignment grid 2
        # np.array( [ [[0,0], [0,1], [0,2], [0,0]],
        #             [[1,0], [1,1], [1,2], [0,0]],
        #             [[2,0], [2,1], [2,2], [0,0]],
        #             [[0,0], [0,0], [0,0], [0,0]],
        #             [[0,0], [0,0], [0,0], [0,0]]
        #             ] )
        true_idx2 = np.array( [ [[0,0], [0,0], [0,0], [0,0]],
                                  [[1,0], [0,1], [0,0], [0,0]],
                                  [[2,0], [1,1], [0,2], [0,0]],
                                  [[2,1], [1,2], [0,0], [0,0]],
                                  [[2,2], [0,0], [0,0], [0,0]],
                                  [[0,0], [0,0], [0,0], [0,0]],
                                  [[0,0], [0,0], [0,0], [0,0]],
                                  [[0,0], [0,0], [0,0], [0,0]],
                                  ] )
        true_mask2 = generate_mask(true_idx2)
        seqlens2 = np.array( [2, 2] )
        
        # alignment grid 3
        # np.array( [ [[0,0], [0,1], [0,2], [0,0]],
        #             [[1,0], [1,1], [1,2], [0,0]],
        #             [[2,0], [2,1], [2,2], [0,0]],
        #             [[3,0], [3,1], [3,2], [0,0]],
        #             [[4,0], [4,1], [4,2], [0,0]]
        #             ] )
        true_idx3 = np.array( [ [[0,0], [0,0], [0,0], [0,0]],
                                  [[1,0], [0,1], [0,0], [0,0]],
                                  [[2,0], [1,1], [0,2], [0,0]],
                                  [[3,0], [2,1], [1,2], [0,0]],
                                  [[4,0], [3,1], [2,2], [0,0]],
                                  [[4,1], [3,2], [0,0], [0,0]],
                                  [[4,2], [0,0], [0,0], [0,0]],
                                  [[0,0], [0,0], [0,0], [0,0]]
                                  ] )
        true_mask3 = generate_mask(true_idx3)
        seqlens3 = np.array( [4, 2] )
        
        # concat all
        true_idx = np.stack( [true_idx1, true_idx2, true_idx3] ) #(B, K, W, 2)
        del true_idx1, true_idx2, true_idx3
        
        true_mask = np.stack( [true_mask1, true_mask2, true_mask3] ) #(B, K, W)
        del true_mask1, true_mask2, true_mask3
        
        seqlens = np.stack([seqlens1, seqlens2, seqlens3]) #(B, 2)
        del seqlens1, seqlens2, seqlens3
        
        # dims
        B = true_idx.shape[0]
        K = true_idx.shape[1]
        W = true_idx.shape[2]
        
        
        ######################
        ### by my function   #
        ######################
        for k in range(K):
            # pred_idx is (B, W, 2)
            # pred_mask is (B, W)
            pred_idx_at_k, pred_mask_at_k = generate_ij_coords_at_diagonal_k(seq_lens = seqlens,
                                                                            diagonal_k = k,
                                                                            widest_diag_W = W)
            
            true_idx_at_k = true_idx[:,k,:,:] #(B, W, 2)
            true_mask_at_k = true_mask[:,k] #(B, W)
            
            npt.assert_allclose( pred_idx_at_k, true_idx_at_k ), k
            npt.assert_allclose( pred_mask_at_k, true_mask_at_k ), k

if __name__ == '__main__':
    unittest.main()