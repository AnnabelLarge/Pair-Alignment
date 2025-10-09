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

from models.latent_class_mixtures.two_dim_forward_algo_helpers import ij_coords_to_wavefront_pos_at_diagonal_k
                                                                             

class TestRecoverWavefrontPos(unittest.TestCase):
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
        align_grid1 = np.array( [ [[0,0], [0,0], [0,0], [0,0]],
                                  [[1,0], [0,1], [0,0], [0,0]],
                                  [[2,0], [1,1], [0,2], [0,0]],
                                  [[3,0], [2,1], [1,2], [0,3]],
                                  [[4,0], [3,1], [2,2], [1,3]],
                                  [[4,1], [3,2], [2,3], [0,0]],
                                  [[4,2], [3,3], [0,0], [0,0]],
                                  [[4,3], [0,0], [0,0], [0,0]]
                                  ] )
        mask1 = generate_mask(align_grid1)
        seqlens1 = np.array( [4, 3] )
        
        # alignment grid 2
        # np.array( [ [[0,0], [0,1], [0,2], [0,0]],
        #             [[1,0], [1,1], [1,2], [0,0]],
        #             [[2,0], [2,1], [2,2], [0,0]],
        #             [[0,0], [0,0], [0,0], [0,0]],
        #             [[0,0], [0,0], [0,0], [0,0]]
        #             ] )
        align_grid2 = np.array( [ [[0,0], [0,0], [0,0], [0,0]],
                                  [[1,0], [0,1], [0,0], [0,0]],
                                  [[2,0], [1,1], [0,2], [0,0]],
                                  [[2,1], [1,2], [0,0], [0,0]],
                                  [[2,2], [0,0], [0,0], [0,0]],
                                  [[0,0], [0,0], [0,0], [0,0]],
                                  [[0,0], [0,0], [0,0], [0,0]],
                                  [[0,0], [0,0], [0,0], [0,0]],
                                  ] )
        mask2 = generate_mask(align_grid2)
        seqlens2 = np.array( [2, 2] )
        
        # alignment grid 3
        # np.array( [ [[0,0], [0,1], [0,2], [0,0]],
        #             [[1,0], [1,1], [1,2], [0,0]],
        #             [[2,0], [2,1], [2,2], [0,0]],
        #             [[3,0], [3,1], [3,2], [0,0]],
        #             [[4,0], [4,1], [4,2], [0,0]]
        #             ] )
        align_grid3 = np.array( [ [[0,0], [0,0], [0,0], [0,0]],
                                  [[1,0], [0,1], [0,0], [0,0]],
                                  [[2,0], [1,1], [0,2], [0,0]],
                                  [[3,0], [2,1], [1,2], [0,0]],
                                  [[4,0], [3,1], [2,2], [0,0]],
                                  [[4,1], [3,2], [0,0], [0,0]],
                                  [[4,2], [0,0], [0,0], [0,0]],
                                  [[0,0], [0,0], [0,0], [0,0]]
                                  ] )
        mask3 = generate_mask(align_grid3)
        seqlens3 = np.array( [4, 2] )
        
        # concat all
        align_grid = np.stack( [align_grid1, align_grid2, align_grid3] ) #(B, K, W, 2)
        del align_grid1, align_grid2, align_grid3
        
        mask = np.stack( [mask1, mask2, mask3] ) #(B, K, W)
        del mask1, mask2, mask3
        
        seqlens = np.stack([seqlens1, seqlens2, seqlens3]) #(B, 2)
        del seqlens1, seqlens2, seqlens3
        
        # dims
        B = align_grid.shape[0]
        K = align_grid.shape[1]
        W = align_grid.shape[2]
        
        
        ######################
        ### by my function   #
        ######################
        for k in range(K):
            pred_pos = ij_coords_to_wavefront_pos_at_diagonal_k(indices = align_grid[:,k,:,:],
                                                                anc_len = seqlens[:,0]) #(B, W)
            
            for b in range(B):
                for w in range(W):
                    if mask[b,k,w]:
                        pred_pair = align_grid[b, k, pred_pos[b, w], :] #(2,)
                        true_pair = align_grid[b, k, w, :] #(2,)
                        npt.assert_allclose(true_pair, pred_pair), f'b={b}, k={k}, w={w}'
                    

if __name__ == '__main__':
    unittest.main()