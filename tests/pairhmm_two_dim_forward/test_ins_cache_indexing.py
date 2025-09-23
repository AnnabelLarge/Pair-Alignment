#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 12:21:08 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
import numpy as np

import numpy.testing as npt
import unittest

from models.simple_site_class_predict.marg_over_alignments_forward_fns import (generate_ij_coords_at_diagonal_k,
                                                                               ij_coords_to_wavefront_pos_at_diagonal_k,
                                                                               index_all_classes_one_state,
                                                                               wavefront_cache_lookup)

def copy_mat(mat):
    mat2 = mat * 10
    mat3 = mat * 100
    mat4 = mat * -1
    mat5 = mat * -10
    mat6 = mat * -100
    return np.stack([mat, mat2, mat3, mat4, mat5, mat6])
    
class TestInsCacheIndexing(unittest.TestCase):
    def test_this(self):
        ### fake alignment grids with values
        # alignment grid 1
        vals1 = np.array( [[ 1,  3,  6, 10],
                           [ 2,  5,  9, 14],
                           [ 4,  8, 13, 17],
                           [ 7, 12, 16, 19],
                           [11, 15, 18, 20]] ) #(L_anc, L_desc)
        vals1 = copy_mat(vals1) #(C_S, L_anc, L_desc)
        
        # diagonalized
        diags1 = np.array([[ 1,  0,  0,  0],
                           [ 2,  3,  0,  0],
                           [ 4,  5,  6,  0],
                           [ 7,  8,  9, 10],
                           [11, 12, 13, 14],
                           [15, 16, 17,  0],
                           [18, 19,  0,  0],
                           [20,  0,  0,  0]]) #(K, W)
        diags1 = copy_mat(diags1) #(C_S, K, W)
        
        # result of (i, j-1)
        true_vals_for_ins1 = np.array([[ 0,  0,  0,  0],
                                       [ 0,  1,  0,  0],
                                       [ 0,  2,  3,  0],
                                       [ 0,  4,  5,  6],
                                       [ 0,  7,  8,  9],
                                       [11, 12, 13,  0],
                                       [15, 16,  0,  0],
                                       [18,  0,  0,  0]]) #(K, W)
        true_vals_for_ins1 = copy_mat(true_vals_for_ins1) #(C_S, K, W)
        
        seqlens1 = np.array( [4, 3] )
        
        # alignment grid 2
        vals2 = np.array( [[22, 24, 27, 0],
                           [23, 26, 29, 0],
                           [25, 28, 30, 0],
                           [ 0,  0,  0, 0],
                           [ 0,  0,  0, 0]] ) #(L_anc, L_desc)
        vals2 = copy_mat(vals2) #(C_S, L_anc, L_desc)
        
        # diagonalized
        diags2 = np.array([[22,  0,  0, 0],
                           [23, 24,  0, 0],
                           [25, 26, 27, 0],
                           [28, 29,  0, 0],
                           [30,  0,  0, 0],
                           [ 0,  0,  0, 0],
                           [ 0,  0,  0, 0],
                           [ 0,  0,  0, 0]]) #(K, W)
        diags2 = copy_mat(diags2) #(C_S, K, W)
        
        # result of (i, j-1)
        true_vals_for_ins2 = np.array([[ 0,  0,  0,  0],
                                       [ 0, 22,  0,  0],
                                       [ 0, 23, 24,  0],
                                       [25, 26,  0,  0],
                                       [28,  0,  0,  0],
                                       [ 0,  0,  0,  0],
                                       [ 0,  0,  0,  0],
                                       [ 0,  0,  0,  0]]) #(K, W)
        true_vals_for_ins2 = copy_mat(true_vals_for_ins2) #(C_S, K, W)
        
        seqlens2 = np.array( [2, 2] )
        
        # alignment grid 3
        vals3 = np.array( [[31, 33, 36, 0],
                           [32, 35, 39, 0],
                           [34, 38, 42, 0],
                           [37, 41, 44, 0],
                           [40, 43, 45, 0]] ) #(L_anc, L_desc)
        vals3 = copy_mat(vals3) #(C_S, L_anc, L_desc)
        
        # diagonalized
        diags3 = np.array([[31,  0,  0,  0],
                           [32, 33,  0,  0],
                           [34, 35, 36,  0],
                           [37, 38, 39,  0],
                           [40, 41, 42,  0],
                           [43, 44,  0,  0],
                           [45,  0,  0,  0],
                           [ 0,  0,  0,  0]]) #(K, W)
        diags3 = copy_mat(diags3) #(C_S, K, W)
        
        # result of (i, j-1)
        true_vals_for_ins3 = np.array([[ 0,  0,  0,  0],
                                       [ 0, 31,  0,  0],
                                       [ 0, 32, 33,  0],
                                       [ 0, 34, 35,  0],
                                       [ 0, 37, 38,  0],
                                       [40, 41,  0,  0],
                                       [43,  0,  0,  0],
                                       [ 0,  0,  0,  0]]) #(K, W)
        true_vals_for_ins3 = copy_mat(true_vals_for_ins3) #(C_S, K, W)
        
        seqlens3 = np.array( [4, 2] )
        
        # concat all
        vals = np.stack([vals1, vals2, vals3]) #(B, C_S, L_anc, L_desc)
        del vals1, vals2, vals3
        
        diags = np.stack([diags1, diags2, diags3]) #(B, C_S, K, W)
        del diags1, diags2, diags3
        
        seq_lens = np.stack([seqlens1, seqlens2, seqlens3]) #(B, 2)
        del seqlens1, seqlens2, seqlens3
        
        true_vals_for_ins = np.stack([true_vals_for_ins1, 
                                      true_vals_for_ins2, 
                                      true_vals_for_ins3]) #(B, C_S, K, W)
        del true_vals_for_ins1, true_vals_for_ins2, true_vals_for_ins3
        
        # dims
        T = 1
        B = diags.shape[0]
        C_S = diags.shape[1]
        K = diags.shape[2]
        W = diags.shape[3]
        L_anc = vals.shape[1]
        L_desc = vals.shape[2]
        
        def reshape_true_answer(mat):
            mat = np.transpose(mat, (2,1,0))
            return mat[:,None,:,:]
        
        for k in range(1,K):
            # diagonals at previous k
            cache_to_read = diags[:, :, k-1, :] #(B, C_S, W)
            cache_to_read = reshape_true_answer(cache_to_read) #(W, T, C_S, B)
            
            # true values of (i, j-1)
            true_ins_cache = true_vals_for_ins[:, :, k, :] #(B, C_S, W)
            true_ins_cache = reshape_true_answer(true_ins_cache) #(W, T, C_S, B)
            
            
            ### predicted values of (i, j-1)
            # align_cell_idxes is (B, W, 2)
            # mask is (B, W)
            align_cell_idxes, wf_pad_mask = generate_ij_coords_at_diagonal_k(seq_lens = seq_lens,
                                                                             diagonal_k = k,
                                                                             widest_diag_W = W)
            
            ins_idxes = align_cell_idxes.at[:, :, 1].add(-1) # (B, W, 2)
            
            pred_ins_cache = wavefront_cache_lookup(ij_needed_for_k = ins_idxes, 
                                                    pad_mask_at_k = wf_pad_mask, 
                                                    cache_for_prev_diagonal = cache_to_read, 
                                                    seq_lens = seq_lens, 
                                                    pad_val = 0) #(W, T, C_S, B)
            
            npt.assert_allclose(pred_ins_cache, true_ins_cache), k
            
if __name__ == '__main__':
    unittest.main()