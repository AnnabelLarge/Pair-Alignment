#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 20:44:28 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
from flax import linen as nn
import numpy as np

import numpy.testing as npt
import unittest
from models.latent_class_mixtures.one_dim_fwd_bkwd_helpers import (joint_loglike_emission_time_grid,
                                                                   joint_loglike_emission_len_per_samp,
                                                                   init_recurs_with_time_grid,
                                                                   init_recurs_with_len_per_samp,
                                                                   flip_alignments)

class TestFwInitFns(unittest.TestCase):
    def setUp(self):
        all_match = jnp.array( [[1, 1, 4],
                                [3, 4, 1],
                                [4, 5, 1],
                                [2, 2, 5],
                                [0, 0, 0]] ) #(L, 3)
        
        all_ins = jnp.array( [[ 1, 1, 4],
                              [43, 4, 2],
                              [ 2, 2, 5],
                              [ 0, 0, 0],
                              [ 0, 0, 0]] ) #(L, 3)
        
        all_del = jnp.array( [[1,  1, 4],
                              [4, 43, 3],
                              [5, 43, 3],
                              [6, 43, 3],
                              [2,  2, 5]] ) #(L, 3)
        
        mix = jnp.array( [[ 1,  1, 4],
                          [ 3,  4, 1],
                          [43,  4, 2],
                          [ 4, 43, 3],
                          [ 2,  2, 5]] ) #(L, 3)
        
        aligned_inputs = jnp.stack([all_match,
                                    all_ins,
                                    all_del,
                                    mix]) #(B, L, 3)
        self.aligned_inputs = flip_alignments(aligned_inputs) #(B, L, 3)
        
        self.B = self.aligned_inputs.shape[0]
        self.L = self.aligned_inputs.shape[1]
        
    
    def test_with_time_grid(self):
        C_trans = 5
        A = 4
        B = self.B
        L = self.L
        T = 10
        
        ### fake scoring matrices
        transit_emit_logits = jax.random.normal( key = jax.random.key(0),
                                                 shape = (T, C_trans, C_trans, 4, 4) ) #(T, C_trans, C_trans, S, S)
        joint_logprob_transit = nn.log_softmax(transit_emit_logits, axis=(-1,-2)) #(T, C_trans, S, S)
        del transit_emit_logits
        
        sub_emit_logits = jax.random.normal( key = jax.random.key(0),
                                              shape = (T, C_trans, A, A) ) #(T, C_trans, A, A)
        joint_logprob_emit_at_match = nn.log_softmax(sub_emit_logits, axis=(-1,-2)) #(T, C_trans, A, A)
        del sub_emit_logits
        
        indel_emit_logits = jax.random.normal( key = jax.random.key(0),
                                              shape = (C_trans, A) ) #(C_trans, A)
        logprob_emit_at_indel = nn.log_softmax(indel_emit_logits, axis=-1) #(C_trans, A)
        del indel_emit_logits
        
        pred_out = init_recurs_with_time_grid( aligned_inputs = self.aligned_inputs,
                                   joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                   logprob_emit_at_indel = logprob_emit_at_indel,
                                   joint_logprob_transit = joint_logprob_transit,
                                   which = 'bkw' ) #(T,C,B)
        
        for b in range(B):
            a = self.aligned_inputs[b,1,0]
            d = self.aligned_inputs[b,1,1]
            s = self.aligned_inputs[b,1,2]
            
            e = joint_loglike_emission_time_grid( aligned_inputs = self.aligned_inputs[b,...][None,...],
                                                  pos = 1,
                                                  joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                                  logprob_emit_at_indel = logprob_emit_at_indel ) #(T, C_prev, 1)
            e = e[...,0] #(T, C_prev)
            
            assert s in [1,2,3]
            true = e + joint_logprob_transit[:, :, -1, s-1, -1] #(T, C_prev)
            
            npt.assert_allclose( pred_out[...,b], true )
                
    def test_uniq_time_per_samp(self):
        C_trans = 10
        A = 4
        B = self.B
        L = self.L
        
        ### fake scoring matrices
        transit_emit_logits = jax.random.normal( key = jax.random.key(0),
                                                 shape = (B, C_trans, C_trans, 4, 4) ) #(B, C_trans, C_trans, S, S)
        joint_logprob_transit = nn.log_softmax(transit_emit_logits, axis=(-1,-2)) #(B, C_trans, S, S)
        del transit_emit_logits
        
        sub_emit_logits = jax.random.normal( key = jax.random.key(0),
                                              shape = (B, C_trans, A, A) ) #(B, C_trans, A, A)
        joint_logprob_emit_at_match = nn.log_softmax(sub_emit_logits, axis=(-1,-2)) #(B, C_trans, A, A)
        del sub_emit_logits
        
        indel_emit_logits = jax.random.normal( key = jax.random.key(0),
                                              shape = (C_trans, A) ) #(C_trans, A)
        logprob_emit_at_indel = nn.log_softmax(indel_emit_logits, axis=-1) #(C_trans, A)
        del indel_emit_logits
        
        pred_out = init_recurs_with_len_per_samp( aligned_inputs = self.aligned_inputs,
                                          joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                          logprob_emit_at_indel = logprob_emit_at_indel,
                                          joint_logprob_transit = joint_logprob_transit,
                                          which = 'bkw') #(C,B)
        
        e = joint_loglike_emission_len_per_samp( aligned_inputs = self.aligned_inputs,
                                                 pos = 1,
                                                 joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                                 logprob_emit_at_indel = logprob_emit_at_indel ) #(C_prev, B)
        
        for b in range(B):
            s = self.aligned_inputs[b,1,2]
            
            assert s in [1,2,3]
            true = e[...,b] + joint_logprob_transit[b, :, -1, s-1, -1] #(C_prev,)
            
            npt.assert_allclose( pred_out[...,b], true )
                    
if __name__ == '__main__':
    unittest.main()  