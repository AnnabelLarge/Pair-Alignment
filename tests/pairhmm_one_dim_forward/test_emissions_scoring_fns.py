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
                                                                   joint_loglike_emission_len_per_samp)



class TestEmissionScoringFns(unittest.TestCase):
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
        
        self.aligned_inputs = jnp.stack([all_match,
                                         all_ins,
                                         all_del,
                                         mix]) #(B, L, 3)
        
        self.B = self.aligned_inputs.shape[0]
        self.L = self.aligned_inputs.shape[1]
        
    
    def test_with_time_grid(self):
        C_trans = 5
        A = 4
        B = self.B
        L = self.L
        T = 4
        
        ### fake scoring matrices
        # use dummy scoring matrices for emissions
        sub_emit_logits = jax.random.normal( key = jax.random.key(0),
                                              shape = (T, C_trans, A, A) ) #(T, C_trans, A, A)
        joint_logprob_emit_at_match = nn.log_softmax(sub_emit_logits, axis=(-1,-2)) #(T, C_trans, A, A)
        del sub_emit_logits
        
        indel_emit_logits = jax.random.normal( key = jax.random.key(0),
                                              shape = (C_trans, A) ) #(C_trans, A)
        logprob_emit_at_indel = nn.log_softmax(indel_emit_logits, axis=-1) #(C_trans, A)
        del indel_emit_logits
        
        for b in range(B):
            for l in range(L):
                a = self.aligned_inputs[b,l,0]
                d = self.aligned_inputs[b,l,1]
                s = self.aligned_inputs[b,l,2]
                
                pred_out = joint_loglike_emission_time_grid( aligned_inputs = self.aligned_inputs[b,...][None,...],
                                                                 pos = l,
                                                                 joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                                                 logprob_emit_at_indel = logprob_emit_at_indel ) #(T, C, 1)
                
                pred_out = pred_out[...,0] # (T, C)
                # invalid tokens
                if s in [0,4,5]:
                    continue
                
                # Match
                elif s == 1:
                    true = joint_logprob_emit_at_match[:, :, a-3, d-3] #(T, C)
                
                # Insert
                elif s==2:
                    true = logprob_emit_at_indel[:, d-3] #(C,)
                    true = jnp.repeat( true[None,:], T, axis=0 ) #(T, C)
                
                # Delete
                elif s==3:
                    true = logprob_emit_at_indel[:, a-3] #(C,)
                    true = jnp.repeat( true[None,:], T, axis=0 ) #(T, C)
                
                npt.assert_allclose( pred_out, true )
                
    
    def test_uniq_time_per_samp(self):
        C_trans = 10
        A = 4
        B = self.B
        L = self.L
        
        ### fake scoring matrices
        # use dummy scoring matrices for emissions
        sub_emit_logits = jax.random.normal( key = jax.random.key(0),
                                              shape = (B, C_trans, A, A) ) #(B, C_trans, A, A)
        joint_logprob_emit_at_match = nn.log_softmax(sub_emit_logits, axis=(-1,-2)) #(B, C_trans, A, A)
        del sub_emit_logits
        
        indel_emit_logits = jax.random.normal( key = jax.random.key(0),
                                              shape = (C_trans, A) ) #(C_trans, A)
        logprob_emit_at_indel = nn.log_softmax(indel_emit_logits, axis=-1) #(C_trans, A)
        del indel_emit_logits
        
        for b in range(B):
            for l in range(L):
                a = self.aligned_inputs[b,l,0]
                d = self.aligned_inputs[b,l,1]
                s = self.aligned_inputs[b,l,2]
                
                pred_out = joint_loglike_emission_len_per_samp( aligned_inputs = self.aligned_inputs[b,...][None,...],
                                                                            pos = l,
                                                                            joint_logprob_emit_at_match = joint_logprob_emit_at_match[b,...][None,...],
                                                                            logprob_emit_at_indel = logprob_emit_at_indel ) #(1, C)
                
                pred_out = pred_out[:,0] # (C)
                
                # invalid tokens
                if s in [0,4,5]:
                    continue
                
                # Match
                elif s == 1:
                    true = joint_logprob_emit_at_match[b, :, a-3, d-3] #(C,)
                
                # Insert
                elif s==2:
                    true = logprob_emit_at_indel[:, d-3] #(C,)
                
                # Delete
                elif s==3:
                    true = logprob_emit_at_indel[:, a-3] #(C,)
                
                npt.assert_allclose( pred_out, true )
        
    
                    
if __name__ == '__main__':
    unittest.main()  