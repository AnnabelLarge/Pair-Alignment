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
from models.simple_site_class_predict.marg_over_alignments_forward_fns import (joint_loglike_emission_at_k_time_grid,
                                                                               joint_loglike_emission_at_k_len_per_samp)



class TestEmissionScoringFns(unittest.TestCase):
    def setUp(self):
        self.anc_toks = jnp.array( [[3, 4, 5, 6, 3],
                                    [4, 4, 4, 5, 3],
                                    [4, 4, 4, 0, 0],
                                    [6, 5, 4, 5, 0]] ) #(B, W)
        
        self.desc_toks = jnp.array( [[6, 5, 4, 3, 3],
                                     [3, 6, 6, 6, 3],
                                     [3, 6, 6, 0, 0],
                                     [3, 4, 5, 1, 0]] ) #(B, W)
        
        self.mask = (self.anc_toks != 0 ) #(B, W)
        
        self.B = self.anc_toks.shape[0]
        self.W = self.anc_toks.shape[1]
        
    
    def test_with_time_grid(self):
        C_trans = 10
        A = 4
        B = self.B
        W = self.W
        T = 3
        
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
        
        
        ### by my function
        pred_out = joint_loglike_emission_at_k_time_grid(self.anc_toks,
                                                         self.desc_toks,
                                                         self.mask,
                                                         joint_logprob_emit_at_match,
                                                         logprob_emit_at_indel) #(W, T, C*S-1, B)
        
        
        ### true value
        for b in range(B):
            for w in range(W):
                a = self.anc_toks[b,w]-3
                d = self.desc_toks[b,w]-3
                score_pos = self.mask[b,w]
                
                if score_pos:
                    true_match_val = joint_logprob_emit_at_match[:, :, a, d] #(T, C_trans)
                    true_ins_val = logprob_emit_at_indel[:, d] #(C_trans,)
                    true_del_val = logprob_emit_at_indel[:, a] #(C_trans,)
                    
                    true_ins_val = jnp.broadcast_to( true_ins_val[None,...], (T, C_trans) ) #(T, C_trans)
                    true_del_val = jnp.broadcast_to( true_del_val[None,...], (T, C_trans) ) #(T, C_trans)
                    
                    true_c_s = jnp.stack( [true_match_val,
                                            true_ins_val,
                                            true_del_val], axis=-1 ) #(T, C_trans, S-1)
                    true_c_s = jnp.reshape(true_c_s, (T, C_trans * 3)) #(T, C_trans*S-1)
                    
                    npt.assert_allclose( pred_out[w, :, :, b], true_c_s ), f'{b}, {w}'
                    
                elif not score_pos:
                    npt.assert_allclose( pred_out[w, :, :, b], 
                                         jnp.zeros( pred_out[w, :, :, b].shape ) ), f'{b}, {w}'
    
    
    def test_uniq_time_per_samp(self):
        C_trans = 10
        A = 4
        B = self.B
        W = self.W
        
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
        
        
        ### by my function
        pred_out = joint_loglike_emission_at_k_len_per_samp(self.anc_toks,
                                                            self.desc_toks,
                                                            self.mask,
                                                            joint_logprob_emit_at_match,
                                                            logprob_emit_at_indel) #(W, C*S-1, B)
        
        ### true value
        for b in range(B):
            for w in range(W):
                a = self.anc_toks[b,w]-3
                d = self.desc_toks[b,w]-3
                score_pos = self.mask[b,w]
                
                if score_pos:
                    true_match_val = joint_logprob_emit_at_match[b, :, a, d] #(C_trans)
                    true_ins_val = logprob_emit_at_indel[:, d] #(C_trans,)
                    true_del_val = logprob_emit_at_indel[:, a] #(C_trans,)
                    
                    true_c_s = jnp.stack( [true_match_val,
                                            true_ins_val,
                                            true_del_val], axis=-1 ) #(C_trans, S-1)
                    true_c_s = jnp.reshape(true_c_s, (C_trans * 3)) #(C_trans*S-1,)
                    
                    npt.assert_allclose( pred_out[w, :, b], true_c_s )
                
                elif not score_pos:
                    npt.assert_allclose( pred_out[w, :, b], 
                                         jnp.zeros( pred_out[w, :, b].shape ) ), f'{b}, {w}'
                  
                    
if __name__ == '__main__':
    unittest.main()  