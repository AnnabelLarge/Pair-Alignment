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
from models.latent_class_mixtures.INPROGRESS import (message_passing_time_grid,
                                                     message_passing_len_per_samp)

class TestMessagePassing(unittest.TestCase):
    def setUp(self):
        self.prev_states = jnp.array([1,2,3]) #(B,)
        self.curr_states = jnp.array([3,2,1]) #(B,)
        
        self.B = self.prev_states.shape[0]
    
    def test_with_time_grid(self):
        C_trans = 5
        A = 4
        B = self.B
        T = 10
        
        ### fake inputs
        transit_emit_logits = jax.random.normal( key = jax.random.key(0),
                                                 shape = (T, C_trans, C_trans, 4, 4) ) #(T, C_trans, C_trans, S, S)
        joint_logprob_transit = nn.log_softmax(transit_emit_logits, axis=(-1,-2)) #(T, C_trans, S, S)
        del transit_emit_logits
        
        prev_message_probs = jax.random.uniform( key = jax.random.key(0),
                                                  shape = (T,C_trans,B),
                                                  minval = 1e-4,
                                                  maxval = 0.999 )
        prev_message_logprobs = jnp.log(prev_message_probs)
        
        # pred value in LOG space
        pred_out = message_passing_time_grid( prev_message = prev_message_logprobs, 
                                              ps = self.prev_states, 
                                              cs = self.curr_states, 
                                              joint_logprob_transit = joint_logprob_transit ) #(T,C,B)
        
        for b in range(B):
            prev_align_state = self.prev_states[b]-1
            curr_align_state = self.curr_states[b]-1
            
            # true value in PROB space
            log_tr_per_class = joint_logprob_transit[..., prev_align_state, curr_align_state] #(T,C_prev,C_curr)
            tr_per_class = jnp.exp(log_tr_per_class) #(T,C_prev,C_curr)
            to_mult = prev_message_probs[...,b] #(T,C_prev)
            to_add = jnp.multiply(tr_per_class, to_mult[:, :, None]) #(T,C_prev,C_curr)
            true = to_add.sum(axis=1) #(T, C_curr)
            
            # compare in log space
            npt.assert_allclose( pred_out[...,b], jnp.log(true), atol=1e-6 )
    
    def test_uniq_time_per_samp(self):
        C_trans = 10
        A = 4
        B = self.B
        
        ### fake inputs
        transit_emit_logits = jax.random.normal( key = jax.random.key(0),
                                                  shape = (B, C_trans, C_trans, 4, 4) ) #(B, C_trans, C_trans, S, S)
        joint_logprob_transit = nn.log_softmax(transit_emit_logits, axis=(-1,-2)) #(B, C_trans, S, S)
        del transit_emit_logits
        
        prev_message_probs = jax.random.uniform( key = jax.random.key(0),
                                                  shape = (C_trans,B),
                                                  minval = 1e-4,
                                                  maxval = 0.999 )
        prev_message_logprobs = jnp.log(prev_message_probs)
        
        # pred value in LOG space
        pred_out = message_passing_len_per_samp( prev_message = prev_message_logprobs, 
                                                 ps = self.prev_states, 
                                                 cs = self.curr_states, 
                                                 joint_logprob_transit = joint_logprob_transit ) #(C,B)
        assert pred_out.shape == (C_trans, B)
        
        for b in range(B):
            prev_align_state = self.prev_states[b]-1
            curr_align_state = self.curr_states[b]-1
            
            # true value in PROB space
            log_tr_per_class = joint_logprob_transit[b, ..., prev_align_state, curr_align_state] #(C_prev,S)
            tr_per_class = jnp.exp(log_tr_per_class) #(C_prev,C_curr)
            to_mult = prev_message_probs[...,b] #(C_prev)
            to_add = jnp.multiply(tr_per_class, to_mult[:, None]) #(C_prev,C_curr)
            true = to_add.sum(axis=0) #(C_curr)
            
            # compare in log space
            npt.assert_allclose( pred_out[...,b], jnp.log(true), atol=1e-6 )
                    
if __name__ == '__main__':
    unittest.main()  