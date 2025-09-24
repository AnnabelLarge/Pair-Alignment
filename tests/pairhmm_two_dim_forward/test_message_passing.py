#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 19:30:31 2025

@author: annabel
"""
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
import flax.linen as nn

import numpy as np
import numpy.testing as npt
import unittest

from models.simple_site_class_predict.transition_models import (TKF92TransitionLogprobs)
from models.simple_site_class_predict.marg_over_alignments_forward_fns import (index_all_classes_one_state,
                                                                               compute_forward_messages_for_state)


class TestMessagePassing(unittest.TestCase):
    def setUp(self):
        # dims
        C_transit = 3
        A = 20
        S = 4
        C_S = C_transit * (S-1) #use this for forward algo carry
        
        # time
        t_array = jnp.array( [1.0, 0.3, 0.2] )
        T = t_array.shape[0]
        
        ########################
        ### scoring matrices   #
        ########################
        # use real model object for transition matrix
        transit_model = TKF92TransitionLogprobs( config={'tkf_function': 'regular_tkf',
                                                         'num_domain_mixtures': 1,
                                                         'num_fragment_mixtures': C_transit},
                                                 name = 'transit_mat' )
        init_params = transit_model.init( rngs = jax.random.key(0),
                                          t_array = t_array,
                                          return_all_matrices = False,
                                          sow_intermediates = False )
        
        out = transit_model.apply( variables = init_params,
                                   t_array = t_array,
                                   return_all_matrices = False,
                                   sow_intermediates = False )
        joint_logprob_transit = out[1]['joint'][:,0,...] #(T, C_transit_prev, C_transit_curr, S_prev, S_curr)
        
        # transpose transition matrix 
        # old: (T, C_transit_prev, C_transit_curr, S_prev, S_curr)
        # new: (T, C_transit_prev, S_prev, C_transit_curr, S_curr)
        joint_logprob_transit = jnp.transpose(joint_logprob_transit, (0,1,3,2,4)) # (T, C_transit_prev, S_prev, C_transit_curr, S_curr)
        joint_logprob_transit_mid_only = jnp.reshape(joint_logprob_transit[:, :, :3, :, :3], (T, C_S, C_S) ) #(T, C*S_prev, C*S_curr)
        

        #################
        ### sequences   #
        #################
        seqs1 = jnp.array( [[1, 1],
                            [3, 3],
                            [5, 2],
                            [3, 3],
                            [2, 0],
                            [0, 0]] )
        
        seqs2 = jnp.array( [[1, 1],
                            [6, 6],
                            [2, 4],
                            [3, 3],
                            [0, 2],
                            [0, 0]] )
        
        # concat
        unaligned_seqs = jnp.stack([seqs1, seqs2], axis=0) #(B, L_seq, 2)
        
        # extra dims
        B = unaligned_seqs.shape[0]
        
        # widest diagonal for wavefront parallelism is min(anc_len, desc_len) + 1
        seq_lens = (unaligned_seqs != 0).sum(axis=1)-2 #(B, 2)
        min_lens = seq_lens.min(axis=1) #(B,)
        W = min_lens.max() + 1 #float
        

        ################################################
        ### Initialize cache for wavefront diagonals   #
        ################################################
        # \tau = state, M/I/D
        # \nu = class (unique to combination of domain+fragment)
        # alpha_{ij}^{s_d} = P(desc_{...j}, anc_{...i}, \tau=s, \nu=d | t)
        prev_alpha = jnp.full( (W, T, C_S, B), jnp.finfo(jnp.float32).min )
        
        # manually init values at (1,0)
        i = 1
        for w in range(W):
            for t_idx in range(T):
                for c_s in range(C_S):
                    for b in range(B):
                        dummy_cache_value = i * -0.001
                        assert dummy_cache_value < 0
                        prev_alpha = prev_alpha.at[w, t_idx, c_s, b].set(dummy_cache_value)
                        i += 1
                        
        
        ### make attributes
        self.T = T
        self.C_transit = C_transit
        self.A = A
        self.S = S
        self.C_S = C_S
        self.B = B
        self.W = W
        
        self.prev_alpha = prev_alpha
        self.logprob_transit_mid_only = joint_logprob_transit_mid_only
    
    def _run_test(self, state_idx):
        W = self.W
        T = self.T
        C_transit = self.C_transit
        C_S = self.C_S
        B = self.B
        prev_alpha = self.prev_alpha
        logprob_transit_mid_only = self.logprob_transit_mid_only
        
        idx_arr_for_state = index_all_classes_one_state(state_idx=state_idx,
                                                num_transit_classes=C_transit) #(C_transit,)
        
        # predicted
        pred = compute_forward_messages_for_state(logprob_transit_mid_only = logprob_transit_mid_only,
                                                      idxes_for_curr_state = idx_arr_for_state,
                                                      cache_for_state = prev_alpha)
        
        ### check against true
        # (W, T, C_transit_curr, B)
        for w in range(W):
            for t_idx in range(T):
                for c_curr, c_s_curr in enumerate(idx_arr_for_state):
                    for b in range(B):
                        true = 0
                        for c_s_prev in range(C_S):
                            
                            # alpha_{i_prev, j_prev}^{c_s_prev}
                            cache_val = prev_alpha[w, t_idx, c_s_prev, b]
                            
                            # logP( c_s_curr | c_s_prev, t )
                            logprob_to_match = logprob_transit_mid_only[t_idx, c_s_prev, c_s_curr]
                            
                            # alpha_{i_prev, j_prev}^{c_s_prev} + logP( c_s_curr | c_s_prev, t )
                            true += np.exp( cache_val + logprob_to_match )
                        
                        npt.assert_allclose( true.item(), np.exp(pred[w,t_idx,c_curr,b]).item() )
                        
        
    def test_match_message_passing(self):
        self._run_test(state_idx = 0)
    
    def test_ins_message_passing(self):
        self._run_test(state_idx = 1)
    
    def test_del_message_passing(self):
        self._run_test(state_idx = 2)
            
if __name__ == '__main__':
    unittest.main()      
