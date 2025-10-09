#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 18:09:01 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
import flax.linen as nn

import numpy as np
import numpy.testing as npt
import unittest

from models.latent_class_mixtures.transition_models import (TKF92TransitionLogprobs)
from models.latent_class_mixtures.two_dim_forward_algo_helpers import (index_all_classes_one_state,
                                                                               init_first_diagonal)


class TestInitFirstDiagonal(unittest.TestCase):
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
        
        # use dummy scoring matrices for emissions
        sub_emit_logits = jax.random.normal( key = jax.random.key(0),
                                             shape = (T, C_transit, A, A) ) #(T, C_transit, A, A)
        joint_logprob_emit_at_match = nn.log_softmax(sub_emit_logits, axis=(-1,-2)) #(T, C_transit, A, A)
        
        indel_emit_logits = jax.random.normal( key = jax.random.key(0),
                                             shape = (C_transit, A) ) #(C_transit, A)
        logprob_emit_at_indel = nn.log_softmax(indel_emit_logits, axis=-1) #(C_transit, A)
        

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
        
        seqs3 = jnp.array( [[1, 1],
                            [6, 6],
                            [2, 2],
                            [0, 0],
                            [0, 0],
                            [0, 0]] )
        
        # concat
        unaligned_seqs = jnp.stack([seqs1, seqs2, seqs3], axis=0) #(B, L_seq, 2)
        
        # extra dims
        B = unaligned_seqs.shape[0]
        
        # widest diagonal for wavefront parallelism is min(anc_len, desc_len) + 1
        seq_lens = (unaligned_seqs != 0).sum(axis=1)-2 #(B, 2)
        min_lens = seq_lens.min(axis=1) #(B,)
        W = min_lens.max() + 1 #float
        

        ################################################
        ### Initialize cache for wavefront diagonals   #
        ################################################
        # fill diagonal k-2: alignment cells (1,0) and (0,1)
        first_diag = init_first_diagonal( cache_size = (W, T, C_S, B), 
                                          unaligned_seqs = unaligned_seqs,
                                          joint_logprob_transit = joint_logprob_transit,
                                          logprob_emit_at_indel = logprob_emit_at_indel )  #(2, W, T, C_S, B)

        
        ### make attributes
        self.T = T
        self.C_transit = C_transit
        self.A = A
        self.S = S
        self.C_S = C_S
        self.B = B
        self.W = W
        
        self.first_diag = first_diag
        self.unaligned_seqs = unaligned_seqs
        self.joint_logprob_transit = joint_logprob_transit
        self.logprob_emit_at_indel = logprob_emit_at_indel
    
    
    def test_shape(self):
        first_diag = self.first_diag
        W = self.W
        T = self.T
        C_S = self.C_S
        B = self.B
        
        npt.assert_allclose( first_diag.shape, (W, T, C_S, B) )
    
    def test_first_deletion_init(self):
        first_diag = self.first_diag
        W = self.W
        T = self.T
        C_transit = self.C_transit
        C_S = self.C_S
        B = self.B
        unaligned_seqs = self.unaligned_seqs
        joint_logprob_transit = self.joint_logprob_transit
        logprob_emit_at_indel = self.logprob_emit_at_indel
        
        cell_1_0 = first_diag[0, ...] # (T, C_S, B)

        # check shape
        npt.assert_allclose( cell_1_0.shape, (T, C_S, B) )

        # cell (1,0) should only have values for delete
        del_idx = index_all_classes_one_state( state_idx = 2,
                                                num_transit_classes = C_transit )
        del_idx = set( np.array( del_idx ) ) #(C_transit)
        assert len(del_idx) == C_transit

        for t_idx in range(T):
            for c_s in range(C_S):
                for b in range(B):
                    cell_value = cell_1_0[t_idx, c_s, b]
                    
                    # cell (1,0) should only have values for delete
                    if c_s not in del_idx:
                        npt.assert_allclose( cell_value, jnp.finfo(jnp.float32).min ), f'{t_idx}, {c_s}, {b}'
                    
                    elif c_s in del_idx:
                        c = c_s // 3
                        logprob_start_to_del = joint_logprob_transit[t_idx, 0, -1, c, 2]
                        anc_tok = unaligned_seqs[b, 1, 0]
                        logprob_anc_tok = logprob_emit_at_indel[c, anc_tok-3]
                        true_val = logprob_start_to_del + logprob_anc_tok
                        npt.assert_allclose(true_val, cell_value), f'{t_idx}, {c_s}, {b}: {cell_value}'
    
    
    def test_first_insert_init(self):
        first_diag = self.first_diag
        W = self.W
        T = self.T
        C_transit = self.C_transit
        C_S = self.C_S
        B = self.B
        unaligned_seqs = self.unaligned_seqs
        joint_logprob_transit = self.joint_logprob_transit
        logprob_emit_at_indel = self.logprob_emit_at_indel
        
        cell_0_1 = first_diag[1, ...] # (T, C_S, B)
        
        # check shape
        npt.assert_allclose( cell_0_1.shape, (T, C_S, B) )
        
        # cell (0,1) should only have values for insert
        ins_idx = index_all_classes_one_state( state_idx = 1,
                                               num_transit_classes = C_transit )
        ins_idx = set( np.array( ins_idx ) ) #(C_transit)
        assert len(ins_idx) == C_transit
        
        for t_idx in range(T):
            for c_s in range(C_S):
                for b in range(B):
                    cell_value = cell_0_1[t_idx, c_s, b]
                    
                    # cell (0,1) should only have values for ins
                    if c_s not in ins_idx:
                        npt.assert_allclose( cell_value, jnp.finfo(jnp.float32).min ), f'{t_idx}, {c_s}, {b}'
                    
                    elif c_s in ins_idx:
                        c = c_s // 3
                        logprob_start_to_ins = joint_logprob_transit[t_idx, 0, -1, c, 1]
                        desc_tok = unaligned_seqs[b, 1, 1]
                        logprob_desc_tok = logprob_emit_at_indel[c, desc_tok-3]
                        true_val = logprob_start_to_ins + logprob_desc_tok
                        npt.assert_allclose(true_val, cell_value), f'{t_idx}, {c_s}, {b}: {cell_value}'
        
        
if __name__ == '__main__':
    unittest.main()