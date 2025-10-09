#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 20:08:43 2025

@author: annabel


TODO: make this pretty later
"""
import jax
from jax import numpy as jnp
import flax.linen as nn

import numpy as np
import numpy.testing as npt
import unittest

from models.latent_class_mixtures.transition_models import TKF92TransitionLogprobs
from models.latent_class_mixtures.two_dim_forward_with_time_grid import two_dim_forward_with_time_grid as forward_fn
from models.latent_class_mixtures.two_dim_forward_algo_helpers import (generate_ij_coords_at_diagonal_k,
                                                               ij_coords_to_wavefront_pos_at_diagonal_k,
                                                               index_all_classes_one_state,
                                                               wavefront_cache_lookup,
                                                               compute_forward_messages_for_state,
                                                               joint_loglike_emission_at_k_time_grid,
                                                               init_first_diagonal,
                                                               init_second_diagonal,
                                                               get_match_transition_message,
                                                               get_ins_transition_message,
                                                               get_del_transition_message,
                                                               update_cache)



class TestAlignSubMatConsistency(unittest.TestCase):
    def setUp(self):
        # make sure this is turned off, or the test will fail
        jax.config.update("jax_enable_x64", False)
        
        ###############################################################################
        ### Fake inputs   #############################################################
        ###############################################################################
        # dims
        C_transit = 2
        A = 20
        S = 4
        C_S = C_transit * (S-1) #use this for forward algo carry
        
        # time
        t_array = jnp.array( [1.0, 0.3] )
        T = t_array.shape[0]
        
        self.C_transit = C_transit
        self.A = A
        self.S = S
        self.C_S = C_S
        self.T = T
        
        
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
        joint_logprob_transit = jnp.transpose(joint_logprob_transit, (0,1,3,2,4)) # (T, C_transit_prev, S_prev, C_transit_curr, S_curr)
        del transit_model, init_params, out
        
        # use dummy scoring matrices for emissions
        sub_emit_logits = jax.random.normal( key = jax.random.key(0),
                                             shape = (T, C_transit, A, A) ) #(T, C_transit, A, A)
        joint_logprob_emit_at_match = nn.log_softmax(sub_emit_logits, axis=(-1,-2)) #(T, C_transit, A, A)
        del sub_emit_logits
        
        indel_emit_logits = jax.random.normal( key = jax.random.key(0),
                                             shape = (C_transit, A) ) #(C_transit, A)
        logprob_emit_at_indel = nn.log_softmax(indel_emit_logits, axis=-1) #(C_transit, A)
        del indel_emit_logits
        
        self.joint_logprob_transit = joint_logprob_transit
        self.joint_logprob_emit_at_match = joint_logprob_emit_at_match
        self.logprob_emit_at_indel = logprob_emit_at_indel
        
        
        #################
        ### sequences   #
        #################
        # AGA -> AGA
        full_seq = jnp.array( [[1, 1],
                               [3, 3],
                               [5, 5],
                               [3, 3],
                               [2, 2]] )
        
        # A -> A
        subseq1 = jnp.array( [[1, 1],
                              [3, 3],
                              [2, 2],
                              [0, 0],
                              [0, 0]] )
        
        # AG -> AG
        subseq2 = jnp.array( [[1, 1],
                              [3, 3],
                              [5, 5],
                              [2, 2],
                              [0, 0]] )
        
        ### concate sequences
        unaligned_seqs = jnp.stack([full_seq, subseq1, subseq2], axis=0) #(B, L_seq, 2)
        del full_seq, subseq1, subseq2
        
        B = unaligned_seqs.shape[0]
        L_seq = unaligned_seqs.shape[1]
        
        # widest diagonal for wavefront parallelism is min(anc_len, desc_len) + 1
        seq_lens = (unaligned_seqs != 0).sum(axis=1)-2 #(B, 2)
        min_lens = seq_lens.min(axis=1) #(B,)
        W = min_lens.max() + 1 #float
        del min_lens
        
        # number of diagonals
        K = (seq_lens.sum(axis=1)).max()
        
        self.unaligned_seqs = unaligned_seqs
        self.seq_lens = seq_lens
        self.B = B
        self.L_seq = L_seq
        self.W = W
        self.K = K
          
    def test_consistency(self):
        _, wavefront_cache = forward_fn(unaligned_seqs = self.unaligned_seqs,
                                        joint_logprob_transit = self.joint_logprob_transit,
                                        joint_logprob_emit_at_match = self.joint_logprob_emit_at_match,
                                        logprob_emit_at_indel = self.logprob_emit_at_indel,
                                        return_full_grid = True) # (K, W, T, C_S, B)
        B = self.B
        
        ### K = 1
        diag1 = wavefront_cache[0] #(W, T, C_S, B)
        
        # make sure first diagonal is same for all
        npt.assert_allclose( diag1[...,0], diag1[...,1])
        npt.assert_allclose( diag1[...,0], diag1[...,2])
        npt.assert_allclose( diag1[...,1], diag1[...,2])
        
        
        ### K = 2
        diag2 = wavefront_cache[1] #(W, T, C_S, B)
        
        # check that cell (1,1) is consistent across all samples
        scores_from_cell_1_1 = diag2[[1, 0, 1], ..., jnp.arange(B)] #(B, T, C_S)
        npt.assert_allclose( scores_from_cell_1_1[0,...], scores_from_cell_1_1[1,...] )
        npt.assert_allclose( scores_from_cell_1_1[2,...], scores_from_cell_1_1[1,...] )
        npt.assert_allclose( scores_from_cell_1_1[0,...], scores_from_cell_1_1[2,...] )
        del scores_from_cell_1_1
        
        # for full and subseq2: check that entire second diagonal matches
        # this encompasses (2,0) and (0,2)
        second_diag_full_and_sub = diag2[..., [0,2]] #(W, T, C_S, B)
        npt.assert_allclose( second_diag_full_and_sub[...,0], second_diag_full_and_sub[...,1] )
        del second_diag_full_and_sub
        
        
        ### K=3
        diag3 = wavefront_cache[2] #(W, T, C_S, B)
        
        # for full and subseq2, check values at (2,1) and (1,2)
        cell_2_1 = diag3[[1, 0], ..., [0, 2]] #(B-1, T, C*S)
        npt.assert_allclose( cell_2_1[0,...], cell_2_1[1,...] )
        del cell_2_1
        
        cell_1_2 = diag3[[2, 1], ..., [0, 2]] #(B-1, T, C*S)
        npt.assert_allclose( cell_1_2[0,...], cell_1_2[1,...] )
        del cell_1_2
        
        ### K=4
        diag4 = wavefront_cache[3] #(W, T, C_S, B)
        
        # for full and subseq2: check values at (2,2)
        cell_2_2 = diag4[[1, 0], ..., [0, 2]] #(B-1, T, C*S)
        npt.assert_allclose( cell_2_2[0,...], cell_2_2[1,...] )
        
        
if __name__ == '__main__':
    unittest.main()
