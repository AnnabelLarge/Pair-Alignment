#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 20:08:43 2025

@author: annabel


TODO: make this pretty later
"""
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
import flax.linen as nn

import numpy as np
import numpy.testing as npt
import unittest

from models.latent_class_mixtures.transition_models import TKF92TransitionLogprobs

from models.latent_class_mixtures.forward_algo_helpers import (generate_ij_coords_at_diagonal_k,
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
        ###############################################################################
        ### Fake inputs   #############################################################
        ###############################################################################
        # dims
        C_transit = 2
        A = 20
        S = 4
        C_S = C_transit * (S-1) #use this for forward algo carry
        
        # time
        t_array = jnp.array( [1.0, 0.3, 0.5] )
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
          
    
    def _recursion(self,
                   k,
                   previous_cache,
                   previous_first_cell_scores,
                   joint_logprob_transit_mid_only):
        W = self.W
        T = self.T
        C_S = self.C_S
        B = self.B
        C_transit = self.C_transit
        
        unaligned_seqs = self.unaligned_seqs
        seq_lens = self.seq_lens

        joint_logprob_emit_at_match = self.joint_logprob_emit_at_match
        logprob_emit_at_indel = self.logprob_emit_at_indel
        
        
        # blank cache to fill
        cache_at_curr_k = jnp.full( (W, T, C_S, B), jnp.finfo(jnp.float32).min ) # (W, T, C*S, B)
        
        # align_cell_idxes is (B, W, 2)
        # pad_mask is (B, W)
        align_cell_idxes, pad_mask = generate_ij_coords_at_diagonal_k(seq_lens = seq_lens,
                                                                      diagonal_k = k,
                                                                      widest_diag_W = W)
        
        
        ### update with transitions
        # match
        # match_idx is (C_trans,) 
        # match_transition_message is (W, T, C_trans, B)
        match_idx, match_transition_message = get_match_transition_message( align_cell_idxes = align_cell_idxes,
                                                                            pad_mask = pad_mask,
                                                                            cache_at_curr_diagonal = cache_at_curr_k,
                                                                            cache_two_diags_prior = previous_cache[1,...],
                                                                            seq_lens = seq_lens,
                                                                            joint_logprob_transit_mid_only = joint_logprob_transit_mid_only,
                                                                            C_transit = C_transit )
        cache_at_curr_k = update_cache(idx_arr_for_state = match_idx, 
                                       transit_message = match_transition_message, 
                                       cache_to_update = cache_at_curr_k) # (W, T, C*S, B)
        
        # ins
        # ins_idx is (C_trans,) 
        # ins_transition_message is (W, T, C_trans, B)
        ins_idx, ins_transition_message = get_ins_transition_message( align_cell_idxes = align_cell_idxes,
                                                                      pad_mask = pad_mask,
                                                                      cache_at_curr_diagonal = cache_at_curr_k,
                                                                      cache_for_prev_diagonal = previous_cache[0,...],
                                                                      seq_lens = seq_lens,
                                                                      joint_logprob_transit_mid_only = joint_logprob_transit_mid_only,
                                                                      C_transit = C_transit )
        cache_at_curr_k = update_cache(idx_arr_for_state = ins_idx, 
                                       transit_message = ins_transition_message, 
                                       cache_to_update = cache_at_curr_k) # (W, T, C*S, B)
        
        # del
        # del_idx is (C_trans,) 
        # del_transition_message is (W, T, C_trans, B)
        del_idx, del_transition_message = get_del_transition_message( align_cell_idxes = align_cell_idxes,
                                                                      pad_mask = pad_mask,
                                                                      cache_at_curr_diagonal = cache_at_curr_k,
                                                                      cache_for_prev_diagonal = previous_cache[0,...],
                                                                      seq_lens = seq_lens,
                                                                      joint_logprob_transit_mid_only = joint_logprob_transit_mid_only,
                                                                      C_transit = C_transit )
        cache_at_curr_k = update_cache(idx_arr_for_state = del_idx, 
                                       transit_message = del_transition_message,  
                                       cache_to_update = cache_at_curr_k) # (W, T, C*S, B)
        
        
        ### update with emissions
        # get emission tokens; at padding positions in diagonal, these will also be pad
        anc_toks_at_diag_k = unaligned_seqs[jnp.arange(B)[:, None], align_cell_idxes[...,0], 0] #(B, W)
        desc_toks_at_diag_k = unaligned_seqs[jnp.arange(B)[:, None], align_cell_idxes[...,1], 1] #(B, W)
        
        # use emissions to index scoring matrices
        #   at invalid positions, this is ZERO (not jnp.finfo(jnp.float32).min)!!!
        #   later, will add this to log-probability of transitions, so at invalid 
        #   positions, adding zero is the same as skipping the operation
        emit_logprobs_at_k = joint_loglike_emission_at_k_time_grid( anc_toks = anc_toks_at_diag_k,
                                                                    desc_toks = desc_toks_at_diag_k,
                                                                    joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                                                    logprob_emit_at_indel = logprob_emit_at_indel, 
                                                                    fill_invalid_pos_with = 0.0 ) # (W, T, C*S, B)
        cache_at_curr_k = cache_at_curr_k + emit_logprobs_at_k # (W, T, C*S, B)
        
        ### Final recordings, updates for next iteration
        # If not padding, then record the first cell of the cache; final 
        #   forward score will be here
        previous_first_cell_scores = jnp.where( pad_mask[:,0][None,None,:],
                                               cache_at_curr_k[0,...],
                                               previous_first_cell_scores ) #(T, C*S, B)
        
        # update cache
        # dim0 = 0 is k-1 (previous diagonal)
        # dim0 = 1 is k-2 (diagonal BEFORE previous diagonal)
        previous_cache = jnp.stack( [cache_at_curr_k, previous_cache[0,...]], axis=0 ) #(2, W, T, C*S, B)
        
        return {'cache_at_curr_k': cache_at_curr_k,
                'previous_first_cell_scores': previous_first_cell_scores,
                'previous_cache': previous_cache} 
    
    
    def test_this(self):
        W = self.W
        T = self.T
        C_S = self.C_S
        B = self.B
        C_transit = self.C_transit
        
        unaligned_seqs = self.unaligned_seqs
        seq_lens = self.seq_lens
        
        joint_logprob_emit_at_match = self.joint_logprob_emit_at_match
        logprob_emit_at_indel = self.logprob_emit_at_indel
        joint_logprob_transit = self.joint_logprob_transit
            

        ################################################
        ### Initialize cache for wavefront diagonals   #
        ################################################
        # \tau = state, M/I/D
        # \nu = class (unique to combination of domain+fragment)
        # alpha_{ij}^{s_d} = P(desc_{...j}, anc_{...i}, \tau=s, \nu=d | t)
        # dim0: 0=previous diagonal, 1=diag BEFORE previous diagonal
        alpha = jnp.full( (2, W, T, C_S, B), jnp.finfo(jnp.float32).min )
        
        ### fill diagonal k-2: alignment cells (1,0) and (0,1)
        alpha = init_first_diagonal( empty_cache = alpha, 
                                     unaligned_seqs = unaligned_seqs,
                                     joint_logprob_transit = joint_logprob_transit,
                                     logprob_emit_at_indel = logprob_emit_at_indel )  #(2, W, T, C_S, B)
        
        # check that cell (1,0) and (0,1) are consistent across all
        npt.assert_allclose( alpha[...,0], alpha[...,1])
        npt.assert_allclose( alpha[...,0], alpha[...,2])
        npt.assert_allclose( alpha[...,1], alpha[...,2])
        
        
        ### fill diag k-1: alignment cells (1,1), and (if applicable) (0,2) and/or (2,0)
        out = init_second_diagonal( cache_with_first_diag = alpha, 
                                    unaligned_seqs = unaligned_seqs,
                                    joint_logprob_transit = joint_logprob_transit,
                                    joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                    logprob_emit_at_indel = logprob_emit_at_indel,
                                    seq_lens = seq_lens ) 
        
        alpha = out[0] #(2, W, T, C_S, B)
        joint_logprob_transit_mid_only = out[1] #(T, C_S_prev, C_S_curr )
        del out 
        
        # check that cell (1,1) is consistent across all samples
        scores_from_cell_1_1 = alpha[0, [1, 0, 1], ..., jnp.arange(B)] #(B, T, C_S)
        
        npt.assert_allclose( scores_from_cell_1_1[0,...], scores_from_cell_1_1[1,...] )
        npt.assert_allclose( scores_from_cell_1_1[2,...], scores_from_cell_1_1[1,...] )
        npt.assert_allclose( scores_from_cell_1_1[0,...], scores_from_cell_1_1[2,...] )
        del scores_from_cell_1_1
        
        # for full and subseq2: check that entire second diagonal matches
        # this encompasses (2,0) and (0,2)
        second_diag_full_and_sub = alpha[0, ..., [0,2]] #(W, B, T, C_S)
        npt.assert_allclose( second_diag_full_and_sub[0,...], second_diag_full_and_sub[1,...] )
        del second_diag_full_and_sub
        
        
        ############################
        ### Continue through k=3   #
        ############################
        previous_cache = alpha #(2, W, T, C_S, B)
        previous_first_cell_scores = alpha[0,0,...] #(T, C_S, B)
        
        out = self._recursion(k=3,
                         previous_cache=previous_cache,
                         previous_first_cell_scores=previous_first_cell_scores,
                         joint_logprob_transit_mid_only=joint_logprob_transit_mid_only)
        
        cache_at_curr_k = out['cache_at_curr_k']
        previous_cache = out['previous_cache']
        previous_first_cell_scores = out['previous_first_cell_scores']
        
        ### for full and subseq2: check values at (2,1) and (1,2)
        cell_2_1 = cache_at_curr_k[[1, 0], ..., [0, 2]] #(B-1, T, C*S)
        npt.assert_allclose( cell_2_1[0,...], cell_2_1[1,...] )
        del cell_2_1
        
        cell_1_2 = cache_at_curr_k[[2, 1], ..., [0, 2]] #(B-1, T, C*S)
        npt.assert_allclose( cell_1_2[0,...], cell_1_2[1,...] )
        del cell_1_2
        
        
        ############################
        ### Continue through k=4   #
        ############################
        out = self._recursion(k=4,
                         previous_cache=previous_cache,
                         previous_first_cell_scores=previous_first_cell_scores,
                         joint_logprob_transit_mid_only=joint_logprob_transit_mid_only)
        
        cache_at_curr_k = out['cache_at_curr_k']
        
        
        ### for full and subseq2: check values at (2,2)
        cell_2_2 = cache_at_curr_k[[1, 0], ..., [0, 2]] #(B-1, T, C*S)
        npt.assert_allclose( cell_2_2[0,...], cell_2_2[1,...] )
        
if __name__ == '__main__':
    unittest.main()
