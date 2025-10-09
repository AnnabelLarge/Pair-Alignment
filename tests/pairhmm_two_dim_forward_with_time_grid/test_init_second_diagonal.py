#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 18:58:08 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
import flax.linen as nn

import numpy as np
import numpy.testing as npt
import unittest

from models.latent_class_mixtures.transition_models import (TKF92TransitionLogprobs)
from models.latent_class_mixtures.two_dim_forward_algo_helpers import (generate_ij_coords_at_diagonal_k,
                                                                               ij_coords_to_wavefront_pos_at_diagonal_k,
                                                                               index_all_classes_one_state,
                                                                               wavefront_cache_lookup,
                                                                               compute_forward_messages_for_state,
                                                                               joint_loglike_emission_at_k_time_grid,
                                                                               init_first_diagonal,
                                                                               init_second_diagonal)


class TestInitSecondDiagonal(unittest.TestCase):
    def setUp(self):
        # dims
        C_transit = 1
        A = 20
        S = 4
        C_S = C_transit * (S-1) #use this for forward algo carry
        
        # time
        t_array = jnp.array( [1.0] )
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
        # has all cells
        seqs1 = jnp.array( [[1, 1],
                            [3, 3],
                            [3, 3],
                            [5, 2],
                            [2, 0],
                            [0, 0]] )
        
        # has all cells
        seqs2 = jnp.array( [[1, 1],
                            [6, 6],
                            [3, 3],
                            [2, 4],
                            [0, 2],
                            [0, 0]] )
        
        # has (1,1)
        seqs3 = jnp.array( [[1, 1],
                            [6, 6],
                            [2, 2],
                            [0, 0],
                            [0, 0],
                            [0, 0]] )
        
        # has (1,1) and (0,2)
        seqs4 = jnp.array( [[1, 1],
                            [6, 6],
                            [2, 3],
                            [0, 2],
                            [0, 0],
                            [0, 0]] )
        
        # has (2,0) and (1,1)
        seqs5 = jnp.array( [[1, 1],
                            [6, 6],
                            [3, 2],
                            [2, 0],
                            [0, 0],
                            [0, 0]] )
        
        # concat
        unaligned_seqs = jnp.stack([seqs1, seqs2, seqs3, seqs4, seqs5], axis=0) #(B, L_seq, 2)
        
        # extra dims
        B = unaligned_seqs.shape[0]
        
        # widest diagonal for wavefront parallelism is min(anc_len, desc_len) + 1
        seq_lens = (unaligned_seqs != 0).sum(axis=1)-2 #(B, 2)
        min_lens = seq_lens.min(axis=1) #(B,)
        W = min_lens.max() + 1 #float
        
        
        ### some samples will only have subset of cells in second diagonal
        # samples to include for cell-specific tests
        samples_with_cell_2_0 = [0,1,4]
        samples_with_cell_0_2 = [0,1,3]
        
        # locations of cells, in order of samples
        idx_of_cell_1_1 = [1, 1, 0, 0, 1]
        idx_of_cell_0_2 = [2, 2, 1]
        

        ################################################
        ### Initialize cache for wavefront diagonals   #
        ################################################
        # fill diagonal k-2: alignment cells (1,0) and (0,1)
        first_diag = init_first_diagonal( cache_size = (W, T, C_S, B), 
                                          unaligned_seqs = unaligned_seqs,
                                          joint_logprob_transit = joint_logprob_transit,
                                          logprob_emit_at_indel = logprob_emit_at_indel ) #(2, W, T, C_S, B)
        
        # fill diag k-1: alignment cells (2,0), (1,1), and (0,2)
        out = init_second_diagonal( cache_for_prev_diagonal = first_diag, 
                                         unaligned_seqs = unaligned_seqs,
                                         joint_logprob_transit = joint_logprob_transit,
                                         joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                         logprob_emit_at_indel = logprob_emit_at_indel,
                                         seq_lens = seq_lens ) 
        
        # second diag is (W, T, C_S, B)
        # joint_logprob_transit_mid_only is (T, C_S, C_S)
        second_diag, joint_logprob_transit_mid_only = out
        del out

        
        ### make attributes
        self.T = T
        self.C_transit = C_transit
        self.A = A
        self.S = S
        self.C_S = C_S
        self.B = B
        self.W = W
        
        self.first_diag = first_diag
        self.second_diag = second_diag 
        self.unaligned_seqs = unaligned_seqs
        self.joint_logprob_transit = joint_logprob_transit
        self.joint_logprob_transit_mid_only = joint_logprob_transit_mid_only
        self.logprob_emit_at_indel = logprob_emit_at_indel
        self.joint_logprob_emit_at_match = joint_logprob_emit_at_match
        
        self.samples_with_cell_2_0 = samples_with_cell_2_0
        self.samples_with_cell_0_2  = samples_with_cell_0_2
        self.idx_of_cell_1_1 = idx_of_cell_1_1
        self.idx_of_cell_0_2 = idx_of_cell_0_2


    def test_shape(self):
        second_diag = self.second_diag
        W = self.W
        T = self.T
        C_S = self.C_S
        B = self.B
        
        npt.assert_allclose( second_diag.shape, (W, T, C_S, B) )
    
    def test_cell_2_0(self):
        first_diag = self.first_diag
        second_diag = self.second_diag
        W = self.W
        T = self.T
        C_transit = self.C_transit
        C_S = self.C_S
        sample_idxes = self.samples_with_cell_2_0
        unaligned_seqs = self.unaligned_seqs[sample_idxes,...]
        joint_logprob_transit = self.joint_logprob_transit_mid_only
        logprob_emit_at_indel = self.logprob_emit_at_indel
        
        cell_2_0 = second_diag[0, ..., sample_idxes] # (B, T, C_S)
        cell_2_0 = jnp.transpose(cell_2_0, (1,2,0) ) #(T, C_S, B)
        B = unaligned_seqs.shape[0]
        
        # check shape
        npt.assert_allclose( cell_2_0.shape, (T, C_S, B) )
        
        # cell (2,0) should only have values for delete
        del_idx = index_all_classes_one_state( state_idx = 2,
                                                num_transit_classes = C_transit )
        del_idx = set( np.array( del_idx ) ) #(C_transit)
        assert len(del_idx) == C_transit
        
        for t_idx in range(T):
            for b in range(B):
                for c_s_curr in range(C_S):
                    cell_value = cell_2_0[t_idx, c_s_curr, b]
                    
                    # cell (2,0) should only have values for del
                    if c_s_curr not in del_idx:
                        npt.assert_allclose( cell_value, jnp.finfo(jnp.float32).min )
                    
                    elif c_s_curr in del_idx:
                        # transitions
                        prob_space_c_s_sum = 0
                        for c_s_prev in range(C_S):
                            # alpha_{1,0}^{c_s_prev}
                            cache = first_diag[0, t_idx, c_s_prev, b]
                            
                            # logP(c_s_curr | c_s_prev, t)
                            logprob_to_del = joint_logprob_transit[t_idx, c_s_prev, c_s_curr]
                            
                            # alpha_{1,0}^{c_s_prev} + logP(c_s_curr | c_s_prev, t),
                            #  but do in probability space
                            prob_space_c_s_sum += np.exp( cache + logprob_to_del )
                        
                        # \sum_{c_s_prev} np.exp(alpha_{1,0}^{c_s_prev}) * P(c_s_curr | c_s_prev, t),
                        #   but transform to log space
                        c_s_lse = np.log(prob_space_c_s_sum)
                        del prob_space_c_s_sum, c_s_prev, cache, logprob_to_del
                        
                        # add logprob emissions Em(x_2); since delete site,
                        #   use ancestor token at position 2
                        anc_tok = unaligned_seqs[b, 2, 0]
                        c_curr = c_s_curr // 3
                        logprob_anc_tok = logprob_emit_at_indel[c_curr, anc_tok-3]
                        true_val = logprob_anc_tok + c_s_lse
                        npt.assert_allclose(true_val, cell_value) 
    
    
    def test_cell_0_2(self):
        first_diag = self.first_diag
        second_diag = self.second_diag
        W = self.W
        T = self.T
        C_transit = self.C_transit
        C_S = self.C_S
        sample_idxes = self.samples_with_cell_0_2
        unaligned_seqs = self.unaligned_seqs[sample_idxes,...]
        joint_logprob_transit = self.joint_logprob_transit_mid_only
        logprob_emit_at_indel = self.logprob_emit_at_indel
        idx_of_cell_0_2 = self.idx_of_cell_0_2
        
        cell_0_2 = second_diag[idx_of_cell_0_2, ..., sample_idxes] # (B, T, C_S)
        cell_0_2 = jnp.transpose(cell_0_2, (1,2,0) ) #(T, C_S, B)
        B = unaligned_seqs.shape[0]
        
        # check shape
        npt.assert_allclose( cell_0_2.shape, (T, C_S, B) )
        
        # cell (0,2) should only have values for ins
        ins_idx = index_all_classes_one_state( state_idx = 1,
                                                num_transit_classes = C_transit )
        ins_idx = set( np.array( ins_idx ) ) #(C_transit)
        assert len(ins_idx) == C_transit
        
        for t_idx in range(T):
            for b in range(B):
                for c_s_curr in range(C_S):
                    cell_value = cell_0_2[t_idx, c_s_curr, b]
                    
                    # cell (0,2) should only have values for ins
                    if c_s_curr not in ins_idx:
                        npt.assert_allclose( cell_value, jnp.finfo(jnp.float32).min )
                    
                    elif c_s_curr in ins_idx:
                        # transitions
                        prob_space_c_s_sum = 0
                        for c_s_prev in range(C_S):
                            # alpha_{0,1}^{c_s_prev}
                            cache = first_diag[1, t_idx, c_s_prev, b]
                            
                            # logP(c_s_curr | c_s_prev, t)
                            logprob_to_ins = joint_logprob_transit[t_idx, c_s_prev, c_s_curr]
                            
                            # alpha_{0,1}^{c_s_prev} + logP(c_s_curr | c_s_prev, t),
                            #  but do in probability space
                            prob_space_c_s_sum += np.exp( cache + logprob_to_ins )
                        
                        # \sum_{c_s_prev} np.exp(alpha_{0,1}^{c_s_prev}) * P(c_s_curr | c_s_prev, t),
                        #   but transform to log space
                        c_s_lse = np.log(prob_space_c_s_sum)
                        del prob_space_c_s_sum, c_s_prev, cache, logprob_to_ins
                        
                        # add logprob emissions Em(y_2); since ins site,
                        #   use descendant token at position 2
                        desc_tok = unaligned_seqs[b, 2, 1]
                        c_curr = c_s_curr // 3
                        logprob_desc_tok = logprob_emit_at_indel[c_curr, desc_tok-3]
                        true_val = logprob_desc_tok + c_s_lse
                        npt.assert_allclose(true_val, cell_value)
    
    
    def test_cell_1_1(self):
        first_diag = self.first_diag
        second_diag = self.second_diag
        W = self.W
        T = self.T
        C_transit = self.C_transit
        C_S = self.C_S
        B = self.B
        unaligned_seqs = self.unaligned_seqs
        idx_of_cell_1_1 = self.idx_of_cell_1_1
        joint_logprob_transit = self.joint_logprob_transit
        joint_logprob_transit_mid_only = self.joint_logprob_transit_mid_only
        joint_logprob_emit_at_match = self.joint_logprob_emit_at_match
        logprob_emit_at_indel = self.logprob_emit_at_indel
        
        cell_1_1 = second_diag[idx_of_cell_1_1, :, :, jnp.arange(B)]  #(B, T, C_S)
        cell_1_1 = jnp.transpose(cell_1_1, (1,2,0) ) #(T, C_S, B)

        # check shape
        npt.assert_allclose( cell_1_1.shape, (T, C_S, B) )

        for t_idx in range(T):
            for c_s in range(C_S):
                s = c_s % 3  
                c = c_s // 3
                for b in range(B):
                    cell_value = cell_1_1[t_idx, c_s, b]
                    
                    ### if cell is match, calculate start -> match
                    if s == 0:
                        logprob_start_to_match = joint_logprob_transit[t_idx, 0, -1, c, 0]
                        anc_tok = unaligned_seqs[b, 1, 0]
                        desc_tok = unaligned_seqs[b, 1, 1]
                        logprob_anc_desc_at_match = joint_logprob_emit_at_match[t_idx, c, anc_tok-3, desc_tok-3]
                        true_val = logprob_start_to_match + logprob_anc_desc_at_match
                        npt.assert_allclose(true_val, cell_value)
                        
                
                    ### if cell is ins, use cache at (1,0)
                    elif s == 1:
                        # transitions
                        prob_space_c_s_sum = 0
                        for c_s_prev in range(C_S):
                            # alpha_{1,0}^{c_s_prev}
                            cache = first_diag[0, t_idx, c_s_prev, b]
                            
                            # logP(c_s | c_s_prev, t)
                            logprob_to_ins = joint_logprob_transit_mid_only[t_idx, c_s_prev, c_s]
                            
                            # alpha_{1,0}^{c_s_prev} + logP(c_s | c_s_prev, t),
                            #  but do in probability space
                            prob_space_c_s_sum += np.exp( cache + logprob_to_ins )
                        
                        # \sum_{c_s_prev} np.exp(alpha_{1,0}^{c_s_prev}) * P(c_s | c_s_prev, t),
                        #   but transform to log space
                        c_s_lse = np.log(prob_space_c_s_sum)
                        del prob_space_c_s_sum, c_s_prev, cache, logprob_to_ins
                        
                        # add logprob emissions Em(y_1); since ins site,
                        #   use descendant token at position 1
                        desc_tok = unaligned_seqs[b, 1, 1]
                        logprob_desc_tok = logprob_emit_at_indel[c, desc_tok-3]
                        true_val = logprob_desc_tok + c_s_lse
                        npt.assert_allclose(true_val, cell_value)
                    
                    
                    ### if cell is del, use cache at (0, 1)
                    elif s == 2:
                        # transitions
                        prob_space_c_s_sum = 0
                        for c_s_prev in range(C_S):
                            # alpha_{0,1}^{c_s_prev}
                            cache = first_diag[1, t_idx, c_s_prev, b]
                            
                            # logP(c_s | c_s_prev, t)
                            logprob_to_del = joint_logprob_transit_mid_only[t_idx, c_s_prev, c_s]
                            
                            # alpha_{0,1}^{c_s_prev} + logP(c_s | c_s_prev, t),
                            #  but do in probability space
                            prob_space_c_s_sum += np.exp( cache + logprob_to_del )
                        
                        # \sum_{c_s_prev} np.exp(alpha_{0,1}^{c_s_prev}) * P(c_s | c_s_prev, t),
                        #   but transform to log space
                        c_s_lse = np.log(prob_space_c_s_sum)
                        del prob_space_c_s_sum, c_s_prev, cache, logprob_to_del
                        
                        # add logprob emissions Em(x_1); since delete site,
                        #   use ancestor token at position 1
                        anc_tok = unaligned_seqs[b, 1, 0]
                        logprob_anc_tok = logprob_emit_at_indel[c, anc_tok-3]
                        true_val = logprob_anc_tok + c_s_lse
                        npt.assert_allclose(true_val, cell_value)
    
        
if __name__ == '__main__':
    unittest.main()