#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 13:47:05 2025

@author: annabel

This works through MARGINALIZING OVER A GRID OF TIMES; T!=B

This also assumes that ancestor and descendant produces an interesting 
  alignment; one has a length of at least 2, and the other has a length 
  of at least 1


sizes:
------
transition matrx: T, C_transit, C_transit, S_prev, S_curr
equilibrium distribution: C_transit, C_sites, A
  > after marginalizing over site-independent C_sites: C_transit, A
substitution emission matrix: T, C_transit, C_sites, K, A, A
  > after marginalizing over site-independent C_sites and K: T, C_transit, A, A


references:
-----------
lecture about wave-front parallelism
    https://snir.cs.illinois.edu/patterns/wavefront.pdf

(bonus) vectorized/rotated version of smith-waterman:
    https://github.com/spetti/SMURF/blob/main/sw_functions.py
    > this keeps the entire alignment matrix (because it's an aligner), but I
      only need JUST enough of the cache to accumulate sums
"""
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
import flax.linen as nn

import numpy as np
from itertools import product
from scipy.special import logsumexp
from tqdm import tqdm

from models.latent_class_mixtures.transition_models import TKF92TransitionLogprobs
from models.latent_class_mixtures.model_functions import joint_only_forward

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


###############################################################################
### Fake inputs   #############################################################
###############################################################################
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
joint_logprob_transit_old_dim_order = out[1]['joint'][:,0,...] #(T, C_transit_prev, C_transit_curr, S_prev, S_curr)
joint_logprob_transit = jnp.transpose(joint_logprob_transit_old_dim_order, (0,1,3,2,4)) # (T, C_transit_prev, S_prev, C_transit_curr, S_curr)
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


#################
### sequences   #
#################
### T -> TC: 5 possible alignment paths
unaligned_seqs = jnp.array( [[1, 1],
                    [6, 6],
                    [2, 4],
                    [0, 2],
                    [0, 0]] )[None,...]

# alignment 1:
# --T
# TC-
align1 = jnp.array( [[ 1,  1,  4],
                     [43,  6,  2],
                     [43,  4,  2],
                     [ 6, 43,  3],
                     [ 2,  2,  5]] )

# alignment 2:
# -T-
# T-C
align2 = jnp.array( [[ 1,  1,  4],
                     [43,  6,  2],
                     [ 6, 43,  3],
                     [43,  4,  2],
                     [ 2,  2,  5]] )

# alignment 3:
# T--
# -TC
align3 = jnp.array( [[ 1,  1,  4],
                     [ 6, 43,  3],
                     [43,  6,  2],
                     [43,  4,  2],
                     [ 2,  2,  5]] )

# alignment 4:
# T-
# TC
align4 = jnp.array( [[ 1,  1,  4],
                     [ 6,  6,  1],
                     [43,  4,  2],
                     [ 2,  2,  5],
                     [ 0,  0,  0]] )

# alignment 5:
# -T
# TC
align5 = jnp.array( [[ 1,  1,  4],
                     [43,  6,  2],
                     [ 6,  4,  1],
                     [ 2,  2,  5],
                     [ 0,  0,  0]] )

aligned_mats = jnp.stack( [align1, align2, align3, align4, align5], axis=0 ) #(num_possible_aligns, L_align, 3)
del align1, align2, align3, align4, align5

# widest diagonal for wavefront parallelism is min(anc_len, desc_len) + 1
seq_lens = (unaligned_seqs != 0).sum(axis=1)-2 #(B, 2)
seq_lens = seq_lens

min_lens = seq_lens.min(axis=1) #(B,)
W = min_lens.max() + 1 #float
del min_lens

# number of diagonals
K = (seq_lens.sum(axis=1)).max()


# ###############################################################################
# ### True scores of each alignment   ###########################################
# ###############################################################################
# compressed_joint_logprob_transit = jnp.squeeze(joint_logprob_transit) #(S_prev, S_to)
# compressed_joint_logprob_emit_at_match = jnp.squeeze(joint_logprob_emit_at_match) #(A, A)
# compressed_logprob_emit_at_indel = jnp.squeeze(logprob_emit_at_indel) #(A,)

# # T = 6
# # C = 4

# ### score each possible alignment by hand

# # --T
# # TC-
# # S -> ins T -> ins C -> del T -> E
# path1 = ( compressed_joint_logprob_transit[3,1] + compressed_logprob_emit_at_indel[6-3] +
#           compressed_joint_logprob_transit[1,1] + compressed_logprob_emit_at_indel[4-3] +
#           compressed_joint_logprob_transit[1,2] + compressed_logprob_emit_at_indel[6-3] +
#           compressed_joint_logprob_transit[2,3] )
         

# # -T-
# # T-C
# # S -> ins T -> del T -> ins C -> E
# path2 = ( compressed_joint_logprob_transit[3,1] + compressed_logprob_emit_at_indel[6-3] +
#           compressed_joint_logprob_transit[1,2] + compressed_logprob_emit_at_indel[6-3] +
#           compressed_joint_logprob_transit[2,1] + compressed_logprob_emit_at_indel[4-3] +
#           compressed_joint_logprob_transit[1,3] )

# # T--
# # -TC
# # S -> del T -> ins T -> ins C -> E
# path3 = ( compressed_joint_logprob_transit[3,2] + compressed_logprob_emit_at_indel[6-3] +
#           compressed_joint_logprob_transit[2,1] + compressed_logprob_emit_at_indel[6-3] +
#           compressed_joint_logprob_transit[1,1] + compressed_logprob_emit_at_indel[4-3] +
#           compressed_joint_logprob_transit[1,3] )

# # T-
# # TC
# # S -> match (T,T) -> ins C -> E
# path4 = ( compressed_joint_logprob_transit[3,0] + compressed_joint_logprob_emit_at_match[6-3, 6-3] +
#           compressed_joint_logprob_transit[0,1] + compressed_logprob_emit_at_indel[4-3] +
#           compressed_joint_logprob_transit[1,3] )

# # -T
# # TC
# # S -> ins T -> match (T, C) -> E
# path5 = ( compressed_joint_logprob_transit[3,1] + compressed_logprob_emit_at_indel[6-3]+
#           compressed_joint_logprob_transit[1,0] + compressed_joint_logprob_emit_at_match[6-3, 4-3] +
#           compressed_joint_logprob_transit[0,3] )


# ### check each with regular 1D forward
# score_per_align = joint_only_forward(aligned_inputs = aligned_mats,
#                           joint_logprob_emit_at_match = joint_logprob_emit_at_match,
#                           logprob_emit_at_indel = logprob_emit_at_indel,
#                           joint_logprob_transit = joint_logprob_transit_old_dim_order,
#                           unique_time_per_sample = False,
#                           return_all_intermeds = False) #(T, num_alignments)

# assert np.allclose( np.array([path1, path2, path3, path4, path5]),
#                     np.squeeze(score_per_align) )

# true_sum_over_aligns = logsumexp(np.array([path1, path2, path3, path4, path5]))

# del path1, path2, path3, path4, path5




###############################################################################
### manually check each cell by hand :(   #####################################
###############################################################################
compressed_joint_logprob_transit = jnp.squeeze(joint_logprob_transit) #(S_prev, S_to)
compressed_joint_logprob_emit_at_match = jnp.squeeze(joint_logprob_emit_at_match) #(A, A)
compressed_logprob_emit_at_indel = jnp.squeeze(logprob_emit_at_indel) #(A,)
# T = 6
# C = 4

#######################
### cell (1,0); K=1   #
#######################
# T
# -
# S -> del T
cell_1_0 = compressed_joint_logprob_transit[3,2] + compressed_logprob_emit_at_indel[6-3]
print(f'cell_1_0: {cell_1_0}')


#######################
### cell (0,1); K=1   #
#######################
# -
# T
# S -> ins T
cell_0_1 = compressed_joint_logprob_transit[3,1] + compressed_logprob_emit_at_indel[6-3]
print(f'cell_0_1: {cell_0_1}')


#######################
### cell (1,1); K=2   #
#######################
# T -
# - T
# S -> del T -> ins T
path1 = ( compressed_joint_logprob_transit[3,2] + compressed_logprob_emit_at_indel[6-3] +
          compressed_joint_logprob_transit[2,1] + compressed_logprob_emit_at_indel[6-3] )

# - T
# T -
# S -> ins T -> del T
path2 = ( compressed_joint_logprob_transit[3,1] + compressed_logprob_emit_at_indel[6-3] +
          compressed_joint_logprob_transit[1,2] + compressed_logprob_emit_at_indel[6-3] )

# T
# T
# S -> match (T,T)
path3 = compressed_joint_logprob_transit[3,0] + compressed_joint_logprob_emit_at_match[6-3, 6-3]

cell_1_1 = logsumexp( [path1, path2, path3] )
print(f'cell_1_1: {cell_1_1}')
del path1, path2, path3


#######################
### cell (0,2); K=2   #
#######################
# - -
# T C
# S -> ins T -> ins C
cell_0_2 = ( compressed_joint_logprob_transit[3,1] + compressed_logprob_emit_at_indel[6-3] +
             compressed_joint_logprob_transit[1,1] + compressed_logprob_emit_at_indel[4-3])
print(f'cell_0_2: {cell_0_2}')


#######################
### cell (1,2); K=3   #
#######################
# --T
# TC-
# S -> ins T -> ins C -> del T 
path1 = ( compressed_joint_logprob_transit[3,1] + compressed_logprob_emit_at_indel[6-3] +
          compressed_joint_logprob_transit[1,1] + compressed_logprob_emit_at_indel[4-3] +
          compressed_joint_logprob_transit[1,2] + compressed_logprob_emit_at_indel[6-3] )
         

# -T-
# T-C
# S -> ins T -> del T -> ins C 
path2 = ( compressed_joint_logprob_transit[3,1] + compressed_logprob_emit_at_indel[6-3] +
          compressed_joint_logprob_transit[1,2] + compressed_logprob_emit_at_indel[6-3] +
          compressed_joint_logprob_transit[2,1] + compressed_logprob_emit_at_indel[4-3] )

# T--
# -TC
# S -> del T -> ins T -> ins C 
path3 = ( compressed_joint_logprob_transit[3,2] + compressed_logprob_emit_at_indel[6-3] +
          compressed_joint_logprob_transit[2,1] + compressed_logprob_emit_at_indel[6-3] +
          compressed_joint_logprob_transit[1,1] + compressed_logprob_emit_at_indel[4-3] )

# T-
# TC
# S -> match (T,T) -> ins C 
path4 = ( compressed_joint_logprob_transit[3,0] + compressed_joint_logprob_emit_at_match[6-3, 6-3] +
          compressed_joint_logprob_transit[0,1] + compressed_logprob_emit_at_indel[4-3] )

# -T
# TC
# S -> ins T -> match (T, C) 
path5 = ( compressed_joint_logprob_transit[3,1] + compressed_logprob_emit_at_indel[6-3]+
          compressed_joint_logprob_transit[1,0] + compressed_joint_logprob_emit_at_match[6-3, 4-3] )

cell_1_2 = logsumexp( [path1, path2, path3, path4, path5] )
del path1, path2, path3

print(f'cell_1_2: {cell_1_2}')











# #%%
# ###############################################################################
# ### 2D forward algo   #########################################################
# ###############################################################################
 

# ################################################
# ### Initialize cache for wavefront diagonals   #
# ################################################
# # \tau = state, M/I/D
# # \nu = class (unique to combination of domain+fragment)
# # alpha_{ij}^{s_d} = P(desc_{...j}, anc_{...i}, \tau=s, \nu=d | t)
# # dim0: 0=previous diagonal, 1=diag BEFORE previous diagonal
# alpha = jnp.full( (2, W, T, C_S, B), jnp.finfo(jnp.float32).min )

# # fill diagonal k-2: alignment cells (1,0) and (0,1)
# alpha = init_first_diagonal( empty_cache = alpha, 
#                              unaligned_seqs = unaligned_seqs,
#                              joint_logprob_transit = joint_logprob_transit,
#                              logprob_emit_at_indel = logprob_emit_at_indel )  #(2, W, T, C_S, B)

# # fill diag k-1: alignment cells (1,1), and (if applicable) (0,2) and/or (2,0)
# out = init_second_diagonal( cache_with_first_diag = alpha, 
#                             unaligned_seqs = unaligned_seqs,
#                             joint_logprob_transit = joint_logprob_transit,
#                             joint_logprob_emit_at_match = joint_logprob_emit_at_match,
#                             logprob_emit_at_indel = logprob_emit_at_indel,
#                             seq_lens = seq_lens ) 

# alpha = out[0] #(2, W, T, C_S, B)
# joint_logprob_transit_mid_only = out[1] #(T, C_S_prev, C_S_curr )
# del out 


# ########################
# ### Start Recurrence   #
# ########################
# previous_cache = alpha #(2, W, T, C_S, B)
# previous_first_cell_scores = alpha[0,0,...] #(T, C_S, B)
# for k in range(3, K+1):
#     # blank cache to fill
#     cache_at_curr_k = jnp.full( (W, T, C_S, B), jnp.finfo(jnp.float32).min ) # (W, T, C*S, B)
    
#     # align_cell_idxes is (B, W, 2)
#     # pad_mask is (B, W)
#     align_cell_idxes, pad_mask = generate_ij_coords_at_diagonal_k(seq_lens = seq_lens,
#                                                                   diagonal_k = k,
#                                                                   widest_diag_W = W)
    
    
#     ### update with transitions
#     # match
#     # match_idx is (C_trans,) 
#     # match_transition_message is (W, T, C_trans, B)
#     match_idx, match_transition_message = get_match_transition_message( align_cell_idxes = align_cell_idxes,
#                                                                         pad_mask = pad_mask,
#                                                                         cache_at_curr_diagonal = cache_at_curr_k,
#                                                                         cache_two_diags_prior = previous_cache[1,...],
#                                                                         seq_lens = seq_lens,
#                                                                         joint_logprob_transit_mid_only = joint_logprob_transit_mid_only,
#                                                                         C_transit = C_transit )
#     cache_at_curr_k = update_cache(idx_arr_for_state = match_idx, 
#                                    transit_message = match_transition_message, 
#                                    cache_to_update = cache_at_curr_k) # (W, T, C*S, B)
    
#     # ins
#     # ins_idx is (C_trans,) 
#     # ins_transition_message is (W, T, C_trans, B)
#     ins_idx, ins_transition_message = get_ins_transition_message( align_cell_idxes = align_cell_idxes,
#                                                                   pad_mask = pad_mask,
#                                                                   cache_at_curr_diagonal = cache_at_curr_k,
#                                                                   cache_for_prev_diagonal = previous_cache[0,...],
#                                                                   seq_lens = seq_lens,
#                                                                   joint_logprob_transit_mid_only = joint_logprob_transit_mid_only,
#                                                                   C_transit = C_transit )
#     cache_at_curr_k = update_cache(idx_arr_for_state = ins_idx, 
#                                    transit_message = ins_transition_message, 
#                                    cache_to_update = cache_at_curr_k) # (W, T, C*S, B)
    
#     # del
#     # del_idx is (C_trans,) 
#     # del_transition_message is (W, T, C_trans, B)
#     del_idx, del_transition_message = get_del_transition_message( align_cell_idxes = align_cell_idxes,
#                                                                   pad_mask = pad_mask,
#                                                                   cache_at_curr_diagonal = cache_at_curr_k,
#                                                                   cache_for_prev_diagonal = previous_cache[0,...],
#                                                                   seq_lens = seq_lens,
#                                                                   joint_logprob_transit_mid_only = joint_logprob_transit_mid_only,
#                                                                   C_transit = C_transit )
#     cache_at_curr_k = update_cache(idx_arr_for_state = del_idx, 
#                                    transit_message = del_transition_message,  
#                                    cache_to_update = cache_at_curr_k) # (W, T, C*S, B)
    
    
#     ### update with emissions
#     # get emission tokens; at padding positions in diagonal, these will also be pad
#     anc_toks_at_diag_k = unaligned_seqs[jnp.arange(B)[:, None], align_cell_idxes[...,0], 0] #(B, W)
#     desc_toks_at_diag_k = unaligned_seqs[jnp.arange(B)[:, None], align_cell_idxes[...,1], 1] #(B, W)
    
#     # use emissions to index scoring matrices
#     #   at invalid positions, this is ZERO (not jnp.finfo(jnp.float32).min)!!!
#     #   later, will add this to log-probability of transitions, so at invalid 
#     #   positions, adding zero is the same as skipping the operation
#     emit_logprobs_at_k = joint_loglike_emission_at_k_time_grid( anc_toks = anc_toks_at_diag_k,
#                                                                 desc_toks = desc_toks_at_diag_k,
#                                                                 joint_logprob_emit_at_match = joint_logprob_emit_at_match,
#                                                                 logprob_emit_at_indel = logprob_emit_at_indel, 
#                                                                 fill_invalid_pos_with = 0.0 ) # (W, T, C*S, B)
#     cache_at_curr_k = cache_at_curr_k + emit_logprobs_at_k # (W, T, C*S, B)
    
    
#     ### Final recordings, updates for next iteration
#     # If not padding, then record the first cell of the cache; final 
#     #   forward score will always be here
#     previous_first_cell_scores = jnp.where( pad_mask[:,0][None,None,:],
#                                            cache_at_curr_k[0,...],
#                                            previous_first_cell_scores ) #(T, C*S, B)
    
#     # update cache
#     # dim0 = 0 is k-1 (previous diagonal)
#     # dim0 = 1 is k-2 (diagonal BEFORE previous diagonal)
#     previous_cache = jnp.stack( [cache_at_curr_k, previous_cache[0,...]], axis=0 ) #(2, W, T, C*S, B)


# ##################################################
# ### Terminate all by multiplying by any -> end   #
# ##################################################
# # joint_logprob_transit is (T, C_transit_prev, S_prev, C_transit_curr, S_curr)
# mid_to_end = joint_logprob_transit[:, :, :3, -1, -1] #(T, C_transit_prev, (S-1)_prev)
# mid_to_end = jnp.reshape(mid_to_end, (T, C_transit*(S-1) ) ) #(T, C_S)
# pred = nn.logsumexp( mid_to_end[...,None] + previous_first_cell_scores, axis=1 ) #(T, B)

# print()
# print(f'PRED: {pred}')
# print(f'TRUE: {true}')

