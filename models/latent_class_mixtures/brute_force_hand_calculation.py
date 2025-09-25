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



###############################################################################
### manually check each cell by hand   ########################################
###############################################################################
# T encoded as: 6
# C encoded as: 4

# scoring matrices, simplified
compressed_joint_logprob_transit = jnp.squeeze(joint_logprob_transit) #(S_prev, S_to)
compressed_joint_logprob_emit_at_match = jnp.squeeze(joint_logprob_emit_at_match) #(A, A)
compressed_logprob_emit_at_indel = jnp.squeeze(logprob_emit_at_indel) #(A,)


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

print(f'cell_1_2: {cell_1_2}')


###############################################################################
### True scores of each alignment   ###########################################
###############################################################################
### check final score per path, with regular 1D forward
score_per_align = joint_only_forward(aligned_inputs = aligned_mats,
                          joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                          logprob_emit_at_indel = logprob_emit_at_indel,
                          joint_logprob_transit = joint_logprob_transit_old_dim_order,
                          unique_time_per_sample = False,
                          return_all_intermeds = False) #(T, num_alignments)

path1_with_eos = path1 + compressed_joint_logprob_transit[2,3]
path2_with_eos = path2 + compressed_joint_logprob_transit[1,3]
path3_with_eos = path3 + compressed_joint_logprob_transit[1,3]
path4_with_eos = path4 + compressed_joint_logprob_transit[1,3]
path5_with_eos = path5 + compressed_joint_logprob_transit[0,3]

assert np.allclose( np.array([path1_with_eos, path2_with_eos, path3_with_eos, path4_with_eos, path5_with_eos]),
                    np.squeeze(score_per_align) )

true_sum_over_aligns = logsumexp(np.array([path1_with_eos, path2_with_eos, path3_with_eos, path4_with_eos, path5_with_eos]))
print(f'True sum: {true_sum_over_aligns}')
del path1, path2, path3, path4, path5
