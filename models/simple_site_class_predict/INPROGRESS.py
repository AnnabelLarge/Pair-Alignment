#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 13:47:05 2025

@author: annabel

sizes:
------
transition matrx: T, C_trans, C_trans, S_from, S_to
equilibrium distribution: C_trans, C_sites, A
  > after marginalizing over site-independent C_sites: C_trans, A
substitution emission matrix: T, C_trans, C_sites, K, A, A
  > after marginalizing over site-independent C_sites and K: T, C_trans, A, A


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
from jax import numpy as jnp
import flax.linen as nn

import numpy as np
from itertools import product
from scipy.special import logsumexp
from tqdm import tqdm

from models.simple_site_class_predict.transition_models import (TKF92TransitionLogprobs)


###############################################################################
### Fake inputs   #############################################################
###############################################################################
# dims
T = 1
C_trans = 2
A = 20
S = 4
C_S = C_trans * (S-1) #use this for forward algo carry

# time
t_array = jnp.array( [1.0] )


########################
### scoring matrices   #
########################
# use real model object for transition matrix
transit_model = TKF92TransitionLogprobs( config={'tkf_function': 'regular_tkf',
                                                 'num_domain_mixtures': 1,
                                                 'num_fragment_mixtures': C_trans},
                                         name = 'transit_mat' )
init_params = transit_model.init( rngs = jax.random.key(0),
                                  t_array = t_array,
                                  return_all_matrices = False,
                                  sow_intermediates = False )

out = transit_model.apply( variables = init_params,
                           t_array = t_array,
                           return_all_matrices = False,
                           sow_intermediates = False )
joint_logprob_transit = out[1]['joint'][:,0,...] #(T, C_trans_from, C_trans_to, S_from, S_to)
del transit_model, init_params, out

# use dummy scoring matrices for emissions
sub_emit_logits = jax.random.normal( key = jax.random.key(0),
                                     shape = (T, C_trans, A, A) ) #(T, C_trans, A, A)
joint_logprob_emit_at_match = nn.log_softmax(sub_emit_logits, axis=(-1,-2)) #(T, C_trans, A, A)
del sub_emit_logits

indel_emit_logits = jax.random.normal( key = jax.random.key(0),
                                     shape = (C_trans, A) ) #(C_trans, A)
logprob_emit_at_indel = nn.log_softmax(indel_emit_logits, axis=-1) #(C_trans, A)
del indel_emit_logits


#################
### sequences   #
#################
### AG -> A: 5 possible alignment paths
seqs1 = jnp.array( [[1, 1],
                    [3, 3],
                    [5, 2],
                    [2, 0],
                    [0, 0]] )

# alignment 1:
# AG-
# --A
align1 = jnp.array( [[ 1,  1,  4],
                     [ 3, 43,  3],
                     [ 5, 43,  3],
                     [43,  3,  2],
                     [ 2,  2,  5]] )

# alignment 2:
# A-G
# -A-
align2 = jnp.array( [[ 1,  1,  4],
                     [ 3, 43,  3],
                     [43,  3,  2],
                     [ 5, 43,  3],
                     [ 2,  2,  5]] )

# alignment 3:
# -AG
# A--
align3 = jnp.array( [[ 1,  1,  4],
                     [43,  3,  2],
                     [ 3, 43,  3],
                     [ 5, 43,  3],
                     [ 2,  2,  5]] )

# alignment 4:
# AG
# A-
align4 = jnp.array( [[ 1,  1,  4],
                     [ 3,  3,  1],
                     [ 5, 43,  3],
                     [ 2,  2,  5],
                     [ 0,  0,  0]] )

# alignment 5:
# AG
# -A
align5 = jnp.array( [[ 1,  1,  4],
                     [ 3, 43,  3],
                     [ 5,  3,  1],
                     [ 2,  2,  5],
                     [ 0,  0,  0]] )

aligned_mats1 = jnp.stack( [align1, align2, align3, align4, align5], axis=0 ) #(num_possible_aligns, L_align, 3)
del align1, align2, align3, align4, align5


### T -> TC: 5 possible alignment paths
seqs2 = jnp.array( [[1, 1],
                    [6, 6],
                    [2, 4],
                    [0, 2],
                    [0, 0]] )

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

aligned_mats2 = jnp.stack( [align1, align2, align3, align4, align5], axis=0 ) #(num_possible_aligns, L_align, 3)
del align1, align2, align3, align4, align5


unaligned_seqs = jnp.stack([seqs1, seqs2], axis=0) #(B, L_seq, 2)
del seqs1, seqs2

B = unaligned_seqs.shape[0]
L_seq = unaligned_seqs.shape[1]

# widest diagonal for wavefront parallelism is min(anc_len, desc_len) + 1
seq_lens = (unaligned_seqs != 0).sum(axis=1) #(B, 2)
min_lens = seq_lens.min(axis=1) #(B,)
W = min_lens.max() + 1 #float
del min_lens
                

###############################################################################
### True scores from manual enumeration   #####################################
###############################################################################
def marginalize_state_and_class(aligns):
    num_aligns = aligns.shape[0]
    L_align = (aligns[...,-1] != 0).sum(axis=-1).max().item()
    score_per_alignment = np.zeros( (T, num_aligns) ) #(T, num_aligns)
    for t in range(T):
        for b in tqdm( range(num_aligns) ):
            sample_seq = aligns[b,:,:]
            
            # all possible path combinations
            invalid_toks = np.array([0,1,2])
            n = (  ~np.isin(sample_seq[:, 0], invalid_toks) ).sum()
            paths = [list(p) for p in product(range(C_trans), repeat= int(n) )]
            
            # manually score each possible path
            score_per_path = []
            for path in paths:
                to_pad = L_align - (len(path)+1)
                path = [-999] + path + [-999]*to_pad
                path_logprob = 0
                prev_state = sample_seq[0, -1]
                for l in range(1,L_align):
                    prev_site_class = path[l-1]
                    curr_site_class = path[l]
                    anc_tok, desc_tok, curr_state = sample_seq[l,:]
                    
                    if curr_state == 0:
                        break
                    
                    curr_state = np.where(curr_state != 5, curr_state, 4)
                    
                    ### emissions
                    e = 0
                    
                    if curr_state == 1:
                        e = joint_logprob_emit_at_match[t, curr_site_class, anc_tok - 3, desc_tok - 3]
                    
                    elif curr_state == 2:
                        e = logprob_emit_at_indel[curr_site_class, desc_tok-3]
                    
                    elif curr_state == 3:
                        e = logprob_emit_at_indel[curr_site_class, anc_tok-3]
                    
                    ### transitions
                    tr = joint_logprob_transit[t, prev_site_class, curr_site_class, prev_state-1, curr_state-1]
                    path_logprob += (tr + e)
                    prev_state = curr_state
                
                score_per_path.append(path_logprob)
                
            sum_over_paths = logsumexp( np.array(score_per_path) )
            score_per_alignment[t,b] = sum_over_paths
    
    sum_over_alignments = logsumexp(score_per_alignment, axis=-1 ) #(T,)
    return sum_over_alignments 

# true_score1 = marginalize_state_and_class(aligned_mats1)
# true_score2 = marginalize_state_and_class(aligned_mats2)



###############################################################################
### 2D forward algo   #########################################################
###############################################################################
def joint_loglike_emission_at_k_time_grid( anc_toks,
                                           desc_toks,
                                           joint_logprob_emit_at_match,
                                           logprob_emit_at_indel ):
    """
    to use when MARGINALIZING over a grid of times; 
        joint_logprob_emit_at_match is (T, C, A, A)
    
    can use this function in forward and backward functions to find 
      emission probabilities (which are site independent)
    
    L: length of pairwise alignment
    T: number of timepoints
    B: batch size
    C: number of latent site clases
    S: number of states (4: match, ins, del, start/end)
    A: alphabet size (20 for proteins, 4 for amino acids)
    
    
    Arguments
    ----------
    anc_toks, desc_toks : ArrayLike, (B,)
        ancestor and descendant tokens at diagonal k
    
    joint_logprob_emit_at_match : ArrayLike, (T, C, A, A)
        logP(anc, desc | c, t); log-probability of emission at match site
    
    logprob_emit_at_indel : ArrayLike, (C, A)
        logP(anc | c) or logP(desc | c); log-equilibrium distribution
        
    Returns
    -------
    joint_e : ArrayLike, (T, C, S-1, B)
        log-probability of emission at given column, across all possible 
        site classes
    """
    # TODO: maybe transpose such that B is first dim?
    # get all possible scores at M/I/D
    joint_emit_if_match = joint_logprob_emit_at_match[..., anc_toks - 3, desc_toks - 3] # (T, C, B) or (C, B)
    emit_if_ins = logprob_emit_at_indel[:, desc_toks - 3] #(C, B)
    emit_if_del = logprob_emit_at_indel[:, anc_toks - 3] #(C, B)
    
    # stack all; add dummy row for Start/End
    emit_if_ins = jnp.broadcast_to( emit_if_ins[None, :, :], (T, C, B) ) #(T, C, B)
    emit_if_del = jnp.broadcast_to( emit_if_del[None, :, :], (T, C, B) ) #(T, C, B)
    joint_emissions = jnp.stack([joint_emit_if_match, 
                                 emit_if_ins, 
                                 emit_if_del], axis=-2) #(T, C, S-1, B)
    
    return joint_emissions


def generate_wavefront_indices_at_k(seq_lens,
                                    diagonal_k,
                                    widest_diag_W,
                                    padding_idx = 0):
    """
    seq_lens does NOT include start/end tokens.
    
    
    Example:
    --------
    AG -> A
        
          <S>     A
    <S>  (0,0)  (0,1) 
     A   (1,0)  (1,1) 
     G   (2,0)  (2,1) 
    
    k=0: (0,0)
    k=1: (1,0), (0,1)
    k=2: (2,0), (1,1)
    k=3: (2,1)
    
    all paths HAVE to end at (2,1)
    
    THEN at end position, add final transition and finish the scan
    
    
    Arguments:
    ------------
    seq_lens : ArrayLike, (B, 2)
        dim1=0: ancestor lengths
        dim1=1: descendant lengths 
    
    diagonal_k: int
        which diagonal to generate indices for
        
    
    Returns:
    ---------
    indices : ArrayLike, (B, W, 2)
    mask : ArrayLike, (B, W)
    """
    B = seq_lens.shape[0]
    
    # unpack ancestor and descendant lengths
    anc_lens = seq_lens[:,0][:, None] #(B,1)
    desc_lens = seq_lens[:,1][:, None] #(B,1)
    
    # widest diagonal width is min(anc_len, desc_len) + 1
    offs = jnp.arange(widest_diag_W)[None, :] #(1, W)
    
    
    ### i limits yield lengths of diagonals
    # 0 <= i <= anc_len
    # 0 <= j <= desc_len
    # k = i + j
    # 0 <= k - i <= desc_len
    
    # i >= k - desc_len
    i_min = jnp.maximum(0, diagonal_k - desc_lens) # (B,1)
    
    # i <= k, and i <= anc_len
    # combine into one: i <= min(anc_len, k) 
    i_max = jnp.minimum(anc_lens, diagonal_k) #(B,1)
    
    # lengths of diagonals
    diag_lengths = i_max - i_min + 1 #(B,1)
    
    
    ### get indices
    # generate i indices, with an offset each diagonal
    i_vals = i_min + offs #(B, W)
    
    # j = k - i
    j_vals = diagonal_k - i_vals #(B, W)

    
    ### mask invalid diagonal positions
    mask = offs < diag_lengths #(B, W)
    i_vals = jnp.multiply( i_vals, mask ) #(B, W)
    j_vals = jnp.multiply( j_vals, mask ) #(B, W)
    indices = jnp.stack( [j_vals, i_vals], axis=-1) #(B, W, 2)
    
    return indices, mask

def index_all_classes_one_state(state_idx):
    """
    combiend class+state indexing in forward carry maps to:
        M_1, I_1, D_1,  M_2, I_2, D_2, ..., M_C, I_C, D_C
    
    get the index of one state across all possible classes. For example, 
    state_idx = 0 would return indices corresponding to M_1, ..., M_C
    """
    return jnp.arange(C_trans) * (S-1) + state_idx


### init
# transpose transition matrix 
# old: (T, C_trans_from, C_trans_to, S_from, S_to)
# new: (T, C_trans_from, S_from, C_trans_to, S_to)
joint_logprob_transit = jnp.transpose(joint_logprob_transit, (0,1,3,2,4)) # (T, C_trans_from, S_from, C_trans_to, S_to)


### cache for wavefront diagonals
# \tau = state, M/I/D
# \nu = class (unique to combination of domain+fragment)
# alpha_{ij}^{s_d} = P(desc_{...j}, anc_{...i}, \tau=s, \nu=d | t)
#
# diagonal runs from lower left to upper right:
# (0,0)  (0,1)  (0,2)  (0,3)
# (1,0)  (1,1)  (1,2)  (1,3)
# (2,0)  (2,1)  (2,2)  (2,3)
#
# k0 = [ (0,0), <pad>, <pad> ]
# k1 = [ (1,0), (0,1), <pad> ]
# k2 = [ (2,0), (1,1), (0,2) ]
# k3 = [ (2,1), (1,2), (0,3) ]
# k4 = [ (2,2), (1,3), <pad> ]
# k5 = [ (2,3), <pad>, <pad> ]
#
# dim0: 0=previous diagonal, 1=diag BEFORE previous diagonal
# dim1: maximum diagonal length, W (pad before and after)
# dim2: time, T
# dim3: transition class and state (not including start/end), C_S
# dim4: batch
alpha = jnp.full( (2, W, T, C_S, B), jnp.finfo(jnp.float32).min )


##########################################################
### init diagonal k-2: alignment cells (1,0) and (0,1)   #
##########################################################
first_anc_toks = unaligned_seqs[:, 1, 0] #(B,)
first_desc_toks = unaligned_seqs[:, 1, 1] #(B,)
k=1
# second_align_cell_idxes is (B, W, 2)
# second_mask is (B, W)
first_align_cell_idxes, first_mask = generate_wavefront_indices_at_k(seq_lens = seq_lens,
                                                          diagonal_k = k,
                                                          widest_diag_W = W,
                                                          padding_idx = 0)

### first insert
# alpha_{i=0, j=1}^{I_d} = Em( y_1 | \tau = I, \nu = d, t ) * Tr( \tau = I, \nu = d | Start, t )
first_ins_emit = logprob_emit_at_indel[:, first_desc_toks - 3] #(C, B)
start_ins_transit = joint_logprob_transit[:, 0, -1, :, 1] #(T, C)
new_value = first_ins_emit[None,...] + start_ins_transit[...,None] #(T, C, B)

#         dim0 = 1: corresponds to k-2
#         dim1 = 0: at cell (1,0), which is the first element of the diagonal
#             dim2: all times, add a new dim to handle indexing
# dim3 = 1, 4, ...: at Ins for all classes, which is encoded as one
#             dim4: all samples in the batch
idx_to_reset = index_all_classes_one_state(1) #(C,)
alpha = alpha.at[1, 0, jnp.arange(T)[:,None], idx_to_reset[None,:], :].set( new_value ) # (2, W, T, C*S, B)
del first_ins_emit, start_ins_transit, new_value, idx_to_reset


### first delete
# alpha_{i=1, j=0}^{D_d} = Em( x_1 | \tau = D, \nu = d, t ) * Tr( \tau = D, \nu = d | Start, t )
first_del_emit = logprob_emit_at_indel[:, first_anc_toks - 3] #(C, B)
start_del_transit = joint_logprob_transit[:, 0, -1, :, 2] #(T, C)
new_value = first_del_emit[None,...] + start_del_transit[...,None] #(T, C, B)

#         dim0 = 1: corresponds to k-2
#         dim1 = 1: at cell (0,1), which is the second element of the diagonal
#             dim2: all times, add a new dim to handle indexing
# dim3 = 2, 5, ...: at Del for all classes, which is encoded as two
#             dim4: all samples in the batch
idx_to_reset = index_all_classes_one_state(2) #(C,)
alpha = alpha.at[1, 1, jnp.arange(T)[:,None], idx_to_reset[None,:], :].set( new_value ) # (2, W, T, C*S, B)
del first_del_emit, start_del_transit, new_value, idx_to_reset


##############################################################
### init diagonal k-1: alignment cells (2,0), (1,1), (0,2)   #
##############################################################
k=2
# second_align_cell_idxes is (B, W, 2)
# second_mask is (B, W)
second_align_cell_idxes, second_mask = generate_wavefront_indices_at_k(seq_lens = seq_lens,
                                                            diagonal_k = k,
                                                            widest_diag_W = W,
                                                            padding_idx = 0)

### match: S->M is only valid move at cell (1,1); fill in manually
# alpha_{i=1, j=1}^{M_d} = Em( x_1, y_1 | \tau = I, \nu = d, t ) * Tr( \tau = M, \nu = d | Start, t )
first_match_emit = joint_logprob_emit_at_match[:, :, first_anc_toks - 3, first_desc_toks - 3] #(T, C, B)
start_match_transit = joint_logprob_transit[:, 0, -1, :, 0] #(T, C)
new_value = first_match_emit + start_match_transit[...,None] #(T, C, B)

#         dim0 = 0: corresponds to k-1
#         dim1 = 1: at cell (1,1), which is the second element of the diagonal
#             dim2: all times, add a new dim to handle indexing
# dim3 = 0, 3, ...: at Match for all classes, which is encoded as zero
#             dim4: all samples in the batch
idx_to_reset = index_all_classes_one_state(0) #(C,)
alpha = alpha.at[0, 1, jnp.arange(T)[:,None], idx_to_reset[None,:], :].set( new_value ) # (2, W, T, C*S, B)
del first_match_emit, start_match_transit, new_value, idx_to_reset


### ins: i, j-1
# TODO: turn this into a function and test it separate from this script
def coords_to_diag_pos(indices, anc_len):
    i = indices[...,0] #(B, W)
    j = indices[...,1] #(B, W)
    
    k = i + j  #(B, W)
    i_max = jnp.minimum(k, anc_len[:,None]) #(B, W)
    pos = i_max - i #(B, W)
    return pos

# these contain (i,j-1) coordinates, which are the previous cache values 
ins_ij_needed = second_align_cell_idxes.at[...,1].add(-1) #(B, W, 2)

# mask out invalid cells (at edges of alignment matrix), combine with previous mask
valid_ij = ( ins_ij_needed[...,1] >= 0 ) #(B, W)
second_mask = second_mask & valid_ij #(B, W)
del valid_ij

# replace invalid cell indices, to avoid nan gradients
ins_ij_needed = jnp.multiply( ins_ij_needed, second_mask[...,None] ) #(B, W, 2)

# positions in cache to index
ins_cache_pos_needed = coords_to_diag_pos( indices = ins_ij_needed,
                                           anc_len = seq_lens[...,0] ) #(B, W)

# use ins_cache_pos_needed to retrieve cells (i, j-1)
# advanced indexing moves batch to fist dim, so transpose that back
ins_cache = alpha[0, ins_cache_pos_needed, :, :, jnp.arange(B)[:,None]] #(B, W, T, C_S)
ins_cache = jnp.where( second_mask[:,:,None,None],
                       ins_cache,
                       jnp.finfo(jnp.float32).min ) #(B, W, T, C_S)
ins_cache = jnp.transpose(ins_cache, (1,2,3,0)) #(W, T, C_S, B)

# next: multiply by transitions and sum out C_S_from
# transition matrx: T, C_S_from, C_S_to
# output should be (W, T, C_S, B)

# multiply by probability of emissions, which will be (W, T, C_S, B); this will
# be the updated alpha for diagonal k!


