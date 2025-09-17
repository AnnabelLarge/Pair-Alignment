#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 13:47:05 2025

@author: annabel

transition matrx: T, C_trans, C_trans, S_from, S_to
equilibrium distribution: C_trans, C_sites, A
  > after marginalizing over site-independent C_sites: C_trans, A
substitution emission matrix: T, C_trans, C_sites, K, A, A
  > after marginalizing over site-independent C_sites and K: T, C_trans, A, A

todo:
------
[X] make fake scoring matrices, inputs 
[X] enumerate small alignment and all possible paths; get true score in prob space
[ip] start working on indexing+scoring function
    > [X] generate indices of diagonals, to scan over

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
                                     shape = (T, C_trans, A) ) #(T, C_trans, A)
logprob_emit_at_indel = nn.log_softmax(indel_emit_logits, axis=-1) #(T, C_trans, A)
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


unaliged_seqs = jnp.stack([seqs1, seqs2], axis=0) #(B, L_seq, 2)
del seqs1, seqs2

B = unaliged_seqs.shape[0]
L_seq = unaliged_seqs.shape[1]

                

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

true_score1 = marginalize_state_and_class(aligned_mats1)
true_score2 = marginalize_state_and_class(aligned_mats2)



###############################################################################
### 2D forward algo   #########################################################
###############################################################################
def get_joint_loglike_emission_time_grid(anc_toks,
                                         desc_toks,
                                         joint_logprob_emit_at_match,
                                         logprob_emit_at_indel):
    """
    to use when MARGINALIZING over a grid of times; 
        joint_logprob_emit_at_match is (T, C, A, A)
    
    can use this function in forward and backward functions to find 
      emission probabilities (which are site independent)
    
    L: length of pairwise alignment
    T: number of timepoints
    B: batch size
    C: number of latent site clases
    A: alphabet size (20 for proteins, 4 for amino acids)
    
    
    Arguments
    ----------
    anc_toks, desc_toks : ArrayLike, (B,)
        ancestor and descendant tokens
    
    joint_logprob_emit_at_match : ArrayLike, (T, C, A, A)
        logP(anc, desc | c, t); log-probability of emission at match site
    
    logprob_emit_at_indel : ArrayLike, (C, A)
        logP(anc | c) or logP(desc | c); log-equilibrium distribution
        
    Returns
    -------
    joint_e : ArrayLike, (T, C, 3, B)
        log-probability of emission at given column, across all possible 
        site classes
    """
    # get all possible scores
    joint_emit_if_match = joint_logprob_emit_at_match[..., anc_toks - 3, desc_toks - 3] # (T, C, B) or (C, B)
    emit_if_ins = logprob_emit_at_indel[:, desc_toks - 3] #(C, B)
    emit_if_del = logprob_emit_at_indel[:, anc_toks - 3] #(C, B)
    
    # stack all
    emit_if_ins = jnp.broadcast_to( emit_if_ins[None, :, :], (T, C, B) ) #(T, C, B)
    emit_if_del = jnp.broadcast_to( emit_if_del[None, :, :], (T, C, B) ) #(T, C, B)
    joint_emissions = jnp.stack([joint_emit_if_match, 
                                 emit_if_ins, 
                                 emit_if_del], axis=-2) #(3, T, C, B)
    
    return joint_emissions


def generate_wavefront_indices_at_k(seq_lens,
                                    diagonal_k, 
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
    widest_diag_W = ( seq_lens.min(axis=1) ).max() + 1
    offs = jnp.arange(widest_diag_W)[None, :] #(1, W)
    
    
    ### i limits yield lengths of diagonals
    # 0 <= i <= anc_len
    # 0 <= j <= desc_len
    # k = i + j
    # 0 <= k - i <= desc_len
    
    # i >= k - desc_len
    i_min = jnp.maximum(0, diagonal_k - desc_lens) # (B,)
    
    # i <= k, and i <= anc_len
    # combine into one: i <= min(anc_len, k) 
    i_max = jnp.minimum(anc_lens, diagonal_k) #(B,)
    
    # lengths of diagonals
    diag_lengths = i_max - i_min + 1 #(B,)
    
    
    ### get indices
    # generate i indices, with an offset each diagonal
    i_vals = i_min[:, None] + offs #(B, W)
    
    # j = k - i
    j_vals = diagonal_k - i_vals #(B, W)
    
    
    ### mask invalid diagonal positions
    mask = offs < diag_lengths[..., None] #(B, W)
    i_vals = jnp.multiply( i_vals, mask ) #(B, W)
    j_vals = jnp.multiply( j_vals, mask ) #(B, W)
    indices = jnp.stack( [i_vals, j_vals], axis=-1) #(B, W, 2)
    
    return indices, mask


### wavefront cache
# cache (alpha_{ij}^{\tau, \nu})
# dim0: 0=diag BEFORE previous diagonal, 1=previous diagonal, 1=current diagonal
# dim1: maximum diagonal length, W
# dim2: 0=match, 1=ins, 2=delete
# dim3: time, T
# dim4: transition class, C_trans
# dim5: batch
W = ( unaliged_seqs.min(axis=1) ).max()
cache = jnp.full( (3, W, 3, T, C_trans, B), jnp.finfo(jnp.float32).min )




