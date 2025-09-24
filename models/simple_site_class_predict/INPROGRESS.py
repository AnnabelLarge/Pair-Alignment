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
from jax import numpy as jnp
import flax.linen as nn

import numpy as np
from itertools import product
from scipy.special import logsumexp
from tqdm import tqdm

from models.simple_site_class_predict.transition_models import (TKF92TransitionLogprobs)

from models.simple_site_class_predict.marg_over_alignments_forward_fns import (generate_ij_coords_at_diagonal_k,
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
T = 1
C_transit = 3
A = 20
S = 4
C_S = C_transit * (S-1) #use this for forward algo carry

# time
t_array = jnp.array( [1.0] )


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
seq_lens = (unaligned_seqs != 0).sum(axis=1)-2 #(B, 2)
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
            paths = [list(p) for p in product(range(C_transit), repeat= int(n) )]
            
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
# transpose transition matrix 
# old: (T, C_transit_prev, C_transit_curr, S_prev, S_curr)
# new: (T, C_transit_prev, S_prev, C_transit_curr, S_curr)
joint_logprob_transit = jnp.transpose(joint_logprob_transit, (0,1,3,2,4)) # (T, C_transit_prev, S_prev, C_transit_curr, S_curr)
  

################################################
### Initialize cache for wavefront diagonals   #
################################################
# \tau = state, M/I/D
# \nu = class (unique to combination of domain+fragment)
# alpha_{ij}^{s_d} = P(desc_{...j}, anc_{...i}, \tau=s, \nu=d | t)
# dim0: 0=previous diagonal, 1=diag BEFORE previous diagonal
alpha = jnp.full( (2, W, T, C_S, B), jnp.finfo(jnp.float32).min )

# fill diagonal k-2: alignment cells (1,0) and (0,1)
alpha = init_first_diagonal( empty_cache = alpha, 
                             unaligned_seqs = unaligned_seqs,
                             joint_logprob_transit = joint_logprob_transit,
                             logprob_emit_at_indel = logprob_emit_at_indel )  #(2, W, T, C_S, B)

# fill diag k-1: alignment cells (1,1), and (if applicable) (0,2) and/or (2,0)
out = init_second_diagonal( cache_with_first_diag = alpha, 
                            unaligned_seqs = unaligned_seqs,
                            joint_logprob_transit = joint_logprob_transit,
                            joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                            logprob_emit_at_indel = logprob_emit_at_indel ) 

alpha = out[0] #(2, W, T, C_S, B)
joint_logprob_transit_mid_only = out[1] #(T, C_S_prev, C_S_curr )
del out 


########################
### Start Recurrence   #
########################
# try this out with k = 3 first, to get feel for jax.lax.scan
k = 3
previous_cache = alpha

# cache to fill
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


# TODO: does this automatically handle end states...?


