#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 12:25:48 2025

@author: annabel

'compute_forward_messages_for_state',
'generate_ij_coords_at_diagonal_k',
'get_del_transition_message',
'get_ins_transition_message',
'get_match_transition_message',
'ij_coords_to_wavefront_pos_at_diagonal_k',
'index_all_classes_one_state',
'init_first_diagonal',
'init_second_diagonal',
'joint_loglike_emission_at_k_len_per_samp',
'joint_loglike_emission_at_k_time_grid',
'replace_invalid_toks_in_emissions',
'update_cache',
'wavefront_cache_lookup'



B: batch
W: width of wavefront cache; equal to longest bottom-left-to-top-right 
 diagonal in the alignment grid
T: time
C_transit: number of latent classes for transitions (domain, fragment)
S: number of alignment states (4: Match, Ins, Del, Start/End)
C_S: C_transit * (S-1), combined dim for state+class, like M_c, I_c, D_c, etc.
A: alphabet size (20 for proteins, 4 for amino acids)


TODO: all of these functions are written for working with a separate grid of 
  times. They probably need editing before applying to case where there's 
  one unique branch length per sample
"""
import jax
from jax import numpy as jnp
import flax.linen as nn


###############################################################################
### SCORING   #################################################################
###############################################################################
def wavefront_cache_lookup(ij_needed_for_k, 
                           pad_mask_at_k, 
                           cache_for_prev_diagonal, 
                           seq_lens, 
                           fill_invalid_pos_with = jnp.finfo(jnp.float32).min):
    """
    Extracts cell value at (i_prev, j_prev) from the cache, which is stored as 
      a diagonal in the wavefront parallelism algo
      
    need (i_prev=i-1, j_prev=j-1) if current state is Match
    need (i_prev=i,   j_prev=j-1) if current state is Ins
    need (i_prev=i-1, j_prev=j)   if current state is Del
    
    B: batch
    L: unaligned sequence length
    W: width of wavefront cache; equal to longest bottom-left-to-top-right 
       diagonal in the alignment grid
    T: time
    C_transit: number of latent classes for transitions (domain, fragment)
    S: number of alignment states (4: Match, Ins, Del, Start/End)
    C_S: C_transit * (S-1), combined dim for state+class, like M_c, I_c, D_c, etc.
    
    
    Arguements:
    ------------
    seq_lens : ArrayLike, (B, L, 2)
        seq_lens[:,:,0] will hold length of ancestor sequences
    
    ij_needed_for_k : ArrayLike, (B, W, 2)
        indices (i_prev=i-1, j_prev=j-1) if current state is Match
        indices (i_prev=i,   j_prev=j-1) if current state is Ins
        indices (i_prev=i-1, j_prev=j)   if current state is Del
        
    pad_mask_at_k : ArrayLike, (B, W)
        for the CURRENT diagonal, what values to mask b/c they are padding tokens
    
    cache_for_prev_diagonal : ArrayLike, (W, T, C_S, B)
        for the PREVIOUS diagonal; the cache you take values from
    
    fill_invalid_pos_with : float
        default = jnp.finfo(jnp.float32).min i.e. as close to log(0) as you can get
    
    
    Returns:
    ---------
    cache, (W, T, C_S, B)
        the (i_prev, j_prev) values needed to update current diagonal values
        mathematically, this is alpha_{i_prev, j_prev}^{S_c} = 
          P(x_{...i_prev}, y_{...j_prev}, tau_{i_prev, j_prev}=S, nu_{i_prev, j_prev}=c)
    
    """
    B = seq_lens.shape[0]
    
    # mask out invalid indexes (at edges of alignment matrix), combine with 
    #   previously generate mask for padding positions
    valid_ij = ( ij_needed_for_k[...,1] >= 0 ) #(B, W)
    mask_at_k = pad_mask_at_k & valid_ij #(B, W)
    del valid_ij, pad_mask_at_k
        
    # replace invalid cell indices with (0,0), to avoid nan gradients due to 
    #   invalid indexing
    ij_needed_for_k = jnp.multiply( ij_needed_for_k, mask_at_k[...,None] ) #(B, W, 2)
    
    # find (i_prev, j_prev) in cache_for_prev_diagonal 
    cache_pos_needed = ij_coords_to_wavefront_pos_at_diagonal_k( indices = ij_needed_for_k,
                                                                 anc_len = seq_lens[...,0] ) #(B, W)
    
    # use cache_pos_needed to retrieve values in the cache
    # advanced indexing moves batch to fist dim, so transpose that back
    cache_to_return = cache_for_prev_diagonal[cache_pos_needed, :, :, jnp.arange(B)[:,None]] #(B, W, T, C_S)
    cache_to_return = jnp.where( mask_at_k[:,:,None,None],
                                 cache_to_return,
                                 fill_invalid_pos_with ) #(B, W, T, C_S)
    cache_to_return = jnp.transpose(cache_to_return, (1,2,3,0)) #(W, T, C_S, B)
    
    return cache_to_return


def compute_forward_messages_for_state(logprob_transit_mid_only,
                                       idxes_for_curr_state,
                                       cache_for_state):
    """
    compute probability of transitioning TO state+class S_d, by summing over 
        all possible source states and classes R_c
        
    \sum_{ R \in \{ M,I,D \}, c \in C_transit } 
        P( S, d | R, c, t ) * \alpha_{ X_{...prev_i}, Y_{...prev_j} } ^ { R_c }
    
    
    B: batch
    W: width of wavefront cache; equal to longest bottom-left-to-top-right 
     diagonal in the alignment grid
    T: time
    C_transit: number of latent classes for transitions (domain, fragment)
    S: number of alignment states (4: Match, Ins, Del, Start/End)
    C_S: C_transit * (S-1), combined dim for state+class, like M_c, I_c, D_c, etc.
    
    
    Arguments:
    ----------
    logprob_transit_mid_only: ArrayLike, ( T, C_transit_prev * (S_prev-1), C_transit_curr * (S_curr-1) )
        compressed transition matrix that yield log-probability of jumping 
        between transit classes and states
        P( curr_transit_class, curr_state |  prev_transit_class, prev_state, t)
    
    idxes_for_curr_state : ArrayLike, (C_transit_curr,)
        indices for current state: S_1, S_2, ..., S_d
    
    cache_for_state : ArrayLike, (W, T, C_trans_prev * (S_prev-1), B)
        alpha_{i_prev,j_prev}^{s_c}; cache for PREVIOUS position
        for match: i_prev = i-1,  j_prev = j-1
        for ins: i_prev = i,  j_prev = j-1
        for del: i_prev = i-1,  j_prev = j
        if index is invalid, then cache value is jnp.finfo(jnp.float32).min
    
    Returns:
    ---------
    lse_alpha_times_transit : ArrayLike, (W, T, C_transit_curr, B)
        updated transition probabilties for CURRENT position
        
    """
    # if jnp.allclose( idxes_for_curr_state, 2 ):
    #     breakpoint()
    
    # P( S, d | R, c, t )
    mid_to_state_transit = logprob_transit_mid_only[:, :, idxes_for_curr_state] #(T, C_S_prev, C_transit_curr)
    
    # NOTE: if you get NaN gradients, it might be because something in this sum is reducing to -np.inf
    # P( S, d | R, c, t ) * \alpha_{ X_{...prev_i}, Y_{...prev_j} } ^ { R_c }
    #(W, T, C_S_prev, C_transit_curr, B)
    alpha_times_transit = cache_for_state[:, :, :, None, :] + mid_to_state_transit[None, :, :, :, None] #(W, T, C_S_prev, C_transit_curr, B)
    
    # \sum_{ R \in \{ M,I,D \}, c \in C_transit } 
    #   P( S, d | R, c, t ) * \alpha_{ X_{...prev_i}, Y_{...prev_j} } ^ { R_c }
    lse_alpha_times_transit = nn.logsumexp(alpha_times_transit, axis=2) #(W, T, C_transit_curr, B)
    
    return lse_alpha_times_transit


def replace_invalid_toks_in_emissions( toks,
                                       start_idx = 1,
                                       end_idx = 2,
                                       seq_padding_idx = 0 ):
    positions_with_invalid_toks = ( (toks == start_idx) |
                                    (toks == end_idx) |
                                    (toks == seq_padding_idx) ) #(B, W)
    positions_with_valid_toks = ~positions_with_invalid_toks #(B, W)
    masked_toks =  jnp.where( positions_with_valid_toks,
                              toks,
                              3 )
    return masked_toks, positions_with_valid_toks


def joint_loglike_emission_at_k_len_per_samp( anc_toks,
                                              desc_toks,
                                              joint_logprob_emit_at_match,
                                              logprob_emit_at_indel, 
                                              fill_invalid_pos_with = 0 ):
    """
    to use when each sample has its own time; 
      joint_logprob_emit_at_match is (B, C_transit, A, A)
    
    
    B: batch; one time per sample
    T: time
        > one time per sample, so T = B
    W: width of wavefront cache; equal to longest bottom-left-to-top-right 
       diagonal in the alignment grid
    C_transit: number of latent classes for transitions (domain, fragment)
    S: number of alignment states (4: Match, Ins, Del, Start/End)
    C_S: C_transit * (S-1), combined dim for state+class, like M_c, I_c, D_c, etc.
    A: alphabet size (20 for proteins, 4 for amino acids)
    
    
    Arguments
    ----------
    anc_toks, desc_toks : ArrayLike, (B,W)
        ancestor and descendant tokens at diagonal k
        will be <pad> wherever padding tokens are in the wavefront cache diagonal
    
    joint_logprob_emit_at_match : ArrayLike, (B, C_transit, A, A)
        logP(anc, desc | c, t); log-probability of emission at match site
    
    logprob_emit_at_indel : ArrayLike, (C, A)
        logP(anc | c) or logP(desc | c); log-equilibrium distribution
        
    fill_invalid_pos_with : float
        where ancestor and/or descendant tokens are start, end, or pad, replace
        emission log-probability with this value
        default = 0
        
        
    Returns
    -------
    joint_emissions : ArrayLike, (W, C_transit*S-1, B)
        log-probability of emission at given column, across all possible 
        site classes
    """
    # infer dims
    C = joint_logprob_emit_at_match.shape[1]
    B = anc_toks.shape[0]
    W = anc_toks.shape[1]
    
    # replace invalid indices to avoid nan errors
    # these will get masked out, so it doesn't matter what they are
    anc_toks, anc_mask = replace_invalid_toks_in_emissions( anc_toks ) #both are (B, W)
    desc_toks, desc_mask = replace_invalid_toks_in_emissions( desc_toks ) #both are (B, W)
    anc_and_desc_mask = anc_mask & desc_mask #(B, W)
    
    ### get all possible scores at M/I/D
    # match
    joint_emit_if_match = jnp.where( anc_and_desc_mask[:,:, None],
                                     joint_logprob_emit_at_match[jnp.arange(B)[:, None], :, anc_toks - 3, desc_toks - 3],
                                     fill_invalid_pos_with ) # (B, W, C_transit)
    joint_emit_if_match = jnp.transpose(joint_emit_if_match, (2,0,1)) #(C_transit, B, W)
    
    # ins
    emit_if_ins = jnp.where( desc_mask[None,:,:],
                             logprob_emit_at_indel[:, desc_toks - 3],
                             fill_invalid_pos_with ) #(C_transit, B, W)
    
    # del
    emit_if_del = jnp.where( anc_mask[None,:,:],
                             logprob_emit_at_indel[:, anc_toks - 3],
                             fill_invalid_pos_with ) #(C_transit, B, W)

    # stack all
    joint_emissions = jnp.stack([joint_emit_if_match, 
                                 emit_if_ins, 
                                 emit_if_del], axis=1) #(C_transit, S-1, B, W)
    
    # transpose, reshape
    joint_emissions = jnp.transpose(joint_emissions, (3, 0, 1, 2) ) #(W, C_transit, S-1, B)
    S_minus_one = joint_emissions.shape[2]
    joint_emissions = jnp.reshape( joint_emissions, (W, C*S_minus_one, B) ) #(W, C_transit*S-1, B)
    
    return joint_emissions


def joint_loglike_emission_at_k_time_grid( anc_toks,
                                           desc_toks,
                                           joint_logprob_emit_at_match,
                                           logprob_emit_at_indel, 
                                           fill_invalid_pos_with = 0 ):
    """
    to use when MARGINALIZING over a grid of times; 
        joint_logprob_emit_at_match is (T, C_transit, A, A)
    
    
    B: batch
    W: width of wavefront cache; equal to longest bottom-left-to-top-right 
       diagonal in the alignment grid
    T: time
    C_transit: number of latent classes for transitions (domain, fragment)
    S: number of alignment states (4: Match, Ins, Del, Start/End)
    C_S: C_transit * (S-1), combined dim for state+class, like M_c, I_c, D_c, etc.
    A: alphabet size (20 for proteins, 4 for amino acids)
    
    
    Arguments
    ----------
    anc_toks, desc_toks : ArrayLike, (B,W)
        ancestor and descendant tokens at diagonal k
        will be <pad> wherever padding tokens are in the wavefront cache diagonal
    
    joint_logprob_emit_at_match : ArrayLike, (T, C_transit, A, A)
        logP(anc, desc | c, t); log-probability of emission at match site
    
    logprob_emit_at_indel : ArrayLike, (C_transit, A)
        logP(anc | c) or logP(desc | c); log-equilibrium distribution
    
    fill_invalid_pos_with : float
        where ancestor and/or descendant tokens are start, end, or pad, replace
        emission log-probability with this value
        default = 0
        
        
    Returns
    -------
    joint_emissions : ArrayLike, (W, T, C_transit*S-1, B)
        log-probability of emission at given column, across all possible 
        site classes
    """
    # infer dims
    T = joint_logprob_emit_at_match.shape[0]
    C = joint_logprob_emit_at_match.shape[1]
    B = anc_toks.shape[0]
    W = anc_toks.shape[1]
    
    # replace start, end, and padding toks to avoid nan errors
    # these will get masked out, so it doesn't really matter what they are
    anc_toks, anc_mask = replace_invalid_toks_in_emissions( anc_toks ) #both are (B, W)
    desc_toks, desc_mask = replace_invalid_toks_in_emissions( desc_toks ) #both are (B, W)
    anc_and_desc_mask = anc_mask & desc_mask #(B, W)
    
    
    ### get all possible scores at M/I/D
    # match
    joint_emit_if_match = jnp.where( anc_and_desc_mask[None,None,:,:],
                                     joint_logprob_emit_at_match[..., anc_toks - 3, desc_toks - 3],
                                     fill_invalid_pos_with ) # (T, C_transit, B, W)
    
    # ins
    emit_if_ins = jnp.where( desc_mask[None,:,:],
                             logprob_emit_at_indel[:, desc_toks - 3],
                             fill_invalid_pos_with ) #(C_transit, B, W)
    
    # del
    emit_if_del = jnp.where( anc_mask[None,:,:],
                             logprob_emit_at_indel[:, anc_toks - 3],
                             fill_invalid_pos_with ) #(C_transit, B, W)
    
    # stack all
    emit_if_ins = jnp.broadcast_to( emit_if_ins[None, :, :, :], (T, C, B, W) ) #(T, C_transit, B, W)
    emit_if_del = jnp.broadcast_to( emit_if_del[None, :, :, :], (T, C, B, W) ) #(T, C_transit, B, W)

    joint_emissions = jnp.stack([joint_emit_if_match, 
                                 emit_if_ins, 
                                 emit_if_del], axis=2) #(T, C_transit, S-1, B, W)
    
    # transpose, reshape
    joint_emissions = jnp.transpose(joint_emissions, (4, 0, 1, 2, 3) ) #(W, T, C_transit, S-1, B)
    S_minus_one = joint_emissions.shape[3]
    joint_emissions = jnp.reshape( joint_emissions, (W, T, C*S_minus_one, B) ) #(W, T, C_transit*S-1, B)
    
    return joint_emissions



###############################################################################
### INDEXING   ################################################################
###############################################################################
def generate_ij_coords_at_diagonal_k(seq_lens,
                                     diagonal_k,
                                     widest_diag_W):
    """
    place (i,j) coordinates on diagonal of the wavefront cache
    
    seq_lens does NOT include start/end tokens
    
    
    B: batch
    W: width of wavefront cache; equal to longest bottom-left-to-top-right 
       diagonal in the alignment grid
    
    
    Example:
    --------
    AG -> A
        
          <S>     A
    <S>  (0,0)  (0,1) 
     A   (1,0)  (1,1) 
     G   (2,0)  (2,1) 
    
    k=0:
    $ generate_ij_coords_at_diagonal_k(seq_lens = [2, 1],
                                       diagonal_k = 0,
                                       widest_diag_W = 2)
    >> (0,0), <pad>
    
    k=1:
    $ generate_ij_coords_at_diagonal_k(seq_lens = [2, 1],
                                       diagonal_k = 1,
                                       widest_diag_W = 2)
    >> (1,0), (0,1)
    
    k=2:
    $ generate_ij_coords_at_diagonal_k(seq_lens = [2, 1],
                                       diagonal_k = 2,
                                       widest_diag_W = 2)
    >> (2,0), (1,1)
    
    k=3:
    $ generate_ij_coords_at_diagonal_k(seq_lens = [2, 1],
                                       diagonal_k = 3,
                                       widest_diag_W = 2)
    >> (2,1), <pad>
         
    
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
        dim2=0: i (row: anc)
        dim2=1: j (column: desc)
        
    mask : ArrayLike, (B, W)
    """
    B = seq_lens.shape[0]
    
    # unpack ancestor and descendant lengths
    anc_lens = seq_lens[:,0][:, None] #(B,1)
    desc_lens = seq_lens[:,1][:, None] #(B,1)
    
    # widest diagonal width is min(anc_len, desc_len) + 1
    offs_ascending = jnp.arange(widest_diag_W)[None, :] #(1, W)
    
    
    ### i limits yield lengths of diagonals
    # 0 <= i <= anc_len
    # 0 <= j <= desc_len
    # k = i + j
    #
    # j = (k - i) <= desc_len
    # 0 <= (k - i) <= desc_len
    
    # minimum i index 
    # i >= k - desc_len
    i_min = jnp.maximum(0, diagonal_k - desc_lens) # (B,1)
    
    # i <= k, and i <= anc_len
    # combine into one: i <= min(anc_len, k) 
    i_max = jnp.minimum(anc_lens, diagonal_k) #(B,1)
    
    # lengths of diagonals
    diag_lengths = i_max - i_min + 1 #(B,1)
    
    
    ### get indices
    # generate i indices, with an offset each diagonal
    i_vals = i_max - offs_ascending #(B, W)
    
    # j = k - i
    j_vals = diagonal_k - i_vals #(B, W)

    
    ### mask invalid diagonal positions
    mask = offs_ascending < diag_lengths #(B, W)
    i_vals = jnp.multiply( i_vals, mask ) #(B, W)
    j_vals = jnp.multiply( j_vals, mask ) #(B, W)
    indices = jnp.stack( [i_vals, j_vals], axis=-1) #(B, W, 2)
    
    return indices, mask

def ij_coords_to_wavefront_pos_at_diagonal_k(indices, anc_len):
    """
    for any (i,j) coordinates, extract the corresponding position in its 
        diagonal of the wavefront cache
    
    anc_len does NOT include start/end tokens
    
    
    B: batch
    W: width of wavefront cache; equal to longest bottom-left-to-top-right 
       diagonal in the alignment grid
    
    
    Example:
    --------
    AG -> A
        
          <S>     A
    <S>  (0,0)  (0,1) 
     A   (1,0)  (1,1) 
     G   (2,0)  (2,1) 
     
     
    for diagonal: [ (1,0), (0,1) ]
    
        (1,0)
        $ ij_coords_to_wavefront_pos( indices=(1,0),
                                      anc_len=2 )
        >> 0
        
        (0,1)
        $ ij_coords_to_wavefront_pos( indices=(0,1),
                                      anc_len=2 )
        >> 1
    
    
    Arguments:
    ------------
    indices : ArrayLike, (B, W, 2)
        dim1=0: ancestor lengths
        dim1=1: descendant lengths 
    
    anc_len: (B,)
        ancestor length
        
    
    Returns:
    ---------
    pos : ArrayLike, (B, W)
        index for entire diagonal
    """
    i = indices[...,0] #(B, W)
    j = indices[...,1] #(B, W)
    
    k = i + j  #(B, W)
    i_max = jnp.minimum(k, anc_len[:,None]) #(B, W)
    pos = i_max - i #(B, W)
    return pos

def index_all_classes_one_state(state_idx,
                                num_transit_classes):
    """
    Returns indices corresponding to one state, all classes

    
    C_transit: number of latent classes for transitions (domain, fragment)
    S: number of alignment states (4: Match, Ins, Del, Start/End)
    C_S: C_transit * (S-1), combined dim for state+class, like M_c, I_c, D_c, etc.


    Example:
    --------
    Values along dimension C_S are stored as:
        arr: [ M_1, I_1, D_1,  M_2, I_2, D_2, ..., M_{C_transit}, I_{C_transit}, D_{C_transit} ]
      index:    0    1    2     3    4    5         C_S-3          C_S-2          C_S-1
    
    $ index_all_classes_one_state(state_idx = 0, num_transit_classes = C_transit)
    >> 0, 3, ..., C_S-3  # corresponds to match
    
    $ index_all_classes_one_state(state_idx = 1, num_transit_classes = C_transit)
    >> 1, 4, ..., C_S-2  # corresponds to ins
    
    $ index_all_classes_one_state(state_idx = 2, num_transit_classes = C_transit)
    >> 2, 5, ..., C_S-1  # corresponds to del
    
    
    Arguments:
    ------------
    state_idx : int
        0 = match
        1 = ins
        2 = del
        (does not include start/end)
    
    num_transit_classes: int
        number of fragment+domain transition classes
        
    
    Returns:
    ---------
    ArrayLike, (C_transit,)
        indices corresponding to state_idx for every transit class
    
    """
    return jnp.arange(num_transit_classes) * 3 + state_idx


###############################################################################
### MESSAGE PASSING FUNCTIONS   ###############################################
###############################################################################
def get_match_transition_message( align_cell_idxes,
                                  pad_mask,
                                  cache_at_curr_diagonal,
                                  cache_two_diags_prior,
                                  seq_lens,
                                  joint_logprob_transit_mid_only,
                                  C_transit ):
    
    # dims
    W = cache_two_diags_prior.shape[0]
    T = cache_two_diags_prior.shape[1]
    C_S = cache_two_diags_prior.shape[2]
    B = cache_two_diags_prior.shape[3]
    
    ### Match: alpha_{i,j}^{M,d} = \sum_{s \in \{M,I,D\}, c in C_transit} Tr(M,d|s,c,t) * alpha_{i-1,j-1}^{s_c}
    # extract (i-1, j-1) indices needed for calculation
    i_minus_one_j_minus_one = align_cell_idxes.at[...,0].add(-1) #(B, W, 2)
    i_minus_one_j_minus_one = i_minus_one_j_minus_one.at[...,1].add(-1) #(B, W, 2)
    
    # extract alpha_{i-1,j-1}^{s_c} from previous diagonal
    #   if i-1 < 0 or j-1 < 0, then the cell is invalid; replace value with as 
    #   close to log(0) as you can get
    cache_for_match = wavefront_cache_lookup(ij_needed_for_k = i_minus_one_j_minus_one, 
                                             pad_mask_at_k = pad_mask, 
                                             cache_for_prev_diagonal = cache_two_diags_prior, 
                                             seq_lens = seq_lens, 
                                             fill_invalid_pos_with = jnp.finfo(jnp.float32).min) #(W, T, C_S_prev, B)
    
    # \sum_{s \in \{M,I,D\}, c in C_transit} Tr(M,d|s,c,t) * alpha_{i-1,j-1}^{s_c}
    match_idx = index_all_classes_one_state(state_idx = 0, num_transit_classes = C_transit) #(C,)
    match_transit_message = compute_forward_messages_for_state( logprob_transit_mid_only = joint_logprob_transit_mid_only,
                                                                idxes_for_curr_state = match_idx,
                                                                cache_for_state = cache_for_match ) #(W, T, C, B)
    
    return match_idx, match_transit_message


def get_ins_transition_message( align_cell_idxes,
                                pad_mask,
                                cache_at_curr_diagonal,
                                cache_for_prev_diagonal,
                                seq_lens,
                                joint_logprob_transit_mid_only,
                                C_transit ):
    
    # dims
    W = cache_for_prev_diagonal.shape[0]
    T = cache_for_prev_diagonal.shape[1]
    C_S = cache_for_prev_diagonal.shape[2]
    B = cache_for_prev_diagonal.shape[3]
    
    ### Ins: alpha_{i,j}^{I,d} = \sum_{s \in \{M,I,D\}, c in C_transit} Tr(I,d|s,c,t) * alpha_{i,j-1}^{s_c}
    # extract (i, j-1) indices needed for calculation
    i_j_minus_one = align_cell_idxes.at[...,1].add(-1) #(B, W, 2)
    
    # extract alpha_{i,j-1}^{s_c} from previous diagonal
    # if j-1 < 0, then the cell is invalid; replace value with as close to log(0) as you can get
    cache_for_ins = wavefront_cache_lookup(ij_needed_for_k = i_j_minus_one, 
                                           pad_mask_at_k = pad_mask, 
                                           cache_for_prev_diagonal = cache_for_prev_diagonal, 
                                           seq_lens = seq_lens, 
                                           fill_invalid_pos_with = jnp.finfo(jnp.float32).min) #(W, T, C_S_prev, B)
    
    # \sum_{s \in \{M,I,D\}, c in C_transit} Tr(I,d|s,c,t) * alpha_{i,j-1}^{s_c}
    ins_idx = index_all_classes_one_state(state_idx = 1, num_transit_classes = C_transit) #(C,)
    ins_transit_message = compute_forward_messages_for_state( logprob_transit_mid_only = joint_logprob_transit_mid_only,
                                                              idxes_for_curr_state = ins_idx,
                                                              cache_for_state = cache_for_ins ) #(W, T, C, B)
    
    return ins_idx, ins_transit_message


def get_del_transition_message( align_cell_idxes,
                                pad_mask,
                                cache_at_curr_diagonal,
                                cache_for_prev_diagonal,
                                seq_lens,
                                joint_logprob_transit_mid_only,
                                C_transit ):
    # dims
    W = cache_for_prev_diagonal.shape[0]
    T = cache_for_prev_diagonal.shape[1]
    C_S = cache_for_prev_diagonal.shape[2]
    B = cache_for_prev_diagonal.shape[3]
    
    ### Del: alpha_{i,j}^{D,d} = \sum_{s \in \{M,I,D\}, c in C_transit} Tr(D,d|s,c,t) * alpha_{i-1,j}^{s_c}
    # extract (i-1, j) indices needed for calculation
    i_minus_one_j = align_cell_idxes.at[...,0].add(-1) #(B, W, 2)
    
    # extract alpha_{i-1,j}^{s_c} from previous diagonal
    # if i-1 < 0, then the cell is invalid; replace value with as close to log(0) as you can get
    cache_for_del = wavefront_cache_lookup(ij_needed_for_k = i_minus_one_j, 
                                           pad_mask_at_k = pad_mask, 
                                           cache_for_prev_diagonal = cache_for_prev_diagonal, 
                                           seq_lens = seq_lens, 
                                           fill_invalid_pos_with = jnp.finfo(jnp.float32).min) #(W, T, C_S_prev, B)
    
    # \sum_{s \in \{M,I,D\}, c in C_transit} Tr(D,d|s,c,t) * alpha_{i-1,j}^{s_c}
    del_idx = index_all_classes_one_state(state_idx = 2, num_transit_classes = C_transit) #(C,)
    del_transit_message = compute_forward_messages_for_state( logprob_transit_mid_only = joint_logprob_transit_mid_only,
                                                              idxes_for_curr_state = del_idx,
                                                              cache_for_state = cache_for_del ) #(W, T, C, B)
    
    return del_idx, del_transit_message


def update_cache(idx_arr_for_state, 
                 transit_message, 
                 cache_to_update):
    # dims
    W = cache_to_update.shape[0]
    T = cache_to_update.shape[1]
    
    # update cache_at_curr_k
    #               dim0: across entire diagonal
    #               dim1: all times
    # dim2 = s, s_3, ...: at State s for all classes, given by idx_arr_for_state
    #               dim3: all samples in the batch
    updated = cache_to_update.at[jnp.arange(W)[:, None, None], 
                                 jnp.arange(T)[None, :, None], 
                                 idx_arr_for_state[None, None, :], 
                                 :].set( transit_message ) # (W, T, C*S, B)
    return updated



###############################################################################
### INITIALIZE THE WAVEFRONT CACHE   ##########################################
###############################################################################
def init_first_diagonal(empty_cache, 
                        unaligned_seqs,
                        joint_logprob_transit,
                        logprob_emit_at_indel):
    """
    with an empty cache, init values at (1,0) and (0,1)
    
    Can only reach (0,1) by Ins
    alpha_{i=0, j=1}^{I_d} = Em( y_1 | \tau = I, \nu = d, t ) * Tr( \tau = I, \nu = d | Start, t )
    
    Can only reach (1,0) by Del
    alpha_{i=1, j=0}^{D_d} = Em( x_1 | \tau = D, \nu = d, t ) * Tr( \tau = D, \nu = d | Start, t )
    
    B: batch
    W: width of wavefront cache; equal to longest bottom-left-to-top-right 
     diagonal in the alignment grid
    T: time
    C_transit: number of latent classes for transitions (domain, fragment)
    S: number of alignment states (4: Match, Ins, Del, Start/End)
    C_S: C_transit * (S-1), combined dim for state+class, like M_c, I_c, D_c, etc.
    A: alphabet size (20 for proteins, 4 for amino acids)


    Arguments:
    ----------
    empty_cache : ArrayLike, ( 2, W, T, C_S, B )
        cache to fill
    
    unaligned_seqs : ArrayLike, ( B, L_seq, 2 )
        dim2=0 is ancestor
        dim2=1 is descendant
    
    joint_logprob_transit : ArrayLike, (T, C_transit_prev, S_prev, C_trans_curr, S_curr)
        transition matrix
    
    logprob_emit_at_indel : ArrayLike, (C_transit, A, A)
        equilibrium distribution per transition calss
    
    
    Returns:
    ---------
    alpha : ArrayLike, ( 2, W, T, C_S, B )
        alpha[1] is now filled with values
    
    """
    # dims
    W = empty_cache.shape[1]
    T = empty_cache.shape[2]
    C_transit = logprob_emit_at_indel.shape[0]
    C_S = empty_cache.shape[3]
    B = empty_cache.shape[4]
    
    # gather emissions
    first_anc_toks = unaligned_seqs[:, 1, 0] #(B,)
    first_desc_toks = unaligned_seqs[:, 1, 1] #(B,)
    del unaligned_seqs
    
    
    ### first delete
    # alpha_{i=1, j=0}^{D_d} = Em( x_1 | \tau = D, \nu = d, t ) * Tr( \tau = D, \nu = d | Start, t )
    first_del_emit = logprob_emit_at_indel[:, first_anc_toks - 3] #(C, B)
    start_del_transit = joint_logprob_transit[:, 0, -1, :, 2] #(T, C)
    new_value = first_del_emit[None,...] + start_del_transit[...,None] #(T, C, B)
    
    #         dim0 = 1: corresponds to k-2
    #         dim1 = 0: at cell (1,0), which is the first element of the diagonal
    #             dim2: all times
    # dim3 = 2, 5, ...: at Del for all classes, which is encoded as two
    #             dim4: all samples in the batch
    idx_to_reset = index_all_classes_one_state(state_idx=2,
                                               num_transit_classes=C_transit) #(C,)
    alpha = empty_cache.at[1, 0, jnp.arange(T)[:,None], idx_to_reset[None,:], :].set( new_value ) # (2, W, T, C*S, B)
    del first_del_emit, start_del_transit, new_value, idx_to_reset, empty_cache
    
    
    ### first insert
    # alpha_{i=0, j=1}^{I_d} = Em( y_1 | \tau = I, \nu = d, t ) * Tr( \tau = I, \nu = d | Start, t )
    first_ins_emit = logprob_emit_at_indel[:, first_desc_toks - 3] #(C, B)
    start_ins_transit = joint_logprob_transit[:, 0, -1, :, 1] #(T, C)
    new_value = first_ins_emit[None,...] + start_ins_transit[...,None] #(T, C, B)
    
    #         dim0 = 1: corresponds to k-2
    #         dim1 = 1: at cell (0,1), which is the second element of the diagonal
    #             dim2: all times
    # dim3 = 1, 4, ...: at Ins for all classes, which is encoded as one
    #             dim4: all samples in the batch
    idx_to_reset = index_all_classes_one_state(state_idx=1,
                                               num_transit_classes=C_transit) #(C,)
    alpha = alpha.at[1, 1, jnp.arange(T)[:,None], idx_to_reset[None,:], :].set( new_value ) # (2, W, T, C*S, B)
    
    return alpha


def init_second_diagonal( cache_with_first_diag, 
                          unaligned_seqs,
                          joint_logprob_transit,
                          joint_logprob_emit_at_match,
                          logprob_emit_at_indel,
                          seq_lens ):
    """
    with cache that only has first diagonal filled, fill the second diagonal
      corresponds to (2,0), (1,1), and (0,2)
    
    Can only reach (1,1) by Match
    alpha_{i=1, j=1}^{I_d} = Em( x_1, y_1 | \tau = M, \nu = d, t ) * Tr( \tau = M, \nu = d | Start, t )
    
    
    B: batch
    W: width of wavefront cache; equal to longest bottom-left-to-top-right 
     diagonal in the alignment grid
    T: time
    C_transit: number of latent classes for transitions (domain, fragment)
    S: number of alignment states (4: Match, Ins, Del, Start/End)
    C_S: C_transit * (S-1), combined dim for state+class, like M_c, I_c, D_c, etc.
    A: alphabet size (20 for proteins, 4 for amino acids)
    
    
    Arguments:
    ----------
    cache_with_first_diag : ArrayLike, ( 2, W, T, C_S, B )
        cache to fill; cache_with_first_diag[1] already has values
    
    unaligned_seqs : ArrayLike, ( B, L_seq, 2 )
        dim2=0 is ancestor
        dim2=1 is descendant
    
    joint_logprob_transit : ArrayLike, (T, C_transit_prev, S_prev, C_trans_curr, S_curr)
        logprob of state+class transitions
        
    joint_logprob_emit_at_match : ArrayLike, (T, C_transit, A, A)
        logprob of substitution, per transition class
        
    logprob_emit_at_indel : ArrayLike, (C_transit, A, A)
        log-transformed equilibrium distribution, per transition class
    
    
    Returns:
    ---------
    alpha : ArrayLike, ( 2, W, T, C_S, B )
        alpha[0] is now filled with values
              
    joint_logprob_transit_mid_only : ArrayLike, (T, C_S, C_S)
        transition matrix with MID transitions only; inner dims combined
    
    """
    # dims, lengths
    W = cache_with_first_diag.shape[1]
    T = cache_with_first_diag.shape[2]
    C_transit = logprob_emit_at_indel.shape[0]
    C_S = cache_with_first_diag.shape[3]
    B = cache_with_first_diag.shape[4]
    
    # align_cell_idxes is (B, W, 2)
    # pad_mask is (B, W)
    align_cell_idxes, pad_mask = generate_ij_coords_at_diagonal_k(seq_lens = seq_lens,
                                                                  diagonal_k = 2,
                                                                  widest_diag_W = W)
    
    # remove S/E transitions, and merge class and state dims
    joint_logprob_transit_mid_only = jnp.reshape(joint_logprob_transit[:, :, :3, :, :3], (T, C_S, C_S) ) #(T, C*S_prev, C*S_curr)
    
    
    ###########################################
    ### Fill second diagonal with transitions #
    ###########################################
    # Ins: alpha_{i,j}^{I,d} = \sum_{s \in \{M,I,D\}, c in C_transit} Tr(I,d|s,c,t) * alpha_{i,j-1}^{s_c}
    to_fill = jnp.full( (W, T, C_S, B), jnp.finfo(jnp.float32).min )
    out = get_ins_transition_message( align_cell_idxes = align_cell_idxes,
                                      pad_mask = pad_mask,
                                      cache_at_curr_diagonal = to_fill,
                                      cache_for_prev_diagonal = cache_with_first_diag[1,...],
                                      seq_lens = seq_lens,
                                      joint_logprob_transit_mid_only = joint_logprob_transit_mid_only,
                                      C_transit = C_transit ) #(W, T, C_S, B)
    
    ins_idx = out[0] #(C,)
    ins_transit_message = out[1] #(W, T, C, B)
    del out
    
    # update stored values in alpha
    #         dim0 = 0: corresponds to k-1
    #             dim1: across entire diagonal
    #             dim2: all times
    # dim3 = 1, 4, ...: at Ins for all classes, which is encoded as one
    #             dim4: all samples in the batch
    alpha = cache_with_first_diag.at[0, 
                                     jnp.arange(W)[:, None, None], 
                                     jnp.arange(T)[None, :, None], 
                                     ins_idx[None, None, :], 
                                     :].set( ins_transit_message ) # (2, W, T, C*S, B)
    del ins_idx, ins_transit_message, to_fill
    
    
    ### Del: alpha_{i,j}^{D,d} = \sum_{s \in \{M,I,D\}, c in C_transit} Tr(D,d|s,c,t) * alpha_{i-1,j}^{s_c}
    # COME BACK HERE
    to_fill = jnp.full( (W, T, C_S, B), jnp.finfo(jnp.float32).min )
    out = get_del_transition_message( align_cell_idxes = align_cell_idxes,
                                      pad_mask = pad_mask,
                                      cache_at_curr_diagonal = to_fill,
                                      cache_for_prev_diagonal = cache_with_first_diag[1,...],
                                      seq_lens = seq_lens,
                                      joint_logprob_transit_mid_only = joint_logprob_transit_mid_only,
                                      C_transit = C_transit ) #(W, T, C_S, B)
    
    del_idx = out[0] #(C,)
    del_transit_message = out[1] #(W, T, C, B)
    del out
    
    # update stored values in alpha
    #         dim0 = 0: corresponds to k-1
    #             dim1: across entire diagonal
    #             dim2: all times
    # dim3 = 2, 5, ...: at del for all classes, which is encoded as two
    #             dim4: all samples in the batch
    alpha = alpha.at[0, 
                     jnp.arange(W)[:, None, None], 
                     jnp.arange(T)[None, :, None], 
                     del_idx[None, None, :], 
                     :].set( del_transit_message ) # (2, W, T, C*S, B)
    del to_fill, del_idx, del_transit_message
    
    
    ### Match: alpha_{i=1, j=1}^{I_d} = Em( x_1, y_1 | \tau = M, \nu = d, t ) * Tr( \tau = M, \nu = d | Start, t )
    start_match_transit = joint_logprob_transit[:, 0, -1, :, 0] #(T, C)

    # along dim W: determine which cell in the diagonal is (1,1)
    mask_for_cell_1_1 = jnp.all(align_cell_idxes == jnp.array([1, 1]), axis=-1)  # (B, W)
    w_idx_for_cell_1_1 = jnp.argmax(mask_for_cell_1_1, axis=1)  # (B,)

    #         dim0 = 0: corresponds to k-1
    #         dim1 = 1: at cell (1,1)
    #             dim2: all times
    # dim3 = 0, 3, ...: at Match for all classes, which is encoded as zero
    #             dim4: all samples in the batch
    match_idx = index_all_classes_one_state( state_idx = 0,
                                             num_transit_classes = C_transit ) #(C,)
    
    alpha = alpha.at[0, 
                     w_idx_for_cell_1_1[None, None, :], 
                     jnp.arange(T)[:, None, None], 
                     match_idx[None,:, None], 
                     jnp.arange(B)[None, None, :] ].set( start_match_transit[..., None] ) # (2, W, T, C*S, B)
    
    del start_match_transit, match_idx
    
    
    ##############################################################
    ### Add logprob of emissions to all cells of second diagonal #
    ##############################################################
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
    
    # add to appropriate positions in the cache
    #         dim0 = 0: corresponds to k-1
    #             dim1: all values in the diagonal
    #             dim2: all times
    #             dim3: all states and caches
    #             dim4: all samples in the batch
    alpha = alpha.at[0, :, :, :, :].add( emit_logprobs_at_k ) # (2, W, T, C*S, B)
    
    return alpha, joint_logprob_transit_mid_only
    
