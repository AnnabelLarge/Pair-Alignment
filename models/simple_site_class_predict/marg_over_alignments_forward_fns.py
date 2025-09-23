#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 12:25:48 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
import flax.linen as nn


###############################################################################
### SCORING   #################################################################
###############################################################################
def joint_loglike_emission_at_k_len_per_samp( anc_toks,
                                              desc_toks,
                                              joint_logprob_emit_at_match,
                                              logprob_emit_at_indel ):
    """
    to use when each sample has its own time; 
      joint_logprob_emit_at_match is (B, C, A, A)
    
    L: length of pairwise alignment
    B: batch size
    W: width of diagonal in wavefront cache
    C: number of latent site clases
    S: number of states (4: match, ins, del, start/end)
    A: alphabet size (20 for proteins, 4 for amino acids)
    
    
    Arguments
    ----------
    anc_toks, desc_toks : ArrayLike, (B,W)
        ancestor and descendant tokens at diagonal k
    
    joint_logprob_emit_at_match : ArrayLike, (B, C, A, A)
        logP(anc, desc | c, t); log-probability of emission at match site
    
    logprob_emit_at_indel : ArrayLike, (C, A)
        logP(anc | c) or logP(desc | c); log-equilibrium distribution
        
    Returns
    -------
    joint_emissions : ArrayLike, (W, C*S-1, B)
        log-probability of emission at given column, across all possible 
        site classes
    """
    # infer dims
    C = joint_logprob_emit_at_match.shape[1]
    B = anc_toks.shape[0]
    W = anc_toks.shape[1]
    
    # emit at match
    joint_emit_if_match = joint_logprob_emit_at_match[jnp.arange(B)[:, None], :, anc_toks - 3, desc_toks - 3] # (B, W, C)
    joint_emit_if_match = jnp.transpose(joint_emit_if_match, (2,0,1)) #(C, B, W)
    
    # emit at indels
    emit_if_ins = logprob_emit_at_indel[:, desc_toks - 3] #(C, B, W)
    emit_if_del = logprob_emit_at_indel[:, anc_toks - 3] #(C, B, W)

    joint_emissions = jnp.stack([joint_emit_if_match, 
                                 emit_if_ins, 
                                 emit_if_del], axis=1) #(C, S-1, B, W)
    
    # transpose, reshape
    joint_emissions = jnp.transpose(joint_emissions, (3, 0, 1, 2) ) #(W, C, S-1, B)
    S_minus_one = joint_emissions.shape[2]
    joint_emissions = jnp.reshape( joint_emissions, (W, C*S_minus_one, B) ) #(W, C*S-1, B)
    
    return joint_emissions


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
    W: width of diagonal in wavefront cache
    C: number of latent site clases
    S: number of states (4: match, ins, del, start/end)
    A: alphabet size (20 for proteins, 4 for amino acids)
    
    
    Arguments
    ----------
    anc_toks, desc_toks : ArrayLike, (B,W)
        ancestor and descendant tokens at diagonal k
    
    joint_logprob_emit_at_match : ArrayLike, (T, C, A, A)
        logP(anc, desc | c, t); log-probability of emission at match site
    
    logprob_emit_at_indel : ArrayLike, (C, A)
        logP(anc | c) or logP(desc | c); log-equilibrium distribution
        
    Returns
    -------
    joint_emissions : ArrayLike, (W, T, C*S-1, B)
        log-probability of emission at given column, across all possible 
        site classes
    """
    # infer dims
    T = joint_logprob_emit_at_match.shape[0]
    C = joint_logprob_emit_at_match.shape[1]
    B = anc_toks.shape[0]
    W = anc_toks.shape[1]
    
    # get all possible scores at M/I/D
    joint_emit_if_match = joint_logprob_emit_at_match[..., anc_toks - 3, desc_toks - 3] # (T, C, B, W)
    emit_if_ins = logprob_emit_at_indel[:, desc_toks - 3] #(C, B, W)
    emit_if_del = logprob_emit_at_indel[:, anc_toks - 3] #(C, B, W)
    
    # stack all
    emit_if_ins = jnp.broadcast_to( emit_if_ins[None, :, :, :], (T, C, B, W) ) #(T, C, B, W)
    emit_if_del = jnp.broadcast_to( emit_if_del[None, :, :, :], (T, C, B, W) ) #(T, C, B, W)

    joint_emissions = jnp.stack([joint_emit_if_match, 
                                 emit_if_ins, 
                                 emit_if_del], axis=2) #(T, C, S-1, B, W)
    
    # transpose, reshape
    joint_emissions = jnp.transpose(joint_emissions, (4, 0, 1, 2, 3) ) #(W, T, C, S-1, B)
    S_minus_one = joint_emissions.shape[3]
    joint_emissions = jnp.reshape( joint_emissions, (W, T, C*S_minus_one, B) ) #(W, T, C*S-1, B)
    return joint_emissions



###############################################################################
### INDEXING   ################################################################
###############################################################################
def generate_ij_coords_at_diagonal_k(seq_lens,
                                     diagonal_k,
                                     widest_diag_W):
    """
    place (i,j) coordinates on diagonal of the wavefront cache
    
    seq_lens does NOT include start/end tokens.
    
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
    indices : ArrayLike, (B, K, W, 2)
        > dim2=0: i (row: anc)
        > dim2=1: j (column: desc)
        
    mask : ArrayLike, (K, W)
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
    
    anc_len does NOT include start/end tokens.
    
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
    state_idx:
        0 = match
        1 = ins
        2 = del
        (does not include start/end)
    
    C = num_transit_classes
    
    combined class+state indexing in forward carry maps to:
        M_1, I_1, D_1,  M_2, I_2, D_2, ..., M_C, I_C, D_C
    
    this function returns indices corresponding to one state, all classes
    
    
    Arguments:
    ------------
    state_idx : ins
        > M/I/D index
    
    num_transit_classes: ins
        > number of fragment+domain transition classes
        
    
    Returns:
    ---------
    ArrayLike, (C,)
        > indices corresponding to state_idx for every transit class
    
    """
    return jnp.arange(num_transit_classes) * 3 + state_idx


def wavefront_cache_lookup(ij_needed_for_k, 
                           pad_mask_at_k, 
                           cache_for_prev_diagonal, 
                           seq_lens, 
                           pad_val = jnp.finfo(jnp.float32).min):
    """
    Extracts cell value at (i_prev, j_prev) from the cache, which is stored as 
      a diagonal in the wavefront parallelism algo
      
    need (i_prev=i-1, j_prev=j-1) if current state is Match
    need (i_prev=i,   j_prev=j-1) if current state is Ins
    need (i_prev=i-1, j_prev=j)   if current state is Del
    
    
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
    
    pad_value : float
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
                                 pad_val ) #(B, W, T, C_S)
    cache_to_return = jnp.transpose(cache_to_return, (1,2,3,0)) #(W, T, C_S, B)
    
    return cache_to_return

