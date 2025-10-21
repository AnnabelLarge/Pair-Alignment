#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 17:18:45 2025

@author: annabel


"""
import jax
from jax import numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.linalg import expm
from jax._src.typing import Array, ArrayLike

from functools import partial
import numpy as np


###############################################################################
### SCORING EMISSIONS   #######################################################
###############################################################################
def joint_loglike_emission_time_grid(aligned_inputs,
                                         pos,
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
    aligned_inputs : ArrayLike, (B, L, 3)
        dim2=0: ancestor
        dim2=1: descendant
        dim2=2: alignment state; M=1, I=2, D=3, S=4, E=5
    
    pos : int
        which alignment column you're at
    
    joint_logprob_emit_at_match : ArrayLike, (T, C, A, A)
        logP(anc, desc | c, t); log-probability of emission at match site
    
    logprob_emit_at_indel : ArrayLike, (C, A)
        logP(anc | c) or logP(desc | c); log-equilibrium distribution
        
    Returns
    -------
    joint_e : ArrayLike, (T, C, B)
        log-probability of emission at given column, across all possible 
        site classes
    """
    T = joint_logprob_emit_at_match.shape[0]
    C = joint_logprob_emit_at_match.shape[1]
    B = aligned_inputs.shape[0]
    
    # unpack
    anc_toks = aligned_inputs[:,pos,0]-3 #(B,)
    desc_toks = aligned_inputs[:,pos,1]-3 #(B,)
    state_at_pos = aligned_inputs[:,pos,2]-1 #(B,)
    
    # get all possible scores
    joint_emit_if_match = joint_logprob_emit_at_match[..., anc_toks, desc_toks] # (T, C, B) or (C, B)
    emit_if_indel_desc = logprob_emit_at_indel[:, desc_toks] #(C, B)
    emit_if_indel_anc = logprob_emit_at_indel[:, anc_toks] #(C, B)
    
    # stack all
    emit_if_indel_desc = jnp.broadcast_to( emit_if_indel_desc[None, :, :], 
                                           (T, C, B) ) #(T, C, B)
    emit_if_indel_anc = jnp.broadcast_to( emit_if_indel_anc[None, :, :], 
                                          (T, C, B) ) #(T, C, B)
    joint_emissions = jnp.stack([joint_emit_if_match, 
                                 emit_if_indel_desc, 
                                 emit_if_indel_anc], axis=0) #(3, T, C, B)
    
    joint_e = joint_emissions[state_at_pos, :, :, jnp.arange(B)] #(B, T, C)
    joint_e = jnp.transpose( joint_e, (1,2,0) ) #(T, C, B)

    return joint_e

def joint_loglike_emission_len_per_samp(aligned_inputs,
                                                   pos,
                                                   joint_logprob_emit_at_match,
                                                   logprob_emit_at_indel):
    """
    ONE branch length per sample
        joint_logprob_emit_at_match is (B, C, A, A)
    
    can use this function in forward and backward functions to find 
      emission probabilities (which are site independent)
    
    L: length of pairwise alignment
    T: number of timepoints
    B: batch size
    C: number of latent site clases
    A: alphabet size (20 for proteins, 4 for amino acids)
    
    
    Arguments
    ----------
    aligned_inputs : ArrayLike, (B, L, 3)
        dim2=0: ancestor
        dim2=1: descendant
        dim2=2: alignment state; M=1, I=2, D=3, S=4, E=5
    
    pos : int
        which alignment column you're at
    
    joint_logprob_emit_at_match : ArrayLike, (B, C, A, A)
        logP(anc, desc | c, t); log-probability of emission at match site
    
    logprob_emit_at_indel : ArrayLike, (C, A)
        logP(anc | c) or logP(desc | c); log-equilibrium distribution
        
    Returns
    -------
    joint_e : ArrayLike, (C, B)
        log-probability of emission at given column, across all possible 
        site classes
    """
    B = joint_logprob_emit_at_match.shape[0]
    C = joint_logprob_emit_at_match.shape[1]
    
    # unpack
    anc_toks = aligned_inputs[:,pos,0]-3
    desc_toks = aligned_inputs[:,pos,1]-3
    state_at_pos = aligned_inputs[:,pos,2]-1
    
    joint_emit_if_match = joint_logprob_emit_at_match[jnp.arange(B), :, anc_toks, desc_toks] #(B, C)
    joint_emit_if_match = joint_emit_if_match.T #(C,B)
    
    # Indels: (C, B)
    emit_if_indel_desc = logprob_emit_at_indel[:, desc_toks] #(C, B)
    emit_if_indel_anc = logprob_emit_at_indel[:, anc_toks] # (C, B)

    joint_emissions = jnp.stack( [joint_emit_if_match,
                                  emit_if_indel_desc,
                                  emit_if_indel_anc], axis=0)  # (3, C, B)
    
    joint_e = joint_emissions[state_at_pos, :, jnp.arange(B)] #(B,C)
    joint_e = joint_e.T #(C,B)

    return joint_e


###############################################################################
### INIT FUNCTIONS   ##########################################################
###############################################################################
def init_recurs_with_time_grid(aligned_inputs,
                               joint_logprob_emit_at_match,
                               logprob_emit_at_indel,
                               joint_logprob_transit,
                               which):
    """
    T: number of timepoints
    B: batch size
    L: length of pairwise alignment
    C: number of latent site clases
    S: number of transitions, 4 (M, I, D, S/E)
    A: alphabet size (20 for proteins, 4 for amino acids)
    
    
    Arguments
    ----------
    aligned_inputs : ArrayLike, (B, L, 3)
        dim2=0: ancestor
        dim2=1: descendant
        dim2=2: alignment state; M=1, I=2, D=3, S=4, E=5
        already reversed, if doing backward algo
    
    joint_logprob_emit_at_match : ArrayLike, (T, C, A, A)
        logP(anc, desc | c, t); log-probability of emission at match site
    
    logprob_emit_at_indel : ArrayLike, (C, A)
        logP(anc | c) or logP(desc | c); log-equilibrium distribution
    
    joint_logprob_transit : ArrayLike, (T, C, C, A, A)
        logP(new state, new class | prev state, prev class, t); the joint 
        transition matrix for finding logP(anc, desc, align | c, t)
    
    which : str
        if "fw", then moving from prev -> curr; return logprob 
        start -> first class and first emission
            
        if "bkw", then moving from curr -> prev; return logprob 
        end -> last class and last emission
    
    Returns
    -------
    ArrayLike, (T, C, B)
        initial value for forward or backward algo
    """
    ### emissions
    e = joint_loglike_emission_time_grid( aligned_inputs=aligned_inputs,
                                          pos=1,
                                          joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                          logprob_emit_at_indel=logprob_emit_at_indel ) # (T, C, B)
    
    
    ### transitions
    state_idx = aligned_inputs[:, 1, 2]-1 #(B,)
    
    # joint_logprob_transit is (T, C_prev, C_curr, S_prev, S_curr)
    if which == 'fw':    
        # initial state is 4 (<start>); take the last row
        # use C_prev=0 for start class (but it doesn't matter, because the 
        # transition probability is the same for all C_prev)
        start_any = joint_logprob_transit[:, 0, :, -1, :] #(T, C_curr, S_curr)
        tr = start_any[:, :, state_idx] #(T, C_curr, B)
    
    elif which == 'bkw':
        # initial state is 5 (<end>); take the last column
        # use C_prev=-1 for end class (but it doesn't matter, because the 
        # transition probability is the same for all C_prev)
        end_any = joint_logprob_transit[:, :, -1, :, -1] #(T, C_prev, S_prev)
        tr = end_any[:, :, state_idx] #(T, C_prev, B)
        
    # carry value
    init_alpha = e + tr #(T, C, B)
    return e + tr


def init_recurs_with_len_per_samp(aligned_inputs,
                                  joint_logprob_emit_at_match,
                                  logprob_emit_at_indel,
                                  joint_logprob_transit,
                                  which):
    """
    B: batch size
    L: length of pairwise alignment
    C: number of latent site clases
    S: number of transitions, 4 (M, I, D, S/E)
    A: alphabet size (20 for proteins, 4 for amino acids)
    
    
    Arguments
    ----------
    aligned_inputs : ArrayLike, (B, L, 3)
        dim2=0: ancestor
        dim2=1: descendant
        dim2=2: alignment state; M=1, I=2, D=3, S=4, E=5
        already reversed, if doing backward algo
    
    joint_logprob_emit_at_match : ArrayLike, (B, C, A, A)
        logP(anc, desc | c, t); log-probability of emission at match site
    
    logprob_emit_at_indel : ArrayLike, (C, A)
        logP(anc | c) or logP(desc | c); log-equilibrium distribution
    
    joint_logprob_transit : ArrayLike, (B, C, C, A, A)
        logP(new state, new class | prev state, prev class, t); the joint 
        transition matrix for finding logP(anc, desc, align | c, t)
    
    which : str
        if "fw", then algo is moving from prev -> curr; return logprob 
        start -> first class and first emission
            
        if "bkw", then algo is moving from curr -> prev; return logprob 
        end -> last class and last emission
        
    
    Returns
    -------
    ArrayLike, (C, B)
        initial value for forward or backward algo
    """
    B = aligned_inputs.shape[0]
    
    ### emissions
    e = joint_loglike_emission_len_per_samp( aligned_inputs=aligned_inputs,
                                             pos=1,
                                             joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                             logprob_emit_at_indel=logprob_emit_at_indel ) # (C_curr, B)
    
    ### transitions
    state_idx = aligned_inputs[:, 1, 2]-1 #(B,)
    
    # joint_logprob_transit is (B, C_prev, C_curr, S_prev, S_curr)
    if which == 'fw':
        # prev = start
        # curr = first alignment column
        # prev -> curr
        #
        # initial state is 4 (<start>); take the last row
        # use C_prev=0 for start class (but it doesn't matter, because the 
        # transition probability is the same for all C_prev)
        start_any = joint_logprob_transit[:, 0, :, -1, :] #(B, C_curr, S_curr)
        tr = start_any[jnp.arange(B), :, state_idx] #(B, C_curr)
        
    elif which == 'bkw':
        # prev = end
        # curr = last alignment column
        # curr -> prev
        #
        # initial state is 5 (<end>); take the last column
        # use C_curr=-1 for end class (but it doesn't matter, because the 
        # transition probability is the same for all C_curr)
        end_any = joint_logprob_transit[:, :, -1, :, -1] #(B, C_prev, S_prev)
        tr = end_any[jnp.arange(B), :, state_idx] #(B, C_prev)
        
    # carry value
    tr = tr.T #(C, B)
    init_alpha = e + tr #(C, B)
    return e + tr    


def init_marginals(aligned_inputs,
                   logprob_emit_at_indel,
                   first_tr,
                   which ):
    # start at pos=1
    anc_toks =   aligned_inputs[:, 1, 0] #(B,)
    desc_toks =  aligned_inputs[:, 1, 1] #(B,)
    curr_state = aligned_inputs[:, 1, 2] #(B,)
    
    ### logP(anc)
    # emissions; only valid if current position is match or delete
    anc_mask = (curr_state == 1) | (curr_state == 3)  # (B,)
    init_anc_e = logprob_emit_at_indel[:, anc_toks - 3] * anc_mask  # (C, B)
    
    # transitions (if anc emitted yet)
    first_anc_emission_flag = anc_mask  # (B,)
    init_anc_tr = first_tr * first_anc_emission_flag  # (C, B)
    init_anc_alpha = init_anc_e + init_anc_tr # (C, B)
    del init_anc_e, init_anc_tr, anc_mask
    
    
    ### logP(desc); (C, B)
    # emissions; only valid if current position is match or ins
    desc_mask = (curr_state == 1) | (curr_state == 2) #(B,)
    init_desc_e = logprob_emit_at_indel[:, desc_toks - 3] * desc_mask # (C, B)
    
    # transitions (if desc emitted yet)
    first_desc_emission_flag = desc_mask # (B,)
    init_desc_tr = first_tr * first_desc_emission_flag # (C, B)
    init_desc_alpha = init_desc_e + init_desc_tr  # (C, B)
    del init_desc_e, init_desc_tr, desc_mask, curr_state
    
    return {'first_anc_emission_flag': first_anc_emission_flag,
            'first_desc_emission_flag': first_desc_emission_flag,
            'init_anc_alpha': init_anc_alpha,
            'init_desc_alpha': init_desc_alpha }


###############################################################################
### MESSAGE PASSING   #########################################################
###############################################################################
def joint_message_passing_len_per_samp(prev_message, 
                                 ps, 
                                 cs, 
                                 joint_logprob_transit,
                                 which):
    """
    joint_logprob_transit is (B, C_prev, C_curr, S_prev, S_curr)
    
    if which == 'fw', then pass message FORWARD (prev -> curr)
    if which == 'bwd', then pass message BACKWARDS (curr -> prev)
    """
    ps = ps-1 #(B,)
    cs = cs-1 #(B,)
    B = ps.shape[0]
    
    tr_per_class = joint_logprob_transit[jnp.arange(B), :, :, ps, cs] #(B, C_prev, C_curr)
    tr_per_class = jnp.transpose(tr_per_class, (1, 2, 0)) #(C_prev, C_curr, B) 
    
    if which == 'fw':
        # prev -> curr
        # prev_message is (C_prev, B)
        # new_message should be (C_curr, B)
        prev_message_expanded = prev_message[:, None, :] #(C_prev, 1, B)
        to_lse = prev_message_expanded + tr_per_class #(C_prev, C_curr, B)
        new_message = logsumexp( to_lse, axis=0 ) #(C_curr, B)
    
    elif which == 'bkw':
        # curr -> prev
        # prev_message is (C_curr, B)
        # new_message should be (C_prev, B)
        prev_message_expanded = prev_message[None, :, :] #(1, C_curr, B)
        to_lse = prev_message_expanded + tr_per_class #(C_prev, C_curr, B)
        new_message = logsumexp( to_lse, axis=1 ) #(C_prev, B)
        
    return new_message


def joint_message_passing_time_grid(prev_message, 
                              ps, 
                              cs, 
                              joint_logprob_transit,
                              which):
    """
    joint_logprob_transit is (T, C_prev, C_curr, S_prev, S_curr)
    
    if which == 'fw', then pass message FORWARD (prev -> curr)
    if which == 'bwd', then pass message BACKWARDS (curr -> prev)
    """
    ps = ps-1 #(B,)
    cs = cs-1 #(B,)
    tr_per_class = joint_logprob_transit[..., ps, cs] #(T, C_prev, C_curr, B)   
    
    
    if which == 'fw':
        # prev -> curr
        # prev_message is (T, C_prev, B)
        # new_message should be (T, C_curr, B)
        prev_message_expanded = prev_message[:, :, None, :] #(T, C_prev, 1, B)
        to_lse = prev_message_expanded + tr_per_class #(T, C_prev, C_curr, B)
        new_message = logsumexp( to_lse, axis=1 ) #(T, C_curr, B)
    
    elif which == 'bkw':
        # curr -> prev
        # prev_message is (C_curr, B)
        # new_message should be (C_prev, B)
        prev_message_expanded = prev_message[:, None, :, :] #(T, 1, C_curr, B)
        to_lse = prev_message_expanded + tr_per_class #(T, C_prev, C_curr, B)
        new_message = logsumexp( to_lse, axis=2 ) #(T, C_prev, B)
    
    return new_message

    
def marginal_message_passing(prev_message, 
                             marginal_logprob_transit,
                             which):
    """
    prev_message is (C_prev, B)
    marginal_logprob_transit is (C_prev, C_curr, 2, 2)
    """
    prev_message_reshaped = prev_message[:,None,:] #(C_prev, 1, B)
    marginal_logprob_transit_reshaped = marginal_logprob_transit[...,0,0][...,None] #(C_prev, C_curr, 1)
    to_logsumexp = prev_message_reshaped + marginal_logprob_transit_reshaped #(C_prev, C_curr, B)
    return logsumexp(to_logsumexp, axis=0) # (C_curr, B)



###############################################################################
### BACKWARD HELPERS   ########################################################
###############################################################################
def flip_alignments(inputs):
    """
    adapted from flax.linen.recurrent.flip_sequences
    https://github.com/google/flax/blob/ \
        c0ea12d3ecae1b87982131dbb637547b9f4eb43a/flax/linen/recurrent.py#L1180
    
    flips along axis 1, but keeps padding at the end!
    
    example:
        
        [[1, 1, 4],
         [3, 4, 1],
         [2, 2, 5],
         [0, 0, 0],
         [0, 0, 0]]
        
             |
             v
             
        [[2, 2, 5],
         [3, 4, 1],
         [1, 1, 4],
         [0, 0, 0],
         [0, 0, 0]]
        
    
    Arguments:
    ------------
    inputs : ArrayLike, (B, L, 3)
        aligned inputs
        dim0 = aligned ancestor
        dim1 = aligned descendant
        dim2 = state
       
    Returns:
    ---------
    outputs : ArrayLike, (B, L, 3)
        inputs, flipped along length axis
        
    """
    B = inputs.shape[0]
    L = inputs.shape[1]
    
    seq_lengths = (inputs[...,0] != 0).sum(axis=1) #(B,)
    max_steps = inputs.shape[1]
    seq_lengths = seq_lengths[:,None,None] #(B, 1, 1)
    
    idxs = jnp.arange(max_steps - 1, -1, -1)  # (L,)
    idxs = jnp.reshape( idxs, (1, max_steps, 1) ) #(1, L, 1)
    idxs = (idxs + seq_lengths) % max_steps  # (B, L, 1)
    idxs = jnp.broadcast_to( idxs, (B, L, 3) ) #(B, L, 3)
    
    outputs = jnp.take_along_axis( inputs, idxs, axis=1 )
    
    return outputs


def flip_backward_outputs_with_time_grid( inputs,
                                           bkw_stacked_outputs ):
    """
    adapted from flax.linen.recurrent.flip_sequences
    https://github.com/google/flax/blob/ \
        c0ea12d3ecae1b87982131dbb637547b9f4eb43a/flax/linen/recurrent.py#L1180
    
    flips along axis 0, but keeps padding at the end!
    
    example: [1,2,3,0,0] -> [3,2,1,0,0]
        
    
    Arguments:
    ------------
    inputs : ArrayLike, (B, L, 3)
        used to determine indexes
    
    bkw_stacked_outputs : ArrayLike, (L, T, C, B) 
        outputs from running backward algorithm
       
    Returns:
    ---------
    outputs : ArrayLike, (L, T, C, B) 
        bkw_stacked_outputs, flipped along length axis
        
    """
    B = inputs.shape[0]
    L = bkw_stacked_outputs.shape[0]
    T = bkw_stacked_outputs.shape[1]
    C = bkw_stacked_outputs.shape[2]
    
    seq_lengths = (inputs[...,0] != 0).sum(axis=1)-1 #(B,)
    seq_lengths = seq_lengths[None,None,None,:] #(1, 1, 1, B)
    
    idxs = jnp.arange(L - 1, -1, -1)  # (L_align,)
    idxs = jnp.reshape( idxs, (L, 1, 1, 1) ) #(L_align, 1, 1, 1)
    idxs = (idxs + seq_lengths) % L  # (L_align, 1, 1, B)
    idxs = jnp.broadcast_to( idxs, (L, T, C, B) ) #(L_align, T, C, B) 
    
    outputs = jnp.take_along_axis( bkw_stacked_outputs, idxs, axis=0 ) #(L_align, T, C, B) 
    
    return outputs


def flip_backward_outputs_with_len_per_samp( inputs,
                                              bkw_stacked_outputs ):
    """
    adapted from flax.linen.recurrent.flip_sequences
    https://github.com/google/flax/blob/ \
        c0ea12d3ecae1b87982131dbb637547b9f4eb43a/flax/linen/recurrent.py#L1180
    
    flips along axis 0, but keeps padding at the end!
    
    example: [1,2,3,0,0] -> [3,2,1,0,0]
        
    
    Arguments:
    ------------
    inputs : ArrayLike, (B, L, 3)
        used to determine indexes
    
    bkw_stacked_outputs : ArrayLike, (L, C, B) 
        outputs from running backward algorithm
       
    Returns:
    ---------
    outputs : ArrayLike, (L, C, B) 
        bkw_stacked_outputs, flipped along length axis
        
    """
    B = inputs.shape[0]
    L = bkw_stacked_outputs.shape[0]
    C = bkw_stacked_outputs.shape[1]
    
    seq_lengths = (inputs[...,0] != 0).sum(axis=1)-1 #(B,)
    max_steps = L
    seq_lengths = seq_lengths[None,None,:] #(1, 1, B)
    
    idxs = jnp.arange(L - 1, -1, -1)  # (L_align,)
    idxs = jnp.reshape( idxs, (L, 1, 1) ) #(L_align, 1, 1)
    idxs = (idxs + seq_lengths) % L  # (L_align, 1, B)
    idxs = jnp.broadcast_to( idxs, (L, C, B) ) #(L_align, C, B) 
    
    outputs = jnp.take_along_axis( bkw_stacked_outputs, idxs, axis=0 ) #(L_align, C, B) 
    
    return outputs