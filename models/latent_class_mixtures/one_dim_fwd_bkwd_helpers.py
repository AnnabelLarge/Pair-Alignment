#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 17:18:45 2025

@author: annabel

'init_fw_len_per_samp',
'init_fw_time_grid',
'init_marginals',
'joint_loglike_emission_len_per_samp',
'joint_loglike_emission_time_grid',
'joint_message_passing_len_per_samp',
'joint_message_passing_time_grid',
'marginal_message_passing',

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
def init_fw_time_grid(aligned_inputs,
                         joint_logprob_emit_at_match,
                         logprob_emit_at_indel,
                         joint_logprob_transit):
    """
    L: length of pairwise alignment
    T: number of timepoints
    B: batch size
    C: number of latent site clases
    S: number of transitions, 4 (M, I, D, S/E)
    A: alphabet size (20 for proteins, 4 for amino acids)
    
    
    Arguments
    ----------
    aligned_inputs : ArrayLike, (B, L, 3)
        dim2=0: ancestor
        dim2=1: descendant
        dim2=2: alignment state; M=1, I=2, D=3, S=4, E=5
    
    joint_logprob_emit_at_match : ArrayLike, (T, C, A, A)
        logP(anc, desc | c, t); log-probability of emission at match site
    
    logprob_emit_at_indel : ArrayLike, (C, A)
        logP(anc | c) or logP(desc | c); log-equilibrium distribution
    
    joint_logprob_transit : ArrayLike, (T, C, C, A, A)
        logP(new state, new class | prev state, prev class, t); the joint 
        transition matrix for finding logP(anc, desc, align | c, t)
    
    
    Returns
    -------
    ArrayLike, (T, C, B)
        logprob of start -> first class and first emission
    """
    # emissions
    e = joint_loglike_emission_time_grid( aligned_inputs=aligned_inputs,
                                          pos=1,
                                          joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                          logprob_emit_at_indel=logprob_emit_at_indel ) # (T, C_curr, B)
    
    # transitions; assume there's never start -> end
    # joint_logprob_transit is (T, C_prev, C_curr, S_prev, S_curr)
    # initial state is 4 (<start>); take the last row
    # use C_prev=0 for start class (but it doesn't matter, because the 
    # transition probability is the same for all C_prev)
    first_state_idx = aligned_inputs[:, 1, 2]-1 #(B,)
    start_any = joint_logprob_transit[:, 0, :, -1, :] #(T, C_curr, S_curr)
    tr = start_any[:, :, first_state_idx] #(T, C_curr, B)
    
    # carry value
    init_alpha = e + tr #(T, C_curr, B)
    return e + tr

def init_fw_len_per_samp(aligned_inputs,
                         joint_logprob_emit_at_match,
                         logprob_emit_at_indel,
                         joint_logprob_transit):
    """
    L: length of pairwise alignment
    T: number of timepoints
    B: batch size
    C: number of latent site clases
    S: number of transitions, 4 (M, I, D, S/E)
    A: alphabet size (20 for proteins, 4 for amino acids)
    
    
    Arguments
    ----------
    aligned_inputs : ArrayLike, (B, L, 3)
        dim2=0: ancestor
        dim2=1: descendant
        dim2=2: alignment state; M=1, I=2, D=3, S=4, E=5
    
    joint_logprob_emit_at_match : ArrayLike, (B, C, A, A)
        logP(anc, desc | c, t); log-probability of emission at match site
    
    logprob_emit_at_indel : ArrayLike, (C, A)
        logP(anc | c) or logP(desc | c); log-equilibrium distribution
    
    joint_logprob_transit : ArrayLike, (B, C, C, A, A)
        logP(new state, new class | prev state, prev class, t); the joint 
        transition matrix for finding logP(anc, desc, align | c, t)
    
    
    Returns
    -------
    ArrayLike, (C, B)
        logprob of start -> first class and first emission
    """
    B = aligned_inputs.shape[0]
    
    # emissions
    e = joint_loglike_emission_len_per_samp( aligned_inputs=aligned_inputs,
                                             pos=1,
                                             joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                             logprob_emit_at_indel=logprob_emit_at_indel ) # (C_curr, B)
    
    # transitions; assume there's never start -> end
    # joint_logprob_transit is (T, C_prev, C_curr, S_prev, S_curr)
    # initial state is 4 (<start>); take the last row
    # use C_prev=0 for start class (but it doesn't matter, because the 
    # transition probability is the same for all C_prev)
    first_state_idx = aligned_inputs[:, 1, 2] #(B,)
    start_any = joint_logprob_transit[:, 0, :, -1, :] #(B, C_curr, S_curr)
    tr = start_any[jnp.arange(B), :, first_state_idx-1] #(B, C_curr)
    tr = tr.T #(C_curr, B)
    
    # carry value
    init_alpha = e + tr #(C_curr, B)
    return e + tr

def init_marginals(aligned_inputs,
                   logprob_emit_at_indel,
                   first_tr ):
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
                                 joint_logprob_transit):
    """
    joint_logprob_transit is (B, C_prev, C_curr, S_prev, S_curr)
    prev_message is (C_prev, B)
    """
    ps = ps-1 #(B,)
    cs = cs-1 #(B,)
    B = ps.shape[0]
    
    tr_per_class = joint_logprob_transit[jnp.arange(B), :, :, ps, cs] #(B, C_prev, C_curr)
    tr_per_class = jnp.transpose(tr_per_class, (1, 2, 0)) #(C_prev, C_curr, B) 
    new_message = logsumexp(prev_message[:, None, :] + tr_per_class, axis=0) #(C_curr, B) 
    return new_message


def joint_message_passing_time_grid(prev_message, 
                              ps, 
                              cs, 
                              joint_logprob_transit):
    """
    joint_logprob_transit is (T, C_prev, C_curr, S_prev, S_curr)
    prev_message is (T, C_prev, B)
    """
    ps = ps-1 #(B,)
    cs = cs-1 #(B,)
    tr_per_class = joint_logprob_transit[..., ps, cs] #(T, C_prev, C_curr, B)   
    new_message = logsumexp(prev_message[:, :, None, :] + tr_per_class, axis=1) #(T, C_curr, B)
    return new_message

    
def marginal_message_passing(prev_message, 
                                     marginal_logprob_transit):
    """
    prev_message is (C_prev, B)
    marginal_logprob_transit is (C_prev, C_curr, 2, 2)
    """
    prev_message_reshaped = prev_message[:,None,:] #(C_prev, 1, B)
    marginal_logprob_transit_reshaped = marginal_logprob_transit[...,0,0][...,None] #(C_prev, C_curr, 1)
    to_logsumexp = prev_message_reshaped + marginal_logprob_transit_reshaped #(C_prev, C_curr, B)
    return logsumexp(to_logsumexp, axis=0) # (C_curr, B)
