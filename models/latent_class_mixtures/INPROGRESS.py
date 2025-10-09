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


def message_passing_len_per_samp(prev_message, 
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

def message_passing_time_grid(prev_message, 
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


def joint_only_forward_len_per_samp(aligned_inputs,
                                    joint_logprob_emit_at_match,
                                    logprob_emit_at_indel,
                                    joint_logprob_transit,
                                    return_all_intermeds: bool = False):
    """
    unique_time_per_sample = True
    
    forward algo ONLY to find joint loglike
    
    L_align: length of pairwise alignment
    T: number of timepoints
    B: batch size
    C_trans = C: number of latent site clases
      > could be C_frag or C_dom * C_frag
    A: alphabet (20 for proteins, 4 for DNA)
    S: possible states; here, this is 4: M, I, D, start/end
    
    Arguments
    ----------
    aligned_inputs : ArrayLike, (B, L, 3)
        dim2=0: ancestor
        dim2=1: descendant
        dim2=2: alignment state; M=1, I=2, D=3, S=4, E=5
    
    joint_logprob_emit_at_match : ArrayLike, (B, C, A, A)
        logP(anc, desc | c, t); log-probability of emission at match site
    
    logprob_emit_at_indel : ArrayLike, (C, A)
        logP(anc | c) or P(desc | c); log-equilibrium distribution
    
    joint_logprob_transit : ArrayLike, (B, C, C, A, A)
        logP(new state, new class | prev state, prev class, t); the joint 
        transition matrix for finding logP(anc, desc, align | c, t)
    
    
    Returns:
    ---------
    stacked_outputs : ArrayLike, (L_align, C, B)
        the cache from the forward algorithm; this is the total log-probability 
        of ending at a given alignment column (l \in L_align) in class C, given
        the observed alignment
        
        to marginalize over all possible combinations of hidden site classes 
        for a given alignment: extract the final element of the length 
        dimension (i.e. stacked_outputs[-1,...]) and do logsumexp over all 
        classes C. This leaves you with the joint probability of the observed 
        alignment, at all branch lengths in T
    """
    B = aligned_inputs.shape[0]
    L_align = aligned_inputs.shape[1]
    
    ### initialize with <start> -> any 
    init_alpha = init_fw_len_per_samp( aligned_inputs,
                                        joint_logprob_emit_at_match,
                                        logprob_emit_at_indel,
                                        joint_logprob_transit) #(C, B)
    
    
    ######################################################
    ### scan down length dimension to end of alignment   #
    ######################################################
    def scan_fn(prev_alpha, pos):
        ### unpack
        anc_toks =   aligned_inputs[:,   pos, 0]
        desc_toks =  aligned_inputs[:,   pos, 1]

        prev_state = aligned_inputs[:, pos-1, 2]
        curr_state = aligned_inputs[:,   pos, 2]
        
        # remove invalid indexing tokens; this doesn't affect the actual '
        #   calculated loglike
        prev_state = jnp.where( prev_state!=5, prev_state, 4 )
        curr_state = jnp.where( curr_state!=5, curr_state, 4 )
        
        
        ### emissions
        # TODO: if there are NaN gradients, might have to alter this
        e = joint_loglike_emission_len_per_samp( aligned_inputs=aligned_inputs,
                                                  pos=pos,
                                                  joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                                  logprob_emit_at_indel=logprob_emit_at_indel ) # (C, B)
        
        ### message passing
        def main_body(in_carry, ps, cs):
            # replace padding idx with 1 to prevent NaN gradients; this doesn't
            #   affect the actual calculated loglike
            ps = jnp.maximum(ps, 1) #(B,)
            cs = jnp.maximum(cs, 1) #(B,)
            accum_sum = message_passing_len_per_samp( prev_message = in_carry, 
                                                      ps = ps, 
                                                      cs = cs, 
                                                      joint_logprob_transit = joint_logprob_transit ) #(C_curr, B)
            return accum_sum + e  #(C_curr, B)
            
        def end(in_carry, ps, arg_not_used):
            # replace padding idx with 1 to prevent NaN gradients; this doesn't
            #   affect the actual calculated loglike
            ps = jnp.maximum(ps, 1) #(B,)
            
            # simple indexing to get end state
            any_to_end = joint_logprob_transit[:, :, -1, :, -1]  # (B, C_prev, S_prev)
            final_tr = any_to_end[jnp.arange(B), :, ps-1] #(B, C_prev)
            final_tr = final_tr.T  # (C_prev, B)
            return final_tr + in_carry #(T, C, B) or (C, B)
        
        
        ### alpha update, in log space ONLY if curr_state is not pad
        new_alpha = jnp.where(curr_state != 0, 
                              jnp.where( curr_state != 4,
                                          main_body(prev_alpha, prev_state, curr_state),
                                          end(prev_alpha, prev_state, curr_state) ),
                              prev_alpha) #(T, C, B) or (C, B)
        
        return (new_alpha, new_alpha)
    
    ### end scan function definition, use scan
    # stacked_outputs is cumulative sum PER POSITION, PER TIME
    idx_arr = jnp.array( [ i for i in range(2, L_align) ] ) #(L_align)
    
    if not return_all_intermeds:
        last_alpha, _ = jax.lax.scan( f = scan_fn,
                                      init = init_alpha,
                                      xs = idx_arr,
                                      length = idx_arr.shape[0] )  #(C, B)
        
        loglike = logsumexp(last_alpha,  # (C, B)
                            axis = 1 if not unique_time_per_sample else 0)
        
        return loglike #(B,)

        
    elif return_all_intermeds:
        _, stacked_outputs = jax.lax.scan( f = scan_fn,
                                            init = init_alpha,
                                            xs = idx_arr,
                                            length = idx_arr.shape[0] )  #(L_align-1, C, B)
        
        # append the first return value (from sentinel -> first alignment column)
        stacked_outputs = jnp.concatenate( [ init_alpha[None,...], #(1, C, B)
                                             stacked_outputs ], #(L_align-1, C, B)
                                          axis=0) #(L_align, C, B)
        
        return stacked_outputs #(L_align, C, B)


def joint_only_forward_time_grid(aligned_inputs,
                                    joint_logprob_emit_at_match,
                                    logprob_emit_at_indel,
                                    joint_logprob_transit,
                                    return_all_intermeds: bool = False):
    """
    unique_time_per_sample = False
    
    forward algo ONLY to find joint loglike
    
    L_align: length of pairwise alignment
    T: number of timepoints
    B: batch size
    C_trans = C: number of latent site clases
      > could be C_frag or C_dom * C_frag
    A: alphabet (20 for proteins, 4 for DNA)
    S: possible states; here, this is 4: M, I, D, start/end
    
    Arguments
    ----------
    aligned_inputs : ArrayLike, (B, L, 3)
        dim2=0: ancestor
        dim2=1: descendant
        dim2=2: alignment state; M=1, I=2, D=3, S=4, E=5
    
    joint_logprob_emit_at_match : ArrayLike, (T, C, A, A) 
        logP(anc, desc | c, t); log-probability of emission at match site
    
    logprob_emit_at_indel : ArrayLike, (C, A)
        logP(anc | c) or P(desc | c); log-equilibrium distribution
    
    joint_logprob_transit : ArrayLike, (T, C, C, S, S) 
        logP(new state, new class | prev state, prev class, t); the joint 
        transition matrix for finding logP(anc, desc, align | c, t)
    
    
    Returns:
    ---------
    stacked_outputs : ArrayLike, (L_align, T, C, B) 
        the cache from the forward algorithm; this is the total log-probability 
        of ending at a given alignment column (l \in L_align) in class C, given
        the observed alignment
        
        to marginalize over all possible combinations of hidden site classes 
        for a given alignment: extract the final element of the length 
        dimension (i.e. stacked_outputs[-1,...]) and do logsumexp over all 
        classes C. This leaves you with the joint probability of the observed 
        alignment, at all branch lengths in T
    """
    B = aligned_inputs.shape[0]
    L_align = aligned_inputs.shape[1]
    
    ### initialize with <start> -> any 
    init_alpha = init_fw_time_grid( aligned_inputs,
                                    joint_logprob_emit_at_match,
                                    logprob_emit_at_indel,
                                    joint_logprob_transit) #(T, C, B)
    
    ######################################################
    ### scan down length dimension to end of alignment   #
    ######################################################
    def scan_fn(prev_alpha, pos):
        ### unpack
        anc_toks =   aligned_inputs[:,   pos, 0]
        desc_toks =  aligned_inputs[:,   pos, 1]

        prev_state = aligned_inputs[:, pos-1, 2]
        curr_state = aligned_inputs[:,   pos, 2]
        
        # remove invalid indexing tokens; this doesn't affect the actual 
        #   calculated loglike
        prev_state = jnp.where( prev_state!=5, prev_state, 4 )
        curr_state = jnp.where( curr_state!=5, curr_state, 4 )
        
        
        ### emissions
        # TODO: if there are NaN gradients, might have to alter this
        e = joint_loglike_emission_time_grid( aligned_inputs=aligned_inputs,
                                              pos=pos,
                                              joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                              logprob_emit_at_indel=logprob_emit_at_indel ) # (T, C, B) 
        
        
        ### message passing
        def main_body(in_carry, ps, cs):
            # replace padding idx with 1 to prevent NaN gradients; this doesn't
            #   affect the actual calculated loglike
            ps = jnp.maximum(ps, 1) #(B,)
            cs = jnp.maximum(cs, 1) #(B,)
            accum_sum = message_passing_time_grid( prev_message = in_carry, 
                                                   ps = ps, 
                                                   cs = cs, 
                                                   joint_logprob_transit = joint_logprob_transit ) #(T, C_curr, B)
            return accum_sum + e  #(T, C_curr, B)
        
        def end(in_carry, ps, cs_not_used):
            # replace padding idx with 1 to prevent NaN gradients; this doesn't
            #   affect the actual calculated loglike
            ps = jnp.maximum(ps, 1)
            
            # simple indexing to get end state
            final_tr = joint_logprob_transit[:, :, -1, ps-1, -1] #(T, C_prev, B)    
            
            return final_tr + in_carry #(T, C, B) or (C, B)
        
        
        ### alpha update, in log space ONLY if curr_state is not pad
        new_alpha = jnp.where(curr_state != 0, 
                              jnp.where( curr_state != 4,
                                          main_body(prev_alpha, prev_state, curr_state),
                                          end(prev_alpha, prev_state, curr_state) ),
                              prev_alpha) #(T, C, B) or (C, B)
        
        return (new_alpha, new_alpha)
    
    ### end scan function definition, use scan
    # stacked_outputs is cumulative sum PER POSITION, PER TIME
    idx_arr = jnp.array( [ i for i in range(2, L_align) ] ) #(L_align)
    
    if not return_all_intermeds:
        last_alpha, _ = jax.lax.scan( f = scan_fn,
                                      init = init_alpha,
                                      xs = idx_arr,
                                      length = idx_arr.shape[0] )  #(T, C, B) 
        
        loglike = logsumexp(last_alpha,  # (T, C, B)  or (C, B)
                            axis = 1 if not unique_time_per_sample else 0)
        
        return loglike #(T, B)

        
    elif return_all_intermeds:
        _, stacked_outputs = jax.lax.scan( f = scan_fn,
                                            init = init_alpha,
                                            xs = idx_arr,
                                            length = idx_arr.shape[0] )  #(L_align-1, T, C, B) 
        
        # append the first return value (from sentinel -> first alignment column)
        stacked_outputs = jnp.concatenate( [ init_alpha[None,...], #(1, T, C, B)
                                             stacked_outputs ], #(L_align-1, T, C, B)
                                          axis=0) #(L_align, T, C, B) 
        
        return stacked_outputs #(L_align, T, C, B) 
    


def joint_only_forward(aligned_inputs,
                       joint_logprob_emit_at_match,
                       logprob_emit_at_indel,
                       joint_logprob_transit,
                       unique_time_per_sample: bool, 
                       return_all_intermeds: bool = False):
    """
    Wrapper; see individual functions for more details
    """
    if unique_time_per_sample:
        return joint_only_forward_len_per_samp(aligned_inputs,
                                            joint_logprob_emit_at_match,
                                            logprob_emit_at_indel,
                                            joint_logprob_transit,
                                            return_all_intermeds)

    elif not unique_time_per_sample:  
        return joint_only_forward_time_grid(aligned_inputs,
                                            joint_logprob_emit_at_match,
                                            logprob_emit_at_indel,
                                            joint_logprob_transit,
                                            return_all_intermeds)
        
    
    
    
    
# def _log_space_dot_prod_helper(alpha,
#                                marginal_logprob_transit):
#     """
#     a helper used in all_loglikes_forward
#     """
#     alpha_reshaped = alpha[:,None,:] #(C_prev, 1, B)
#     marginal_logprob_transit_reshaped = marginal_logprob_transit[...,0,0][...,None] #(C_prev, C_curr, 1)
#     to_logsumexp = alpha_reshaped + marginal_logprob_transit_reshaped #(C_prev, C_curr, B)
#     return logsumexp(to_logsumexp, axis=0) # (C_curr, B)


# def all_loglikes_forward(aligned_inputs,
#                          logprob_emit_at_indel,
#                          joint_logprob_emit_at_match,
#                          all_transit_matrices,
#                          unique_time_per_sample: bool):
#     """
#     TODO: this should be ALMOST ready for nested TKF92 model... but come back  
#           and check this out later
#           > without domains: crude memory variables to remember if start -> emit
#             has been seen yet; this handles alignments that start with a 
#             start -> ins transitions
#           > with domains: ??? 
    
#     forward algo to find joint, conditional, and both single-sequence marginal 
#         loglikeihoods
    
#     IMPORANT: I never carry gradients through this!!!
    
    
#     L_align: length of pairwise alignment
#     T: number of timepoints
#     B: batch size
#     C_trans = C: number of latent site clases
#       > could be C_frag or C_dom * C_frag
#     A: alphabet (20 for proteins, 4 for DNA)
#     S: possible states; here, this is 4: M, I, D, start/end
    
#     Arguments
#     ----------
#     aligned_inputs : ArrayLike, (B, L, 3)
#         dim2=0: ancestor
#         dim2=1: descendant
#         dim2=2: alignment state; M=1, I=2, D=3, S=4, E=5
    
#     joint_logprob_emit_at_match : ArrayLike, (T, C, A, A)
#         logP(anc, desc | c, t); log-probability of emission at match site
    
#     logprob_emit_at_indel : ArrayLike, (C, A)
#         logP(anc | c) or P(desc | c); log-equilibrium distribution
    
#     all_transit_matrices : dict[ArrayLike]
#         all_transit_matrices['joint'] : ArrayLike, (T, C, C, S, S)
#             logP(new state, new class | prev state, prev class, t); the joint 
#             transition matrix for finding logP(anc, desc, align | c, t)
        
#         all_transit_matrices['marginal'] : ArrayLike, (C, C, 2, 2)
#             logP(new state, new class | prev state, prev class, t); the marginal 
#             transition matrix for finding logP(anc | c, t) or logP(desc | c, t)
    
#     unique_time_per_sample : Bool 
#         whether or not you have unqiue times per sample; affects indexing
        
#     Returns:
#     ---------
    
#     """
#     joint_logprob_transit = all_transit_matrices['joint']
#     marginal_logprob_transit = all_transit_matrices['marginal'] 
    
#     # decide which version of the functions you're going to use
#     if not unique_time_per_sample:
#         # output from this is (T, C, B)
#         get_joint_loglike_emission = get_joint_loglike_emission_time_grid
        
#     elif unique_time_per_sample:
#         # output from this is (C, B)
#         get_joint_loglike_emission = get_joint_loglike_emission_branch_len_per_samp
    
#     B = aligned_inputs.shape[0]
#     L_align = aligned_inputs.shape[1]
#     C = logprob_emit_at_indel.shape[0]
    
#     # memory for single-sequence marginals
#     anc_alpha = jnp.zeros( (C, B) ) #(C, B)
#     desc_alpha = jnp.zeros( (C, B) ) #(C, B)
#     md_seen = jnp.zeros( B, ).astype(bool) #(B,)
#     mi_seen = jnp.zeros( B, ).astype(bool) #(B,)
    
#     ######################################################
#     ### initialize with <start> -> any (curr pos is 1)   #
#     ######################################################
#     pos = 1
#     anc_toks =   aligned_inputs[:, pos, 0] #(B,)
#     desc_toks =  aligned_inputs[:, pos, 1] #(B,)
#     curr_state = aligned_inputs[:, pos, 2] #(B,)

    
#     ### joint: P(anc, desc, align)
#     # emissions; 
#     joint_e = get_joint_loglike_emission( aligned_inputs=aligned_inputs,
#                                     pos=pos,
#                                     joint_logprob_emit_at_match=joint_logprob_emit_at_match,
#                                     logprob_emit_at_indel=logprob_emit_at_indel ) # (T, C, B)
    
#     # transitions; assume there's never start -> end; 
#     # joint_logprob_transit is (T, C_prev, C_curr, S_prev, S_curr)
#     # initial state is 4 (<start>); take the last row
#     # use C_prev=0 for start class (but it doesn't matter, because the 
#     # transition probability is the same for all C_prev)
#     curr_state_idx = curr_state - 1         # (B,)
#     start_any = joint_logprob_transit[:, 0, :, -1, :] #(T, C_curr, S_curr) or (B, C_curr, S_curr)
    
#     if not unique_time_per_sample:
#         joint_tr = start_any[..., curr_state_idx] #(T, C_curr, B)
    
#     elif unique_time_per_sample:
#         # joint_logprob_transit: (B, C_curr, S_curr)
#         # goal: (C_curr, B)
#         joint_tr = jnp.take_along_axis(
#             start_any, 
#             curr_state_idx[:, None, None],  # shape (B, 1)
#             axis=-1
#         ) 
#         joint_tr = joint_tr[...,0].T #(C_curr, B)
        
#     # carry value
#     init_joint_alpha = joint_e + joint_tr # (T, C, B) or (C, B)
#     del joint_e, joint_tr, start_any
    
    
#     ### logP(anc)
#     # emissions; only valid if current position is match or delete
#     anc_mask = (curr_state == 1) | (curr_state == 3)  # (B,)
#     init_anc_e = logprob_emit_at_indel[:, anc_toks - 3] * anc_mask  # (C, B)
    
#     # transitions
#     # marginal_logprob_transit is (C_prev, C_curr, S_prev, S_curr), where:
#     #   (S_prev=0, S_curr=0) is emit->emit
#     #   (S_prev=1, S_curr=0) is <s>->emit
#     #   (S_prev=0, S_curr=1) is emit-><e>
#     # use C_prev=0 for start class (but it doesn't matter, because the 
#     # transition probability is the same for all C_prev)
#     # transition prob for <s>->emit
#     first_anc_emission_flag = (~md_seen) & anc_mask  # (B,)
#     anc_first_tr = marginal_logprob_transit[0,:,1,0][...,None] #(C_curr, 1)
    
#     # transition prob for emit->emit
#     continued_anc_emission_flag = md_seen & anc_mask  # (B,)
#     anc_cont_tr = _log_space_dot_prod_helper(alpha = anc_alpha,
#                                             marginal_logprob_transit = marginal_logprob_transit)  # (C_curr, B)
    
#     # possibilities are: <s>->emit transition, emit->emit transition, or  
#     #   nothing happened (at an indel site where ancestor was not emitted yet)
#     init_anc_tr = ( anc_cont_tr * continued_anc_emission_flag +
#                     anc_first_tr * first_anc_emission_flag ) # (C, B)
    
#     # things to remember are:
#     #   alpha: for forward algo
#     #   md_seen: used to remember if <s> -> emit has been used yet
#     #   (there could be gaps in between <s> and first emission)
#     init_anc_alpha = init_anc_e + init_anc_tr # (C, B)
#     del init_anc_e, init_anc_tr, anc_mask
    
    
#     ### logP(desc); (C, B)
#     # emissions; only valid if current position is match or ins
#     desc_mask = (curr_state == 1) | (curr_state == 2) #(B,)
#     init_desc_e = logprob_emit_at_indel[:, desc_toks - 3] * desc_mask # (C, B)
    
#     # transitions
#     first_desc_emission_flag = (~mi_seen) & desc_mask # (B,)
#     desc_first_tr = marginal_logprob_transit[0,:,1,0][...,None] #(C_curr, 1)
    
#     continued_desc_emission_flag = mi_seen & desc_mask # (B,)
#     desc_cont_tr = _log_space_dot_prod_helper(alpha = desc_alpha,
#                                              marginal_logprob_transit = marginal_logprob_transit)  # (C_curr, B)
    
#     init_desc_tr = ( desc_cont_tr * continued_desc_emission_flag +
#                      desc_first_tr * first_desc_emission_flag ) # (C, B)
    
#     # things to remember are:
#     #   alpha: for forward algo
#     #   mi_seen: used to remember if <s> -> emit has been used yet
#     #   (there could be gaps in between <s> and first emission)
#     init_desc_alpha = init_desc_e + init_desc_tr  # (C, B)
#     del init_desc_e, init_desc_tr, desc_mask, curr_state
    
#     init_dict = {'joint_alpha': init_joint_alpha, # (T, C, B) or (C, B)
#                  'anc_alpha': init_anc_alpha,  # (C, B)
#                  'desc_alpha': init_desc_alpha,  # (C, B),
#                  'md_seen': first_anc_emission_flag, # (B,)
#                  'mi_seen': first_desc_emission_flag} # (B,)
    
    
#     ######################################################
#     ### scan down length dimension to end of alignment   #
#     ######################################################
#     def scan_fn(carry_dict, pos):
#         ### unpack 
#         # carry dict
#         prev_joint_alpha = carry_dict['joint_alpha'] #(T, C, B) or (C, B)
#         prev_anc_alpha = carry_dict['anc_alpha'] #(C, B)
#         prev_desc_alpha = carry_dict['desc_alpha'] #(C, B)
#         prev_md_seen = carry_dict['md_seen'] #(B,)
#         prev_mi_seen = carry_dict['mi_seen'] #(B,)
        
#         # batch
#         anc_toks =   aligned_inputs[:,   pos, 0] #(B,)
#         desc_toks =  aligned_inputs[:,   pos, 1] #(B,)
#         prev_state = aligned_inputs[:, pos-1, 2] #(B,)
#         curr_state = aligned_inputs[:,   pos, 2] #(B,)
#         curr_state = jnp.where( curr_state!=5, curr_state, 4 ) #(B,)
        
        
#         ### emissions
#         joint_e = get_joint_loglike_emission( aligned_inputs=aligned_inputs,
#                                               pos=pos,
#                                               joint_logprob_emit_at_match=joint_logprob_emit_at_match,
#                                               logprob_emit_at_indel=logprob_emit_at_indel ) #(T, C, B)
        
#         anc_mask = (curr_state == 1) | (curr_state == 3) #(B,)
#         anc_e = logprob_emit_at_indel[:, anc_toks - 3] * anc_mask  #(C,B)

#         desc_mask = (curr_state == 1) | (curr_state == 2)  #(B,)
#         desc_e = logprob_emit_at_indel[:, desc_toks - 3] * desc_mask #(C,B)
        
        
#         ### flags needed for transitions
#         # first_emission_flag: is the current position <s> -> emit?
#         # continued_emission_flag: is the current postion emit -> emit?
#         # need these because gaps happen in between single sequence 
#         #   emissions...
#         first_anc_emission_flag = (~prev_md_seen) & anc_mask  #(B,)
#         continued_anc_emission_flag = prev_md_seen & anc_mask  #(B,)
#         first_desc_emission_flag = (~prev_mi_seen) & desc_mask  #(B,)
#         continued_desc_emission_flag = (prev_mi_seen) & desc_mask  #(B,)
        
        
#         ### transition probabilities
#         def main_body(joint_carry, anc_carry, desc_carry):
#             # logP(anc, desc, align)
#             if not unique_time_per_sample:
#                 # joint_logprob_transit is (T, C_prev, C_curr, S_prev, S_curr)
#                 joint_tr_per_class = joint_logprob_transit[..., prev_state-1, curr_state-1] #(T, C_prev, C_curr, B)   
#                 to_add = logsumexp(joint_carry[:, :, None, :] + joint_tr_per_class, axis=1) #(T, C_curr, B)
            
#             elif unique_time_per_sample:
#                 # joint_logprob_transit is (B, C_prev, C_curr, S_prev, S_curr)
#                 ps_idx = (prev_state - 1)[:, None, None, None, None] #(B, 1, 1, 1, 1)
#                 cs_idx = (curr_state - 1)[:, None, None, None, None] #(B, 1, 1, 1, 1)
#                 transit_ps = jnp.take_along_axis(joint_logprob_transit, ps_idx, axis=3)  # (B, C_prev, C_curr, 1, S_curr)
#                 transit_ps_cs = jnp.take_along_axis(transit_ps, cs_idx, axis=4)  # (B, C_prev, C_curr, 1, 1)
#                 transit_ps_cs = transit_ps_cs[:,:,:,0,0] # (B, C_prev, C_curr)
#                 joint_tr_per_class = jnp.transpose(transit_ps_cs, (1, 2, 0)) #(C_prev, C_curr, B) 
#                 to_add = logsumexp(joint_carry[:, None, :] + joint_tr_per_class, axis=0) #(C_curr, B)
                         
#             joint_out = joint_e + to_add #(T, C_curr, B) or (C_curr, B)
            
            
#             # logP(anc)
#             anc_first_tr = marginal_logprob_transit[0,:,1,0][...,None] #(C_curr, 1)
#             anc_cont_tr = _log_space_dot_prod_helper(alpha = anc_carry,
#                                                     marginal_logprob_transit = marginal_logprob_transit) #(C_curr, B)
#             anc_tr = ( anc_cont_tr * continued_anc_emission_flag +
#                        anc_first_tr * first_anc_emission_flag ) # (C_curr, B)
#             anc_out = anc_e + anc_tr # (C, B)
            
            
#             # logP(desc)
#             desc_first_tr = marginal_logprob_transit[0,:,1,0][...,None] #(C_curr, 1)
#             desc_cont_tr = _log_space_dot_prod_helper(alpha = desc_carry,
#                                                     marginal_logprob_transit = marginal_logprob_transit) #(C_curr, B)
#             desc_tr = ( desc_cont_tr * continued_desc_emission_flag +
#                         desc_first_tr * first_desc_emission_flag ) # (C_curr, B)
#             desc_out = desc_e + desc_tr # (C, B)
            
#             return (joint_out, anc_out, desc_out)
        
#         def end(joint_carry, anc_carry, desc_carry):
#             # note for all: if end, then curr_state = -1 (<end>)
#             # logP(anc, desc, align)
#             if not unique_time_per_sample:
#                 joint_tr_per_class = joint_logprob_transit[..., -1, prev_state-1, -1] #(T, C_prev, B)    
            
#             elif unique_time_per_sample:
#                 sliced = joint_logprob_transit[:, :, -1, :, -1]  # (B, C_prev, S_prev)
#                 ps_idx = (prev_state - 1)[:, None]  # (B, 1)
#                 gathered = jnp.take_along_axis(sliced, ps_idx[:, None, :], axis=2)  # (B, C_prev, 1)
#                 gathered = gathered[:,:,0]# (B, C_prev)
#                 joint_tr_per_class = gathered.T  # (C_prev, B)
            
#             joint_out = joint_tr_per_class + joint_carry #(T,C,B) or (C,B)
            
            
#             # logP(anc)
#             final_anc_tr = marginal_logprob_transit[:,-1,0,1] #(C,)
#             final_anc_tr = jnp.broadcast_to( final_anc_tr[:,None], anc_carry.shape ) #(C, B)
#             anc_out = anc_carry + final_anc_tr #(C, B)
            
            
#             # logP(desc)
#             final_desc_tr = marginal_logprob_transit[:,-1,0,1] #(C,)
#             final_desc_tr = jnp.broadcast_to( final_desc_tr[:,None], desc_carry.shape ) #(C,B)
#             desc_out = desc_carry + final_desc_tr #(C,B)
            
#             return (joint_out, anc_out, desc_out)
        
        
#         ### alpha updates, in log space 
#         continued_out = main_body( prev_joint_alpha, 
#                                    prev_anc_alpha, 
#                                    prev_desc_alpha )
#         end_out = end( prev_joint_alpha, 
#                        prev_anc_alpha, 
#                        prev_desc_alpha )
        
#         # joint: update ONLY if curr_state is not pad
#         new_joint_alpha = jnp.where( curr_state != 0,
#                                      jnp.where( curr_state != 4,
#                                                 continued_out[0],
#                                                 end_out[0] ),
#                                      prev_joint_alpha )
        
#         # anc marginal; update ONLY if curr_state is not pad or ins
#         new_anc_alpha = jnp.where( (curr_state != 0) & (curr_state != 2),
#                                      jnp.where( curr_state != 4,
#                                                 continued_out[1],
#                                                 end_out[1] ),
#                                      prev_anc_alpha )
        
#         # desc margianl; update ONLY if curr_state is not pad or del
#         new_desc_alpha = jnp.where( (curr_state != 0) & (curr_state != 3),
#                                     jnp.where( curr_state != 4,
#                                                continued_out[2],
#                                                end_out[2] ),
#                                     prev_desc_alpha )
        
#         out_dict = { 'joint_alpha': new_joint_alpha, #(T, C, B) or (C, B)
#                      'anc_alpha': new_anc_alpha, # (C, B)
#                      'desc_alpha': new_desc_alpha, # (C, B)
#                      'md_seen': (first_anc_emission_flag + prev_md_seen).astype(bool), #(B,)
#                      'mi_seen': (first_desc_emission_flag + prev_mi_seen).astype(bool) } #(B,)
        
#         return (out_dict, None)

#     ### scan over remaining length
#     idx_arr = jnp.array( [i for i in range(2, L_align)] )
#     out_dict, _ = jax.lax.scan( f = scan_fn,
#                                                init = init_dict,
#                                                xs = idx_arr,
#                                                length = idx_arr.shape[0] )
#     final_joint_alpha = out_dict['joint_alpha'] #(T, C, B) or #(C, B)
#     joint_neg_logP = -logsumexp(final_joint_alpha, 
#                                 axis = 1 if not unique_time_per_sample else 0) #(T, B) or (B,)
    
#     final_anc_alpha = out_dict['anc_alpha'] #(C, B)
#     anc_neg_logP = -logsumexp(final_anc_alpha, axis=0) # (B,)
    
#     final_desc_alpha = out_dict['desc_alpha'] #(C, B)
#     desc_neg_logP = -logsumexp(final_desc_alpha, axis=0) # (B,)
    
#     loglike_dict = {'joint_neg_logP': joint_neg_logP,  #(T, B) or (B,)
#                     'anc_neg_logP': anc_neg_logP, # (B,)
#                     'desc_neg_logP': desc_neg_logP} # (B,)
    
#     return loglike_dict