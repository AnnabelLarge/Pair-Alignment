#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 13:02:43 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.linalg import expm
from jax._src.typing import Array, ArrayLike

import numpy as np

from models.latent_class_mixtures.one_dim_fwd_bkwd_helpers import (init_recurs_with_len_per_samp,
                                                                   init_recurs_with_time_grid,
                                                                   joint_loglike_emission_len_per_samp,
                                                                   joint_loglike_emission_time_grid,
                                                                   joint_message_passing_len_per_samp,
                                                                   joint_message_passing_time_grid,
                                                                   flip_backward_outputs_with_time_grid,
                                                                   flip_backward_outputs_with_len_per_samp,
                                                                   flip_alignments)

def joint_only_one_dim_backward_time_grid(aligned_inputs,
                                          joint_logprob_emit_at_match,
                                          logprob_emit_at_indel,
                                          joint_logprob_transit):
    """
    unique_time_per_sample = False
    
    backward algo ONLY to find joint loglike
    
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
        the cache from the backward algorithm
    """
    which = 'bkw'
    B = aligned_inputs.shape[0]
    L_align = aligned_inputs.shape[1]
    
    # flip inputs
    flipped_aligned_inputs = flip_alignments(aligned_inputs)  #(B, L, 3)
    del aligned_inputs

    # init   
    init_alpha = init_recurs_with_time_grid( aligned_inputs = flipped_aligned_inputs,
                                             joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                             logprob_emit_at_indel = logprob_emit_at_indel,
                                             joint_logprob_transit = joint_logprob_transit,
                                             which = which ) #(T, C, B)
    
    ### recursion
    def scan_fn(prev_alpha, pos):
        ### unpack
        anc_toks =   flipped_aligned_inputs[:,   pos, 0]
        desc_toks =  flipped_aligned_inputs[:,   pos, 1]
    
        curr_state = flipped_aligned_inputs[:, pos-1, 2]
        prev_state = flipped_aligned_inputs[:,   pos, 2]
        
        # remove invalid indexing tokens; this doesn't affect the actual 
        #   calculated loglike
        curr_state = jnp.where( curr_state!=5, curr_state, 4 )
        prev_state = jnp.where( prev_state!=5, prev_state, 4 )
        
        
        ### emissions
        # TODO: if there are NaN gradients, might have to alter this
        e = joint_loglike_emission_time_grid( aligned_inputs=flipped_aligned_inputs,
                                              pos=pos,
                                              joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                              logprob_emit_at_indel=logprob_emit_at_indel ) # (T, C, B) 
        
        ### message passing
        def main_body(in_carry, ps, cs):
            # replace padding idx with 1 to prevent NaN gradients; this doesn't
            #   affect the actual calculated loglike
            cs = jnp.maximum(cs, 1) #(B,)
            ps = jnp.maximum(ps, 1) #(B,)
            accum_sum = joint_message_passing_time_grid( prev_message = in_carry, 
                                                   ps = ps, 
                                                   cs = cs, 
                                                   joint_logprob_transit = joint_logprob_transit,
                                                   which = which ) #(T, C_prev, B)
            return accum_sum + e  #(T, C_prev, B)
        
        def end(in_carry, ps_not_used, cs):
            # replace padding idx with 1 to prevent NaN gradients; this doesn't
            #   affect the actual calculated loglike
            cs = jnp.maximum(cs, 1) #(B,)
            
            # simple indexing to get start state 
            final_tr = joint_logprob_transit[:, -1, :, -1, cs-1] # (B, T, C)
            final_tr = jnp.transpose(final_tr, (1,2,0)) #(T, C, B)
            
            return final_tr + in_carry #(T, C, B) 
        
        
        ### alpha update, in log space ONLY if prev_state is not pad
        new_alpha = jnp.where(prev_state != 0, 
                              jnp.where( prev_state != 4,
                                         main_body(prev_alpha, prev_state, curr_state),
                                         end(prev_alpha, prev_state, curr_state) ),
                              prev_alpha) #(T, C, B) 
        
        return (new_alpha, new_alpha)
    
    ### end scan function definition, use scan
    # stacked_outputs is cumulative sum PER POSITION, PER TIME
    idx_arr = jnp.array( [ i for i in range(2, L_align) ] ) #(L_align)
    
    _, stacked_outputs = jax.lax.scan( f = scan_fn,
                                        init = init_alpha,
                                        xs = idx_arr,
                                        length = idx_arr.shape[0] )  #(L_align-2, T, C, B) 
    
    # append the first return value (from sentinel -> first alignment column)
    stacked_outputs = jnp.concatenate( [ init_alpha[None,...], #(1, T, C, B)
                                         stacked_outputs ], #(L_align-1, T, C, B)
                                      axis=0) #(L_align-1, T, C, B) 
    
    ### flip this along L_align
    # padding posititions will have same value as l=0
    flipped_stacked_outputs = flip_backward_outputs_with_time_grid( inputs = flipped_aligned_inputs,
                                                                     bkw_stacked_outputs = stacked_outputs ) #(L_align, T, C, B) 
        
    return flipped_stacked_outputs


def joint_only_one_dim_backward_len_per_samp(aligned_inputs,
                                          joint_logprob_emit_at_match,
                                          logprob_emit_at_indel,
                                          joint_logprob_transit):
    """
    unique_time_per_sample = False
    
    backward algo ONLY to find joint loglike
    
    L_align: length of pairwise alignment
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
    
    joint_logprob_transit : ArrayLike, (B, C, C, S, S) 
        logP(new state, new class | prev state, prev class, t); the joint 
        transition matrix for finding logP(anc, desc, align | c, t)
    
    
    Returns:
    ---------
    stacked_outputs : ArrayLike, (L_align, T, C, B) 
        the cache from the backward algorithm
    """
    which = 'bkw'
    B = aligned_inputs.shape[0]
    L_align = aligned_inputs.shape[1]
    
    # flip inputs
    flipped_aligned_inputs = flip_alignments(aligned_inputs)  #(B, L, 3)
    del aligned_inputs

    # init   
    init_alpha = init_recurs_with_len_per_samp( aligned_inputs = flipped_aligned_inputs,
                                             joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                             logprob_emit_at_indel = logprob_emit_at_indel,
                                             joint_logprob_transit = joint_logprob_transit,
                                             which = which ) #(C, B)


    ### recursion
    def scan_fn(prev_alpha, pos):
        ### unpack
        anc_toks =   flipped_aligned_inputs[:,   pos, 0]
        desc_toks =  flipped_aligned_inputs[:,   pos, 1]

        curr_state = flipped_aligned_inputs[:, pos-1, 2]
        prev_state = flipped_aligned_inputs[:,   pos, 2]
        
        # remove invalid indexing tokens; this doesn't affect the actual '
        #   calculated loglike
        curr_state = jnp.where( curr_state!=5, curr_state, 4 )
        prev_state = jnp.where( prev_state!=5, prev_state, 4 )
        
        
        ### emissions
        # TODO: if there are NaN gradients, might have to alter this
        e = joint_loglike_emission_len_per_samp( aligned_inputs=flipped_aligned_inputs,
                                                  pos=pos,
                                                  joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                                  logprob_emit_at_indel=logprob_emit_at_indel ) # (C, B)
        
        ### message passing
        def main_body(in_carry, ps, cs):
            # replace padding idx with 1 to prevent NaN gradients; this doesn't
            #   affect the actual calculated loglike
            cs = jnp.maximum(cs, 1) #(B,)
            ps = jnp.maximum(ps, 1) #(B,)
            accum_sum = joint_message_passing_len_per_samp( prev_message = in_carry, 
                                                      ps = ps, 
                                                      cs = cs, 
                                                      joint_logprob_transit = joint_logprob_transit,
                                                      which = which) #(C_prev, B)
            return accum_sum + e  #(C_prev, B)
            
        def end(in_carry, ps_not_used, cs):
            # replace padding idx with 1 to prevent NaN gradients; this doesn't
            #   affect the actual calculated loglike
            cs = jnp.maximum(cs, 1) #(B,)
            
            # simple indexing to get end state
            any_to_start = joint_logprob_transit[:, -1, :, -1, :]  # (B, C_prev, S_prev)
            final_tr = any_to_start[jnp.arange(B), :, cs-1] #(B, C_prev)
            final_tr = final_tr.T  # (C_prev, B)
            return final_tr + in_carry #(C, B)
        
        
        ### alpha update, in log space ONLY if curr_state is not pad
        new_alpha = jnp.where(prev_state != 0, 
                              jnp.where( prev_state != 4,
                                          main_body(prev_alpha, prev_state, curr_state),
                                          end(prev_alpha, prev_state, curr_state) ),
                              prev_alpha) #(C, B)
        
        return (new_alpha, new_alpha)
    
    
    ### end scan function definition, use scan
    # stacked_outputs is cumulative sum PER POSITION, PER TIME
    idx_arr = jnp.array( [ i for i in range(2, L_align) ] ) #(L_align)
    
    _, stacked_outputs = jax.lax.scan( f = scan_fn,
                                        init = init_alpha,
                                        xs = idx_arr,
                                        length = idx_arr.shape[0] )  #(L_align-1, C, B) 
    
    # append the first return value (from sentinel -> first alignment column)
    stacked_outputs = jnp.concatenate( [ init_alpha[None,...], #(1, C, B)
                                         stacked_outputs ], #(L_align-1, C, B)
                                      axis=0) #(L_align, C, B) 
    
    ### flip this along L_align
    # padding posititions will have same value as l=0
    flipped_stacked_outputs = flip_backward_outputs_with_len_per_samp( inputs = flipped_aligned_inputs,
                                                                  bkw_stacked_outputs = stacked_outputs ) #(L_align, C, B) 
    
    return flipped_stacked_outputs


def joint_only_one_dim_backward(aligned_inputs,
                       joint_logprob_emit_at_match,
                       logprob_emit_at_indel,
                       joint_logprob_transit,
                       unique_time_per_sample: bool):
    """
    Wrapper; see individual functions for more details
    """
    if unique_time_per_sample:
        return joint_only_one_dim_backward_len_per_samp(aligned_inputs,
                                                        joint_logprob_emit_at_match,
                                                        logprob_emit_at_indel,
                                                        joint_logprob_transit)

    elif not unique_time_per_sample:  
        return joint_only_one_dim_backward_time_grid(aligned_inputs,
                                                     joint_logprob_emit_at_match,
                                                     logprob_emit_at_indel,
                                                     joint_logprob_transit)
    