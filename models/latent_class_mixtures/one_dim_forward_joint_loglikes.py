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

from models.latent_class_mixtures.one_dim_fwd_bkwd_helpers import (init_fw_len_per_samp,
                                                                   init_fw_time_grid,
                                                                   joint_loglike_emission_len_per_samp,
                                                                   joint_loglike_emission_time_grid,
                                                                   joint_message_passing_len_per_samp,
                                                                   joint_message_passing_time_grid)


def joint_only_one_dim_forward_len_per_samp(aligned_inputs,
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
            accum_sum = joint_message_passing_len_per_samp( prev_message = in_carry, 
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
            return final_tr + in_carry #(C, B)
        
        
        ### alpha update, in log space ONLY if curr_state is not pad
        new_alpha = jnp.where(curr_state != 0, 
                              jnp.where( curr_state != 4,
                                          main_body(prev_alpha, prev_state, curr_state),
                                          end(prev_alpha, prev_state, curr_state) ),
                              prev_alpha) #(C, B)
        
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


def joint_only_one_dim_forward_time_grid(aligned_inputs,
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
    # note to self: CAN'T make this a parted function, because that will
    # trigger a new jit-compilation EVERY time this function is called :(
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
            accum_sum = joint_message_passing_time_grid( prev_message = in_carry, 
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
    

def joint_only_one_dim_forward(aligned_inputs,
                       joint_logprob_emit_at_match,
                       logprob_emit_at_indel,
                       joint_logprob_transit,
                       unique_time_per_sample: bool, 
                       return_all_intermeds: bool = False):
    """
    Wrapper; see individual functions for more details
    """
    if unique_time_per_sample:
        return joint_only_one_dim_forward_len_per_samp(aligned_inputs,
                                            joint_logprob_emit_at_match,
                                            logprob_emit_at_indel,
                                            joint_logprob_transit,
                                            return_all_intermeds)

    elif not unique_time_per_sample:  
        return joint_only_one_dim_forward_time_grid(aligned_inputs,
                                            joint_logprob_emit_at_match,
                                            logprob_emit_at_indel,
                                            joint_logprob_transit,
                                            return_all_intermeds)
