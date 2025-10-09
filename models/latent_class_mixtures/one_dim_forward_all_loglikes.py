#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 21:25:31 2025

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
                                                                   init_marginals,
                                                                   joint_loglike_emission_len_per_samp,
                                                                   joint_loglike_emission_time_grid,
                                                                   joint_message_passing_len_per_samp,
                                                                   joint_message_passing_time_grid,
                                                                   marginal_message_passing)

def all_loglikes_one_dim_forward_len_per_samp(aligned_inputs,
                                      logprob_emit_at_indel,
                                      joint_logprob_emit_at_match,
                                      all_transit_matrices):
    """
    unique_time_per_sample = True
    
    forward algo to find joint, conditional, and both single-sequence marginal 
        loglikeihoods
    
    IMPORANT: I never carry gradients through this!!!
    
    
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
    
    all_transit_matrices : dict[ArrayLike]
        all_transit_matrices['joint'] : ArrayLike, (B, C, C, S, S)
            logP(new state, new class | prev state, prev class, t); the joint 
            transition matrix for finding logP(anc, desc, align | c, t)
        
        all_transit_matrices['marginal'] : ArrayLike, (C, C, 2, 2)
            logP(new state, new class | prev state, prev class, t); the marginal 
            transition matrix for finding logP(anc | c, t) or logP(desc | c, t)
    
    """
    # unpack
    joint_logprob_transit = all_transit_matrices['joint']  # (B, C, C, S, S)
    marginal_logprob_transit = all_transit_matrices['marginal']  # (C, C, 2, 2)
    
    # get dims
    B = aligned_inputs.shape[0]
    L_align = aligned_inputs.shape[1]
    C = logprob_emit_at_indel.shape[0]
    
    ### initialize with <start> -> any
    # joint: P(anc, desc, align)
    init_joint_alpha = init_fw_len_per_samp( aligned_inputs,
                                        joint_logprob_emit_at_match,
                                        logprob_emit_at_indel,
                                        joint_logprob_transit) #(C, B)
    # logP(anc), logP(desc)
    first_tr = marginal_logprob_transit[0,:,1,0][...,None] #(C, 1)
    out = init_marginals(aligned_inputs = aligned_inputs,
                         logprob_emit_at_indel = logprob_emit_at_indel,
                         first_tr = marginal_logprob_transit[0,:,1,0][...,None] )
    first_anc_emission_flag = out['first_anc_emission_flag']
    first_desc_emission_flag = out['first_desc_emission_flag']
    init_anc_alpha = out['init_anc_alpha']
    init_desc_alpha = out['init_desc_alpha']
    del out
    
    init_dict = {'joint_alpha': init_joint_alpha, # (T, C, B) or (C, B)
                  'anc_alpha': init_anc_alpha,  # (C, B)
                  'desc_alpha': init_desc_alpha,  # (C, B),
                  'md_seen': first_anc_emission_flag, # (B,)
                  'mi_seen': first_desc_emission_flag} # (B,)
    
    
    ######################################################
    ### scan down length dimension to end of alignment   #
    ######################################################
    def scan_fn(carry_dict, pos):
        ### unpack 
        # carry dict
        prev_joint_alpha = carry_dict['joint_alpha'] #(C, B)
        prev_anc_alpha = carry_dict['anc_alpha'] #(C, B)
        prev_desc_alpha = carry_dict['desc_alpha'] #(C, B)
        prev_md_seen = carry_dict['md_seen'] #(B,)
        prev_mi_seen = carry_dict['mi_seen'] #(B,)
        
        # batch
        anc_toks =   aligned_inputs[:,   pos, 0] #(B,)
        desc_toks =  aligned_inputs[:,   pos, 1] #(B,)
        prev_state = aligned_inputs[:, pos-1, 2] #(B,)
        curr_state = aligned_inputs[:,   pos, 2] #(B,)
        curr_state = jnp.where( curr_state!=5, curr_state, 4 ) #(B,)
        
        
        ### emissions
        joint_e = joint_loglike_emission_len_per_samp( aligned_inputs=aligned_inputs,
                                              pos=pos,
                                              joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                              logprob_emit_at_indel=logprob_emit_at_indel ) #(C, B)
        
        anc_mask = (curr_state == 1) | (curr_state == 3) #(B,)
        anc_e = logprob_emit_at_indel[:, anc_toks - 3] * anc_mask  #(C,B)

        desc_mask = (curr_state == 1) | (curr_state == 2)  #(B,)
        desc_e = logprob_emit_at_indel[:, desc_toks - 3] * desc_mask #(C,B)
        
        
        ### flags needed for transitions
        # first_emission_flag: is the current position <s> -> emit?
        # continued_emission_flag: is the current postion emit -> emit?
        # need these because gaps happen in between single sequence 
        #   emissions...
        first_anc_emission_flag = (~prev_md_seen) & anc_mask  #(B,)
        continued_anc_emission_flag = prev_md_seen & anc_mask  #(B,)
        first_desc_emission_flag = (~prev_mi_seen) & desc_mask  #(B,)
        continued_desc_emission_flag = (prev_mi_seen) & desc_mask  #(B,)
        
        
        ### transition probabilities
        def main_body(joint_carry, anc_carry, desc_carry):
            ### logP(anc, desc, align)
            # replace padding idx with 1 to prevent NaN gradients; this doesn't
            #   affect the actual calculated loglike
            ps = jnp.maximum(prev_state, 1) #(B,)
            cs = jnp.maximum(curr_state, 1) #(B,)
            accum_sum = joint_message_passing_len_per_samp( prev_message = joint_carry, 
                                                      ps = ps, 
                                                      cs = cs, 
                                                      joint_logprob_transit = joint_logprob_transit ) #(C_curr, B)
            joint_out = accum_sum + joint_e  #(C_curr, B)
            
            
            ### logP(anc)
            anc_cont_tr = marginal_message_passing(prev_message = anc_carry,
                                                   marginal_logprob_transit = marginal_logprob_transit) #(C_curr, B)
            anc_tr = ( anc_cont_tr * continued_anc_emission_flag +
                        first_tr * first_anc_emission_flag ) # (C_curr, B)
            anc_out = anc_e + anc_tr # (C, B)
            
            
            ### logP(desc)
            desc_cont_tr = marginal_message_passing(prev_message = desc_carry,
                                                    marginal_logprob_transit = marginal_logprob_transit) #(C_curr, B)
            desc_tr = ( desc_cont_tr * continued_desc_emission_flag +
                        first_tr * first_desc_emission_flag ) # (C_curr, B)
            desc_out = desc_e + desc_tr # (C, B)
            
            return (joint_out, anc_out, desc_out)
        
        def end(joint_carry, anc_carry, desc_carry):
            ### logP(anc, desc, align)
            # replace padding idx with 1 to prevent NaN gradients; this doesn't
            #   affect the actual calculated loglike
            ps = jnp.maximum(prev_state, 1) #(B,)
            
            # simple indexing to get end state
            any_to_end = joint_logprob_transit[:, :, -1, :, -1]  # (B, C_prev, S_prev)
            final_tr = any_to_end[jnp.arange(B), :, ps-1] #(B, C_prev)
            final_tr = final_tr.T  # (C_prev, B)
            joint_out = final_tr + joint_carry #(T, C, B)
        
        
            ### logP(anc), logP(desc)
            final_tr = marginal_logprob_transit[:,-1,0,1] #(C,)
            anc_out = anc_carry + final_tr[:,None] #(C, B)
            desc_out = desc_carry + final_tr[:,None] #(C,B)
            
            return (joint_out, anc_out, desc_out)
        
        
        ### alpha updates, in log space 
        continued_out = main_body( prev_joint_alpha, 
                                    prev_anc_alpha, 
                                    prev_desc_alpha )
        end_out = end( prev_joint_alpha, 
                        prev_anc_alpha, 
                        prev_desc_alpha )
        
        # joint: update ONLY if curr_state is not pad
        new_joint_alpha = jnp.where( curr_state != 0,
                                      jnp.where( curr_state != 4,
                                                continued_out[0],
                                                end_out[0] ),
                                      prev_joint_alpha )
        
        # anc marginal; update ONLY if curr_state is not pad or ins
        new_anc_alpha = jnp.where( (curr_state != 0) & (curr_state != 2),
                                      jnp.where( curr_state != 4,
                                                continued_out[1],
                                                end_out[1] ),
                                      prev_anc_alpha )
        
        # desc margianl; update ONLY if curr_state is not pad or del
        new_desc_alpha = jnp.where( (curr_state != 0) & (curr_state != 3),
                                    jnp.where( curr_state != 4,
                                                continued_out[2],
                                                end_out[2] ),
                                    prev_desc_alpha )
        
        out_dict = { 'joint_alpha': new_joint_alpha, #(T, C, B) or (C, B)
                      'anc_alpha': new_anc_alpha, # (C, B)
                      'desc_alpha': new_desc_alpha, # (C, B)
                      'md_seen': (first_anc_emission_flag + prev_md_seen).astype(bool), #(B,)
                      'mi_seen': (first_desc_emission_flag + prev_mi_seen).astype(bool) } #(B,)
        
        return (out_dict, None)

    ### scan over remaining length
    idx_arr = jnp.array( [i for i in range(2, L_align)] )
    out_dict, _ = jax.lax.scan( f = scan_fn,
                                init = init_dict,
                                xs = idx_arr,
                                length = idx_arr.shape[0] )
    final_joint_alpha = out_dict['joint_alpha'] #(C, B) or #(C, B)
    joint_neg_logP = -logsumexp(final_joint_alpha, axis = 0) #(B,)
    
    final_anc_alpha = out_dict['anc_alpha'] #(C, B)
    anc_neg_logP = -logsumexp(final_anc_alpha, axis=0) # (B,)
    
    final_desc_alpha = out_dict['desc_alpha'] #(C, B)
    desc_neg_logP = -logsumexp(final_desc_alpha, axis=0) # (B,)
    
    loglike_dict = {'joint_neg_logP': joint_neg_logP,  #(B,)
                    'anc_neg_logP': anc_neg_logP, # (B,)
                    'desc_neg_logP': desc_neg_logP} # (B,)
    
    return loglike_dict    
    

def all_loglikes_one_dim_forward_time_grid(aligned_inputs,
                          logprob_emit_at_indel,
                          joint_logprob_emit_at_match,
                          all_transit_matrices):
    """
    unique_time_per_sample = False
    
    forward algo to find joint, conditional, and both single-sequence marginal 
        loglikeihoods
    
    IMPORANT: I never carry gradients through this!!!
    
    
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
    
    all_transit_matrices : dict[ArrayLike]
        all_transit_matrices['joint'] : ArrayLike, (T, C, C, S, S)
            logP(new state, new class | prev state, prev class, t); the joint 
            transition matrix for finding logP(anc, desc, align | c, t)
        
        all_transit_matrices['marginal'] : ArrayLike, (C, C, 2, 2)
            logP(new state, new class | prev state, prev class, t); the marginal 
            transition matrix for finding logP(anc | c, t) or logP(desc | c, t)
    
    unique_time_per_sample : Bool 
        whether or not you have unqiue times per sample; affects indexing
        
    Returns:
    ---------
    
    """
    # unpack
    joint_logprob_transit = all_transit_matrices['joint']  # (T, C, C, S, S)
    marginal_logprob_transit = all_transit_matrices['marginal']  # (C, C, 2, 2)
    
    # get dims
    T = joint_logprob_transit.shape[0]
    B = aligned_inputs.shape[0]
    L_align = aligned_inputs.shape[1]
    C = logprob_emit_at_indel.shape[0]
    
    ### initialize with <start> -> any
    # joint: P(anc, desc, align)
    init_joint_alpha = init_fw_time_grid( aligned_inputs,
                                        joint_logprob_emit_at_match,
                                        logprob_emit_at_indel,
                                        joint_logprob_transit) #(C, B)
    
    
    # logP(anc), logP(desc)
    first_tr = marginal_logprob_transit[0,:,1,0][...,None] #(C, 1)
    out = init_marginals(aligned_inputs = aligned_inputs,
                         logprob_emit_at_indel = logprob_emit_at_indel,
                         first_tr = marginal_logprob_transit[0,:,1,0][...,None] )
    first_anc_emission_flag = out['first_anc_emission_flag']
    first_desc_emission_flag = out['first_desc_emission_flag']
    init_anc_alpha = out['init_anc_alpha']
    init_desc_alpha = out['init_desc_alpha']
    del out
    
    init_dict = {'joint_alpha': init_joint_alpha, # (T, C, B) 
                  'anc_alpha': init_anc_alpha,  # (C, B)
                  'desc_alpha': init_desc_alpha,  # (C, B),
                  'md_seen': first_anc_emission_flag, # (B,)
                  'mi_seen': first_desc_emission_flag} # (B,)
    
    
    ######################################################
    ### scan down length dimension to end of alignment   #
    ######################################################
    def scan_fn(carry_dict, pos):
        ### unpack 
        # carry dict
        prev_joint_alpha = carry_dict['joint_alpha'] #(T, C, B) 
        prev_anc_alpha = carry_dict['anc_alpha'] #(C, B)
        prev_desc_alpha = carry_dict['desc_alpha'] #(C, B)
        prev_md_seen = carry_dict['md_seen'] #(B,)
        prev_mi_seen = carry_dict['mi_seen'] #(B,)
        
        # batch
        anc_toks =   aligned_inputs[:,   pos, 0] #(B,)
        desc_toks =  aligned_inputs[:,   pos, 1] #(B,)
        prev_state = aligned_inputs[:, pos-1, 2] #(B,)
        curr_state = aligned_inputs[:,   pos, 2] #(B,)
        curr_state = jnp.where( curr_state!=5, curr_state, 4 ) #(B,)
        
        
        ### emissions
        joint_e = joint_loglike_emission_time_grid( aligned_inputs=aligned_inputs,
                                              pos=pos,
                                              joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                              logprob_emit_at_indel=logprob_emit_at_indel ) #(T, C, B)
        
        anc_mask = (curr_state == 1) | (curr_state == 3) #(B,)
        anc_e = logprob_emit_at_indel[:, anc_toks - 3] * anc_mask  #(C,B)

        desc_mask = (curr_state == 1) | (curr_state == 2)  #(B,)
        desc_e = logprob_emit_at_indel[:, desc_toks - 3] * desc_mask #(C,B)
        
        
        ### flags needed for transitions
        # first_emission_flag: is the current position <s> -> emit?
        # continued_emission_flag: is the current postion emit -> emit?
        # need these because gaps happen in between single sequence 
        #   emissions...
        first_anc_emission_flag = (~prev_md_seen) & anc_mask  #(B,)
        continued_anc_emission_flag = prev_md_seen & anc_mask  #(B,)
        first_desc_emission_flag = (~prev_mi_seen) & desc_mask  #(B,)
        continued_desc_emission_flag = (prev_mi_seen) & desc_mask  #(B,)
        
        
        ### transition probabilities
        def main_body(joint_carry, anc_carry, desc_carry):
            ### logP(anc, desc, align)
            # replace padding idx with 1 to prevent NaN gradients; this doesn't
            #   affect the actual calculated loglike
            ps = jnp.maximum(prev_state, 1) #(B,)
            cs = jnp.maximum(curr_state, 1) #(B,)
            accum_sum = joint_message_passing_time_grid( prev_message = joint_carry, 
                                                      ps = ps, 
                                                      cs = cs, 
                                                      joint_logprob_transit = joint_logprob_transit ) #(T, C_curr, B)
            joint_out = accum_sum + joint_e  #(T, C_curr, B)
            
            
            ### logP(anc)
            anc_cont_tr = marginal_message_passing(prev_message = anc_carry,
                                                   marginal_logprob_transit = marginal_logprob_transit) #(C_curr, B)
            anc_tr = ( anc_cont_tr * continued_anc_emission_flag +
                      first_tr * first_anc_emission_flag ) # (C_curr, B)
            anc_out = anc_e + anc_tr # (C, B)
            
            
            ### logP(desc)
            desc_cont_tr = marginal_message_passing(prev_message = desc_carry,
                                                    marginal_logprob_transit = marginal_logprob_transit) #(C_curr, B)
            desc_tr = ( desc_cont_tr * continued_desc_emission_flag +
                        first_tr * first_desc_emission_flag ) # (C_curr, B)
            desc_out = desc_e + desc_tr # (C, B)
            
            return (joint_out, anc_out, desc_out)
        
        def end(joint_carry, anc_carry, desc_carry):
            ### logP(anc, desc, align)
            # replace padding idx with 1 to prevent NaN gradients; this doesn't
            #   affect the actual calculated loglike
            ps = jnp.maximum(prev_state, 1) #(B,)
            
            # simple indexing to get end state
            final_tr = joint_logprob_transit[:, :, -1, ps-1, -1] #(T, C_prev, B)    
            joint_out = final_tr + joint_carry #(T,C,B) 
            
            
            ### logP(anc), logP(desc)
            final_tr = marginal_logprob_transit[:,-1,0,1] #(C,)
            anc_out = anc_carry + final_tr[:,None] #(C, B)
            desc_out = desc_carry + final_tr[:,None] #(C,B)
            
            return (joint_out, anc_out, desc_out)
        
        
        ### alpha updates, in log space 
        continued_out = main_body( prev_joint_alpha, 
                                    prev_anc_alpha, 
                                    prev_desc_alpha )
        end_out = end( prev_joint_alpha, 
                        prev_anc_alpha, 
                        prev_desc_alpha )
        
        # joint: update ONLY if curr_state is not pad
        new_joint_alpha = jnp.where( curr_state != 0,
                                      jnp.where( curr_state != 4,
                                                continued_out[0],
                                                end_out[0] ),
                                      prev_joint_alpha )
        
        # anc marginal; update ONLY if curr_state is not pad or ins
        new_anc_alpha = jnp.where( (curr_state != 0) & (curr_state != 2),
                                      jnp.where( curr_state != 4,
                                                continued_out[1],
                                                end_out[1] ),
                                      prev_anc_alpha )
        
        # desc margianl; update ONLY if curr_state is not pad or del
        new_desc_alpha = jnp.where( (curr_state != 0) & (curr_state != 3),
                                    jnp.where( curr_state != 4,
                                                continued_out[2],
                                                end_out[2] ),
                                    prev_desc_alpha )
        
        out_dict = { 'joint_alpha': new_joint_alpha, #(T, C, B)
                      'anc_alpha': new_anc_alpha, # (C, B)
                      'desc_alpha': new_desc_alpha, # (C, B)
                      'md_seen': (first_anc_emission_flag + prev_md_seen).astype(bool), #(B,)
                      'mi_seen': (first_desc_emission_flag + prev_mi_seen).astype(bool) } #(B,)
        
        return (out_dict, None)

    ### scan over remaining length
    idx_arr = jnp.array( [i for i in range(2, L_align)] )
    out_dict, _ = jax.lax.scan( f = scan_fn,
                                init = init_dict,
                                xs = idx_arr,
                                length = idx_arr.shape[0] )
    final_joint_alpha = out_dict['joint_alpha'] #(T, C, B)
    joint_neg_logP = -logsumexp(final_joint_alpha, axis = 1) #(T, B)
    
    final_anc_alpha = out_dict['anc_alpha'] #(C, B)
    anc_neg_logP = -logsumexp(final_anc_alpha, axis=0) # (B,)
    
    final_desc_alpha = out_dict['desc_alpha'] #(C, B)
    desc_neg_logP = -logsumexp(final_desc_alpha, axis=0) # (B,)
    
    loglike_dict = {'joint_neg_logP': joint_neg_logP,  #(T, B)
                    'anc_neg_logP': anc_neg_logP, # (B,)
                    'desc_neg_logP': desc_neg_logP} # (B,)
    
    return loglike_dict

def all_loglikes_one_dim_forward(aligned_inputs,
                         logprob_emit_at_indel,
                         joint_logprob_emit_at_match,
                         all_transit_matrices,
                         unique_time_per_sample):
    """
    wrapper fn
    """
    if unique_time_per_sample:
        return all_loglikes_one_dim_forward_len_per_samp(aligned_inputs,
                                                 logprob_emit_at_indel,
                                                 joint_logprob_emit_at_match,
                                                 all_transit_matrices)
    elif not unique_time_per_sample:
        return all_loglikes_one_dim_forward_time_grid(aligned_inputs,
                                              logprob_emit_at_indel,
                                              joint_logprob_emit_at_match,
                                              all_transit_matrices)
    

