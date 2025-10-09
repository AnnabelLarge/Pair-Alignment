#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 18:32:24 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
import flax.linen as nn

from models.latent_class_mixtures.two_dim_forward_algo_helpers import (generate_ij_coords_at_diagonal_k,
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

def two_dim_forward_with_time_grid(unaligned_seqs,
                                   joint_logprob_transit,
                                   joint_logprob_emit_at_match,
                                   logprob_emit_at_indel,
                                   return_full_grid = False):
    """
    Marginalize over alignments and latent site classes; later, marginalize
      over times in a geometrically-distributed grid
    
    Implement this with a wavefront approach: move in a diagonal from top-left
      to bottom-right of alignment grid
    
    
    B: batch
    W: width of wavefront cache; equal to longest bottom-left-to-top-right 
     diagonal in the alignment grid
    T: times in the grid
    C_transit: number of latent classes for transitions (domain, fragment)
    S: number of alignment states (4: Match, Ins, Del, Start/End)
    C_S: C_transit * (S-1), combined dim for state+class, like M_c, I_c, D_c, etc.
    A: alphabet size (20 for proteins, 4 for amino acids)
    
    
    Arguements:
    -----------
    unaligned_seqs : ArrayLike,
    
    joint_logprob_transit : ArrayLike, (T, C_transit, S_prev, C_transit, S_curr)
        transition logprobs
        MAY HAVE TO TRANSPOSE THIS, depending on transition function

    joint_logprob_emit_at_match : ArrayLike, (T, C_transit, A, A)
        substitution logprobs; used to score emissions from match sites

    logprob_emit_at_indel : ArrayLike, (C_transit, A)
        equilibrium distributions; used to score emissions from indel sites
    
    return_full_grid : bool
        in debugging, return all the diagonals of the wavefront cache
    
    
    Returns:
    --------
    forward_logprob : ArrayLike, (T, B)
        logP( anc, desc | t ); score per sample and per time
    
    """
    # widest diagonal for wavefront parallelism is min(anc_len, desc_len) + 1
    seq_lens = (unaligned_seqs != 0).sum(axis=1)-2 #(B, 2)
    min_lens = seq_lens.min(axis=1) #(B,)
    
    # infer dims
    B = unaligned_seqs.shape[0]
    T = joint_logprob_transit.shape[0]
    S = joint_logprob_transit.shape[-1]
    C_transit = logprob_emit_at_indel.shape[0]
    A = logprob_emit_at_indel.shape[1]
    C_S = C_transit * (S-1) 
    W = min_lens.max() + 1 
    K = (seq_lens.sum(axis=1)).max()
    del min_lens
    
    
    ################################################
    ### Initialize cache for wavefront diagonals   #
    ################################################
    # fill diagonal k=1: alignment cells (1,0) and (0,1)
    diag_k1 = init_first_diagonal( cache_size = (W, T, C_S, B),
                                   unaligned_seqs = unaligned_seqs,
                                   joint_logprob_transit = joint_logprob_transit,
                                   logprob_emit_at_indel = logprob_emit_at_indel )  #(2, W, T, C_S, B)
    
    # fill diag k-1: alignment cells (1,1), and (if applicable) (0,2) and/or (2,0)
    out = init_second_diagonal( cache_for_prev_diagonal = diag_k1, 
                                unaligned_seqs = unaligned_seqs,
                                joint_logprob_transit = joint_logprob_transit,
                                joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                logprob_emit_at_indel = logprob_emit_at_indel,
                                seq_lens = seq_lens ) 
    
    diag_k2 = out[0] #(W, T, C_S, B)
    joint_logprob_transit_mid_only = out[1] #(T, C_S_prev, C_S_curr )
    del out 
    
    
    ########################
    ### Start Recurrence   #
    ########################
    def scan_fn( carry: dict, 
                 k : int ):
        ############
        ### init   #
        ############
        ### unpack the carry
        # diagonal at k-1 (previous diagonal; used to get [i, j-1] and [i-1, j] )
        cache_for_prev_diagonal = carry['cache_for_prev_diagonal'] #(W, T, C_S, B)
    
        # diagonal at k-2 (diagonal BEFORE previous diagonal; used to get [i-1, j-1] )
        cache_two_diags_prior = carry['cache_two_diags_prior'] #(W, T, C_S, B)
        
        # at the end, this will ALWAYS be the bottom-right cell, at [i=anc_len, j=desc_len]
        prev_alignment_score = carry['alignment_score_per_class_and_state'] #(T, C_S, B)
        del carry
        
        
        #### prep for recurrence 
        # blank diagonal k; fill this in
        cache_at_curr_k = jnp.full( (W, T, C_S, B), jnp.finfo(jnp.float32).min ) # (W, T, C*S, B)
        
        # align_cell_idxes is (B, W, 2)
        # pad_mask is (B, W)
        # pad_mask is True at valid cells, False at padding locations
        align_cell_idxes, pad_mask = generate_ij_coords_at_diagonal_k(seq_lens = seq_lens,
                                                                      diagonal_k = k,
                                                                      widest_diag_W = W)
        
        
        ##################################
        ### message passing, per state   #
        ##################################
        ### c, d: latent site class
        ### S: some alignment state, M/I/D (not start/end)
        
        ### match: 
        ### \sum_{s, c} Tr( curr_state = Match, curr_class = d | prev_state = S, prev_class = c, t ) * \alpha_{i-1, j-1}^{S_c}
        out = get_match_transition_message( align_cell_idxes = align_cell_idxes,
                                            pad_mask = pad_mask,
                                            cache_at_curr_diagonal = cache_at_curr_k,
                                            cache_two_diags_prior = cache_two_diags_prior,
                                            seq_lens = seq_lens,
                                            joint_logprob_transit_mid_only = joint_logprob_transit_mid_only,
                                            C_transit = C_transit )
        match_idx = out[0] #(C,)
        match_transit_message = out[1] #(W, T, C, B)
        del out
        
        # after this step, cache contains: 
        # P(curr_state = Match, curr_class = d, anc_{...,i-1}, desc_{...,j-1} | t )
        cache_at_curr_k = update_cache(idx_arr_for_state = match_idx, 
                                       transit_message = match_transit_message, 
                                       cache_to_update = cache_at_curr_k) # (W, T, C*S, B)
        
        
        ### ins
        ### \sum_{s, c} Tr( curr_state = Ins, curr_class = d | prev_state = S, prev_class = c, t ) * \alpha_{i, j-1}^{S_c}
        out = get_ins_transition_message( align_cell_idxes = align_cell_idxes,
                                          pad_mask = pad_mask,
                                          cache_at_curr_diagonal = cache_at_curr_k,
                                          cache_for_prev_diagonal = cache_for_prev_diagonal,
                                          seq_lens = seq_lens,
                                          joint_logprob_transit_mid_only = joint_logprob_transit_mid_only,
                                          C_transit = C_transit )
        ins_idx = out[0] #(C_transit,)
        ins_transit_message = out[1] #(W, T, C_transit, B)
        del out
        
        # after this step, cache contains: 
        # P(curr_state = Match, curr_class = d, anc_{...,i-1}, desc_{...,j-1} | t )
        # P(curr_state = Ins,   curr_class = d, anc_{...,i},   desc_{...,j-1} | t )
        cache_at_curr_k = update_cache(idx_arr_for_state = ins_idx, 
                                       transit_message = ins_transit_message, 
                                       cache_to_update = cache_at_curr_k) # (W, T, C*S, B)
        
        
        ### del
        ### \sum_{s, c} Tr( curr_state = Del, curr_class = d | prev_state = S, prev_class = c, t ) * \alpha_{i-1, j}^{S_c}
        out = get_del_transition_message( align_cell_idxes = align_cell_idxes,
                                          pad_mask = pad_mask,
                                          cache_at_curr_diagonal = cache_at_curr_k,
                                          cache_for_prev_diagonal = cache_for_prev_diagonal,
                                          seq_lens = seq_lens,
                                          joint_logprob_transit_mid_only = joint_logprob_transit_mid_only,
                                          C_transit = C_transit )
        del_idx = out[0] #(C,)
        del_transit_message = out[1] #(W, T, C, B)
        del out
        
        # after this step, cache contains: 
        # P(curr_state = Match, curr_class = d, anc_{...,i-1}, desc_{...,j-1} | t )
        # P(curr_state = Ins,   curr_class = d, anc_{...,i},   desc_{...,j-1} | t )
        # P(curr_state = Del,   curr_class = d, anc_{...,i-1}, desc_{...,j}   | t )
        cache_at_curr_k = update_cache(idx_arr_for_state = del_idx, 
                                       transit_message = del_transit_message,  
                                       cache_to_update = cache_at_curr_k) # (W, T, C*S, B)
        
        
        ######################################
        ### update messages with emissions   #
        ######################################
        # get emission tokens; at padding positions in diagonal, these will also be pad
        anc_toks_at_diag_k = unaligned_seqs[jnp.arange(B)[:, None], align_cell_idxes[...,0], 0] #(B, W)
        desc_toks_at_diag_k = unaligned_seqs[jnp.arange(B)[:, None], align_cell_idxes[...,1], 1] #(B, W)
        
        # use emissions to index scoring matrices
        # at invalid positions, this is ZERO (not jnp.finfo(jnp.float32).min)!
        emit_logprobs_at_k = joint_loglike_emission_at_k_time_grid( anc_toks = anc_toks_at_diag_k,
                                                                    desc_toks = desc_toks_at_diag_k,
                                                                    joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                                                    logprob_emit_at_indel = logprob_emit_at_indel, 
                                                                    fill_invalid_pos_with = 0.0 ) # (W, T, C*S, B)
        
        # after this step, cache contains: 
        # P(curr_state = Match, curr_class = d, anc_{...,i}, desc_{...,j} | t )
        # P(curr_state = Ins,   curr_class = d, anc_{...,i}, desc_{...,j} | t )
        # P(curr_state = Del,   curr_class = d, anc_{...,i}, desc_{...,j} | t )
        cache_at_curr_k = cache_at_curr_k + emit_logprobs_at_k # (W, T, C*S, B)
        
        
        ####################################################
        ### Final recordings, updates for next iteration   #
        ####################################################
        # if at final diagonal, then cell [i = anc_len, j = desc_len] will be at W=0
        #   if at an intermediate diagonal, this contains an intermediate probability
        #   if at padding, then toss this value
        # pad_mask is True at valid cells, False at padding locations
        new_alignment_score = cache_at_curr_k[0,...] #(T, C*S, B)
        new_alignment_score = jnp.where( pad_mask[:,0][None,None,:],
                                         new_alignment_score,
                                         prev_alignment_score ) #(T, C*S, B)
        
        # build a new carry
        new_carry = {'cache_for_prev_diagonal': cache_at_curr_k, #(W, T, C_S, B)
                     'cache_two_diags_prior': cache_for_prev_diagonal, #(W, T, C_S, B)
                     'alignment_score_per_class_and_state': new_alignment_score} #(T, C_S, B)
         
        return new_carry, cache_at_curr_k
    
    # do scan fn
    init_carry = {'cache_for_prev_diagonal': diag_k2, #( W, T, C_S, B)
                  'cache_two_diags_prior': diag_k1, #( W, T, C_S, B)
                  'alignment_score_per_class_and_state': diag_k2[0,...]} #( T, C_S, B)
    xs = jnp.arange(3, K+1)
    
    # final_carry is a dictionary
    # all_diags is (K-2, W, T, C_S, B)
    if return_full_grid:
        final_carry, all_diags = jax.lax.scan( f = scan_fn,
                                               init = init_carry,
                                               xs = xs,
                                               length = xs.shape[0] )
    
    elif not return_full_grid:
        final_carry, _ = jax.lax.scan( f = scan_fn,
                                       init = init_carry,
                                       xs = xs,
                                       length = xs.shape[0] )
    
    alignment_score_per_class_and_state = final_carry['alignment_score_per_class_and_state'] #(T, C_S, B)
    del final_carry, joint_logprob_transit_mid_only, init_carry
    
    
    #################################
    ### Multiplying by any -> end   #
    #################################
    # reminder: joint_logprob_transit was (T, C_transit_prev, S_prev, C_transit_curr, S_curr)
    mid_to_end = joint_logprob_transit[:, :, :3, -1, -1] #(T, C_transit_prev, (S-1)_prev)
    mid_to_end = jnp.reshape(mid_to_end, (T, C_S ) ) #(T, C_S)
    forward_logprob = nn.logsumexp( mid_to_end[...,None] + alignment_score_per_class_and_state, axis=1 ) #(T, B)
    
    if return_full_grid:
        all_diags = jnp.concatenate( [diag_k1[None, ...],
                                      diag_k2[None, ...],
                                      all_diags], axis=0 )  # (K, W, T, C_S, B)
        return forward_logprob, all_diags

    elif not return_full_grid:
        return forward_logprob #(T, B)
