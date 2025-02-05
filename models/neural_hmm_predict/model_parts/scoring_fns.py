#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:18:44 2024

@author: annabel
"""
import jax
from jax import numpy as jnp


SMALLEST_FLOAT32 = jnp.finfo('float32').smallest_normal

#############
### helpers #
#############
def matrix_indexing_fn(logprob_mat, index_vec, offset):
    row = index_vec[0]-offset
    col = index_vec[1]-offset
    return logprob_mat[row, col]

vmapped_matrix_indexing = jax.vmap(matrix_indexing_fn, in_axes = (0,0, None))


# def trans_mat_at_eos_indexing_fn(t_mat, index_vec, offset=3):
#     """
#     final transition probability = 1 - exp( logP(x -> ins) )
    
#     offset should do the following re-mapping-
#     Match: match_tok -> 0
#     Ins: ins_tok -> 1
#     Del: del_tok -> 2
#     Start: start_tok -> 3
#     End: end_tok -> 3
#     """
#     row_idx = index_vec[0]-offset
#     prob_X_to_end = 1 - jnp.exp(t_mat[row_idx, 1])
#     prob_X_to_end = jnp.where(prob_X_to_end != 0,
#                               prob_X_to_end,
#                               SMALLEST_FLOAT32)
#     logprob_X_to_end = jnp.log(prob_X_to_end)
#     return logprob_X_to_end
# vmapped_eos_trans_mat_indexing = jax.vmap(trans_mat_at_eos_indexing_fn,
#                                           in_axes = (0,0,None))



############################
### main scoring functions #
############################
def score_transitions(alignment_state, 
                      trans_mat, 
                      padding_idx = 0):
    """
    inputs:
    -------
    alignment_state: (B, max_align_len-1, 2)
      > dim2=0: prev position's state
      > dim2=1: curr position's state (the position you're trying to predict)
      
    trans_mat: (T, B, max_align_len-1, 4, 4) OR (T, 1, 1, 4, 4)
    
    padding_idx: int
      > zero for align_path
    
    
    output sizes:
    --------------
    final_logprobs: (T, B, max_align_len-1)
     
    """
    ### For indexing the (4x4) matrix, change <eos> to 4
    alignment_state = jnp.where(alignment_state == 5, 4, alignment_state)
    
    
    ### dims
    # batch and length from align_path_offset
    B = alignment_state.shape[0]
    L = alignment_state.shape[1] # max_align_len - 1
    
    # time and possible transitions from trans_mat
    T = trans_mat.shape[0]
    num_possible_trans = trans_mat.shape[3] #should be equal to 4
    
    
    ########################
    ### Sequential Scoring #
    ########################
    ### make new views of trans_mat for vmapped function
    # if there's one transition matrix for all lengths, 
    #   need to broadcast to B and L:
    #   (T, B, 1, 4, 4) -> (T, B, L, 4, 4)
    if trans_mat.shape[2] == 1:
        new_shape = (T, B, L, num_possible_trans, num_possible_trans)
        trans_mat = jnp.broadcast_to(trans_mat, new_shape)
        del new_shape 
    
    # reshape: (T, B, L, 4, 4) -> (T*B*L, 4, 4)
    new_shape = (T*B*L, num_possible_trans, num_possible_trans)
    trans_mat_reshape = trans_mat.reshape( new_shape )
    del new_shape
    

    ###  make new views of align_path_offset for vmapped function
    # broadcast to times T: (B, L, 2) -> (T, B, L, 2)
    new_shape = (T, B, L, alignment_state.shape[2])
    alignment_state_reshape = jnp.broadcast_to(alignment_state[None,:,:,:],
                                               new_shape)
    del new_shape
    
    # reshape: (T, B, L, 2) -> (T*B*L, 2)
    new_shape = ( T*B*L, alignment_state_reshape.shape[3] )
    alignment_state_reshape = alignment_state_reshape.reshape( new_shape )
    del new_shape
    
    
    ### use vmapped function (could probably turn into vmapped take, but
    ###    do that later)
    # offset of 1 to map:
    # Match: 1 -> 0
    # Ins: 2 -> 1
    # Del: 3 -> 2
    # S/E: 4 -> 3
    out_by_vmap_raw = vmapped_matrix_indexing(trans_mat_reshape, 
                                              alignment_state_reshape,
                                              1)
    
    # reshape back to original dims: (T*B*L, ) -> (T, B, L)
    raw_logprobs = out_by_vmap_raw.reshape( ( T, B, L ) )
    
    # clean up variables
    del trans_mat_reshape, alignment_state_reshape, out_by_vmap_raw
    
    
    #####################
    ### Mask and return #
    #####################
    # padding shouldn't contribute to scoring each position
    padding_mask =  ( (alignment_state[:,:,0] != padding_idx) &
                      (alignment_state[:,:,1] != padding_idx) )
    
    final_logprobs = jnp.where( padding_mask[None,:,:],
                                raw_logprobs,
                                0)
    
    return final_logprobs



def score_substitutions(true_out,
                        subs_mat,
                        token_offset = 3):
    """
    inputs:
    -------
    true_out: (B, max_align_len-1, 2)
      > (dim0=0): gapped ancestor seq
      > (dim0=1): gapped descendant seq
    
    subs_mat: (T, B, max_align_len-1, alph, alph)
      > this was already broadcasted to full (T, B, max_align_len-1, alph, alph)
        by MatchEmissionsLogprobs class
     
    token_offset: int; used to map
        A: n -> 0
        B: n+1 -> 1
        C: n+2 -> 2
    and so on, for rest of alphabet; usually this is 3
    
    
    output sizes:
    --------------
    final_logprobs: (T, B, max_align_len-1)
      
    """
    ### dims
    # get B and L from anc_desc_pairs
    B = true_out.shape[0]
    L = true_out.shape[1]
    
    # get number of times and alphabet size from subs_mat
    T = subs_mat.shape[0]
    alph = subs_mat.shape[3]
    
    
    ### make new views of subs_mat for vmapped function
    # reshape: (T, B, L, alph, alph) -> (T*B*L, alph, alph)
    new_shape = (T*B*L, alph, alph)
    subs_mat_reshaped = subs_mat.reshape( new_shape )
    del new_shape
    
    
    ### make new views of anc_desc_pairs for vmapped functions
    # broadcast to times T: (B, L, 2) -> (T, B, L, 2)
    new_shape = (T, B, L, true_out.shape[2])
    true_out_reshaped = jnp.broadcast_to(true_out, new_shape)
    del new_shape
    
    # reshape: (T, B, L, 2) -> (T*B*L, 2)
    new_shape = ( T*B*L, true_out_reshaped.shape[3] )
    true_out_reshaped = true_out_reshaped.reshape( new_shape )
    del new_shape
    
    
    ### index the substitution matrix
    # offset of token_offset to map:
    # A: n -> 0
    # B: n+1 -> 1
    # C: n+2 -> 2
    # and so on, for rest of alphabet; usually this is 3
    raw_logprobs = vmapped_matrix_indexing(subs_mat_reshaped,
                                           true_out_reshaped,
                                           token_offset)
    
    # reshape back to original dims: (T*B*L, ) -> (T, B, L)
    final_logprobs = raw_logprobs.reshape( ( T, B, L ) )
    
    return final_logprobs



def score_indels(true_out: jnp.array, 
                 scoring_vec: jnp.array, 
                 which_seq: int,
                 token_offset: int=3):
    """
    inputs:
    -------
    true_out: (B, max_align_len - 1, 2)
      > dim2=0: gapped ancestor
      > dim2=1: gapped descendant
      
    scoring_vec: (B, max_align_len - 1, alph) OR (1, 1, alph)
    
    which_seq: 0 to score ancestor, 1 to score descendant
     
    token_offset: int; used to map
        A: n -> 0
        B: n+1 -> 1
        C: n+2 -> 2
    and so on, for rest of alphabet; usually this is 3
    
    
    output sizes:
    --------------
    final_logprobs: (T, B, max_align_len - 1)
      
    """
    residue_tokens = true_out[:,:,which_seq] - token_offset
    
    # dims
    B = residue_tokens.shape[0]
    L = residue_tokens.shape[1]
    alph = scoring_vec.shape[2]
    
    ### if one scoring_vec for all positions, need new view of scoring_vec 
    ###   for take_along_axis
    if scoring_vec.shape[1] == 1:
        # broadcast up: (B, 1, alph) -> (B, L, alph)
        new_shape = (B, L, alph)
        scoring_vec = jnp.broadcast_to(scoring_vec, new_shape)
        del new_shape

    final_logprobs = jnp.take_along_axis(arr=scoring_vec, 
                                         indices=residue_tokens[:,:,None], 
                                         axis=-1)[:,:,0]
    
    return final_logprobs