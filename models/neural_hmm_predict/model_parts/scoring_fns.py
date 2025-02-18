#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:18:44 2024

@author: annabel
"""
import jax
from jax import numpy as jnp


SMALLEST_FLOAT32 = jnp.finfo('float32').smallest_normal

def score_transitions(alignment_state, 
                      logprob_trans_mat, 
                      token_offset = 1,
                      padding_idx = 0):
    """
    inputs:
    -------
    alignment_state: (B, max_align_len-1, 2)
      > dim2=0: prev position's state
      > dim2=1: curr position's state (the position you're trying to predict)
      
    logprob_trans_mat: (T, B, max_align_len-1, 4, 4) OR (T, 1, 1, 4, 4)
    
    padding_idx: int
      > zero for align_path
    
    
    output sizes:
    --------------
    final_logprobs: (T, B, max_align_len-1)
     
    """
    ### For indexing the (4x4) matrix, change <eos> to 4
    alignment_state = jnp.where(alignment_state == 5, 4, alignment_state)
    
    
    ### reshape if needed
    # dims
    T = logprob_trans_mat.shape[0]
    B = alignment_state.shape[0]
    L = alignment_state.shape[1] # max_align_len - 1
    num_possible_trans = logprob_trans_mat.shape[3] #should be equal to 4
    
    # final mat
    final_mat_shape = (T, B, L, num_possible_trans, num_possible_trans)
    logprob_trans_mat = jnp.broadcast_to(logprob_trans_mat, final_mat_shape)
    
    
    ### scoring
    # get rows: T, B, L, 1, num_possible_trans
    prev_state = alignment_state[..., 0][None, ..., None,None] - token_offset
    interm = jnp.take_along_axis(logprob_trans_mat, 
                                 prev_state,
                                 axis=-2)  
    
    # get columns: T, B, L, 1, 1
    curr_state = alignment_state[..., 1][None, ..., None,None] - token_offset
    raw_logprobs = jnp.take_along_axis(interm, 
                                       curr_state, 
                                       axis=-1)  
    
    # squash to (T, B, L); mask
    raw_logprobs = raw_logprobs[...,0,0] 
    
    
    ### mask and return
    padding_mask =  ( (alignment_state[...,0] != padding_idx) &
                      (alignment_state[...,1] != padding_idx) )
    
    final_logprobs = raw_logprobs * padding_mask[None, ...]
    
    return final_logprobs


def score_substitutions(true_out,
                        logprob_subs_mat,
                        token_offset = 3,
                        padding_idx = -1):
    """
    inputs:
    -------
    true_out: (B, max_align_len-1, 2)
      > (dim0=0): gapped ancestor seq
      > (dim0=1): gapped descendant seq
    
    logprob_subs_mat: (T, B, max_align_len-1, alph, alph)
      > this was already broadcasted to full (T, B, max_align_len-1, alph, alph)
     
    token_offset: int; used to map
        A: n -> 0
        B: n+1 -> 1
        C: n+2 -> 2
    and so on, for rest of alphabet; usually this is 3
    
    
    output sizes:
    --------------
    final_logprobs: (T, B, max_align_len-1)
      
    """
    # get rows: T, B, L, 1, num_possible_trans
    anc_idx = true_out[..., 0][None, ..., None,None] - token_offset
    interm = jnp.take_along_axis(logprob_subs_mat, 
                                 anc_idx,
                                 axis=-2)  
    
    # get columns: T, B, L, 1, 1
    desc_idx = true_out[..., 1][None, ..., None,None] - token_offset
    raw_logprobs = jnp.take_along_axis(interm, 
                                       desc_idx, 
                                       axis=-1)  
    
    # squash to (T, B, L); mask
    raw_logprobs = raw_logprobs[...,0,0] 
    
    
    ### mask and return
    padding_mask =  ( (true_out[...,0] != padding_idx) &
                      (true_out[...,1] != padding_idx) )
    
    final_logprobs = raw_logprobs * padding_mask[None, ...]
    
    return final_logprobs


def score_indels(true_out: jnp.array, 
                 logprob_scoring_vec: jnp.array, 
                 which_seq: str,
                 token_offset: int=3,
                 padding_idx: int=0):
    """
    inputs:
    -------
    true_out: (B, max_align_len - 1, 2)
      > dim2=0: gapped ancestor
      > dim2=1: gapped descendant
      
    logprob_scoring_vec: (B, max_align_len - 1, alph) OR (1, 1, alph)
    
    which_seq: 'anc' to score ancestor, 'desc' to score descendant
     
    token_offset: int; used to map
        A: n -> 0
        B: n+1 -> 1
        C: n+2 -> 2
    and so on, for rest of alphabet; usually this is 3
    
    
    output sizes:
    --------------
    final_logprobs: (T, B, max_align_len - 1)
      
    """
    ### determine which to index
    if which_seq == 'anc':
        which_seq = 0
    
    elif which_seq == 'desc':
        which_seq = 1
    
    else:
        which_seq = None
    
    
    ### reshape if needed
    # dims
    B = true_out.shape[0]
    L = true_out.shape[1]
    alph = logprob_scoring_vec.shape[-1]
    
    # reshape
    new_shape = (B, L, alph)
    logprob_scoring_vec = jnp.broadcast_to(logprob_scoring_vec, new_shape)
    
    
    ### index
    residue_tokens = true_out[:,:,which_seq] - token_offset
    final_logprobs = jnp.take_along_axis(arr=logprob_scoring_vec, 
                                         indices=residue_tokens[...,None], 
                                         axis=-1)[...,0]
    
    
    ### mask
    padding_mask = (residue_tokens != padding_idx)
    final_logprobs = final_logprobs * padding_mask[None,...]
    
    return final_logprobs






if __name__ == '__main__':
    import jax
    from jax import numpy as jnp
    import numpy as np
    
    
    T = 6
    B = 3
    L = 5
    alph = 4
    
    rngkey = jax.random.key(42)
    
    mat = jax.random.randint( key=rngkey,
                              shape=(T,B,L,alph,alph),
                              minval = 1,
                              maxval = 1000 )
    
    indices = jnp.array([ [[1,2,3,4,1],
                           [4,3,2,1,4]] ,
                         
                          [[2,3,4,1,0],
                           [3,2,1,4,0]] ,
                         
                          [[1,0,0,0,0],
                           [4,0,0,0,0]] ]     
                         )
    indices = jnp.transpose( indices,
                             (0,2,1) )
    
    mask = (indices != 0)[...,0]
    
    
    ### true answer by loop
    true = np.zeros( (T,B,L) )
    for t in range(T):
        for b in range(B):
            for l in range(L):
                one_mat = mat[t,b,l,...]
                anc_idx, desc_idx = indices[b,l,...]
                to_fill = one_mat[anc_idx-1, desc_idx-1]
                true[t,b,l,...] = to_fill
    true = true * mask[None, :, :]
    
    
    ### answer as-is, with existing function
    by_func = score_substitutions(true_out = indices,
                                  logprob_subs_mat = mat,
                                  token_offset = 1)
    by_func = by_func * mask[None, :, :]
    
    assert jnp.allclose(by_func, true) 