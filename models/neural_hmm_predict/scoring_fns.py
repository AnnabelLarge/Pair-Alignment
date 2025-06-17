#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:18:44 2024

@author: annabel
"""
import jax
from jax import numpy as jnp



###############################################################################
### functions for marginalizing over time grid   ##############################
###############################################################################
def score_transitions_marg_over_times(alignment_state, 
                                      logprob_trans_mat,
                                      padding_idx = 0):
    """
    T: number of times
    B: batch size
    L_align: length of alignment
    S: number of transition states, here, it's 4
    
    
    Arguments
    ----------
    alignment_state : ArrayLike, (B, L_align-1, 2)
      > dim2=0: prev position's state
      > dim2=1: curr position's state (the position you're trying to predict)
      > <pad>: 0
      > Match, M: 1
      > Insert, I: 2
      > Delete, D: 3
      > Start, S: 4
      > End, E: 5
      
    logprob_trans_mat : ArrayLike, (T, B, L_align-1, 4, 4) OR (T, 1, 1, 4, 4)
      > order of rows and columns: M, I, D, S/E
    
    padding_idx: int
      > default = 0
    
    
    Returns:
    ----------
    final_logprobs : ArrayLike(T, B, L_align-1)
    """
    # dims
    T = logprob_trans_mat.shape[0]
    B = alignment_state.shape[0]
    L = alignment_state.shape[1]  #this is L_align-1
    S = logprob_trans_mat.shape[-1] #should be equal to 4
    
    
    ### preprocess
    # get padding mask, do this BEFORE adjusting the encoding!!!
    padding_mask =  ( (alignment_state[...,0] != padding_idx) &
                      (alignment_state[...,1] != padding_idx) ) #(B, L_align-1)
    
    # adjust encoding
    # end: write as combined start/end token (5 -> 4)
    # pad: write as delete, so that indexing invalid positions doesn't cause 
    #      jax gradients to be NaN (0 -> 3)
    alignment_state_adj = jnp.where(alignment_state != 5, alignment_state, 4) # (B, L_align-1, 2)
    alignment_state_adj = jnp.where(alignment_state_adj != 0, alignment_state_adj, 3) # (B, L_align-1, 2)
    
    # by how much do you offset tokens for indexing the transition matrix? 
    # default offset is 1, which remaps tokens to:
    # > Match, M: 0
    # > Insert, I: 1
    # > Delete, D, and pad: 2
    # > Start, S, and End, E: 3
    token_offset = 1
    
    # move all positions down
    alignment_state_adj = alignment_state_adj - token_offset
    
    
    ### Scoring
    # global: one transition matrix for all samples, all positions
    if logprob_trans_mat.shape == (T, 1, 1, S, S):
        logprob_trans_mat = logprob_trans_mat[:,0,0,...] #(T, S, S)

        prev_state = alignment_state_adj[..., 0][None,...] #(1, B, L_align-1)
        prev_state = jnp.broadcast_to(prev_state, (T, B, L))

        curr_state = alignment_state_adj[..., 1][None,...] #(1, B, L_align-1)        
        curr_state = jnp.broadcast_to(curr_state, (T, B, L))
        
        raw_logprobs = logprob_trans_mat[:, prev_state, curr_state]  # (T, B, L_align-1)
                
    
    # local: unique transition matrix for each sample, each position
    else:
        # get rows: T, B, L_align-1, 1, S
        prev_state = alignment_state_adj[..., 0][None, ..., None,None]
        interm = jnp.take_along_axis(logprob_trans_mat, 
                                     prev_state,
                                     axis=-2)  
        
        # get columns: T, B, L_align-1, 1, 1
        curr_state = alignment_state_adj[..., 1][None, ..., None,None]
        raw_logprobs = jnp.take_along_axis(interm, 
                                           curr_state, 
                                           axis=-1)  
        
        # squash to (T, B, L_align-1); mask and return
        raw_logprobs = raw_logprobs[...,0,0] 
    
    
    ### mask and return
    final_logprobs = raw_logprobs * padding_mask[None, ...]
    return final_logprobs # (T, B, L_align-1)


def score_transitions_t_per_samp(alignment_state, 
                                 logprob_trans_mat,
                                 padding_idx = 0):
    """
    T: number of times
    B: batch size
    L_align: length of alignment
    S: number of transition states, here, it's 4
    
    
    Arguments
    ----------
    alignment_state : ArrayLike, (B, L_align-1, 2)
      > dim2=0: prev position's state
      > dim2=1: curr position's state (the position you're trying to predict)
      > <pad>: 0
      > Match, M: 1
      > Insert, I: 2
      > Delete, D: 3
      > Start, S: 4
      > End, E: 5
      
    logprob_trans_mat : ArrayLike, (B, L_align-1, 4, 4) OR (B, 1, 4, 4)
      > order of rows and columns: M, I, D, S/E
    
    padding_idx: int
      > default = 0
    
    
    Returns:
    ----------
    final_logprobs : ArrayLike(B, L_align-1)
    """
    # dims
    B = alignment_state.shape[0]
    L = alignment_state.shape[1]  #this is L_align-1
    S = logprob_trans_mat.shape[-1] #should be equal to 4
    
    
    ### preprocess
    # get padding mask, do this BEFORE adjusting the encoding!!!
    padding_mask =  ( (alignment_state[...,0] != padding_idx) &
                      (alignment_state[...,1] != padding_idx) ) #(B, L_align-1)
    
    # adjust encoding
    # end: write as combined start/end token (5 -> 4)
    # pad: write as delete, so that indexing invalid positions doesn't cause 
    #      jax gradients to be NaN (0 -> 3)
    alignment_state_adj = jnp.where(alignment_state != 5, alignment_state, 4) # (B, L_align-1, 2)
    alignment_state_adj = jnp.where(alignment_state_adj != 0, alignment_state_adj, 3) # (B, L_align-1, 2)
    
    # by how much do you offset tokens for indexing the transition matrix? 
    # default offset is 1, which remaps tokens to:
    # > Match, M: 0
    # > Insert, I: 1
    # > Delete, D, and pad: 2
    # > Start, S, and End, E: 3
    token_offset = 1
    
    # move all positions down
    alignment_state_adj = alignment_state_adj - token_offset
    
    
    ### Scoring
    # global: one transition matrix for all samples, all positions
    if logprob_trans_mat.shape == (B, 1, S, S):
        logprob_trans_mat = logprob_trans_mat[:,0,...] # (B, S, S)

        prev_state = alignment_state_adj[..., 0] # (B, L_align-1)
        curr_state = alignment_state_adj[..., 1] # (B, L_align-1) 
        
        batch_idx = jnp.arange(B)[:, None]  # (B, 1)
        raw_logprobs = logprob_trans_mat[batch_idx, prev_state, curr_state]  # (B, L_align-1)
                
    
    # local: unique transition matrix for each sample, each position
    else:
        # get rows: B, L_align-1, 1, S
        prev_state = alignment_state_adj[..., 0][..., None,None] # (B, L_align-1, 1, 1)
        interm = jnp.take_along_axis(logprob_trans_mat, 
                                     prev_state,
                                     axis=-2)  # (B, L_align-1, 1, S)
        
        # get columns: B, L_align-1, 1, 1
        curr_state = alignment_state_adj[..., 1][..., None,None] # (B, L_align-1, 1, 1)
        raw_logprobs = jnp.take_along_axis(interm, 
                                           curr_state, 
                                           axis=-1)  # (B, L_align-1, 1, 1) 
        
        # squash to (B, L_align-1); mask and return
        raw_logprobs = raw_logprobs[...,0,0] 
    
    
    ### mask and return
    final_logprobs = raw_logprobs * padding_mask
    return final_logprobs # (B, L_align-1)


def score_indels(true_out: jnp.array, 
                 logprob_scoring_vec: jnp.array, 
                 which_seq: str,
                 padding_idx: int=0):
    """
    T: number of times
    B: batch size
    L_align: length of alignment
    A: alphabet size
    
    
    Arguments
    ----------
    true_out: (B, L_align - 1, 2)
      > dim2=0: gapped ancestor
      > dim2=1: gapped descendant
      
    logprob_scoring_vec: (B, L_align - 1, A) OR (1, 1, A)
    
    which_seq: 'anc' to score ancestor, 'desc' to score descendant
    
    
    Returns:
    ---------
    final_logprobs: (B, L_align - 1)
      
    """
    ### preprocess
    # dims
    B = true_out.shape[0]
    L = true_out.shape[1] #this is L_align-1
    A = logprob_scoring_vec.shape[-1]
    
    # determine which to index
    if which_seq == 'anc':
        residue_tokens = true_out[...,0] #(B, L-align-1)
    
    elif which_seq == 'desc':
        residue_tokens = true_out[...,1] #(B, L-align-1)
    
    # create padding mask BEFORE rempping tokens
    padding_mask = (residue_tokens != padding_idx) #(B, L-align-1)
    
    # map <pad>, <bos>, and <eos> to last token, so that jax doesn't
    # have invalid indexing and NaN gradients
    residue_tokens_adj = jnp.where(residue_tokens != 0, residue_tokens, A)
    residue_tokens_adj = jnp.where(residue_tokens_adj != 1, residue_tokens_adj, A)
    residue_tokens_adj = jnp.where(residue_tokens_adj != 2, residue_tokens_adj, A)
    
    # remap tokens, to account for the <pad>, <bos>, <eos> tokens in the 
    # alphabet; for example, for proteins:
    # A: 0
    # C: 1
    # D: 2
    # (etc.)
    token_offset = 3
    residue_tokens_adj = residue_tokens_adj - token_offset #(B, L_align-1)
    
    
    ### score
    logprob_scoring_vec = jnp.broadcast_to( logprob_scoring_vec,
                                            (B,L,A) ) #(B,L_align-1,A)
    
    raw_logprobs = jnp.take_along_axis(arr=logprob_scoring_vec, 
                                         indices=residue_tokens_adj[...,None], 
                                         axis=-1)[...,0] #(B, L_align-1)
        
    
    ### mask and return
    final_logprobs = raw_logprobs * padding_mask #(B, L_align-1)
    
    return final_logprobs #(B, L_align-1)





"""
TODO: different substitution scoring function, depending on if you're using 
  full GTR or abbreviated F81

Check the size of these matrices
"""



#%%
def score_substitutions(true_out,
                        logprob_subs_mat,
                        token_offset = 3,
                        padding_idx = -1):
    """
    inputs:
    -------
    true_out: (B, L_align-1, 2)
      > (dim0=0): gapped ancestor seq
      > (dim0=1): gapped descendant seq
    
    logprob_subs_mat: (T, B, L_align-1, A, A)
      > this was already broadcasted to full (T, B, L_align-1, A, A)
     
    token_offset: int; used to map
        A: n -> 0
        B: n+1 -> 1
        C: n+2 -> 2
    and so on, for rest of alphabet; usually this is 3
    
    
    output sizes:
    --------------
    final_logprobs: (T, B, L_align-1)
      
    """
    # get rows: T, B, L, 1, S
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









if __name__ == '__main__':
    import jax
    from jax import numpy as jnp
    import numpy as np
    
    
    T = 6
    B = 3
    L = 5
    A = 4
    
    rngkey = jax.random.key(42)
    
    mat = jax.random.randint( key=rngkey,
                              shape=(T,B,L,A,A),
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