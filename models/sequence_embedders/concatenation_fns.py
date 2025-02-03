#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:32:50 2023

@author: annabel_large

ABOUT:
=======
These contain the concatenation functions used combine position-specific
  embeddings. 

"""
from flax import linen as nn
import jax
import jax.numpy as jnp



def extract_embs(anc_encoded, 
                 desc_encoded, 
                 idx_lst, 
                 align_idx_padding = -9):
    """
    extract embeddings, according to coordinates given by idx_lst
    need this as a class in order to initialize the special indexing function
      once
    
    inputs:
        - anc_encoded: full ancestor sequence embeddings, from 
          full-context encoding modules
          > (batch, seq_len, hid_dim)
          
        - desc_encoded: full descendant sequence embeddings, from 
          causal-context encoding modules
          > (batch, seq_len, hid_dim)
        
        - idx_lst: indices to concatenate; ancestor indices are in first 
          column, descendant indices are in the second
          > (batch, seq_len, 2)
        
    outputs:
        - tuple of (anc_embs, desc_embs)
          > both of size (batch, alignment_len, hid_dim)
        - mask for alignment positions
    """
    # get indexes needed
    anc_idxes = idx_lst[:,:,0] #(B, L)
    desc_idxes = idx_lst[:,:,1] #(B, L)
    
    # from the idx list, padding characters correspond to 
    #   align_idx_padding (different from general padding index)
    # (B,L,2) -> (B, L)
    masking_vec = jnp.where( idx_lst!=align_idx_padding, True, False)[:,:,0]

    # indexing, but along the batch dimension
    # TODO: COULD THIS BE REPLACED BY JNP.TAKE SOMEHOW, FOR SPEED UPS?
    def index_perbatch(idx_lst, enc_mat, masking_mat):
        # do the indexing
        raw_out = enc_mat[idx_lst,:]
        
        # reshape the mask to match the hidden dimension
        masking_mat = jnp.expand_dims(masking_mat, 1)
        masking_mat = jnp.repeat(masking_mat, raw_out.shape[1], axis=1)
        
        # apply mask
        masked_out = raw_out * masking_mat
        return masked_out
    init_vmapped_indexer = jax.vmap(index_perbatch, 
                                    in_axes=0, 
                                    out_axes=0)
    
    # do indexing along batch dimension and get a masking matrix
    anc_selected = init_vmapped_indexer(anc_idxes,
                                        anc_encoded,
                                        masking_vec)
    
    desc_selected = init_vmapped_indexer(desc_idxes, 
                                         desc_encoded,
                                         masking_vec)
    
    return ([anc_selected, desc_selected], masking_vec)
    




### not used yet
# def concat_all_embs(anc_encoded, desc_encoded):
#     """
#     return all possible combinations of embeddings

#     inputs:
#         - anc_encoded: full ancestor sequence embeddings, from 
#           full-context encoding modules
#           > (batch, seq_len, hid_dim)
          
#         - desc_encoded: full descendant sequence embeddings, from 
#           causal-context encoding modules
#           > (batch, seq_len, hid_dim)
        
#     outputs:
#         - concatenated sequence embeddings: concat([desc_embeds, anc_embeds])
#           > (batch, desc_positions * anc_positions, hid_dim)

#     """
#
#     # TODO: DON'T USE JNP.REPEAT!!!
#
#     # copy each hidden dim of n across multiple new columns 
#     # (repeat across new dim=2)
#     desc_expanded = jnp.expand_dims(desc_encoded, axis=2)
#     desc_expanded = jnp.repeat(desc_expanded, desc_encoded.shape[1], axis=2)
    
#     # copy each hidden dim of m across multiple new rows 
#     # (repeat across new dim=1)
#     anc_expanded = jnp.expand_dims(anc_encoded, axis=1)
#     anc_expanded = jnp.repeat(anc_expanded, anc_encoded.shape[1], axis=1)
    
#     # concatenate these together along last dimension
#     concat_mat = jnp.concatenate([desc_expanded, anc_expanded], axis=3)
    
#     # for compatibility, reshape to (batch, desc_len * anc_len, 2*hid)
#     reshaped = jnp.reshape(concat_mat, (concat_mat.shape[0],
#                                         concat_mat.shape[1]*concat_mat.shape[2],
#                                         concat_mat.shape[3]))
#     return reshaped


 
  