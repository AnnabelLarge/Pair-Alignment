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
                 extra_features,
                 idx_lst, 
                 align_idx_padding: int = -9,
                 **kwargs):
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
        - tuple of (anc_embs, desc_embs, extra_features)
          > both of size (batch, alignment_len, hid_dim)
        - mask for alignment positions
    """
    # get indexes needed; each are (B, L_align, 1)
    anc_idxes = idx_lst[:,:,0][...,None]
    desc_idxes = idx_lst[:,:,1][...,None] 
    masking_vec = ( anc_idxes != align_idx_padding )
    
    # index with jnp take
    anc_selected = jnp.take_along_axis(anc_encoded, anc_idxes, axis=1)
    anc_selected = anc_selected * masking_vec
    
    desc_selected = jnp.take_along_axis(desc_encoded, desc_idxes, axis=1)
    desc_selected = desc_selected * masking_vec
    
    out_lst = [anc_selected, desc_selected]
    
    if extra_features is not None:
        out_lst.append(extra_features)
    
    return (out_lst, masking_vec[...,0])








if __name__ == '__main__':
    import numpy as np
    
    
    unaligned_ancs = jnp.array( [[1, 3, 4, 5, 2],
                                 [1, 4, 6, 2, 0]] )
    unaligned_ancs = jnp.repeat(unaligned_ancs[...,None], 10, axis=-1)
    
    unaligned_descs = jnp.array( [[1, 5, 3, 4, 2],
                                  [1, 6, 6, 2, 0]] )
    unaligned_descs = jnp.repeat(unaligned_descs[...,None], 10, axis=-1)
    
    extra_feats = jnp.array([[4, 2, 1, 1, 3, 5],
                             [4, 3, 1, 2, 5, 0]] )[...,None]
    
    m_idxes = jnp.array( [[1, 1, 2, 3,  4, -9],
                          [1, 2, 3, 3, -9, -9]] )[...,None]
    
    n_idxes = jnp.array( [[0,1,2,3,3,-9],
                          [0,0,1,2,-9,-9]] )[...,None]
    
    idx_lst = jnp.concatenate([m_idxes, n_idxes], axis=-1)
    
    out = extract_embs(anc_encoded = unaligned_ancs, 
                      desc_encoded = unaligned_descs, 
                      extra_features = extra_feats,
                      idx_lst = idx_lst, 
                      align_idx_padding = -9)
    
    inspect = jnp.concatenate( out[0], axis=-1 )[..., [0, 10, -1]]
    inspect = np.array( inspect * out[1][...,None] )
    
    true_1 = np.array( [[3,1,4],
                        [3,5,2],
                        [4,3,1],
                        [5,4,1],
                        [2,4,3],
                        [0,0,0]] )
    
    true_2 = np.array( [[4,1,4],
                        [6,1,3],
                        [2,6,1],
                        [2,6,2],
                        [0,0,0],
                        [0,0,0]] )
    
    assert np.allclose(inspect[0], true_1)
    assert np.allclose(inspect[1], true_2)
    
    













# def combine_one_hot_embeddings(anc_encoded, 
#                                desc_encoded,
#                                seq_padding_idx = 0,
#                                *args,
#                                **kwargs):
#     """
#     ignore idx_lst, and just return embeddings as-is
    
#     when used in TKF92, this is essentially one-hot encoding the alignment
#     itself
#     """
#     masking_vec = anc_encoded != seq_padding_idx
#     return ([anc_encoded, desc_encoded], masking_vec)

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


 
  