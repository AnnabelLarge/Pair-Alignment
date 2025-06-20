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
                 align_idx_padding: int = -9,
                 **kwargs):
    """
    B = batch size
    L_align = length of alignment
    H = hidden dim
    
    
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
          > both of size (B, L_align, H)
        - mask for alignment positions (B, L_align)
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
    
    return [anc_selected, desc_selected]
    








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
