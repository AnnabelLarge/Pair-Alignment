#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 20:26:36 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
import numpy as np

def ref_func(m, n):
    """
    generate indices in a dumb loop
    """
    max_len = min(m, n) + 1
    idxs = []
    mask = []
    for k in range(m+n+1):
        # 0 <= i <= anc_len
        # 0 <= j <= desc_len
        # k = i + j
        # 0 <= k - i <= desc_len
        # 
        # smallest i: i >= k - desc_len
        # largest i: 
        #   i <= k, and i <= anc_len
        #   combine into one: i <= min(anc_len, k) 
        #   add one because range does not include last element
        cells = [(i, k-i) for i in range(max(0, k-n), min(m, k)+1)]
        pad = max_len - len(cells)
        idxs.append(cells + [(0,0)]*pad)
        mask.append([1]*len(cells) + [0]*pad)
    return np.array(idxs), np.array(mask)


def generate_wavefront_indices(unaligned_seqs,
                               padding_idx = 0):
    """
    Arguments:
    ------------
    unaligned_seqs : ArrayLike, (B, L_seq, 2)
        dim2=0: ancestor
        dim2=1: descendant
    
    Returns:
    ---------
    indices : ArrayLike, (B, K, W, 2)
    mask : ArrayLike, (B, K, W)
    """
    B = unaligned_seqs.shape[0]
    
    # ancestor and descendant lengths
    seq_lens = (unaligned_seqs != padding_idx).sum(axis=1) #(B, 2)
    anc_lens = seq_lens[:,0] #(B,)
    desc_lens = seq_lens[:,1] #(B,)
    
    # k = i + j
    # need K number of diagonals
    num_diags_K = (anc_lens + desc_lens).max() + 1
    
    # widest diagonal width is min(anc_len, desc_len) + 1
    widest_diag_W = ( seq_lens.min(axis=1) ).max() + 1 
    
    # possible k diagonal indices
    k_idx = jnp.arange(num_diags_K)[None, :] #(1, K)
    
    # expand anc_lens, desc_lens
    anc_lens = anc_lens[:, None] #(B, 1)
    desc_lens = desc_lens[:, None] #(B, 1)
    
    
    ### i limits yield lengths of diagonals
    # 0 <= i <= anc_len
    # 0 <= j <= desc_len
    # k = i + j
    # 0 <= k - i <= desc_len
    
    # i >= k - desc_len
    i_min = jnp.maximum(0, k_idx - desc_lens) # (B, K)
    
    # i <= k, and i <= anc_len
    # combine into one: i <= min(anc_len, k) 
    i_max = jnp.minimum(anc_lens, k_idx) #(B, K)
    
    # lengths of diagonals
    diag_lengths = i_max - i_min + 1 #(B, K)
    
    
    ### get indices
    # generate i indices, with an offset each diagonal
    offs = jnp.arange(widest_diag_W)[None, None, :] #(1, 1, W)
    i_vals = i_min[..., None] + offs #(B, K, W)
    
    # j = k - i
    j_vals = k_idx[..., None] - i_vals #(B, K, W)
    
    
    ### mask invalid diagonal positions
    mask = offs < diag_lengths[..., None] #(B, K, W)
    i_vals = jnp.multiply( i_vals, mask ) #(B, K, W)
    j_vals = jnp.multiply( j_vals, mask ) #(B, K, W)
    indices = jnp.stack( [i_vals, j_vals], axis=-1) #(B, K, W, 2)
    
    return indices, mask


seq1 = jnp.array( [[1, 1],
                   [1, 0],
                   [0, 0],
                   [0, 0]] )

seq2 = jnp.array( [[1, 1],
                   [1, 1],
                   [1, 1],
                   [0, 1]] )

example_input = jnp.stack([seq1, seq2]) #(B, L_seq, 2)
pred_out = generate_wavefront_indices( example_input  ) #(B, K, W, 2)
all_pred_idx, all_pred_mask = pred_out
del pred_out


for b in range(example_input.shape[0]):
    anc = example_input[b, :, 0]
    desc = example_input[b, :, 1]
    
    anc_len = (anc != 0).sum()
    desc_len = (desc != 0).sum()
    
    true_idxes, true_mask = ref_func( anc_len.item(), desc_len.item() )
    pred_idxes = all_pred_idx[b]
    pred_mask = all_pred_mask[b]
    
    dim0_padding_len = pred_idxes.shape[0] - true_idxes.shape[0]
    dim1_padding_len = pred_idxes.shape[1] - true_idxes.shape[1]
    
    true_idxes = np.pad( true_idxes, 
                         ( (0,dim0_padding_len),
                           (0,dim1_padding_len),
                           (0,0)) 
                         )
    
    true_mask = np.pad( true_mask, 
                         ( (0,dim0_padding_len),
                           (0,dim1_padding_len)) 
                         )
    
    assert np.allclose( true_idxes, pred_idxes )
    assert np.allclose( true_mask, pred_mask )


def generate_wavefront_indices_at_k(unaligned_seqs,
                                    diagonal_k, 
                                    padding_idx = 0):
    """
    Arguments:
    ------------
    unaligned_seqs : ArrayLike, (B, L_seq, 2)
        dim2=0: ancestor sequence (includes start/end!)
        dim2=1: descendant sequence (includes start/end!)
    
    diagonal_k: int
        which diagonal to generate indices for
        
    
    Returns:
    ---------
    indices : ArrayLike, (B, W, 2)
    mask : ArrayLike, (B, W)
    """
    # no need to redo this every iteration
    seq_lens = (unaligned_seqs!=padding_idx).sum(axis=1) #(B, 2)
    
    B = seq_lens.shape[0]
    
    # unpack ancestor and descendant lengths
    anc_lens = seq_lens[:,0][:, None] #(B,1)
    desc_lens = seq_lens[:,1][:, None] #(B,1)
    
    # widest diagonal width is min(anc_len, desc_len)  + 1
    widest_diag_W = ( seq_lens.min(axis=1) ).max() + 1
    offs = jnp.arange(widest_diag_W)[None, :] #(1, W)
    
    
    ### i limits yield lengths of diagonals
    # 0 <= i <= anc_len
    # 0 <= j <= desc_len
    # k = i + j
    # 0 <= k - i <= desc_len
    
    # i >= k - desc_len
    i_min = jnp.maximum(0, diagonal_k - desc_lens) # (B,)
    
    # i <= k, and i <= anc_len
    # combine into one: i <= min(anc_len, k) 
    i_max = jnp.minimum(anc_lens, diagonal_k) #(B,)
    
    # lengths of diagonals
    diag_lengths = i_max - i_min + 1 #(B,)
    
    
    ### get indices
    # generate i indices, with an offset each diagonal
    i_vals = i_min[:, None] + offs #(B, W)
    
    # j = k - i
    j_vals = diagonal_k - i_vals #(B, W)
    
    
    ### mask invalid diagonal positions
    mask = offs < diag_lengths[..., None] #(B, W)
    i_vals = jnp.multiply( i_vals, mask ) #(B, W)
    j_vals = jnp.multiply( j_vals, mask ) #(B, W)
    indices = jnp.stack( [i_vals, j_vals], axis=-1) #(B, W, 2)
    
    return indices, mask

true_idx = all_pred_idx.copy()
del all_pred_idx

true_mask = all_pred_mask.copy()
del all_pred_mask


K = true_idx.shape[1]
for k in range(K):
    pred_out = generate_wavefront_indices_at_k( unaligned_seqs = example_input,
                                                diagonal_k = k )
    pred_idx, pred_mask = pred_out
    del pred_out
    
    for b in range(example_input.shape[0]):
        sample_true_idx_at_k = true_idx[b, k, ...] #(W, 2)
        sample_pred_idx_at_k = pred_idx[b, ...] #(W, 2)
        assert np.allclose( sample_true_idx_at_k, sample_pred_idx_at_k )
        
        sample_true_mask_at_k = true_mask[b, k, ...] #(W, 2)
        sample_pred_mask_at_k = pred_mask[b, ...] #(W, 2)
        assert np.allclose( sample_true_mask_at_k, sample_pred_mask_at_k )
        
        
        
        
        
        
        
    
