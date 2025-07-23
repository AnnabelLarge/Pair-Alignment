#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:22:19 2025

@author: annabel

"""
import jax
from jax import numpy as jnp
import numpy as np


###############################################################################
### RAW STRINGS -> TENSORS OF ALIGNMENTS   ####################################
###############################################################################
def to_alignment_aug_alphabet(token: str, 
                              amino_acids_in_order: str = 'ACDEFGHIKLMNPQRSTWYV',
                              num_special_toks: int = 3,
                              gap_idx = 43):
    token = token.upper()
    
    return amino_acids_in_order.index(token) + num_special_toks


def encode_one_alignment(alignment: list[str],
                         gap_alph: list[str] = ['-','.'],
                         gap_idx = 43):
    anc_len, desc_len = [len(s) for s in alignment]
    assert anc_len == desc_len, 'inconsistent alignment length'
    
    pair_encoded = np.zeros( (1, anc_len, 3) )
    for l in range(anc_len):
        anc_tok = alignment[0][l]
        desc_tok = alignment[1][l]
        
        if (anc_tok in gap_alph) and (desc_tok in gap_alph):
            raise ValueError(f'Empty column: both ancestor and descendant are gaps')
        
        # match = 1
        if (anc_tok not in gap_alph) and (desc_tok not in gap_alph):
            pair_encoded[0,l,0] = to_alignment_aug_alphabet(anc_tok)
            pair_encoded[0,l,1] = to_alignment_aug_alphabet(desc_tok)
            pair_encoded[0,l,2] = 1
        
        # insert = 2
        if (anc_tok in gap_alph) and (desc_tok not in gap_alph):
            pair_encoded[0,l,0] = gap_idx
            pair_encoded[0,l,1] = to_alignment_aug_alphabet(desc_tok)
            pair_encoded[0,l,2] = 2
        
        # delete = 3
        if (anc_tok not in gap_alph) and (desc_tok in gap_alph):
            pair_encoded[0,l,0] = to_alignment_aug_alphabet(anc_tok)
            pair_encoded[0,l,1] = gap_idx
            pair_encoded[0,l,2] = 3
    return pair_encoded
            

def str_aligns_to_tensor(alignments: list[ tuple[ str ] ]):
    """
    returns a jax numpy tensor of size (B, L, 3)
    
    dim2=0: ancestor
    dim2=1: descendant
    dim2=2: alignment type (<pad>=0, M=1, I=2, D=3, <start>=4, <end>=5)
    """
    B = len(alignments)
    
    ### encode
    list_of_np_arrays = []
    L = 0
    for a in alignments:
        one_sample = encode_one_alignment(a)
        one_sample = np.concatenate( [np.array([1,1,4])[None,None,:], 
                                      one_sample], axis=1 )
        one_sample = np.concatenate( [one_sample, 
                                      np.array([2,2,5])[None,None,:]], 
                                    axis=1 )
        L = one_sample.shape[1] if (one_sample.shape[1] > L) else L
        list_of_np_arrays.append(one_sample)
    
    out_tensor = []
    for samp in list_of_np_arrays:
        padding = ( (0,0),
                    (0,L-samp.shape[1]),
                    (0,0) )
        out_tensor.append( np.pad(samp, padding) )
    
    return jnp.concatenate(out_tensor, axis=0).astype(int)


###############################################################################
### TENSORS OF ALIGNMENTS -> COUNTS TENSORS   #################################
###############################################################################
def summarize_alignment(align,
                        num_special_toks: int = 3,
                        alphabet_size: int = 20):
    """
    align is one pairwise alignment with shape: (L, 3)
    
    vmap this over entire batch
    """
    
    def counting_func( prev_dict, align_column):
        """
        scan this over length (dim0) of align
        prev_dict is a dictionary with:
        ===============================
        state
        match_counts
        ins_counts
        del_counts
        transit_counts
        emit_counts
        STATIC ARGUMENT alphabet_size
        STATIC ARGUMENT num_special_toks
        
        """
        ### unpack arguments
        
        # static arguments
        alphabet_size = prev_dict['alphabet_size']
        num_special_toks = prev_dict['num_special_toks']
        
        # reduct insert chars to match chars
        align_column = jnp.where(align_column > alphabet_size + num_special_toks,
                                 align_column - alphabet_size,
                                 align_column)
        
        # parts of alignment column
        anc_tok = align_column[0]
        desc_tok = align_column[1]
        curr_state = align_column[2]
        
        
        ### aggregate counts
        updated_match_counts = jnp.where( curr_state == 1,
                                          prev_dict['match_counts'].at[anc_tok, desc_tok].add(1),
                                          prev_dict['match_counts'] )
        
        updated_ins_counts = jnp.where( curr_state == 2,
                                        prev_dict['ins_counts'].at[desc_tok].add(1),
                                        prev_dict['ins_counts'] )
        
        updated_del_counts = jnp.where( curr_state == 3,
                                        prev_dict['del_counts'].at[anc_tok].add(1),
                                        prev_dict['del_counts'] )
        
        updated_emit_counts = prev_dict['emit_counts'].at[anc_tok].add(1)
        updated_emit_counts = updated_emit_counts.at[desc_tok].add(1)
        
        updated_transit_counts = prev_dict['transit_counts'].at[ prev_dict['state'], curr_state].add(1)
        
        ### save to dictionary; this is the carry through jax.lax.scan
        updated_dict = {'state': curr_state,
                        'match_counts': updated_match_counts,
                        'ins_counts': updated_ins_counts,
                        'del_counts': updated_del_counts,
                        'emit_counts': updated_emit_counts,
                        'transit_counts': updated_transit_counts,
                        'alphabet_size': alphabet_size,
                        'num_special_toks': num_special_toks}
        
        return updated_dict, None
    
    
    ### jax scan time
    # init the carry dictionary
    # align is (L, 3)
    mat_vec_dim = alphabet_size + num_special_toks
    init_dict = {'state': align[0,2],
                 'match_counts': jnp.zeros( (mat_vec_dim, mat_vec_dim) ),
                 'ins_counts': jnp.zeros( (mat_vec_dim,) ),
                 'del_counts': jnp.zeros( (mat_vec_dim,) ),
                 'emit_counts': jnp.zeros( (mat_vec_dim,) ),
                 'transit_counts': jnp.zeros( (6,6) ),
                 'alphabet_size': alphabet_size,
                 'num_special_toks': num_special_toks}
    
    # scan
    counts_dict, _ = jax.lax.scan(f = counting_func,
                                  init = init_dict,
                                  xs = align[1:, :],
                                  length = align.shape[0]-1)
    
    # remove information from special tokens
    counts_dict['match_counts'] = counts_dict['match_counts'][num_special_toks:, num_special_toks:]
    counts_dict['ins_counts'] = counts_dict['ins_counts'][num_special_toks:]
    counts_dict['del_counts'] = counts_dict['del_counts'][num_special_toks:]
    counts_dict['emit_counts'] = counts_dict['emit_counts'][num_special_toks:]
    counts_dict['transit_counts'] = counts_dict['transit_counts'][ 1:-1, [1,2,3,5] ]
    
    # remove extra info
    del counts_dict['state']
    del counts_dict['num_special_toks']
    del counts_dict['alphabet_size']
    return counts_dict


###############################################################################
### TENSORS OF ALIGNMENTS -> TENSOR OF UNALIGNED SEQUENCES   ##################
###############################################################################
def aligned_to_seq(encoded_seq: np.array,
                   gap_idx: int = 43,
                   num_special_toks: int = 3,
                   alphabet_size: int = 20):
    ungapped = encoded_seq[ (encoded_seq != gap_idx) ]
    
    unaligned = np.where( ungapped < num_special_toks + alphabet_size,
                          ungapped,
                          ungapped - 20)
    return unaligned
    
def align_tensor_to_seq_tensor(alignment_tensor: jnp.array,
                               gap_idx: int=43):
    """
    returns a tensor of size (B, L, 2)
    
    dim2=0: ancestor
    dim2=1: descendant
    """
    
    B, L, _ = alignment_tensor.shape
    
    out_tensor = np.zeros( (B,L,2) )
    for b in range(B):
        gapped_anc = alignment_tensor[b,:,0]
        ungapped_anc = aligned_to_seq(gapped_anc)
        out_tensor[b, :ungapped_anc.shape[0], 0] = ungapped_anc
        
        gapped_desc = alignment_tensor[b,:,1]
        ungapped_desc = aligned_to_seq(gapped_desc)
        out_tensor[b, :ungapped_desc.shape[0], 1] = ungapped_desc
    
    out_tensor = out_tensor[ :, (out_tensor.sum(axis=2) != 0)[0], :]
    return jnp.array( out_tensor ).astype(int)




if __name__ == '__main__':
    # visually check this
    alignment = [('AC-', 'D-E'),
                 ('AC-', 'D-E')]
    encoded_alignment = str_aligns_to_tensor(alignment)
    
    vmapped_summarize_alignment = jax.vmap(summarize_alignment, 
                                           in_axes=0, 
                                           out_axes=0)
    counts_dict = vmapped_summarize_alignment(encoded_alignment)
    counts_dict = {key: np.array(value) for key,value in counts_dict.items()}
    
    match_counts = counts_dict['match_counts']
    ins_counts = counts_dict['ins_counts']    
    del_counts = counts_dict['del_counts']    
    transit_counts = counts_dict['transit_counts']
    
