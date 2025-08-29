#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 14:19:55 2025

@author: annabel_large
"""
import numpy as np

def main(data_dir, prefix):
    path = f'{data_dir}/{prefix}'
    
    # 20 amino acids, <bos>, <eos>, <pad> (remove later)
    in_alphabet_size = 23 # A
    
    # 20 aas+match, 20 aas+ins, gap, <bos>, <eos>, <pad> (remove later)
    out_alphabet_size = 44 #A_aug
    
    
    
    ### read files
    with open(f'{path}_aligned_mats.npy','rb') as f:
        aligned_mats = np.load(f)
    
    with open(f'{path}_seqs_unaligned.npy','rb') as f:
        seqs_unaligned = np.load(f)
    
    del f
    
    
    ### declare dims
    B = aligned_mats.shape[0]
    L_align = aligned_mats.shape[1]
    
    
    ### get the state path
    is_ins = (aligned_mats[...,0] == 43) #(B, L_align)
    m_idx = aligned_mats[...,2] #(B, L_align)
    
    desc_tok_counts = np.zeros((out_alphabet_size)) #(A_aug,)
    desc_given_current_anc_tok_counts = np.zeros((in_alphabet_size, out_alphabet_size)) #(A, A_aug)
    for b in tqdm( range(B) ):
        this_sample_m_idx = m_idx[b,...] #(L_align)
        this_sample_anc = seqs_unaligned[b,...,0] #(L_anc)
        this_sample_anc_left_aligned = this_sample_anc[this_sample_m_idx] #(L_align)
        assert (this_sample_anc_left_aligned != 43).all()
        del this_sample_m_idx, this_sample_anc
        
        for l in range(1,L_align):
            current_a = this_sample_anc_left_aligned[l-1]
            current_d_or_gap = aligned_mats[b,l,1]
            current_col_is_ins = is_ins[b,l] 
            
            if current_col_is_ins:
                current_d_or_gap += 20
                
            desc_tok_counts[current_d_or_gap] += 1
            desc_given_current_anc_tok_counts[current_a, current_d_or_gap] += 1
    
    del m_idx, b, current_a, current_d_or_gap, l
    del this_sample_anc_left_aligned
    
    
    ### remove padding counts
    desc_tok_counts[0] = 0
    desc_given_current_anc_tok_counts[0,:] = 0
    desc_given_current_anc_tok_counts[:,0] = 0
    
    
    ### final checksums
    # encoding the same total counts
    assert np.allclose( desc_given_current_anc_tok_counts.sum(axis=0), desc_tok_counts )
    
    # <eos> <-> <eox> only appears once per sample
    assert desc_tok_counts[2] == B
    assert desc_given_current_anc_tok_counts[2,2] == B
    
    # total aligned tokens matches input file
    total_aligned_toks = ( aligned_mats[...,0] != 0 ) & ( aligned_mats[...,0] != 1 )
    total_aligned_toks = total_aligned_toks.sum()
    assert desc_tok_counts.sum() == total_aligned_toks
    assert desc_given_current_anc_tok_counts.sum() == total_aligned_toks
    
    
    ### write files
    with open(f'{data_dir}/{prefix}_desc-align_counts.npy','wb') as g:
        np.save( g, desc_tok_counts )
        
    with open(f'{data_dir}/{prefix}_desc-align_given_current_anc_counts.npy','wb') as g:
        np.save( g, desc_given_current_anc_tok_counts )
                
if __name__ == '__main__':
    from tqdm import tqdm
    
    
    data_dir = 'DATA_cherries'
    
    for p in [f'split{i}' for i in range(10)] + ['OOD_valid']:
        prefix = f'FAMCLAN-CHERRIES_{p}'
        print(prefix)
        
        main(data_dir, prefix)
        