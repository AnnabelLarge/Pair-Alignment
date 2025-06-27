#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 13:53:31 2025

@author: annabel
"""
import numpy as np
import pandas as pd


#######################
### True Alignments   #
#######################
A = 4

# unaligned seqs
ancs = np.array( [[1, 3, 4, 5, 2, 0], 
                  [1, 3, 2, 0, 0, 0],
                  [1, 3, 4, 5, 6, 2],
                  [1, 3, 4, 5, 6, 2],
                  [1, 3, 4, 2, 0, 0]] )

descs = np.array( [[ 1,  3,  2,  0,  0,  0],
                   [ 1,  3,  4,  5,  2,  0],
                   [ 1,  3,  4,  5,  6,  2],
                   [ 1,  3,  4,  5,  6,  2],
                   [ 1,  3,  4,  2,  0,  0]] )

unaligned_mats = np.stack( [ancs, descs], axis=-1 )
del ancs, descs

L_seq = unaligned_mats.shape[1]

# aligned matrices
ancs_aligned = np.array( [[1,  3,  4,  5,  2,  0,  0],
                          [1, 43,  3, 43,  2,  0,  0],
                          [1,  3,  4,  5,  6, 43,  2],
                          [1, 43,  3,  4,  5,  6,  2],
                          [1,  3,  4,  2,  0,  0,  0]] )

descs_aligned = np.array( [[ 1, 43,  3, 43,  2,  0,  0],
                           [ 1,  3,  4,  5,  2,  0,  0],
                           [ 1, 43,  3,  4,  5,  6,  2],
                           [ 1,  3,  4,  5,  6, 43,  2],
                           [ 1,  3,  4,  2,  0,  0,  0]] )

state = np.array( [[4,3,1,3,5,0,0],
                   [4,2,1,2,5,0,0],
                   [4,3,1,1,1,2,5],
                   [4,2,1,1,1,3,5],
                   [4,1,1,5,0,0,0]] )

m_idx = np.array( [[1,  2,  3,  4, -9, -9, -9],
                   [1,  1,  2,  2, -9, -9, -9],
                   [1,  2,  3,  4,  5,  5, -9],
                   [1,  1,  2,  3,  4,  5, -9],
                   [1,  2,  3, -9, -9, -9, -9]] )

n_idx = np.array( [[0,  0,  1,  1, -9, -9, -9],
                   [0,  1,  2,  3, -9, -9, -9],
                   [0,  0,  1,  2,  3,  4, -9],
                   [0,  1,  2,  3,  4,  4, -9],
                   [0,  1,  2, -9, -9, -9, -9]] )

aligned_mats = np.stack( [ancs_aligned, descs_aligned, state, m_idx, n_idx], axis=-1 )
B = aligned_mats.shape[0]
L_align = aligned_mats.shape[1]

del m_idx, n_idx, ancs_aligned, descs_aligned, state

# metadata
meta_df = pd.DataFrame( {'pairID': [f'pair{i}' for i in range(B)], 
                         'ancestor': [f'anc{i}' for i in range(B)],
                         'descendant': [f'desc{i}' for i in range(B)],
                         'pfam': 'PF00000',
                         'anc_seq_len': [3, 1, 4, 4, 2],
                         'desc_seq_len': [1, 3, 4, 4, 2], 
                         'alignment_len': [3, 3, 5, 5, 2], 
                         'num_matches': [1, 1, 3, 3, 2], 
                         'num_ins': [0,2, 1, 1, 0], 
                         'num_del': [2, 0, 1, 1, 0]} )

# times
copy_df = meta_df.copy()['pairID']
copy_df['times'] = np.arange( 1, B+1 ) * 0.1


############################
### make counts matrices   #
############################
# calculate true values by loop
true_subs = np.zeros( (B, A, A) )
true_ins = np.zeros( (B, A) )
true_del = np.zeros( (B, A) )
true_trans = np.zeros( (B, 5, 5) )
true_emissions = np.zeros( (A,) )
true_emit_from_match = np.zeros( (A,) )

for b in range(B):
    # first position is start
    prev_state = 4
    
    for l in range(1, L_align):
        anc_tok = aligned_mats[b,l,0]
        desc_tok = aligned_mats[b,l,1]
        
        # padding
        if anc_tok == 0:
            break
        
        # end
        if (anc_tok == 2) and (desc_tok == 2):
            curr_state = 5
        
        # match
        elif (anc_tok != 43) and (desc_tok != 43):
            curr_state = 1
            true_subs[b, anc_tok-3, desc_tok-3] += 1
            true_emissions[anc_tok-3] += 1
            true_emissions[desc_tok-3] += 1
            true_emit_from_match[anc_tok-3] += 1
            true_emit_from_match[desc_tok-3] += 1
        
        # ins
        elif (anc_tok == 43) and (desc_tok != 43):
            curr_state = 2
            true_ins[b, desc_tok-3] += 1
            true_emissions[desc_tok-3] += 1
        
        # del
        elif (anc_tok != 43) and (desc_tok == 43):
            curr_state = 3
            true_del[b, anc_tok-3] += 1
            true_emissions[anc_tok-3] += 1
        
        # update transitions
        true_trans[b, prev_state-1, curr_state-1] += 1
        prev_state = curr_state
     

################
### save all   #
################
arrs = [unaligned_mats,
        aligned_mats,
        true_subs,
        true_ins,
        true_del,
        true_trans,
        true_emissions,
        true_emit_from_match]

suffixes = ['seqs_unaligned',
            'aligned_mats',
            'subCounts',
            'insCounts',
            'delCounts',
            'transCounts_five_by_five',
            'NuclCounts',
            'NuclCounts_subsOnly']

prefix = 'fake_inputs_dna'
for i in range( len(arrs) ):
    a = arrs[i]
    s = suffixes[i]
    
    with open(f'{prefix}_{s}.npy','wb') as g:
        np.save(g, a)

meta_df.to_csv(f'{prefix}_metadata.tsv', sep='\t')
copy_df.to_csv(f'{prefix}_pair-times.tsv', header=False, index=False, sep='\t')

    

