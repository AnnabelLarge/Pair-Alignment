#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 19:18:50 2025

@author: annabel
"""
import numpy as np
import pandas as pd
from tqdm import tqdm


save_mats = True
remove_eos_counts = False


########################
### load training set  #
########################
def load_counts(data_folder, prefix_lst, suffix):
    all_mats = []
    for pre in prefix_lst:
        all_mats.append( np.load(f'{data_folder}/{pre}_{suffix}.npy') )
    return np.stack(all_mats, axis=0).sum(axis=0)

def load_scoring_mats(prefix_lst, remove_eos=True):
    ### P(desc, align | current_anc, prev_desc)
    counts = load_counts( data_folder = 'example_data',
                                         prefix_lst = prefix_lst,
                                         suffix = 'desc-align_given_curr-anc-prev-desc_counts') #(A, A, A_aug)
    
    # for this test, don't count beginning or end of alignments
    # however, DO score (eos, any) pairs i.e. inserts at the end of the alignment
    counts[0, :, :] = 0
    counts[:, 0, :] = 0
    counts[:, :, 0] = 0
    counts[1, :, 1] = 0
    if remove_eos:
        counts[2, :, 2] = 0
    
    denom = counts.sum(axis=2, keepdims=True)  # shape (A, A, 1)
    freqs = counts / np.where(denom == 0, 1, denom) #(A, A, A_aug)
    
    
    ### finally, get P(len) for a geometric sequence (from the metadata file)
    all_lens = []
    for prefix in prefix_lst:
        metadata_file = f'example_data/{prefix}_metadata.tsv'
        lens_mat = pd.read_csv( metadata_file, 
                               sep='\t',
                               usecols=['desc_seq_len'] ).to_numpy() #(B,)
        all_lens.append(lens_mat)
        
    all_lens = np.concatenate(all_lens, axis=0) #(B,)
    
    geom_prob_end_seq = all_lens.shape[0] / ( all_lens.shape[0] + all_lens.sum() )
    
    return (freqs, geom_prob_end_seq)

out = load_scoring_mats(['threeSampShort'],
                        remove_eos_counts)
freqs, geom_prob_end_seq = out
del out

with open(f'geom_prob_end_seq_eos.tsv','w') as g:
    g.write(f'P(end token): {geom_prob_end_seq}\n')

with open(f'frequencies_from_train_set.npy','wb') as g:
    np.save(g, freqs)
    

###################################################################
### score conditionals with a very dumb loop, because I'm tired   #
###################################################################
def score(prefix, freq_mat, save, use_trans_score=True):
    with open(f'example_data/{prefix}_seqs_unaligned.npy','rb') as f:
        seqs_unaligned = np.load(f) #(B, L, 2)
    
    with open(f'example_data/{prefix}_aligned_mats.npy','rb') as f:
        aligned_mats = np.load(f) #(B, L, 4)
    desc_lengths = (~np.isin(aligned_mats[...,1], [0,1,2,43])).sum(axis=1) #(B,)
    
    # augment descendant
    ins_loc = (aligned_mats[:,:,0] == 43) #(B, L_align)
    align_aug_desc = np.where( ins_loc,
                               aligned_mats[:,:,1] + 20,
                               aligned_mats[:,:,1]) #(B, L_align)
    
    # unpack
    anc = seqs_unaligned[:, :, 0] #(B, L_anc)
    desc = seqs_unaligned[:, :, 1] #(B, L_desc)
    m_idx = aligned_mats[:, :, 2] #(B, L_align)
    n_idx = aligned_mats[:, :, 3] #(B, L_align)
    mask = (aligned_mats[:, :, 2] != -9) #(B, L_align)
    
    # scoring matrix
    scoring_mat = np.log( np.where(freq_mat != 0,
                                   freq_mat,
                                   1) ) #(A, A_aug)
    
    
    ### emissions
    B = aligned_mats.shape[0]
    L = aligned_mats.shape[1]

    emit_score = np.zeros(B)
    for b in tqdm( range(B) ):
        anc_seq_unaligned = anc[b] #(L_seq)
        desc_seq_unaligned = desc[b] #(L_seq)
        curr_anc = anc_seq_unaligned[ m_idx[b] ] #(L_align)
        prev_desc = desc_seq_unaligned[ n_idx[b] ] #(L_align)
        mask_vec = mask[b] #(L_align)
        desc_seq_augmented_with_alignment = align_aug_desc[b] #(L_align)
        
        del anc_seq_unaligned, desc_seq_unaligned
        
        for l in range(1,L):
            # stop at final end place
            if not mask_vec[l-1]:
                break
            
            # NOT start
            curr_align_aug_desc_tok = desc_seq_augmented_with_alignment[l]

            # alignment tuple is one behind current alignment position
            curr_anc_tok = curr_anc[l-1]
            prev_desc_tok = prev_desc[l-1]
            
            # add to emit score
            emit_score[b] += scoring_mat[curr_anc_tok, prev_desc_tok, curr_align_aug_desc_tok]
            
            
            
    ### add transitions and compile final scores
    if use_trans_score:
        transit_score = desc_lengths * np.log( 1-geom_prob_end_seq ) + np.log( geom_prob_end_seq ) #(B,)
    else:
        transit_score = 0
        
    score = emit_score + transit_score
    
    # metadata
    metadata_file = f'example_data/{prefix}_metadata.tsv'
    out_df = pd.read_csv( metadata_file, 
                          sep='\t',
                          usecols=['pairID','ancestor','descendant','pfam'] )
    
    out_df['desc_seq_len'] = desc_lengths
    out_df['cond_loglike'] = score
    out_df['cond_loglike_seqlen_normed'] = score / desc_lengths
        
    # save
    if save:
        out_file = f'desc-align_given_curr-anc-prev-desc_cond_{dset}.tsv'
        out_df.to_csv(out_file, sep='\t')
    
    return out_df

test_df = []
for i in [0]:
    dset = 'threeSampShort'
    print(dset)
    df = score(prefix = dset, 
               freq_mat = freqs,
               save = save_mats,
               use_trans_score = remove_eos_counts)
    print()
    test_df.append(df)


################
### Postproc   #
################
# combine dataframes 
all_test_scores = pd.concat(test_df, axis=0)

if save_mats:
    all_test_scores.to_csv('test_desc-align_given_curr-anc-prev-desc_scores.tsv', sep='\t')

# get stats of interest
def maybe_flip_sign(vec):
    if (vec <= 0).all():
        return -vec
    else:
        return vec
    
def get_stats(df, outfile, dset_name):
    neg_loglikes = maybe_flip_sign( df['cond_loglike'] )
    neg_loglikes_len_normed = maybe_flip_sign( df['cond_loglike_seqlen_normed'] )
    
    
    sum_loglikes = neg_loglikes.sum()
    ave_loss = neg_loglikes.mean()
    ave_loss_seqlen_normed = neg_loglikes_len_normed.mean()
    ece = np.exp(ave_loss_seqlen_normed)
    perplexity = np.exp( neg_loglikes_len_normed ).mean()

    out_dict = {'sum_cond_loglikes': sum_loglikes,
                'ave_cond_loss': ave_loss,
                'ave_cond_loss_seqlen_normed': ave_loss_seqlen_normed,
                'cond_ece': ece,
                'cond_perplexity': perplexity}
    
    with open(outfile,'w') as g:
        g.write(f'RUN\t{dset_name}\n')
        for key, val in out_dict.items():
            g.write(f'{key}\t{val}\n')
    
    return out_dict
    
test_summary_stats = get_stats(all_test_scores,
                                'TEST-AVE_DESC-ALIGN_GIVEN_CURR-ANC-PREV-DESC_LOGLIKES.tsv',
                                'test_set_desc-align_given_curr-anc-prev-desc_loglikes')

