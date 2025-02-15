#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:06:14 2025

@author: annabel
"""
import jax
from jax.scipy.special import logsumexp
from jax import numpy as jnp

import pandas as pd
pd.options.mode.chained_assignment = None # this is annoying
import numpy as np
import argparse
import json
from tqdm import tqdm
import os
from torch.utils.data import DataLoader

from dloaders.FullLenDset import FullLenDset 
from dloaders.FullLenDset import jax_collator as collator


def safe_log(x):
    return jnp.where( x > 0,
                      jnp.log( x ),
                      jnp.log( jnp.finfo('float32').smallest_normal )
                      )

def get_feedforward_controls(args):
    ###############################################################################
    ### MAKE DIR, LOAD DATA   #####################################################
    ###############################################################################
    if args.training_wkdir not in os.listdir():
        os.mkdir(args.training_wkdir)
    
    
    # train data
    print(f'Training dset: {args.train_dset_splits}')
    assert type(args.train_dset_splits) == list
    training_dset = FullLenDset( data_dir = args.data_dir, 
                                split_prefixes = args.train_dset_splits,
                                times_from_array = np.ones( (1,) ),
                                single_time_from_file = False,
                                toss_alignments_longer_than = args.toss_alignments_longer_than,
                                pred_model_type = 'neural_hmm',
                                use_scan_fns = False )
    
    training_dl = DataLoader( training_dset, 
                              batch_size = args.batch_size, 
                              shuffle = False,
                              collate_fn = collator
                             )
    
    # test data
    print(f'Test dset: {args.test_dset_splits}')
    assert type(args.test_dset_splits) == list
    test_dset = FullLenDset( data_dir = args.data_dir, 
                            split_prefixes = args.test_dset_splits,
                            times_from_array = np.ones( (1,) ),
                            single_time_from_file = False,
                            toss_alignments_longer_than = args.toss_alignments_longer_than,
                            pred_model_type = 'neural_hmm',
                            use_scan_fns = False )
    
    test_dl = DataLoader( test_dset, 
                          batch_size = args.batch_size, 
                          shuffle = False,
                          collate_fn = collator
                         )
    
    
    ###############################################################################
    ### GET SCORING MATRICES FROM TRAIN SET   #####################################
    ###############################################################################
    base_alphabet_size = args.base_alphabet_size if 'base_alphabet_size' in dir(args) else 23
    full_alphabet_size = args.full_alphabet_size if 'full_alphabet_size' in dir(args) else 44
    emission_alphabet_size = args.emission_alphabet_size if 'emission_alphabet_size' in dir(args) else 20
    
    training_aligned_mat = jnp.array( training_dset.aligned_mat[:,:,:3] )
    
    
    def scan_fn(carry_dict, col_idx):
        curr_col = training_aligned_mat[:,col_idx,:]
        for_ins_updates = carry_dict['for_ins_updates']
        counts = carry_dict['counts']
        
        # updates counts dict
        row_idx = jnp.where( curr_col[:,-1] != 2,
                             curr_col[:,0],
                             for_ins_updates )
        
        col_idx = jnp.where( curr_col[:,-1] !=2,
                             curr_col[:,1],
                             curr_col[:,1]+20 )
        
        to_add = jnp.ones( (curr_col.shape[0],) )
        new_counts = counts.at[row_idx, col_idx].add(to_add)
        
        # update last_seen_valid anc
        to_update = (curr_col[:,-1] == 1) | (curr_col[:,-1] == 3) | (curr_col[:,-1] == 5)
        for_ins_updates = jnp.where(to_update, 
                                    curr_col[:,0], 
                                    for_ins_updates )
        
        out_dict = {'for_ins_updates': for_ins_updates,
                    'counts': new_counts}
        
        return out_dict, None
    
    init_dict = {'for_ins_updates': jnp.zeros( (training_aligned_mat.shape[0]), 
                                                   dtype=int),
                 'counts': jnp.zeros( (base_alphabet_size, full_alphabet_size) )
                 }
    
    idxes = jnp.flip( jnp.arange(training_aligned_mat.shape[1]) )
    out_dict, _ = jax.lax.scan(f = scan_fn,
                               init = init_dict,
                               xs = idxes,
                               length = idxes.shape[0])
    joint_counts = out_dict['counts']
    del out_dict
    
    
    # ### loop version of what I want to do; uncomment to check jax.lax.scan
    # num_updates = 0
    # true_joint_counts = np.zeros( (base_alphabet_size, full_alphabet_size) )
    # training_aligned_mat = np.array( training_aligned_mat )
    # for_ins_updates = np.zeros( (training_aligned_mat.shape[0]), dtype=int)
    # for l in tqdm( range(training_aligned_mat.shape[1]-1, -1, -1) ):
    #     curr_col = training_aligned_mat[:, l, :]
    #     for b in range(training_aligned_mat.shape[0]):
    #         curr_anc, curr_desc, align = curr_col[b,:]
    #         last_seen_valid_anc = for_ins_updates[b]
            
    #         # at ins, use previously seen ancestor, add 20 to descendant
    #         if align == 2:
    #             true_joint_counts[last_seen_valid_anc, curr_desc + 20] += 1
            
    #         # otherwise, add as normal
    #         elif align in [1,3,5]:
    #             true_joint_counts[curr_anc, curr_desc] += 1
    #             for_ins_updates[b] = curr_anc
            
    #         elif align in [0,4]:
    #             true_joint_counts[curr_anc, curr_desc] += 1
                
    #     num_updates += 1
    
    # assert num_updates == training_aligned_mat.shape[1]
    # assert true_joint_counts.sum() == (training_aligned_mat.shape[0] *
    #                               training_aligned_mat.shape[1])
    # assert jnp.allclose( true_joint_counts, joint_counts )
    
    
    # main scoring matrices
    anc_marginal_counts = joint_counts.sum(axis=1)[:,None]
    freq_based_desc_given_anc_probs = jnp.where(  anc_marginal_counts > 0,
                                            joint_counts / marginal_counts,
                                            0 )
    freq_based_desc_given_anc_logprobs = safe_log( freq_based_desc_given_anc_probs )
    
    # assert jnp.allclose( freq_based_desc_given_anc_probs.sum(axis=1),
    #                      jnp.ones(marginal_counts.shape) )
    
    desc_marginal_counts = joint_counts.sum(axis=0)
    freq_based_desc_marginal_probs = desc_marginal_counts / desc_marginal_counts.sum()
    freq_based_desc_marginal_logprobs = safe_log( freq_based_desc_marginal_probs )
    
    
    # padding tokens shouldn't add to score
    freq_based_desc_given_anc_logprobs = freq_based_desc_given_anc_logprobs.at[:,0].set(0)
    freq_based_desc_given_anc_logprobs = freq_based_desc_given_anc_logprobs.at[0,:].set(0) 
    freq_based_desc_marginal_logprobs = freq_based_desc_marginal_logprobs.at[0].set(0)
    
    
    ###############################################################################
    ### SCORE, WRITE   ############################################################
    ###############################################################################
    ### training set
    all_train_desc_given_anc_losses = []
    all_train_desc_given_anc_perpl = []
    all_train_desc_marginal_losses = []
    all_train_desc_marginal_perpl = []
    
    for idx, batch in enumerate( tqdm(training_dl) ):
        # don't include <bos>
        batch_aligned_mat = batch[1][:,1:,:3]
        
        def scan_fn(carry_dict, col_idx):
            curr_col = batch_aligned_mat[:,col_idx,:]
            for_ins_updates = carry_dict['for_ins_updates']
            desc_given_anc_score = carry_dict['desc_given_anc_score']
            desc_marginal_score = carry_dict['desc_marginal_score']
            
            # get new indices
            row_idx = jnp.where( curr_col[:,-1] != 2,
                                 curr_col[:,0],
                                 for_ins_updates )
            
            col_idx = jnp.where( curr_col[:,-1] !=2,
                                 curr_col[:,1],
                                 curr_col[:,1]+20 )
            
            # retrieve score 
            new_desc_given_anc_score = desc_given_anc_score + freq_based_desc_given_anc_logprobs[row_idx, col_idx]
            new_desc_marginal_score = desc_marginal_score + freq_based_desc_marginal_logprobs[col_idx]
            
            # update last_seen_valid anc
            to_update = (curr_col[:,-1] == 1) | (curr_col[:,-1] == 3) | (curr_col[:,-1] == 5)
            for_ins_updates = jnp.where(to_update, 
                                        curr_col[:,0], 
                                        for_ins_updates )
            
            out_dict = {'for_ins_updates': for_ins_updates,
                        'desc_given_anc_score': new_desc_given_anc_score,
                        'desc_marginal_score': new_desc_marginal_score}
            
            return out_dict, None
    
        init_dict = {'for_ins_updates': jnp.zeros( (batch_aligned_mat.shape[0]), 
                                                       dtype=int),
                     'desc_given_anc_score': jnp.zeros( (batch_aligned_mat.shape[0]) ),
                     'desc_marginal_score': jnp.zeros( (batch_aligned_mat.shape[0]) ),
                     }
    
        idxes = jnp.flip( jnp.arange(batch_aligned_mat.shape[1]) )
        out_dict, _ = jax.lax.scan(f = scan_fn,
                                   init = init_dict,
                                   xs = idxes,
                                   length = idxes.shape[0])
        
        desc_given_anc_scores = out_dict['desc_given_anc_score']
        desc_marginal_scores = out_dict['desc_marginal_score']
        
        
        # ### loop version of what I want to do; uncomment to check jax.lax.scan
        # true_desc_given_anc_score = np.zeros( (batch_aligned_mat.shape[0], ) )
        # true_desc_marginal_score = np.zeros( (batch_aligned_mat.shape[0], ) )
                                   
        # for_ins_updates = np.zeros( (batch_aligned_mat.shape[0]), dtype=int)
        # for l in tqdm( range(batch_aligned_mat.shape[1]-1, -1, -1) ):
        #     curr_col = batch_aligned_mat[:, l, :]
        #     for b in range(batch_aligned_mat.shape[0]):
        #         curr_anc, curr_desc, align = curr_col[b,:]
        #         last_seen_valid_anc = for_ins_updates[b]
                
        #         # at ins, use previously seen ancestor, add 20 to descendant indexing
        #         if align == 2:
        #             desc_given_anc_to_add = freq_based_desc_given_anc_logprobs[ last_seen_valid_anc, 
        #                                                             curr_desc + 20 ]
        #             desc_marginal_to_add = freq_based_desc_marginal_logprobs[curr_desc + 20]
        #             true_desc_given_anc_score[b] += desc_given_anc_to_add
        #             true_desc_marginal_score[b] += desc_marginal_to_add
                
        #         # otherwise, add as normal
        #         elif align in [1,3,5]:
        #             desc_given_anc_to_add = freq_based_desc_given_anc_logprobs[curr_anc, curr_desc]
        #             desc_marginal_to_add = freq_based_desc_marginal_logprobs[curr_desc]
        #             true_desc_given_anc_score[b] += desc_given_anc_to_add
        #             true_desc_marginal_score[b] += desc_marginal_to_add
        #             for_ins_updates[b] = curr_anc
                
        #         elif align in [0,4]:
        #             desc_given_anc_to_add = freq_based_desc_given_anc_logprobs[curr_anc, curr_desc]
        #             desc_marginal_to_add = freq_based_desc_marginal_logprobs[curr_desc]
        #             true_desc_given_anc_score[b] += desc_given_anc_to_add
        #             true_desc_marginal_score[b] += desc_marginal_to_add
                    
        # assert jnp.allclose( true_desc_given_anc_score, desc_given_anc_scores )
        # assert jnp.allclose( true_desc_marginal_score, desc_marginal_scores )
        
        
        # normalizing length
        if args.norm_loss_by == 'desc_len':
            length = ( (batch_aligned_mat[:,:,1] != 0) & (batch_aligned_mat[:,:,1] != 43)).sum(axis=1)
            
        elif args.norm_loss_by == 'align_len':
            length = ( batch_aligned_mat[:,:,1] ).sum(axis=1)
        
        # output two separate dataframes for each control
        template = training_dset.retrieve_sample_names(batch[-1])
        
        train_desc_given_anc_df = template.copy()
        train_desc_given_anc_df['cond_raw_scores'] = desc_given_anc_scores
        train_desc_given_anc_df['cond_normed_scores'] = desc_given_anc_scores / length
        train_desc_given_anc_df['cond_perplexity'] = jnp.exp( -(desc_given_anc_scores / length) )
        train_desc_given_anc_df.to_csv( (f'{args.training_wkdir}/'+
                                   f'desc-given-anc_train-set_pt{idx}_FINAL-LOGLIKES.tsv'),
                                 sep='\t'
                                 )
        all_train_desc_given_anc_losses.append( (desc_given_anc_scores / length) )
        all_train_desc_given_anc_perpl.append( ( jnp.exp( -(desc_given_anc_scores / length) ) ) )
        
        train_desc_marginal_df = template.copy()
        train_desc_marginal_df['cond_raw_scores'] = desc_marginal_scores
        train_desc_marginal_df['cond_normed_scores'] = desc_marginal_scores / length
        train_desc_marginal_df['cond_perplexity'] = jnp.exp( -(desc_marginal_scores / length) )
        train_desc_marginal_df.to_csv( (f'{args.training_wkdir}/'+
                               f'desc-marginals_train-set_pt{idx}_FINAL-LOGLIKES.tsv'),
                              sep='\t'
                              )
        all_train_desc_marginal_losses.append( ( desc_marginal_scores / length ) )
        all_train_desc_marginal_perpl.append( ( jnp.exp( -(desc_marginal_scores / length) ) ) )
        
        
        
    ### test set
    all_test_desc_given_anc_losses = []
    all_test_desc_given_anc_perpl = []
    all_test_desc_marginal_losses = []
    all_test_desc_marginal_perpl = []

    for idx, batch in enumerate( tqdm(training_dl) ):
        # don't include <bos>
        batch_aligned_mat = batch[1][:,1:,:3]
        
        def scan_fn(carry_dict, col_idx):
            curr_col = batch_aligned_mat[:,col_idx,:]
            for_ins_updates = carry_dict['for_ins_updates']
            desc_given_anc_score = carry_dict['desc_given_anc_score']
            desc_marginal_score = carry_dict['desc_marginal_score']
            
            # get new indices
            row_idx = jnp.where( curr_col[:,-1] != 2,
                                 curr_col[:,0],
                                 for_ins_updates )
            
            col_idx = jnp.where( curr_col[:,-1] !=2,
                                 curr_col[:,1],
                                 curr_col[:,1]+20 )
            
            # retrieve score 
            new_desc_given_anc_score = desc_given_anc_score + freq_based_desc_given_anc_logprobs[row_idx, col_idx]
            new_desc_marginal_score = desc_marginal_score + freq_based_desc_marginal_logprobs[col_idx]
            
            # update last_seen_valid anc
            to_update = (curr_col[:,-1] == 1) | (curr_col[:,-1] == 3) | (curr_col[:,-1] == 5)
            for_ins_updates = jnp.where(to_update, 
                                        curr_col[:,0], 
                                        for_ins_updates )
            
            out_dict = {'for_ins_updates': for_ins_updates,
                        'desc_given_anc_score': new_desc_given_anc_score,
                        'desc_marginal_score': new_desc_marginal_score}
            
            return out_dict, None

        init_dict = {'for_ins_updates': jnp.zeros( (batch_aligned_mat.shape[0]), 
                                                       dtype=int),
                     'desc_given_anc_score': jnp.zeros( (batch_aligned_mat.shape[0]) ),
                     'desc_marginal_score': jnp.zeros( (batch_aligned_mat.shape[0]) ),
                     }

        idxes = jnp.flip( jnp.arange(batch_aligned_mat.shape[1]) )
        out_dict, _ = jax.lax.scan(f = scan_fn,
                                   init = init_dict,
                                   xs = idxes,
                                   length = idxes.shape[0])
        
        desc_given_anc_scores = out_dict['desc_given_anc_score']
        desc_marginal_scores = out_dict['desc_marginal_score']
        
        # normalizing length
        if args.norm_loss_by == 'desc_len':
            length = ( (batch_aligned_mat[:,:,1] != 0) & (batch_aligned_mat[:,:,1] != 43)).sum(axis=1)
            
        elif args.norm_loss_by == 'align_len':
            length = ( batch_aligned_mat[:,:,1] ).sum(axis=1)
        
        # output two separate dataframes for each control
        template = training_dset.retrieve_sample_names(batch[-1])
        
        test_desc_given_anc_df = template.copy()
        test_desc_given_anc_df['cond_raw_scores'] = desc_given_anc_scores
        test_desc_given_anc_df['cond_normed_scores'] = desc_given_anc_scores / length
        test_desc_given_anc_df['cond_perplexity'] = jnp.exp( -(desc_given_anc_scores / length) )
        test_desc_given_anc_df.to_csv( (f'{args.training_wkdir}/'+
                                   f'desc-given-anc_test-set_pt{idx}_FINAL-LOGLIKES.tsv'),
                                 sep='\t'
                                 )
        all_test_desc_given_anc_losses.append( (desc_given_anc_scores / length) )
        all_test_desc_given_anc_perpl.append( ( jnp.exp( -(desc_given_anc_scores / length) ) ) )
        
        test_desc_marginal_df = template.copy()
        test_desc_marginal_df['cond_raw_scores'] = desc_marginal_scores
        test_desc_marginal_df['cond_normed_scores'] = desc_marginal_scores / length
        test_desc_marginal_df['cond_perplexity'] = jnp.exp( -(desc_marginal_scores / length) )
        test_desc_marginal_df.to_csv( (f'{args.training_wkdir}/'+
                               f'desc-marginals_test-set_pt{idx}_FINAL-LOGLIKES.tsv'),
                              sep='\t'
                              )
        all_test_desc_marginal_losses.append( ( desc_marginal_scores / length ) )
        all_test_desc_marginal_perpl.append( ( jnp.exp( -(desc_marginal_scores / length) ) ) )  
    
    
    ### write scores 
    # anc and desc-based
    to_write = { 'RUN': args.training_wkdir,
                 'train_ave_cond_loss_seqlen_normed': jnp.concatenate( all_train_desc_given_anc_losses ).mean(),
                 'train_perplexity': jnp.concatenate( all_train_desc_given_anc_perpl ).mean(),
                 'train_ece': jnp.exp( -jnp.concatenate( all_train_desc_given_anc_losses ).mean() ),
                 
                 'test_ave_cond_loss_seqlen_normed': jnp.concatenate( all_test_desc_given_anc_losses ).mean(),
                 'test_perplexity': jnp.concatenate( all_test_desc_given_anc_perpl ).mean(),
                 'test_ece': jnp.exp( -jnp.concatenate( all_test_desc_given_anc_losses ).mean() ),
                 }
    
    with open(f'{args.training_wkdir}/desc-given-anc_AVE-LOSSES.tsv', 'w') as g:
        for k, v in to_write.items():
            g.write(f'{k}\t{v}\n')
    del to_write
            
    # only desc-based
    to_write = { 'RUN': args.training_wkdir,
                 'train_ave_cond_loss_seqlen_normed': jnp.concatenate( all_train_desc_marginal_losses ).mean(),
                 'train_perplexity': jnp.concatenate( all_train_desc_marginal_perpl ).mean(),
                 'train_ece': jnp.exp( -jnp.concatenate( all_train_desc_marginal_losses ).mean() ),
                 
                 'test_ave_cond_loss_seqlen_normed': jnp.concatenate( all_test_desc_marginal_losses ).mean(),
                 'test_perplexity': jnp.concatenate( all_test_desc_marginal_perpl ).mean(),
                 'test_ece': jnp.exp( -jnp.concatenate( all_test_desc_marginal_losses ).mean() ),
                 }
    
    with open(f'{args.training_wkdir}/desc-marginals_AVE-LOSSES.tsv', 'w') as g:
        for k, v in to_write.items():
            g.write(f'{k}\t{v}\n')
    del to_write
            
    
    ### write scoring matrices
    def write_matrix(mat, filename):
        with open(f'{args.training_wkdir}/{filename}', 'wb') as g:
            jnp.save(g, mat)
    
    write_matrix(mat = freq_based_desc_given_anc_logprobs, 
                 filename = 'desc_given_anc_logprobs.npy')
    
    write_matrix(mat = freq_based_desc_marginals_logprobs, 
                 filename = 'desc_marginals_logprobs.npy')
