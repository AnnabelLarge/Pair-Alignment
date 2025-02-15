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

from dloaders.CountsDset import CountsDset 
from dloaders.CountsDset import jax_collator as collator


def safe_log(x):
    return jnp.where( x > 0,
                      jnp.log( x ),
                      jnp.log( jnp.finfo('float32').smallest_normal )
                      )

def get_pairhmm_controls(args):
    ###############################################################################
    ### MAKE DIR, LOAD DATA   #####################################################
    ###############################################################################
    if args.training_wkdir not in os.listdir():
        os.mkdir(args.training_wkdir)
    
    
    # train data
    print(f'Training dset: {args.train_dset_splits}')
    assert type(args.train_dset_splits) == list
    training_dset = CountsDset( data_dir = args.data_dir, 
                                split_prefixes = args.train_dset_splits,
                                times_from_array = np.ones( (1,) ),
                                single_time_from_file = False,
                                toss_alignments_longer_than = args.toss_alignments_longer_than,
                                bos_eos_as_match = args.bos_eos_as_match)
    
    training_dl = DataLoader( training_dset, 
                              batch_size = args.batch_size, 
                              shuffle = False,
                              collate_fn = collator
                             )
    
    # test data
    print(f'Test dset: {args.test_dset_splits}')
    assert type(args.test_dset_splits) == list
    test_dset = CountsDset( data_dir = args.data_dir, 
                            split_prefixes = args.test_dset_splits,
                            times_from_array = np.ones( (1,) ),
                            single_time_from_file = False,
                            toss_alignments_longer_than = args.toss_alignments_longer_than,
                            bos_eos_as_match = args.bos_eos_as_match)
    
    test_dl = DataLoader( test_dset, 
                          batch_size = args.batch_size, 
                          shuffle = False,
                          collate_fn = collator
                         )
    
    
    ###############################################################################
    ### GET SCORING MATRICES FROM TRAIN SET   #####################################
    ###############################################################################
    ### based on frequencies
    # aggregate counts
    all_transits = jnp.array( training_dset.transCounts.sum(axis=0) )
    all_cond_matches = jnp.array( training_dset.subCounts.sum(axis=0) )
    all_ins_counts = jnp.array( training_dset.insCounts.sum(axis=0) )
    all_del_counts = jnp.array( training_dset.delCounts.sum(axis=0) )
    
    # make probability matrices
    transits_logprobs = safe_log( all_transits / all_transits.sum() )
    cond_match_logprobs = safe_log( all_cond_matches / all_cond_matches.sum() )
    ins_logprobs = safe_log( all_ins_counts / all_ins_counts.sum() )
    del_logprobs = safe_log( all_del_counts / all_del_counts.sum() )
    del all_transits, all_cond_matches, all_ins_counts, all_del_counts
    
    # joint = cond + marginal(anc)
    anc_marginal_logprobs = logsumexp( cond_match_logprobs, axis=1 )
    joint_match_logprobs = cond_match_logprobs + anc_marginal_logprobs[:, None]
    
    
    
    ###############################################################################
    ### SCORE, WRITE   ############################################################
    ###############################################################################
    ### training set
    all_train_cond_losses = []
    all_train_joint_losses = []
    all_train_cond_perpl = []
    
    for idx, batch in enumerate( tqdm(training_dl) ):
        batch_subCounts = batch[0]
        batch_insCounts = batch[1]
        batch_delCounts = batch[2]
        batch_transCounts = batch[3]
        
        # conditional pairHMM
        match_cond_emit = jnp.einsum( 'bij, ij -> b',
                                      batch_subCounts,
                                      cond_match_logprobs )
        
        ins_emit = jnp.einsum( 'bj, j -> b',
                                batch_insCounts,
                                ins_logprobs )
        
        transits = jnp.einsum( 'bmn, mn -> b',
                               batch_transCounts,
                               transits_logprobs )
        
        cond_scores = ( match_cond_emit +
                        ins_emit +
                        transits )
        
        # joint pairHMM
        match_joint_emit = jnp.einsum( 'bij, ij -> b',
                                       batch_subCounts,
                                       joint_match_logprobs )
        
        dels_omit = jnp.einsum( 'bi, i -> b',
                                batch_delCounts,
                                del_logprobs )
        
        joint_scores = ( match_joint_emit +
                         ins_emit +
                         dels_omit +
                         transits )
        
        # normalizing length
        if args.norm_loss_by == 'desc_len':
            length = ( batch_subCounts.sum(axis=(1,2)) +
                       batch_insCounts.sum(axis=1)
                       )
            
        elif args.norm_loss_by == 'align_len':
            length += batch_delCounts.sum(axis=1)
        
        # add one for <eos>
        length += 1
        
        train_df = training_dset.retrieve_sample_names(batch[-1])
        train_df['cond_raw_scores'] = cond_scores
        train_df['cond_normed_scores'] = cond_scores / length
        train_df['cond_perplexity'] = jnp.exp( -(cond_scores / length) )
        train_df['joint_raw_scores'] = joint_scores
        train_df['joint_normed_scores'] = joint_scores / length
        train_df.to_csv( (f'{args.training_wkdir}/'+
                          f'train-set_pt{idx}_FINAL-LOGLIKES.tsv'),
                          sep='\t'
                        )
        
        all_train_cond_losses.append( (cond_scores / length) )
        all_train_cond_perpl.append( jnp.exp( -(cond_scores / length) ) )
        all_train_joint_losses.append( (joint_scores / length) )
        
        
    ### test set
    all_test_cond_losses = []
    all_test_joint_losses = []
    all_test_cond_perpl = []
    
    for idx,batch in enumerate( tqdm(test_dl) ):
        batch_subCounts = batch[0]
        batch_insCounts = batch[1]
        batch_delCounts = batch[2]
        batch_transCounts = batch[3]
        
        # conditional pairHMM
        match_cond_emit = jnp.einsum( 'bij, ij -> b',
                                      batch_subCounts,
                                      cond_match_logprobs )
        
        ins_emit = jnp.einsum( 'bj, j -> b',
                                batch_insCounts,
                                ins_logprobs )
        
        transits = jnp.einsum( 'bmn, mn -> b',
                               batch_transCounts,
                               transits_logprobs )
        
        cond_scores = ( match_cond_emit +
                        ins_emit +
                        transits )
        
        # joint pairHMM
        match_joint_emit = jnp.einsum( 'bij, ij -> b',
                                       batch_subCounts,
                                       joint_match_logprobs )
        
        dels_omit = jnp.einsum( 'bi, i -> b',
                                batch_delCounts,
                                del_logprobs )
        
        joint_scores = ( match_joint_emit +
                         ins_emit +
                         dels_omit +
                         transits )
        
        # normalizing length
        if args.norm_loss_by == 'desc_len':
            length = ( batch_subCounts.sum(axis=(1,2)) +
                       batch_insCounts.sum(axis=1)
                       )
            
        elif args.norm_loss_by == 'align_len':
            length += batch_delCounts.sum(axis=1)
    
        # add one for <eos>
        length += 1
    
        test_df = training_dset.retrieve_sample_names(batch[-1])
        test_df['cond_raw_scores'] = cond_scores
        test_df['cond_normed_scores'] = cond_scores / length
        test_df['cond_perplexity'] = jnp.exp( -(cond_scores / length) )
        test_df['joint_raw_scores'] = joint_scores
        test_df['joint_normed_scores'] = joint_scores / length
        test_df.to_csv( (f'{args.training_wkdir}/'+
                          f'test-set_pt{idx}_FINAL-LOGLIKES.tsv'),
                          sep='\t'
                        )
        
        all_test_cond_losses.append( (cond_scores / length) )
        all_test_cond_perpl.append( jnp.exp( -(cond_scores / length) ) )
        all_test_joint_losses.append( (joint_scores / length) )
        
        
        
    ### write scores 
    to_write = [ jnp.concatenate( all_train_joint_losses ).mean(),
                 jnp.concatenate( all_train_cond_losses ).mean(),
                 jnp.concatenate( all_train_cond_perpl ).mean(),
                 jnp.exp( -jnp.concatenate( all_train_cond_losses ).mean() ),
                 
                 jnp.concatenate( all_test_joint_losses ).mean(),
                 jnp.concatenate( all_test_cond_losses ).mean(),
                 jnp.concatenate( all_test_cond_perpl ).mean(),
                 jnp.exp( -jnp.concatenate( all_test_cond_losses ).mean() )
                 ]
    
    names = ['train_ave_joint_loss_seqlen_normed',
             'train_ave_cond_loss_seqlen_normed',
             'train_perplexity',
             'train_ece',
             
             'test_ave_joint_loss_seqlen_normed',
             'test_ave_cond_loss_seqlen_normed',
             'test_perplexity',
             'test_ece']
    
    with open(f'{args.training_wkdir}/AVE-LOSSES.tsv', 'w') as g:
        g.write(f'RUN\t')
        g.write(f'{args.training_wkdir}\n')
        
        for i in range(len(names)):
            g.write(f'{names[i]}\t')
            g.write(f'{to_write[i]}\n')
    
        
    ### write scoring matrices
    def write_matrix(mat, filename):
        with open(f'{args.training_wkdir}/{filename}', 'wb') as g:
            jnp.save(g, mat)
    
    write_matrix(mat = transits_logprobs, 
                 filename = 'transits_logprobs.npy')
    
    write_matrix(mat = cond_match_logprobs, 
                 filename = 'cond_match_logprobs.npy')
    
    write_matrix(mat = joint_match_logprobs, 
                 filename = 'joint_match_logprobs.npy')
    
    write_matrix(mat = ins_logprobs, 
                 filename = 'ins_logprobs.npy')
        
    write_matrix(mat = del_logprobs, 
                 filename = 'del_logprobs.npy')
    
    
