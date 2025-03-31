#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:39:48 2023

@author: annabel

About:
======
Custom pytorch dataset object for giving pfam data to length-based models


outputs:
========
1. unaligned sequences: (batch_size, seq_max_len, 2)

2. aligned inputs: (batch_size, align_max_len, d)
   > for pairHMM models: d = 3
     >> dim2 = 0: gapped ancestor
     >> dim2 = 1: gapped descendant
     >> dim2 = 2: categorically-encoded alignment (<pad>, M, I, D, <bos>, <eos>)
  
    > for feedforward head: d = 4
    >> dim2 = 0: descendant, under alignment-augmented alphabet (ins + A)
    >> dim2 = 1: categorically-encoded alignment (<pad>, M, I, D, <bos>, <eos>)
    >> dim2 = 2: m-indices, precalculated from alignment
    >> dim2 = 3: n-indices, precalculated from alignment
  
   > for neural pairHMM models: d = 5
     >> dim2 = 0: gapped ancestor
     >> dim2 = 1: gapped descendant
     >> dim2 = 2: categorically-encoded alignment (<pad>, M, I, D, <bos>, <eos>)
     >> dim2 = 3: m-indices, precalculated from alignment
     >> dim2 = 4: n-indices, precalculated from alignment
         
3. times to use for pair: (num_timepoints, batch_size) or None

4. sample indices: (batch_size,)
    
use FullLenDset.retrieve_sample_names(sample_indices) to retrieve pairID,
  names of both sequences, and the pfam name


Data to be read:
=================
1. Numpy matrix of unaligned inputs: a tensor of size (num_pairs, max_len, 2),
   where dim2 corresponds to-
    - (dim2=0): ungapped ancestor sequence
    - (dim2=0): ungapped descendant sequence

2. Numpy matrix of aligned inputs: a tensor of size (num_pairs, max_len, 4),
   where dim2 corresponds to-
    - (dim2=0): aligned ancestor sequence
    - (dim2=1): aligned descendant sequence
    - (dim2=2): m indexes (indices for ancestor alignment)
    - (dim2=3): n indexes (indices for descendant alignment)

3. Pandas dataframe of metadata
   > note: alignment length in this dataframe does NOT include 
     sentinel tokens!

4 (optional) different times to associate with each pair
  > plain .tsv file with two columns; no header and no index
  > first column is pairID
  > second column is time

"""
# general python
import numpy as np
import pandas as pd

# jax stuff
import jax 
from jax import numpy as jnp
from jax.tree_util import tree_map

# pytorch stuff
import torch
from torch.utils.data import Dataset, DataLoader,default_collate


###############################################################################
### pytorch collator   ########################################################
###############################################################################
def default_collate_to_jax_array(mat):
    """
    kind of cumbersome, but conversion path is 
        tuple -> pytorch tensor -> numpy array -> jax array
    """
    pytorch_tensor = default_collate(mat)
    numpy_mat = pytorch_tensor.numpy()
    return jnp.array( numpy_mat )
    

def jax_collator(batch):
    ### unpack batch
    sample_unaligned_seqs, sample_aligned_mat, sample_time, sample_idx = zip(*batch)
    
    
    ### handle unaligned_seqs, aligned_mat, and idx with default_collate
    collated_unaligned_seqs = default_collate_to_jax_array( sample_unaligned_seqs )
    collated_aligned_mat = default_collate_to_jax_array( sample_aligned_mat )
    collated_idx = default_collate_to_jax_array( sample_idx )
    
    
    ### handle time, which could be none
    if (sample_time[0] is not None):
        collated_times = default_collate_to_jax_array( sample_time )
        collated_times = jnp.transpose(collated_times, (1,0) ) # want (T, B)
    
    elif (sample_time[0] is None):
        collated_times = None
        
        
    return (collated_unaligned_seqs, 
            collated_aligned_mat,
            collated_times,
            collated_idx)


###############################################################################
### some helpers   ############################################################
###############################################################################
def remove_excess_padding(seqs, 
                          padding_tok: int):
    """
    trim excess padding
    """
    global_max_len = np.where(seqs != padding_tok,
                              True,
                              False).sum(axis=1).max()
    clipped_seqs = seqs[:, :global_max_len, ...]
    return clipped_seqs, global_max_len


def add_padding_dim_1(mat,
                      padding_length: int,
                      padding_tok: int):
    """
    add padding to dim1 of matrix (usually length)
    """
    final_dtype = mat.dtype
    
    new_shape = (mat.shape[0], padding_length, mat.shape[2])
    padding = np.ones( new_shape, dtype =  final_dtype) * padding_tok
    padded_mat = np.concatenate( [mat, padding], axis=1)
    return padded_mat


def pad_to_length_divisible_by_chunk_len(aligned_mat,
                                         padding_tok: int,
                                         chunk_length: int = 512):
    """
    to make sure seqs is divisible by chunk_length, may need to 
      pad with extra tokens
    
    this is used when padding alignment_mats for use with loss functions
      that use jax.lax.scan
    """
    global_max_len = aligned_mat.shape[1]
    
    num_chunks = 1
    while (chunk_length * num_chunks) < global_max_len:
        num_chunks += 1
    
    # add 1 for <bos>
    padding_length = ((chunk_length * num_chunks) - global_max_len) + 1
    
    final_aligned_mat = add_padding_dim_1(mat = aligned_mat,
                                          padding_length = padding_length,
                                          padding_tok = padding_tok)
    
    return final_aligned_mat, padding_length
    



###############################################################################
### functions to load raw data   ##############################################
###############################################################################
def load_aligned_mats(data_dir, 
                      split, 
                      pred_model_type,
                      emission_alphabet_size ,
                      toss_alignments_longer_than = None, 
                      gap_idx = 43,
                      bos_idx = 1,
                      eos_idx = 2):
    ### load data
    with open(f'{data_dir}/{split}_aligned_mats.npy','rb') as f:
        mat = np.load(f)
       
        
    ### if alignments are longer than toss_alignments_longer_than, 
    ###   then toss the samples
    if toss_alignments_longer_than:
        eos_locs = np.argwhere(mat[:,:,0] == 2)
        idxes_to_keep = eos_locs[ eos_locs[:, 1] <= toss_alignments_longer_than ][:, 0]
        mat = mat[idxes_to_keep, :, :]
        
        if len(idxes_to_keep) == 0:
            raise RuntimeError(f"no samples to keep from {split}!")
            
    else:
        idxes_to_keep = None
    
    
    ### encode alignment state; bos and eos are shifted to the end, 
    ###   to match overleaf document
    # <pad> = 0
    # M = 1
    # I = 2
    # D = 3
    # <bos> = 4
    # <eos> = 5
    
    # find match pos; (B, L)
    gapped_seqs = mat[:,:,[0,1]]
    tmp = np.where( (gapped_seqs >= 3) & (gapped_seqs <= 22), 1, 0 ).sum(axis=2) 
    matches = np.where(tmp == 2, 1, 0)
    del tmp
    
    # find ins pos i.e. where ancestor is gap; (B,L)
    ins = np.where(gapped_seqs[:,:,0] == gap_idx, 2, 0)
    
    # find del pos i.e. where descendant is gap; (B,L)
    dels = np.where(gapped_seqs[:,:,1] == gap_idx, 3, 0)
    
    # bos, eos
    bos = jnp.where(mat == bos_idx, 4, 0)[:,:,0]
    eos = jnp.where(mat == eos_idx, 5, 0)[:,:,0]
    
    
    ### concatenate
    # categorical encoding
    alignment = bos + eos + matches + ins + dels
    
    # these use padding token of 0: (B, L, 3)
    zero_padded_mat = np.concatenate([gapped_seqs, alignment[...,None]], axis=-1)
    
    # these use -9 as padding token: (B, L, 2)
    neg_nine_padded_mat = mat[:,:,[2,3]]
    
    
    ### model-specific transformations
    # feedforward: add 20 to insert sites in descendant, toss ancestor
    if pred_model_type == 'feedforward':
        ins_pos = np.argwhere( zero_padded_mat[:,:,0] == gap_idx )
        zero_padded_mat[ins_pos[:,0], ins_pos[:,1], 1] += emission_alphabet_size
    
    # pairHMM: don't need alignment indices
    elif pred_model_type.startswith('pairhmm'):
        neg_nine_padded_mat = None
    
    
    # (neural pairHMM: keep everything as-is)
    return zero_padded_mat, neg_nine_padded_mat, idxes_to_keep


def load_unaligned(data_dir, 
                   split, 
                   idxes_to_keep=None):
    with open(f'{data_dir}/{split}_seqs_unaligned.npy','rb') as f:
        mat = jnp.load(f)
    
    if (idxes_to_keep is not None):
        mat = mat[idxes_to_keep, :, :]
    
    return mat


def load_metadata(data_dir, 
                  split, 
                  idxes_to_keep=None):
    cols_to_keep = ['pairID',
                    'ancestor',
                    'descendant',
                    'pfam', 
                    'anc_seq_len', 
                    'desc_seq_len', 
                    'alignment_len',
                    'num_matches',
                    'num_ins',
                    'num_del']
    
    df = pd.read_csv( f'./{data_dir}/{split}_metadata.tsv', 
                     sep='\t', 
                     index_col=0,
                     usecols=cols_to_keep) 
    
    if (idxes_to_keep is not None):
        df = df.iloc[idxes_to_keep]
    
    return df


###############################################################################
### functions to postprocess   ################################################
###############################################################################
def postprocess_aligned_mats(zero_padded_aligned_mats_lst,
                             neg_nine_padded_aligned_mats_lst,
                             divisible_by_chunk_length: bool,
                             chunk_length: int = 512,
                             seq_padding_idx: int = 0,
                             align_padding_idx: int = -9):
    """
    zero_padded_aligned_mats_lst: list of matrices to concatenate, which use 
        zero as the padding token
    
    neg_nine_padded_aligned_mats_lst: list of matrices to concatenate, which  
        use -9 as the padding token
    
    divisible_by_chunk_length [BOOL]: True if using scanned version of
        loss function; False otherwise
    
    chunk_length [INT=512]: used for lengths in scan and determining number of 
        jit-compiled functions; if not provided, use 512
    
    seq_padding_idx, align_padding_idx: what the padding tokens are
    """
    # concat
    zero_padded_aligned_mats = np.concatenate(zero_padded_aligned_mats_lst,
                                              axis=0)
    if neg_nine_padded_aligned_mats_lst is not None:
        neg_nine_padded_aligned_mats = np.concatenate(neg_nine_padded_aligned_mats_lst,
                                                      axis=0)
    else:
        neg_nine_padded_aligned_mats = None
    
    del zero_padded_aligned_mats_lst, neg_nine_padded_aligned_mats_lst
    
    
    ### first half; adjust gapped ancestor and descendant seqs
    # remove excess padding
    out = remove_excess_padding(seqs = zero_padded_aligned_mats, 
                                padding_tok = 0)
    final_mat, align_max_len_without_padding = out
    del out
    
    # if you want this to be divisible by chunk_length, may need to 
    #   add more padding tokens (0)
    if divisible_by_chunk_length:
        out = pad_to_length_divisible_by_chunk_len(aligned_mat = final_mat,
                                                   padding_tok = 0,
                                                   chunk_length = chunk_length)
        final_mat, extra_padding_to_add = out
        del out
        
        
    ### second half; adjust precomputed alignment indices
    if neg_nine_padded_aligned_mats is not None:
        # remove excess padding; already calculated the length for this, so
        #   just reuse that
        second_half = neg_nine_padded_aligned_mats[:, :align_max_len_without_padding, :]
        
        # if you want this to be divisible by chunk_length, may need to 
        #   add more padding tokens (-9); again, already calculated length for this
        if divisible_by_chunk_length:
            second_half = add_padding_dim_1(mat = second_half,
                                            padding_length = extra_padding_to_add,
                                            padding_tok = -9)
        
        final_mat = np.concatenate([final_mat, second_half], axis=-1)
    
    return final_mat


def postprocess_unaligned_seqs(in_lst,
                               seq_padding_idx: int = 0):
    unaligned_seqs = np.concatenate(in_lst, axis=0)
    unaligned_seqs, _ = remove_excess_padding(seqs = unaligned_seqs, 
                                              padding_tok = seq_padding_idx)
    return unaligned_seqs
    
    
def postprocess_metadata(in_lst):
    metadata_df = pd.concat(in_lst)
    metadata_df = metadata_df.reset_index(drop=True)
    
    return metadata_df




###############################################################################
### dataloader   ##############################################################
###############################################################################
class FullLenDset(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 split_prefixes: list, 
                 pred_model_type: str,
                 use_scan_fns: bool,
                 emission_alphabet_size: int,
                 times_from_array: np.array = None,
                 single_time_from_file: bool = False,
                 chunk_length: int = 512,
                 toss_alignments_longer_than = None,
                 seq_padding_idx: int = 0,
                 align_padding_idx: int = -9,
                 gap_idx: int = 43):
        """
        data locations, format:
        ------------------------
        data_dir [STR]: where to find data
        
        split_prefixes [LIST of STR]: prefixes of splits to use
        
        
        loss fn to use:
        ---------------
        pred_model_type: different tensors needed for feedforward head vs 
            tkf92 neural head
        
        use_scan_fns [BOOL]: which loss functions you'll use- the ones built
            with jax.lax.scan, or the ones without
        
        
        managing time:
        --------------
        times_from_array [arr or None]: if not None, then provide a time array
            to use for all samples; array should be (T,)
        
        single_time_from_file [BOOL = False]: if true, then read from file;  
            provides one time for each sample; array should be (B,)
        
        (if times_from_array = None and single_time_from_file = False, then
             don't use time at all)
        
        
        managing alignment length:
        ---------------------------
        chunk_length: how to chunk along the length of the sequence; used for
            a couple things-
                1.) jit-compiling different versions of the train and test fns
                2.) calculating loss in tkf92, in chunks along sequence
            > if None, use maximum length without padding (i.e. do 
              only one jax.lax.scan iteration)
        
        toss_alignments_longer_than: if you want to enforce a global max length and TOSS
            ANY SAMPLES LONGER THAN THIS, provide a value here
        
        
        special tokens, alphabet sizes 
        (set defaults for my given data):
        ----------------------------------
        > seq_padding_idx: padding token, usually zero
        > align_padding_idx: padding for precomputed alignments; I use -9
        > gap_idx: I use 43
        > emission_alphabet_size: 20 for proteins, 4 for DNA
        
        
        final attributes are:
        ----------------------
          self.unaligned_seqs
          self.aligned_mat
          self.names_df
          self.times
          self.emit_counts
          self.global_seq_max_length
          self.global_align_max_length
            > global_align_max_length is divisible by chunk_length 
              if using scan version of functions
        """
        ###############
        ### read data #
        ###############
        # always read
        zero_padded_aligned_mats_lst = []
        neg_nine_padded_aligned_mats_lst = []
        unaligned_seqs_lst = []
        metadata_lst = []
        self.emit_counts = np.zeros( (emission_alphabet_size,) )
            
        if emission_alphabet_size == 20:
            counts_suffix = 'AAcounts'
        
        elif emission_alphabet_size == 4:
            counts_suffix = 'NuclCounts'
            
        # optionally read
        if single_time_from_file:
            times_lst = []
        
        for split in split_prefixes:
            ### aligned inputs: alignment, and precalculated (m,n) indices
            ###   remove any samples with alignments greater than toss_alignments_longer_than
            out = load_aligned_mats(data_dir = data_dir, 
                                    split = split, 
                                    toss_alignments_longer_than = toss_alignments_longer_than, 
                                    pred_model_type = pred_model_type,
                                    gap_idx = gap_idx,
                                    emission_alphabet_size = emission_alphabet_size)
            zero_padded_mat, neg_nine_padded_mat, idxes_to_keep = out
            del out
            
            zero_padded_aligned_mats_lst.append( zero_padded_mat )
            if neg_nine_padded_mat is not None:
                neg_nine_padded_aligned_mats_lst.append( neg_nine_padded_mat )
                
            del zero_padded_mat, neg_nine_padded_mat
            
            
            ### unaligned inputs (the sequences themselves)
            ###   remove any samples with alignments greater than toss_alignments_longer_than
            unaligned_seqs = load_unaligned(data_dir = data_dir, 
                                            split = split,
                                            idxes_to_keep = idxes_to_keep)
            
            unaligned_seqs_lst.append(unaligned_seqs)
            del unaligned_seqs
            
            
            ### metadata
            meta_df = load_metadata(data_dir = data_dir, 
                                    split = split,
                                    idxes_to_keep = idxes_to_keep)
            metadata_lst.append(meta_df)
            del meta_df
            
            
            ### counts of amino acids
            ###   TODO: this could still include the counts of emissions from
            ###   tossed samples... fix this later
            with open(f'{data_dir}/{split}_{counts_suffix}.npy','rb') as f:
                self.emit_counts += np.load(f)
            
            
            ### (optional) time; assume time is in same order as samples in
            ###   metadata
            if single_time_from_file:
                times = pd.read_csv(f'{data_dir}/{split}_pair-times.tsv', 
                                    sep='\t',
                                    header=None,
                                    names=['pairID','time'],
                                    index_col=None)

                if (idxes_to_keep is not None):
                    times = times.iloc[idxes_to_keep]

                times_lst += times['time'].tolist()
                del times
                
                
        #################
        ### postprocess #
        #################
        # matrix of alignment info
        lst2 = None if len(neg_nine_padded_aligned_mats_lst) == 0 else neg_nine_padded_aligned_mats_lst
        self.aligned_mat = postprocess_aligned_mats(zero_padded_aligned_mats_lst = zero_padded_aligned_mats_lst,
                                                    neg_nine_padded_aligned_mats_lst = lst2,
                                                    divisible_by_chunk_length = use_scan_fns,
                                                    chunk_length = chunk_length,
                                                    seq_padding_idx = seq_padding_idx,
                                                    align_padding_idx = align_padding_idx)
        self.global_align_max_length = self.aligned_mat.shape[1]
        del zero_padded_aligned_mats_lst, neg_nine_padded_aligned_mats_lst, lst2
        
        # ungapped seqs
        self.unaligned_seqs = postprocess_unaligned_seqs(in_lst = unaligned_seqs_lst,
                                                         seq_padding_idx = seq_padding_idx)
        self.global_seq_max_length = self.unaligned_seqs.shape[1]
        del unaligned_seqs_lst
        
        # metadata
        self.names_df = postprocess_metadata(in_lst = metadata_lst)
        del metadata_lst
        
        # (optional) time
        if single_time_from_file:
            self.times = np.array(times_lst) #(B,)
            self.func_to_retrieve_time = self.return_single_time_per_samp
            del times_lst
        
        elif (not single_time_from_file) and (times_from_array is not None):
            self.times = times_from_array #(T,)
            self.func_to_retrieve_time = self.return_time_array
        
        else:
            self.func_to_retrieve_time = self.return_no_time


        ######################################################
        ### check all matrices encode same number of samples #
        ######################################################
        nsamps_from_align = self.aligned_mat.shape[0]
        nsamps_from_unalign = self.unaligned_seqs.shape[0]
        nsamps_from_meta = len(self.names_df)

        seen_nsamps = set( [nsamps_from_align, nsamps_from_unalign, nsamps_from_meta] )
        assert len(seen_nsamps) == 1

        if single_time_from_file:
            nsamps_from_times = self.times.shape[0]
            assert seen_nsamps.pop() == nsamps_from_times
        

    def __len__(self):
        return self.aligned_mat.shape[0]

    def __getitem__(self, idx):
        sample_unaligned_seqs = self.unaligned_seqs[idx, :, :]
        sample_aligned_mat = self.aligned_mat[idx, :, :]
        sample_time = self.func_to_retrieve_time(idx)
        sample_idx = idx
        return (sample_unaligned_seqs, 
                sample_aligned_mat, 
                sample_time, 
                sample_idx)
    
    def retrieve_sample_names(self, idxes):
        # used the list of sample indices to query the original names_df
        return self.names_df.iloc[idxes]
    
    def write_split_indices(self, idxes):
        # this is a method for lazy loading, but not early loading
        raise NotImplementedError
    
    
    ################################
    ### functions to manage time   #
    ################################
    def return_single_time_per_samp(self, idx):
        # self.times is (B,)
        return self.times[idx, None] #integer value
    
    def return_time_array(self, idx=None):
        # self.times is (T,)
        return self.times #array of size (T,)
    
    def return_no_time(self, idx=None):
        return None
    
    def retrieve_num_timepoints(self, times_from):
        if times_from in ['geometric', 't_array_from_file']:
            return self.times.shape[0]
        
        elif times_from == 'one_time_per_sample_from_file':
            return 1
        
        elif (times_from is None):
            return 0
    
    