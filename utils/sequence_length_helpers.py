#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 22:03:47 2025

@author: annabel
"""
import jax
import jax.numpy as jnp



def selective_squeeze(mat):
    """
    jnp.squeeze, but ignore batch dimension (dim0)
    """
    new_shape = tuple( [mat.shape[0]] + [s for s in mat.shape[1:] if s != 1] )
    return jnp.reshape(mat, new_shape)


def length_without_padding(seqs, 
                           padding_idx=0):
    return jnp.where(seqs != padding_idx,
                     True,
                     False).sum(axis=1).max()


def clip_by_bins(batch_seqs, 
                 chunk_length: int = 512, 
                 padding_idx = 0):
    """
    Clip excess paddings by binning according to chunk_length
    
    For example, if chunk_length is 3, then possible places to clip include:
        > up to length 3, if longest sequence is <= 3 in length
        > up to length 6, if longest sequence is > 3 and <= 6 in length
        > up to length 9, if longest sequence is > 6 and <= 9 in length
        > etc., until maximum length of batch_seqs
    
    overall, this helps jit-compile different versions of the functions
      for different max lengths (semi-dynamic batching)
    """
    # lengths
    #   if batch_seqs is alignments, then it will be max_anc_len - 1 already
    #   if batch_seqs is unaligned seqs, then this will be include bos
    max_len = batch_seqs.shape[1]
    max_len_without_padding = length_without_padding(seqs = batch_seqs, 
                                                     padding_idx = padding_idx)
    
    # determine the number of chunks
    def cond_fun(num_chunks):
        return chunk_length * num_chunks < max_len_without_padding

    def body_fun(num_chunks):
        return num_chunks + 1
    
    num_chunks = jax.lax.while_loop(cond_fun, body_fun, 1)
    length_with_all_chunks = chunk_length * num_chunks
    
    # if length_with_all_chunks is greater than max_len, 
    #   use max_len instead
    clip_to = jnp.where( length_with_all_chunks > max_len,
                         max_len,
                         length_with_all_chunks )
    return clip_to


def determine_seqlen_bin(batch,
                         chunk_length: int,
                         seq_padding_idx: int = 0):
    unaligned_seqs = batch[0]
    batch_max_seqlen = clip_by_bins(batch_seqs = unaligned_seqs, 
                                    chunk_length = chunk_length, 
                                    padding_idx = seq_padding_idx)
    return batch_max_seqlen


def determine_alignlen_bin(batch,
                           chunk_length: int,
                           seq_padding_idx: int = 0):
    # use the first sequence from aligned matrix for this (gapped ancestor for 
    #   neural_pairhmm, alignment-augmented descendant for feedforward); 
    #   exclude <bos> for the clip_by_bins function
    gapped_seq = batch[1][:, 1:, 0]
    
    # get length
    batch_max_alignlen = clip_by_bins(batch_seqs = gapped_seq, 
                                      chunk_length = chunk_length, 
                                      padding_idx = seq_padding_idx)
      
    # add one again, to re-include <bos>
    return batch_max_alignlen + 1

