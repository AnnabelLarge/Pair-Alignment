#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:12:00 2023

@author: annabel_large

ABOUT:
======
Sequence embedders with no params: one-hot encoding and a placeholder class

"""
from typing import Optional

from flax import linen as nn
import jax
from jax import numpy as jnp

from models.BaseClasses import SeqEmbBase


class Placeholder(SeqEmbBase):
    """
    Returns nothing
    """
    embedding_which: str
    config: dict
    name: str
    
    def __call__(self, *args, **kwargs):
        return None
    

class EmptyEmb(SeqEmbBase):
    """
    Returns an empty matrix
    
    
    init with:
    ==========
    config (dict): will be an empty dictionary
    name (str): like "ANCESTOR EMBEDDER" or "DESCENDANT EMBEDDER"
    
    
    call arguments are:
    ===================
    datamat: matrix of sequences (B, L)
    training: NOT USED
    sow_intermediates: NOT USED
    
    
    outputs:
    ========
    datamat (altered matrix): placeholder matrix: size (B, L, 0)
    
    """
    embedding_which: str
    config: dict
    name: str
    
    @nn.compact
    def __call__(self, 
                 datamat, 
                 sow_intermediates: bool=False, 
                 training: bool=False):
        return jnp.empty( (datamat.shape[0], datamat.shape[1], 0) )
     
     
class OneHotEmb(SeqEmbBase):
    """
    Only one-hot encoding
    
    
    init with:
    ==========
    config (dict): config to pass to each subsequent module
    name (str): "ANCESTOR EMBEDDER" or "DESCENDANT EMBEDDER"
    
    
    config will have:
    =================
    base_alphabet_size: 23 for proteins, 7 for DNA
    
    
    call arguments are:
    ===================
    datamat: matrix of sequences (B, L)
    training: NOT USED
    sow_intermediates: NOT USED
    
    
    outputs:
    ========
    datamat (altered matrix): one-hot encodings for all sequences 
                              (B, L, base_alphabet_size)
    """
    embedding_which: str
    config: dict
    name: str
    
    
    def setup(self):
        self.base_alphabet_size = self.config.get('base_alphabet_size', 23)
        self.seq_padding_idx = self.config.get('seq_padding_idx', 0)
    
    def __call__(self, 
                 datamat, 
                 *args,
                 **kwargs):
        """
        Arguments
        ----------
        datamat : ArrayLike, (B, L)
            > encoded with tokens from 1 to base_alphabet_size; padding is 
              assumed to be zero
        """
        padding_mask = (datamat != self.seq_padding_idx) #(B,L)
        
        # flax's one-hot will start one-hot encoding at token 0 (padding)
        #   run the one-hot encoding with an extra class, mask it, then remove 
        #   the empty leading column
        raw_one_hot = nn.one_hot(datamat, 
                                 n_classes = self.base_alphabet_size,
                                 axis=-1) #(B, L, base_alphabet_size)
        one_hot_masked = raw_one_hot * padding_mask  #(B, L, base_alphabet_size)
        one_hot_final = one_hot_masked[..., 1:] #(B, L, base_alphabet_size - 1)
        return one_hot_final
        


class MaskingEmb(SeqEmbBase):
    """
    Return (B, L, 1) matrix of indicators:
        - ones at real positions
        - zeros at padding sites
    (like a sequence mask)
    
    Use this for desc entropy unit test
    
    
    init with:
    ==========
    config (dict): config to pass to each subsequent module
    name (str): "ANCESTOR EMBEDDER" or "DESCENDANT EMBEDDER"
    
    
    config will have:
    =================
    seq_padding_idx: used to create indicator matrix
    
    
    call arguments are:
    ===================
    datamat: matrix of sequences (B, L)
    training: NOT USED
    sow_intermediates: NOT USED
    
    
    outputs:
    ========
    datamat (altered matrix): indicator for all sequences 
                              (B, L, 1)
    """
    embedding_which: str
    config: dict
    name: str
    
    
    def setup(self):
        self.seq_padding_idx = self.config.get('seq_padding_idx', 0)
    
    def __call__(self, 
                 datamat, 
                 sow_intermediates: bool=False, 
                 training: bool=False):
        
        out_mat = (datamat != self.seq_padding_idx)
        out_mat = out_mat[..., None]
        return out_mat
    
    
