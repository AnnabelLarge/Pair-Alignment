#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:12:00 2023

@author: annabel_large

ABOUT:
======
Sequence embedders with no params: one-hot encoding and a placeholder class

"""
from flax import linen as nn
import jax
from jax import numpy as jnp

from models.BaseClasses import SeqEmbBase


class Placeholder(SeqEmbBase):
    """
    Returns nothing
    """
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
    config: dict
    name: str
    
    
    def setup(self):
        self.base_alphabet_size = self.config.get('base_alphabet_size', 23)
    
    def __call__(self, 
                 datamat, 
                 sow_intermediates: bool=False, 
                 training: bool=False):
        # (B,L) -> (B, L, 23)
        return nn.one_hot(datamat, self.base_alphabet_size)


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
    config: dict
    name: str
    
    
    def setup(self):
        self.seq_padding_idx = self.config.get('seq_padding_idx', 0)
    
    def __call__(self, 
                 datamat, 
                 sow_intermediates: bool=False, 
                 training: bool=False):
        
        out_mat = jnp.where(datamat != self.seq_padding_idx,
                            True,
                            False)
        out_mat = out_mat[:,:,None]
        return out_mat
    
    
