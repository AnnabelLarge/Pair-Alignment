#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 18:19:15 2024

@author: annabel

ABOUT:
======
Mamba-based embedding module for ancestor and descendant sequences

"""
# general python
import logging
from typing import Any, Callable, Sequence, Union, Tuple
from dataclasses import field
import math
from functools import reduce
import einops

# flax n jax
import jax
import jax.numpy as jnp
import flax.linen as nn

# custom imports
from models.BaseClasses import SeqEmbBase


class MambaSeqEmb(SeqEmbBase):
    """
    Mamba-based embedding module for ancestor and descendant sequences
    
    
    init with:
    ==========
    initial_embed_module (callable): module for initial projection to hidden dim
    first_block_module (callable): first mamba block
    subsequent_block_module (callable): subsequent mamba blocks, if desired
    causal (bool): true if working with the descendant sequence; false otherwise
    config (dict): config to pass to each subsequent module
    name (str): "ANCESTOR EMBEDDER" or "DESCENDANT EMBEDDER"
    
    
    call arguments are:
    ===================
    datamat: matrix of sequences (B, L)
    training: controls dropout behavior
    sow_intermediates: if you want to capture intermediates for debugging
    
    
    outputs:
    ========
    datamat (altered matrix): position-specific encodings for all 
                             sequences (B, L, H)
    
    """
    initial_embed_module: callable
    first_block_module: callable
    subsequent_block_module: callable
    causal: bool
    config: dict
    name: str


    def setup(self):
        # first module projects (B,L) -> (B,L,H)
        name = f'{self.name} 0/initial embed'
        self.initial_embed = self.initial_embed_module(config = self.config,
                                                       causal = self.causal,
                                                       name = name)
        del name
        
        # second module does the first sequence embedding: (B,L,H) -> (B,L,H)
        name = f'{self.name} 1/Mamba Block 0'
        self.first_block = self.first_block_module(config = self.config,
                                                   name = name)
        del name
        
        # may have additional blocks: (B,L,H) -> (B,L,H)
        subsequent_blocks = []
        for i in range(self.config["num_blocks"]-1):
            layer_idx = i + 2
            block_idx = i + 1
            name = f'{self.name} {layer_idx}/Mamba Block {block_idx}'
            l = self.subsequent_block_module(config = self.config,
                                         name = name)
            del name
            subsequent_blocks.append(l)
        self.subsequent_blocks = subsequent_blocks
    
    
    def __call__(self, 
                 datamat, 
                 sow_intermediates: bool, 
                 training: bool):
        ### initial embedding: (B,L) -> (B,L,H)
        datamat, padding_mask = self.initial_embed(datamat)
        
        
        ### first convolution: (B, L, H) -> (B, L, H)
        datamat = self.first_block(datamat = datamat,
                                   padding_mask = padding_mask,
                                   sow_intermediates = sow_intermediates, 
                                   training = training)
        
        # optionally, sow the intermediate values (as long as this isn't
        # the last block)
        if sow_intermediates:
            label = f'{self.name} 1/Mamba Block 0/after block'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
        
        
        ### apply successive blocks; these start at layernum=2, Mamba Block 1
        # (B, L, H) -> (B, L, H)
        for i,block in enumerate(self.subsequent_blocks):
            layer_idx = i+2
            block_idx = i+1
            datamat = block(datamat = datamat,
                            padding_mask = padding_mask,
                            sow_intermediates = sow_intermediates, 
                            training = training)
            
            # optionally, sow the intermediate values (as long as this isn't
            # the last block)
            if sow_intermediates:
                label = (f'{self.name} {layer_idx}/'+
                         f'Mamba Block {block_idx}/'+
                         f'after block')
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
        
        # output is (B, L, H)
        return datamat
