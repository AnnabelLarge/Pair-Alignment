#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:12:00 2023

@author: annabel_large

Custom layers to throw into larger CNN sequence embedders


configs will have:
-------------------
- hidden_dim (int): length of the embedded vector

- kern_size_lst (list): list of kernel sizes 
  >> these are 1D convolutions, so each elem will be a one-element 
     list of integers: [int]

- base_alphabet_size (int = 23): <pad>, <bos>, <eos>, then all alphabet 
                              (20 for amino acids, 4 for DNA)

- dropout (float = 0.0): dropout rate


"""
# general python
from typing import Callable

# flaxy and jaxy
from flax import linen as nn
import jax
import jax.numpy as jnp

# custom
from models.model_utils.BaseClasses import ModuleBase


class ConvnetBlock(ModuleBase):
    """
    one Conv Block:
        
       |
       v
      in --------- 
       |         |
       v         |
      norm       |
       |         |
       v         |
      conv       |
       |         |
       v         |
      relu       |
       |         | 
       v         |
    dropout      |
       |         |
       v         |
       ---> + <---
            |
            v
           out
    
    then, padding positions in "out" are reset to zeros
       
    
    (B, L, H) -> (B, L, H)
    """
    config: dict
    kern_size: int
    causal: bool
    name: str
    
    def setup(self):
        # !!!  hard set
        # activations
        self.act_type = 'silu'
        self.act = nn.silu
        
        # normalization
        self.norm_type = 'layer'
        if self.causal:
            self.norm = nn.LayerNorm(reduction_axes=-1, feature_axes=-1)
        elif not self.causal:
            self.norm = nn.LayerNorm(reduction_axes= (-2,-1), 
                                     feature_axes=-1)
            
            
        ### unpack from config
        self.hidden_dim = self.config['hidden_dim']
        self.dropout = self.config.get('dropout', 0.0)
        
        
        ### other layers
        self.conv = nn.Conv(features = self.hidden_dim,
                           kernel_size = self.kern_size,
                           strides = 1,
                           padding =  'CAUSAL' if self.causal else 'SAME')
        self.dropout_layer = nn.Dropout(rate=self.dropout)
        
        
    def __call__(self, datamat, padding_mask, sow_intermediates:bool, training:bool):
        skip = datamat

        # record
        if sow_intermediates:
            self.sow_histograms_scalars(mat = datamat, 
                                        label = f'{self.name}/before conv block', 
                                        which=['scalars'])
        
        ### norm
        datamat = self.norm(datamat, 
                            mask = padding_mask)
        
        # manually mask again, because layernorm leaves NaNs
        datamat = jnp.where( padding_mask,
                            datamat,
                            0)
        
        # record
        if sow_intermediates:
            self.sow_histograms_scalars(mat = datamat, 
                                        label = f'{self.name}/after {self.norm_type}Norm', 
                                        which=['scalars'])
        
        
        ### convolution + relu
        datamat = self.conv(datamat)
        datamat = self.act(datamat)
        
        # record
        if sow_intermediates:
            self.sow_histograms_scalars(mat = datamat, 
                                        label = f'{self.name}/after {self.act_type}', 
                                        which=['scalars'])
        
        
        ### dropout
        datamat = self.dropout_layer(datamat, 
                                     deterministic = not training)
        
        
        ### residual add to before step 1 (the raw block input)
        datamat = datamat + skip

        # record
        if sow_intermediates:
            self.sow_histograms_scalars(mat = datamat, 
                                        label = f'{self.name}/after conv block', 
                                        which=['scalars'])
        
        ### zero out positions corresponding to padding tokens
        # mask is (B, L, H)
        datamat = jnp.multiply(datamat, padding_mask)
        
        return datamat

