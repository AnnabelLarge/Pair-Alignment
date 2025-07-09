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
from typing import Callable

from flax import linen as nn
import jax
import jax.numpy as jnp

from models.BaseClasses import ModuleBase
from models.neural_utils.custom_normalization_layers import ( LayerNormOverLastTwoDims )


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
      silu       |
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
        ### unpack from config
        # ignore config['kern_size_lst'] here
        self.hidden_dim = self.config['hidden_dim']
        self.dropout = self.config.get('dropout', 0.0)
        
        
        ### set up layers of the CNN block
        # normalization
        if self.causal:
            self.norm = nn.LayerNorm(reduction_axes=-1, feature_axes=-1)
            self.norm_type = 'Instance'
            
        elif not self.causal:
            # self.norm = nn.LayerNorm(reduction_axes= (-2,-1), feature_axes=-1)
            self.norm = LayerNormOverLastTwoDims()
            self.norm_type = 'Layer'
        
        # convolution
        self.conv = nn.Conv(features = self.hidden_dim,
                           kernel_size = self.kern_size,
                           strides = 1,
                           padding =  'CAUSAL' if self.causal else 'SAME')
        
        # activation
        self.act_type = 'silu'
        self.act = nn.silu
        
        # dropout
        self.dropout_layer = nn.Dropout(rate=self.dropout)
        
        
    def __call__(self, 
                 datamat: jnp.array, #(B, L, H_in)
                 padding_mask: jnp.array, #(B, L) 
                 sow_intermediates:bool, 
                 training:bool):
        # mask for padding tokens; broadcast to the datamat input (which will
        # not change in shape through this whole operation)
        mask = jnp.broadcast_to( padding_mask[...,None], datamat.shape ) #(B, L, H_in)
        datamat = jnp.multiply(datamat, mask) #(B, L, H_in)
        
        # skip connection
        skip = datamat #(B, L, H_in)

        if sow_intermediates:
            self.sow_histograms_scalars(mat = datamat, 
                                        label = f'{self.name}/before conv block', 
                                        which=['scalars'])
        
        ### 1.) norm, mask padding tokens
        mask_for_norm = padding_mask if not self.causal else None
        datamat = self.norm(datamat, mask=mask_for_norm)  #(B, L, H_in)
        datamat = jnp.multiply(datamat, mask) #(B, L, H_in)
        
        if sow_intermediates:
            self.sow_histograms_scalars(mat = datamat, 
                                        label = f'{self.name}/after {self.norm_type}Norm', 
                                        which=['scalars'])
        
        
        ### 2.) convolution, mask padding tokens
        datamat = self.conv(datamat) #(B, L, H_in)
        datamat = jnp.multiply(datamat, mask) #(B, L, H_in)
        
        if sow_intermediates:
            self.sow_histograms_scalars(mat = datamat, 
                                        label = f'{self.name}/after conv', 
                                        which=['scalars'])
        
        
        ### 3.) activation (silu)
        datamat = self.act(datamat) #(B, L, H_in)
        
        if sow_intermediates:
            self.sow_histograms_scalars(mat = datamat, 
                                        label = f'{self.name}/after {self.act_type}', 
                                        which=['scalars'])
        
        
        ### 4.) dropout
        datamat = self.dropout_layer(datamat, 
                                     deterministic = not training) #(B, L, H_in)
        
        if sow_intermediates and (self.dropout > 0):
            self.sow_histograms_scalars(mat = datamat, 
                                        label = f'{self.name}/after dropout', 
                                        which=['scalars'])
        
        
        ### 5.) residual add to the block input; again, mask padding tokens
        datamat = datamat + skip
        datamat = jnp.multiply(datamat, mask) #(B, L, H_in)

        return datamat

