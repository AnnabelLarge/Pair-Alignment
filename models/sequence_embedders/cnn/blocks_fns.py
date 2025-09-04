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

- in_alph_size (int = 23): <pad>, <bos>, <eos>, then all alphabet 
                              (20 for amino acids, 4 for DNA)

- dropout (float = 0.0): dropout rate


"""
from typing import Callable

import flax
from flax import linen as nn
import jax
import jax.numpy as jnp

from models.BaseClasses import ModuleBase


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
        self.hidden_dim = self.config['hidden_dim']
        self.dropout = self.config.get('dropout', 0.0)
        
        
        ### set up layers of the CNN block
        # normalization
        self.norm = nn.LayerNorm(reduction_axes=-1, feature_axes=-1)
        self.norm_type = 'Instance'
        
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
                 datamat: jnp.array, #(B, L, H)
                 padding_mask: jnp.array, #(B, L) 
                 sow_intermediates:bool, 
                 training:bool):
        # mask for padding tokens; broadcast to the datamat input (which will
        # not change in shape through this whole operation)
        mask = jnp.broadcast_to( padding_mask[...,None], datamat.shape ) #(B, L, H)
        datamat = jnp.multiply(datamat, mask) #(B, L, H)
        
        # skip connection
        skip = datamat #(B, L, H)

        if sow_intermediates:
            self.sow_histograms_scalars(mat = datamat, 
                                        label = f'{self.name}/before conv block', 
                                        which=['scalars'])
        
        ### 1.) norm, mask padding tokens
        datamat = self.norm(datamat)  #(B, L, H)
        datamat = jnp.multiply(datamat, mask) #(B, L, H)
        
        if sow_intermediates:
            self.sow_histograms_scalars(mat = datamat, 
                                        label = f'{self.name}/after {self.norm_type}Norm', 
                                        which=['scalars'])
        
        
        ### 2.) convolution, mask padding tokens
        datamat = self.conv(datamat) #(B, L, H)
        datamat = jnp.multiply(datamat, mask) #(B, L, H)
        
        if sow_intermediates:
            self.sow_histograms_scalars(mat = datamat, 
                                        label = f'{self.name}/after conv', 
                                        which=['scalars'])
        
        
        ### 3.) activation (silu)
        datamat = self.act(datamat) #(B, L, H)
        
        if sow_intermediates:
            self.sow_histograms_scalars(mat = datamat, 
                                        label = f'{self.name}/after {self.act_type}', 
                                        which=['scalars'])
        
        
        ### 4.) dropout
        datamat = self.dropout_layer(datamat, 
                                     deterministic = not training) #(B, L, H)
        
        if (sow_intermediates) and (self.dropout > 0):
            self.sow_histograms_scalars(mat = datamat, 
                                        label = f'{self.name}/after dropout', 
                                        which=['scalars'])
        
        
        ### 5.) residual add to the block input; again, mask padding tokens
        datamat = datamat + skip
        datamat = jnp.multiply(datamat, mask) #(B, L, H)
        
        return datamat


class FakeConvnetBlock(ModuleBase):
    """
    make conv block that behaves like a deterministic one-hot encoder
    """
    config: dict
    kern_size: int
    causal: bool
    name: str
    
    def setup(self):
        ### unpack from config
        self.hidden_dim = self.config['hidden_dim']
        
        
        ### set up layers of the CNN block
        # normalization
        self.norm = lambda x: x
        self.norm_type = None
        
        # convolution
        self.conv = nn.Conv(features = self.hidden_dim,
                            kernel_size = self.kern_size,
                            strides = 1,
                            padding = 'CAUSAL' if self.causal else 'SAME',
                            use_bias = False)
        
        self.custom_kernel = self.kernel_fn( self.kern_size, 
                                             self.hidden_dim, 
                                             self.hidden_dim, 
                                             causal=self.causal ) #(self.kernel_size, H, H)
        
        # activation
        self.act = lambda x: x
        self.act_type = None
        
        # dropout
        self.dropout_layer = nn.Dropout(rate=0.0)
        
        
    def __call__(self, 
                 datamat: jnp.array, #(B, L, H)
                 padding_mask: jnp.array, #(B, L) 
                 *args,
                 **kwargs):
        # mask for padding tokens; broadcast to the datamat input (which will
        # not change in shape through this whole operation)
        mask = jnp.broadcast_to( padding_mask[...,None], datamat.shape ) #(B, L, H)
        datamat = jnp.multiply(datamat, mask) #(B, L, H)
        
        # skip connection
        skip = datamat #(B, L, H)

        # fake norm
        datamat = self.norm(datamat)  #(B, L, H)
        datamat = jnp.multiply(datamat, mask) #(B, L, H)
        
        
        ### convolution with deterministic kernel
        # custom kernel
        params = self.conv.variables.get("params", {})
        if len(params) > 0:
            params = params.unfreeze()
            
        params["kernel"] = self.custom_kernel  #(self.kernel_size, H, H)
        frozen_params = flax.core.freeze(params)
    
        # Apply conv with overridden kernel
        datamat = self.conv.apply({"params": frozen_params}, datamat)
        datamat = jnp.multiply(datamat, mask) #(B, L, H)
        
        
        ### rest of the fake network
        # fake activation, dropout with zero rate
        datamat = self.act(datamat) #(B, L, H)
        datamat = self.dropout_layer(datamat, 
                                     deterministic = True) #(B, L, H)
        
        # residual add
        datamat = datamat + skip
        datamat = jnp.multiply(datamat, mask) #(B, L, H)
        
        # 1s where values exist; 0s otherwise
        datamat = jnp.where( datamat > 0,
                             1.0,
                             0.0 ) #(B, L, H)
        
        return datamat
    
    def kernel_fn(self,
                  kernel_size: int, 
                  in_dim: int, 
                  out_dim: int, 
                  causal: bool):
        """
        mimic the one-hot encoding function
        """
        focal_point = kernel_size - 1 if causal else kernel_size // 2
        W = jnp.zeros((kernel_size, in_dim, out_dim))

        n = min(in_dim, out_dim)  
        W = W.at[focal_point, jnp.arange(n), jnp.arange(n)].set(1.0)

        return W

