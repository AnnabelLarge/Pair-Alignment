#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 18:13:13 2024

@author: annabel


"""
import logging
from typing import Any, Callable, Sequence, Union, Tuple
from dataclasses import field
import math
from functools import reduce
import einops

import jax
import jax.numpy as jnp
import flax.linen as nn

from models.BaseClasses import ModuleBase
from models.sequence_embedders.mamba.model_parts import (UnidirecMambaModule, 
                                                         BidirecMambaModule)


###############################################################################
### Mamba in one direction   ##################################################
###############################################################################
class UnidirectMamba(ModuleBase):
    """
    Full Mamba residual block (using UnidirecMambaModule)
    Using pre-norm structure, similar to a pre-norm transformer
    
     |
     v
    in --------- 
     |         |
     v         |
    norm       |
     |         |
     v         |
    Mamba      |
    module     |
     |         |
     v         |
  dropout      |
     |         |
     ---> + <---
          |
          v
         out
    
    make self.init_mamba_layer a method that can be over-written later
    """
    config: dict
    name: str
    
    def setup(self):
        # !!! defaults
        self.norm_type = 'rms'
        self.norm = nn.RMSNorm(reduction_axes=-1, 
                               feature_axes=-1)
            
        ### unpack config
        self.dropout = self.config.get("dropout", 0.0)
        
        
        ### declare layers to use
        self.init_mamba_layer()
        self.dropout_layer = nn.Dropout(rate = self.dropout)
    
    
    def init_mamba_layer(self):
        self.mamba = UnidirecMambaModule(config = self.config,
                                         name = self.name)
    
    
    def __call__(self, 
                 datamat, #(B, L, H)
                 padding_mask,  #(B, L)
                 sow_intermediates:bool, 
                 training:bool):
        B = datamat.shape[0]
        L = datamat.shape[1]
        
        # mask padding tokens of input
        padding_mask = jnp.broadcast_to( padding_mask[...,None], datamat.shape ) #(B, L, H)
        datamat = jnp.multiply(datamat, padding_mask) #(B, L, H)
        
        # skip connection
        skip = datamat
        
        
        ### norm
        datamat = self.norm(datamat) #(B, L, H)
        datamat = jnp.multiply(datamat, padding_mask) #(B, L, H)
        
        # record
        if sow_intermediates:
            label = f'{self.name}/after initial {self.norm_type}Norm'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
        
        
        ### Mamba
        datamat = self.mamba(datamat=datamat, 
                             padding_mat=padding_mask, 
                             sow_intermediates=sow_intermediates)
                       
        
        ### Optional dropout
        datamat = self.dropout_layer(datamat,
                               deterministic = not training)
        
        
        ### residual add and return
        datamat = datamat + skip
        
        return datamat
    
    
class UnidirectMambaWithFeedforward(ModuleBase):
    """
    Mamba residual block (using UnidirecMambaModule) and feedforward
    Using pre-norm structure, similar to a pre-norm transformer
    
             |
             v
             in 
             |
             v
     UnidirectResidualMambaLayer
             |
             v
        after_mamba------
              |         |
              v         |
             norm       |
              |         |
              v         |
            dense       |
              |         | 
              v         |
          activation    |
              |         | 
              v         |
            dense       |
              |         | 
              v         |
           dropout      |
              |         |
              v         |
              ---> + <---
                   |
                   v
                  out
    
    note: original Mamba doesn't use any dropout... leave as zero for now
    
    make self.init_mamba_layer a method that can be over-written later
    """
    config: dict
    name: str
    
    def setup(self):
        # !!! defaults
        # normalization
        self.norm_type = 'rms'
        self.norm = nn.RMSNorm(reduction_axes=-1, 
                               feature_axes=-1)
        # activations
        self.act_type = 'silu'
        self.act = nn.silu
        self.kernel_init = nn.initializers.lecun_normal()

        
        ### unpack config
        self.hidden_dim = self.config['hidden_dim']
        self.dropout = self.config.get("dropout", 0.0)
        
        
        ### declare layers to use
        # mamba
        self.init_mamba_layer()
        
        # dropout
        self.dropout_layer = nn.Dropout(rate = self.dropout)
        
        # dense layers (in final feedforward)
        self.first_feedforward_dense = nn.Dense(self.hidden_dim,
                                                kernel_init = self.kernel_init,
                                                use_bias=True)
        
        self.second_feedforward_dense = nn.Dense(self.hidden_dim,
                                                 kernel_init = self.kernel_init,
                                                 use_bias=True)
    
    
    def init_mamba_layer(self):
        self.mamba = UnidirectResidualMambaLayer(config = self.config,
                                                 name = self.name)
    
    
    def __call__(self, 
                 datamat, 
                 padding_mask, 
                 sow_intermediates:bool, 
                 training:bool):
        B = datamat.shape[0]
        L = datamat.shape[1]
        
        ################
        ### mamba part #
        ################
        datamat = self.mamba(datamat = datamat, 
                             padding_mask = padding_mask, 
                             sow_intermediates = sow_intermediates, 
                             training = training)
        
        if sow_intermediates:
            self.sow_histograms_scalars(mat = datamat, 
                                        label = f'{self.name}/after mamba half', 
                                        which=['scalars'])
        
        
        ######################
        ### feedforward part #
        ######################
        skip = datamat
        
        
        ### Norm
        # mask padding tokens of input
        padding_mask = jnp.broadcast_to( padding_mask[...,None], datamat.shape ) #(B, L, H)
        datamat = self.norm(datamat) #(B, L, H)
        datamat = jnp.multiply(datamat, padding_mask) #(B, L, H)
        
        if sow_intermediates:
            label = f'{self.name}/after second {self.norm_type}Norm'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
        
        
        ### feedforward: dense -> relu -> dense
        datamat = self.first_feedforward_dense(datamat) #(B, L, H)
        
        datamat = self.act(datamat) #(B, L, H)
        datamat = jnp.multiply(datamat, padding_mask) #(B, L, H)
        
        if sow_intermediates:
            label = f'{self.name}/in feedforward, after {self.act_type}'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
        
        datamat = self.second_feedforward_dense(datamat) #(B, L, H)
        
        
        ### dropout and residual add
        datamat = self.dropout_layer(datamat,
                                     deterministic = not training) #(B, L, H)
        
        datamat = skip + datamat #(B, L, H)
        datamat = jnp.multiply(datamat, padding_mask) #(B, L, H)
        
        if sow_intermediates:
            label = f'{self.name}/after feedforward half'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars']) 
            del label
        
        return datamat
    
    
    
    
###############################################################################
### Mamba in both directions   ################################################
###############################################################################
class BidirectMamba(UnidirectMamba):
    """
    Full Mamba residual block (using BidirecMambaModule)
    
    practically, just replace self.init_mamba_layer
    """
    def init_mamba_layer(self):
        self.mamba = BidirecMambaModule(config = self.config,
                                         name = self.name) 


class BidirectMambaWithFeedforward(UnidirectMambaWithFeedforward):
    """
    Mamba residual block (using BidirecMambaModule) and feedforward
    
    practically, just replace self.init_mamba_layer
    """
    def init_mamba_layer(self):
        self.mamba = BidirectResidualMambaLayer(config = self.config,
                                                name = self.name) 


    