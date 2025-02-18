#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:22:44 2024

@author: annabel_large
    

modules to project (B, L) -> (B, L, H), before sending to main architecture

"""
# general python
from typing import Callable

# flaxy and jaxy
from flax import linen as nn
import jax
import jax.numpy as jnp

# custom
from models.model_utils.BaseClasses import ModuleBase


class PlaceholderEmbedding(nn.Module):
    """
    for debugging; take in a (B,L) matrix and repeat entries
      to (B, L, hidden_dim)
    """
    config: dict
    name: str
    causal: bool
    
    @nn.compact
    def __call__(self, datamat, training: bool = None):
        ### unpack
        hidden_dim = self.config['hidden_dim']
        padding_idx = self.config.get('seq_padding_idx', 0)
        
        ### run
        padding_mask_template = jnp.where(datamat == seq_padding_idx, 
                                          False, 
                                          True)[:,:,None]
        new_shape = (padding_mask_template.shape[0],
                     padding_mask_template.shape[1],
                     hidden_dim)
        padding_mask = jnp.broadcast_to(padding_mask_template, new_shape)
        del new_shape
        
        datamat = datamat[:,:,None]
        new_shape = (datamat.shape[0],
                     datamat.shape[1],
                     hidden_dim)
        datamat = jnp.broadcast_to(datamat, new_shape)
        del new_shape
        
        return (datamat, padding_mask)
        

class EmbeddingWithPadding(ModuleBase):
    """
    replicated torch's embedding function, with padding_idx option 
    
    doesn't really matter if it's causal or not; keeping here to preserve trace
    
    configs have (at minimum):
    --------------------------
    hidden_dim (int): length of the embedded vector
    padding_idx (int = 0): padding token
    base_alphabet_size (int = 23): <pad>, <bos>, <eos>, then all alphabet 
                                  (20 for amino acids, 4 for DNA)
                              
    """
    config: dict
    name: str
    causal: bool
    
    def setup(self):
        ### unpack config
        self.features = self.config['hidden_dim']
        
        # these have default values
        self.vocab_size = self.config.get('base_alphabet_size', 23)
        self.padding_idx = self.config.get('seq_padding_idx', 0)
        
        
        ### layers to use
        self.initial_embedding = nn.Embed(num_embeddings = self.vocab_size, 
                                          features = self.features)
        
        
    def __call__(self, datamat, training: bool = None):
        padding_mask_template = jnp.where(datamat == self.padding_idx, False, True)[:,:,None]
        
        # (B,L) -> (B, L, H)
        datamat = self.initial_embedding(datamat)
        
        # mask positions with padding tokens
        # mask is also (B, L, H)
        new_shape = (padding_mask_template.shape[0],
                     padding_mask_template.shape[1],
                     self.features)
        padding_mask = jnp.broadcast_to(padding_mask_template, new_shape)
        del new_shape
        
        datamat = jnp.multiply(datamat, padding_mask)
        
        # return masked matrix, and the masking used
        return (datamat, padding_mask)



class TAPEEmbedding(ModuleBase):
    """
    replicated the embedding function used in the TAPE transformer, with the 
    caveat that I also add the padding_idx option 
    
    doesn't really matter if it's causal or not; keeping here to preserve trace
    
    
    configs have (at minimum):
    --------------------------
    hidden_dim (int): length of the embedded vector
    padding_idx (int = 0): padding token
    dropout (float = 0.0): dropout rate
    max_len (int = 3000): maximum protein length
    base_alphabet_size (int = 23): <pad>, <bos>, <eos>, then all alphabet 
                                  (20 for amino acids, 4 for DNA)
                              
    """
    config: dict
    name: str
    causal: bool
    
    def setup(self):
        ### unpack config
        self.features = self.config['hidden_dim']
        
        # these have default values
        self.vocab_size = self.config.get('base_alphabet_size', 23)
        self.padding_idx = self.config.get('seq_padding_idx', 0)
        self.max_len = self.config.get('max_len', 3000)
        self.dropout = self.config.get('dropout', 0.0)
        
        
        ### layers to use
        self.seq_initial_embedding = nn.Embed(num_embeddings = self.vocab_size, 
                                              features = self.features)
        self.pos_initial_embedding = nn.Embed(num_embeddings = self.max_len, 
                                              features = self.features)
        self.final_layernorm =  nn.LayerNorm(reduction_axes=-1, 
                                             feature_axes=-1)
        self.final_dropout = nn.Dropout(rate = self.dropout)
        
        
    def __call__(self, datamat, training):
        padding_mask_template = jnp.where(datamat == self.padding_idx, False, True)[:,:,None]
    
        ### create a position matrix
        datamat_batch_size, datamat_max_len = datamat.shape
        
        datamat_max_len = datamat.shape[1]
        posmat = jnp.arange(0, datamat_max_len)[None, :]
        
        new_shape = (datamat_batch_size,
                     posmat.shape[1])
        posmat = jnp.broadcast_to(posmat, new_shape)
        del new_shape
        
        
        ### first, embed the input data itself
        # (B,L) -> (B, L, H)
        datamat = self.seq_initial_embedding(datamat)
        
        # mask positions with padding tokens
        # mask is also (B, L, H)
        new_shape = (padding_mask_template.shape[0],
                     padding_mask_template.shape[1],
                     self.features)
        padding_mask = jnp.broadcast_to(padding_mask_template, new_shape)
        del new_shape
        
        datamat = jnp.multiply(datamat, padding_mask)
        
        
        ### second, embed the position matrix (and mask)
        # (B,L) -> (B, L, H)
        posmat = self.pos_initial_embedding(posmat)
        posmat = jnp.multiply(posmat, padding_mask)
        
        
        ### add, layernorm, and dropout
        # padding positions should already be zeros
        out = datamat + posmat
        out = self.final_layernorm(out)
        out = self.final_dropout(out, 
                                 deterministic = not training)
        
        # return masked matrix, and the masking used
        return (out, padding_mask)



class ConvEmbedding(ModuleBase):
    """
    one-hot encode, then use convolution to expand to H
    
    this captures some surrounding sequence context, so need to guard against
      look-ahead for causal models
    
    
    configs have (at minimum):
    --------------------------
    hidden_dim (int): length of the embedded vector
    conv_emb_kernel_width (int): width of convolution
    padding_idx (int = 0): padding token
    base_alphabet_size (int = 23): <pad>, <bos>, <eos>, then all alphabet 
                                  (20 for amino acids, 4 for DNA)
       
    """
    config: dict
    causal: bool
    name: str
    
    def setup(self):
        ### unpack config
        self.vocab_size = self.config['base_alphabet_size']
        self.features = self.config['hidden_dim']
        self.conv_emb_kernel_size = self.config['conv_emb_kernel_size']
        
        # these have default values
        self.padding_idx = self.config.get('seq_padding_idx', 0)
        
        
        ### layers to use
        self.conv = nn.Conv(features = self.features,
                            kernel_size = self.conv_emb_kernel_size,
                            strides = 1,
                            padding = 'CAUSAL' if self.causal else 'SAME')
        
        
    def __call__(self, datamat, training = None):
        ### use this for building padding masks
        padding_mask_template = jnp.where(datamat == self.padding_idx, False, True)[:,:,None]
        
        
        ### one-hot encode
        # (B,L) -> (B, L, base_alphabet_size)
        datamat = nn.one_hot(datamat, self.vocab_size)
        
        # mask positions with padding tokens
        # mask is also (B, L, base_alphabet_size)
        new_shape = (padding_mask_template.shape[0],
                     padding_mask_template.shape[1],
                     self.vocab_size)
        padding_mask_for_OH = jnp.broadcast_to(padding_mask_template, new_shape)
        del new_shape
        
        datamat = jnp.multiply(datamat, padding_mask_for_OH)
        
        
        ### conv to full hidden dimension
        # (B, L, base_alphabet_size) -> (B, L, H)
        datamat = self.conv(datamat)
        
        # mask positions with padding tokens
        # mask is also (B, L, H)
        new_shape = (padding_mask_template.shape[0],
                     padding_mask_template.shape[1],
                     self.features)
        final_padding_mask = jnp.broadcast_to(padding_mask_template, new_shape)
        del new_shape
        
        datamat = jnp.multiply(datamat, final_padding_mask)
        
        # return masked matrix, and the masking used (the one that projected to H, not the first one)
        return (datamat, final_padding_mask)
    