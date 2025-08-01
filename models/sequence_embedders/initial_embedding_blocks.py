#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:22:44 2024

@author: annabel_large
    

modules to project (B, L) -> (B, L, H), before sending to main architecture

"""
# general python
from typing import Callable, Optional, Any, Dict

# flaxy and jaxy
from flax import linen as nn
import jax
import jax.numpy as jnp

# custom
from models.BaseClasses import ModuleBase, SeqEmbBase


class PlaceholderEmbedding(nn.Module):
    """
    for debugging; take in a (B,L) matrix and repeat entries
      to (B, L, hidden_dim)
    """
    config: Dict
    name: str
    embedding_which: Optional[Any] = None
    causal: Optional[Any] = None
    
    @nn.compact
    def __call__(self, 
                 datamat: jnp.array,  #(B, L)
                 training: Optional[Any] = None):
        ### unpack
        hidden_dim = self.config['hidden_dim'] #H
        seq_padding_idx = self.config.get('seq_padding_idx', 0)
        
        ### run
        B = datamat.shape[0]
        L = datamat.shape[1]
        final_shape = (B, L, hidden_dim)
        
        # padding mask
        padding_mask = (datamat != seq_padding_idx) #(B, L)
        padding_mask_expanded = jnp.broadcast_to(padding_mask[..., None], final_shape) # (B, L, H)
        
        # expand the data matrix and mask
        datamat = jnp.broadcast_to(datamat[...,None], final_shape)  #(B, L, H)
        datamat = jnp.multiply(datamat, padding_mask_expanded)  #(B, L, H)
        del padding_mask_expanded
        
        # datamat is (B, L, H)
        # padding_mask is (B, L)
        return (datamat, padding_mask)
    
    
class OneHotEmb(SeqEmbBase):
    """
    Only one-hot encoding
    
    
    init with:
    ==========
    config (dict): config to pass to each subsequent module
    name (str): "ANCESTOR EMBEDDER" or "DESCENDANT EMBEDDER"
    
    
    config will have:
    =================
    in_alph_size: 23 for proteins, 7 for DNA
    
    
    call arguments are:
    ===================
    datamat: matrix of sequences (B, L)
    training: NOT USED
    sow_intermediates: NOT USED
    
    
    outputs:
    ========
    datamat (altered matrix): one-hot encodings for all sequences 
                              (B, L, in_alph_size)
    """
    embedding_which: str
    config: dict
    name: str
    causal: Optional[Any] = None
    
    def setup(self):
        self.in_alph_size = self.config.['in_alph_size']
        self.seq_padding_idx = self.config.get('seq_padding_idx', 0)
    
    def __call__(self, 
                 datamat, 
                 *args,
                 **kwargs):
        """
        Arguments
        ----------
        datamat : ArrayLike, (B, L)
            > encoded with tokens from 1 to in_alph_size; padding is 
              assumed to be zero
        """
        padding_mask = (datamat != self.seq_padding_idx) #(B, L)
        padding_mask_template = padding_mask[...,None] #(B,L,1)
        
        # flax's one-hot will start one-hot encoding at token 0 (padding)
        #   run the one-hot encoding with an extra class, mask it, then remove 
        #   the empty leading column
        raw_one_hot = nn.one_hot(datamat, 
                                 num_classes = self.in_alph_size,
                                 axis=-1) #(B, L, in_alph_size)
        
        seq_mask = jnp.broadcast_to(padding_mask_template, 
                                    raw_one_hot.shape) #(B, L, in_alph_size)
        one_hot_final = raw_one_hot * seq_mask  #(B, L, in_alph_size)
        return one_hot_final, padding_mask

class EmbeddingWithPadding(ModuleBase):
    """
    replicated torch's embedding function, with padding_idx option 
    
    doesn't really matter if it's causal or not; keeping here to preserve trace
    
    configs have (at minimum):
    --------------------------
    hidden_dim (int): length of the embedded vector
    padding_idx (int = 0): padding token
    args.in_alph_size (int): <pad>, <bos>, <eos>, then all alphabet 
                                  (20 for amino acids, 4 for DNA)
                              
    """
    embedding_which: str
    config: Dict
    name: str
    causal: Optional[Any] = None
    
    def setup(self):
        # unpack config
        self.features = self.config['hidden_dim'] #H
        self.vocab_size = self.config['in_alph_size']
        self.seq_padding_idx = self.config.get('seq_padding_idx', 0)
        
        # layers to use
        self.initial_embedding = nn.Embed(num_embeddings = self.vocab_size, 
                                          features = self.features)
        
        
    def __call__(self, 
                 datamat: jnp.array,  #(B, L)
                 training: Optional[Any] = None):
        B = datamat.shape[0]
        L = datamat.shape[1]
        final_shape = (B, L, self.features)
        
        # padding mask
        padding_mask = (datamat != self.seq_padding_idx) #(B, L)
        padding_mask_expanded = jnp.broadcast_to(padding_mask[..., None], final_shape) # (B, L, H)
        
        # embed: (B,L) -> (B, L, H)
        datamat = self.initial_embedding(datamat) # (B, L, H)
        datamat = jnp.multiply(datamat, padding_mask_expanded) # (B, L, H)
        
        # datamat is (B, L, H)
        # padding_mask is (B, L)
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
    args.in_alph_size (int): <pad>, <bos>, <eos>, then all alphabet 
                                  (20 for amino acids, 4 for DNA)
                              
    """
    embedding_which: str
    config: Dict
    name: str
    causal: Optional[Any] = None
    
    def setup(self):
        # unpack config
        self.features = self.config['hidden_dim']
        self.vocab_size = self.config['in_alph_size']
        self.padding_idx = self.config.get('seq_padding_idx', 0)
        self.max_len = self.config.get('max_len', 3000)
        self.dropout = self.config.get('dropout', 0.0)
        
        # layers to use
        self.seq_initial_embedding = nn.Embed(num_embeddings = self.vocab_size, 
                                              features = self.features)
        self.pos_initial_embedding = nn.Embed(num_embeddings = self.max_len, 
                                              features = self.features)
        self.final_instancenorm =  nn.LayerNorm(reduction_axes=-1, 
                                                feature_axes=-1)
        self.final_dropout = nn.Dropout(rate = self.dropout)
        
        
    def __call__(self, 
                 datamat: jnp.array,  #(B, L)
                 training: bool):
        B = datamat.shape[0]
        L = datamat.shape[1]
        final_shape = (B, L, self.features)
        
        # padding mask
        padding_mask = (datamat != self.seq_padding_idx) #(B, L)
        padding_mask_expanded = jnp.broadcast_to(padding_mask[..., None], final_shape) # (B, L, H)
        
        
        # create a position matrix
        posmat = jnp.arange(0, L) #(L,)
        posmat = jnp.broadcast_to(posmat[None, :], (B, L) ) #(B, L)
        
        
        ### 1.) embed the input data itself: (B,L) -> (B, L, H)
        datamat = self.seq_initial_embedding(datamat) #(B, L, H)
        datamat = jnp.multiply(datamat, padding_mask_expanded) #(B, L, H)
        
        
        ### 2.) embed the position matrix: (B,L) -> (B, L, H)
        posmat = self.pos_initial_embedding(posmat) #(B, L, H)
        posmat = jnp.multiply(posmat, padding_mask_expanded) #(B, L, H)
        
        
        ### 3.) add, layernorm, and dropout
        out = datamat + posmat #(B, L, H)
        
        out = self.final_instancenorm(out) #(B, L, H)
        out = jnp.multiply(out, padding_mask_expanded) #(B, L, H)
        
        out = self.final_dropout(out, deterministic = not training) #(B, L, H)
        
        # datamat is (B, L, H)
        # padding_mask is (B, L)
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
    args.in_alph_size (int): <pad>, <bos>, <eos>, then all alphabet 
                                  (20 for amino acids, 4 for DNA)
       
    """
    embedding_which: str
    config: Dict
    causal: bool
    name: str
    
    def setup(self):
        # unpack config
        self.vocab_size = self.config['in_alph_size'] #A
        self.features = self.config['hidden_dim'] #H
        self.conv_emb_kernel_size = self.config['conv_emb_kernel_size']
        self.seq_padding_idx = self.config.get('seq_padding_idx', 0)
        assert self.seq_padding_idx == 0
        
        # layers to use
        self.conv = nn.Conv(features = self.features,
                            kernel_size = self.conv_emb_kernel_size,
                            strides = 1,
                            padding = 'CAUSAL' if self.causal else 'SAME')
        
        
    def __call__(self, 
                 datamat: jnp.array, #(B, L) 
                 training: bool):
        B = datamat.shape[0]
        L = datamat.shape[1]
        final_shape = (B, L, self.features)
        
        # padding mask
        padding_mask = (datamat != self.seq_padding_idx) #(B, L)
        
        
        ### one-hot encode first
        # (B,L) -> (B, L, A)
        datamat = nn.one_hot(datamat, self.vocab_size) #(B, L, A)
        padding_mask_expanded = jnp.broadcast_to(padding_mask[...,None], 
                                                 (B, L, self.vocab_size) ) # (B, L, A)
        datamat = jnp.multiply(datamat, padding_mask_expanded) #(B, L, A)
        del padding_mask_expanded
        
        # remove embeddings associated with padding token (should be 0)
        datamat = datamat[..., 1:] #(B, L, A - 1)
        
        
        ### conv to full hidden dimension
        # (B, L, A - 1) -> (B, L, H)
        datamat = self.conv(datamat) # (B, L, H)
        padding_mask_expanded = jnp.broadcast_to(padding_mask[...,None], final_shape ) # (B, L, H)
        datamat = jnp.multiply(datamat, padding_mask_expanded) #(B, L, H)
        
        # datamat is (B, L, H)
        # padding_mask is (B, L)
        return (datamat, padding_mask)
    