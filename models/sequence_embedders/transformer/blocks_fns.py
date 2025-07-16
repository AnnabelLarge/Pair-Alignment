#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:12:00 2023

@author: annabel_large

ABOUT:
=======
Main transformer blocks:
    - Pre-norm with sinusoidal embedding
    - Pre-norm with rotary embedding
    - TAPE transformer


config contains (for all transformers):
========================================
required:
---------
num_heads (int): number of heads for self-attention
hidden_dim (int): size of hidden layer


optional:
---------
dropout (float=0.0): dropout rate



"""
# general python
import numpy as np
from functools import partial

# flax n jax
import jax
from jax import lax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights

# custom
from models.BaseClasses import ModuleBase
from models.sequence_embedders.transformer.model_parts import (expand_padding_mask,
                                                      PositionalEncoding,
                                                      RotaryEmbeddingSelfAttention)



###############################################################################
### Pre-norm transformer ######################################################
###############################################################################
class TransfBaseBlock(ModuleBase):
    """
    one Transformer block (no positional encoding yet):
    
        |
        v
       in --------- 
        |         |
        v         |
       norm       |
        |         |
        v         |
    multihead     |
    self-attn     |
        |         | 
        v         |
     dropout      |
        |         |
        v         |
        ---> + <---
             |
             v
         after_attn--------- 
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
    
    """
    config: dict
    causal: bool
    name: str
    
    def setup(self):
        ### unpack from config
        # required
        self.num_heads = self.config['num_heads']
        self.hidden_dim = self.config['hidden_dim']
        
        # have defaults
        self.dropout = self.config.get('dropout', 0.0)
        self.output_attn_weights = self.config.get('output_attn_weights', False)
        self.max_position_embeddings = self.config.get('max_position_embeddings', 3000)
        
        
        ### if causal, have a causal mask ready to go
        if self.causal:
            # causal_mask is (1, 1, max_position_embeddings, max_position_embeddings)
            self.causal_mask_template = nn.make_causal_mask( jnp.ones( (1, self.max_position_embeddings) ), 
                                                    dtype="bool" )
        
        
        ### set up layers
        # self-attention
        self.setup_attn_layer()
        
        # dropout
        self.dropout_layer = nn.Dropout(rate=self.dropout)
        
        # dense layers (in final feedforward)
        self.first_feedforward_dense = nn.Dense(self.hidden_dim,
                                                kernel_init = nn.initializers.lecun_normal(),
                                                use_bias=True)
        
        self.second_feedforward_dense = nn.Dense(self.hidden_dim,
                                                 kernel_init = nn.initializers.lecun_normal(),
                                                 use_bias=True)
        
        # activation
        self.act_type = 'silu'
        self.act = nn.silu

        # normalization
        self.norm = nn.LayerNorm(reduction_axes=-1, feature_axes=-1)
        self.norm_type = 'Instance'
        
        
    def setup_attn_layer(self):
        """
        for now, this is the only difference between sinusoidal
        embedding transformer and RoPE transformer
        
        so make this a method that can be overwritten
        """
        self.self_attn = nn.MultiHeadDotProductAttention(num_heads = self.num_heads,
                                                         qkv_features=self.hidden_dim, 
                                                         out_features=self.hidden_dim, 
                                                         dropout_rate=self.dropout, 
                                                         decode=False, 
                                                         normalize_qk=False,
                                                         use_bias=True)
        
    def __call__(self, 
                 datamat, #(B, L, H)
                 padding_mask,  #(B, L)
                 sow_intermediates:bool, 
                 training:bool):  
        B = datamat.shape[0]
        L = datamat.shape[1]
        
        # mask padding tokens of input
        seq_padding_mask = jnp.broadcast_to( padding_mask[...,None], datamat.shape ) #(B, L, H)
        datamat = jnp.multiply(datamat, seq_padding_mask) #(B, L, H)
        

        #######################
        ### 1: attention part #
        #######################        
        skip = datamat #(B, L, H)

        ### 1.1) norm, mask
        datamat = self.norm(datamat)  #(B, L, H)
        datamat = jnp.multiply(datamat, seq_padding_mask) #(B, L, H)
        
        if sow_intermediates:
            label = f'{self.name}/after first {self.norm_type}Norm'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
        
        
        ### 1.2) make masks for attention (padding, plus optional causal)
        # padding mask is: (B,1,L,L)
        attn_padding_mask = expand_padding_mask(padding_mask) # (B,1,L,L)
        
        # causal mask is: (B,1,L,L)
        # (1,1,max_position_embeddings,max_position_embeddings) -> 
        #   (B,1,max_position_embeddings,max_position_embeddings) -> 
        #   (B,1,L,L)
        if self.causal:
            # causal_mask is (1, 1, max_position_embeddings, max_position_embeddings)
            #(B,1,max_position_embeddings,max_position_embeddings)
            out_shape = (B, 
                         self.causal_mask_template.shape[1], 
                         self.causal_mask_template.shape[2], 
                         self.causal_mask_template.shape[3])
            causal_mask = jnp.broadcast_to( self.causal_mask_template, out_shape)[:, :, :L, :L] #(B,1,L,L)
            attention_mask = nn.combine_masks(attn_padding_mask, 
                                              causal_mask,
                                              dtype=bool) #(B,1,L,L)
        
        elif not self.causal:
            attention_mask = attn_padding_mask.astype(bool) #(B,1,L,L)
        
        
        ### 1.3) self-attention
        datamat = self.self_attn(inputs_q = datamat, 
                                 mask=attention_mask, 
                                 deterministic=not training,
                                 sow_weights=self.output_attn_weights) #(B, L, H)
        
        
        ### 1.4) dropout and residual add
        datamat = self.dropout_layer(datamat,
                                     deterministic = not training)  #(B, L, H)
        datamat = skip + datamat  #(B, L, H)
        
        if sow_intermediates:
            label = f'{self.name}/after self-attention half'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
        
        
        #########################
        ### 2: feedforward part #
        #########################
        skip = datamat #(B, L, H)
        
        
        ### 2.1) norm, mask
        datamat = self.norm(datamat)  #(B, L, H)
        datamat = jnp.multiply(datamat, seq_padding_mask) #(B, L, H)
        
        if sow_intermediates:
            label = f'{self.name}/after second {self.norm_type}Norm'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
        
        
        ### 2.2) small MLP: dense -> silu -> mask -> dense
        datamat = self.first_feedforward_dense(datamat) #(B, L, H)
        
        datamat = self.act(datamat) #(B, L, H)
        datamat = jnp.multiply(datamat, seq_padding_mask) #(B, L, H)
        
        if sow_intermediates:
            label = f'{self.name}/in feedforward, after {self.act_type}'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
        
        datamat = self.second_feedforward_dense(datamat) #(B, L, H)
        
        
        ### 2.3) dropout, residual add mask again just in case
        datamat = self.dropout_layer(datamat,
                                     deterministic = not training)  #(B, L, H)
        datamat = skip + datamat  #(B, L, H)
        datamat = jnp.multiply(datamat, seq_padding_mask) #(B, L, H)
        
        if sow_intermediates:
            label = f'{self.name}/after feedforward half'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars']) 
            del label
        
        return datamat



class TransfBaseBlockWithAbsPosEmbedding(ModuleBase):
    """
    embed with sinusoidal embedding, then run TransfBaseBlock
    
    use this as the "first block"
    """
    config: dict
    causal: bool
    name: str
    
    @nn.compact
    def __call__(self, 
                 datamat,  #(B, L, H)
                 padding_mask,  #(B, L)
                 sow_intermediates:bool, 
                 training:bool):  
        datamat = PositionalEncoding( hidden_dim = self.config['hidden_dim'],
                                      max_len = self.config.get('max_position_embeddings',3000) )(x = datamat)
        datamat = TransfBaseBlock( config=self.config,
                                   causal=self.causal,
                                   name=self.name )(datamat = datamat, 
                                                    padding_mask = padding_mask, 
                                                    training = training, 
                                                    sow_intermediates = sow_intermediates)
        return datamat



###############################################################################
### Pre-norm transformer with Rotational Embeddings ###########################
###############################################################################
class RoPETransfBlock(TransfBaseBlock):
    """
    one Transformer block with Rotational Embeddings
    pretty much the same as TransfBaseBlock, but with different attention layer
      (functionally, change this by replacing self.setup_attn_layer)
    
    
        |
        v
       in --------- 
        |         |
        v         |
       norm       |
        |         |
        v         |
      RoPE        |
    self-attn     |
        |         | 
        v         |
     dropout      |
        |         |
        v         |
        ---> + <---
             |
             v
         after_attn--------- 
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
    
    """
    def setup_attn_layer(self):
        self.self_attn = RotaryEmbeddingSelfAttention(num_heads = self.num_heads,
                                                      hidden_dim = self.hidden_dim,
                                                      causal = self.causal,
                                                      output_attn_weights = self.output_attn_weights,
                                                      max_position_embeddings = self.max_position_embeddings,
                                                      use_bias = True,
                                                      dropout = self.dropout)
        
        
###############################################################################
### TAPE transformer ##########################################################
###############################################################################
class TapeTransfBlock(TransfBaseBlock):
    """
    the transformer block from TAPE, based on ProteinBERT (which was a 
    post-norm transformer)
    
    use the same setup method from TransfBaseBlock, but implement
      a different __call__
    
    other notes:
      - make sure to run TAPEEmbedding first!!!
      - if you want to match TAPE exactly, use gelu and layerNorm
    
    
        |
        v
       in --------- 
        |         |
        v         |
    multihead     |
    self-attn     |
        |         | 
        v         |
     dropout      |
        |         |
        |         |
        ---> + <---
             |
             v
            norm
             |
             v
         after_attn--------- 
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
                 |         |
                 ---> + <---
                      |
                      v
                    norm
                      |
                      v
                     out
    """
    config: dict
    causal: bool
    name: str
    
    def __call__(self, 
                 datamat, #(B, L, H)
                 padding_mask,  #(B, L)
                 sow_intermediates:bool, 
                 training:bool):
        B = datamat.shape[0]
        L = datamat.shape[1]
        
        # mask padding tokens of input
        seq_padding_mask = jnp.broadcast_to( padding_mask[...,None], datamat.shape ) #(B, L, H)
        datamat = jnp.multiply(datamat, seq_padding_mask) #(B, L, H)
        
        
        #######################
        ### 1: attention part #
        #######################
        skip = datamat
        
        ### 1.1) make masks
        padding_mask = expand_padding_mask(padding_mask) # (B,1,L,L)
        
        # causal mask is: (B,1,L,L)
        # (1,1,max_position_embeddings,max_position_embeddings) -> 
        #   (B,1,max_position_embeddings,max_position_embeddings) -> 
        #   (B,1,L,L)
        if self.causal:
            # causal_mask is (1, 1, max_position_embeddings, max_position_embeddings)
            #(B,1,max_position_embeddings,max_position_embeddings)
            out_shape = (B, 
                         self.causal_mask_template.shape[1], 
                         self.causal_mask_template.shape[2], 
                         self.causal_mask_template.shape[3])
            causal_mask = jnp.broadcast_to( self.causal_mask_template, out_shape)[:, :, :L, :L] #(B,1,L,L)
            attention_mask = nn.combine_masks(attn_padding_mask, 
                                              causal_mask,
                                              dtype=bool) #(B,1,L,L)
        
        elif not self.causal:
            attention_mask = padding_mask.astype(bool) #(B,1,L,L)
        
        
        ### 1.2) self-attention
        datamat = self.self_attn(inputs_q = datamat, 
                                 mask=attention_mask, 
                                 deterministic=not training,
                                 sow_weights=self.output_attn_weights)
        
        
        ### 1.3) dropout and residual add
        datamat = self.dropout_layer(datamat,
                                     deterministic = not training)  #(B, L, H)
        datamat = skip + datamat  #(B, L, H)
        
        if sow_intermediates:
            label = f'{self.name}/after self-attention half'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
        
        
        ##################
        ### First Norm   #
        ##################
        datamat = self.norm(datamat)  #(B, L, H)
        datamat = jnp.multiply(datamat, seq_padding_mask) #(B, L, H)
        
        if sow_intermediates:
            label = f'{self.name}/after first {self.norm_type}Norm'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
            
        
        #########################
        ### 2: feedforward part #
        #########################
        skip = datamat
        
        ### 2.1) small MLP: dense -> gelu -> mask -> dense
        datamat = self.first_feedforward_dense(datamat)
        
        datamat = self.nn.gelu(datamat) #(B, L, H)
        datamat = jnp.multiply(datamat, seq_padding_mask) #(B, L, H)
        
        if sow_intermediates:
            label = f'{self.name}/in feedforward, after gelu'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
        
        datamat = self.second_feedforward_dense(datamat)
        
        
        ### 2.2) dropout and residual add
        datamat = self.dropout_layer(datamat,
                                     deterministic = not training)  #(B, L, H)
        datamat = skip + datamat  #(B, L, H)
        
        if sow_intermediates:
            label = f'{self.name}/after feedforward half'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars']) 
            del label
        
        
        #################
        ### Second Norm #
        #################
        datamat = self.norm(datamat)  #(B, L, H)
        datamat = jnp.multiply(datamat, seq_padding_mask) #(B, L, H)
        
        return datamat
        
        
            