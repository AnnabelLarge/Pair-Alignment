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
from models.model_utils.BaseClasses import ModuleBase
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
        # !!! hard code
        # activations
        self.act_type = 'silu'
        self.act = nn.silu
        self.kernel_init = nn.initializers.lecun_normal()

        # normalization
        self.norm_type = 'layer'
        if self.causal:
            self.norm = nn.LayerNorm(reduction_axes=-1, 
                                     feature_axes=-1,
                                     epsilon=1e-5)
        elif not self.causal:
            self.norm = nn.LayerNorm(reduction_axes= (-2,-1), 
                                     feature_axes=-1)
        
        # other
        self.output_attn_weights = False
        self.use_bias = True
        self.max_len = 3000
        
        
        ### unpack from config
        # required
        self.num_heads = self.config['num_heads']
        self.hidden_dim = self.config['hidden_dim']
        
        # have defaults
        self.dropout = self.config.get('dropout', 0.0)
        
        
        ### if causal, have a causal mask ready to go
        #   (1, 1, max_position_embeddings, max_position_embeddings)
        if self.causal:
            self.causal_mask = nn.make_causal_mask(jnp.ones( (1, self.max_len), 
                                                            dtype="bool"), 
                                                    dtype="bool")
        
        ### other layers
        # self-attention
        self.setup_attn_layer()
        
        # dropout
        self.dropout_layer = nn.Dropout(rate=self.dropout)
        
        # dense layers (in final feedforward)
        self.first_feedforward_dense = nn.Dense(self.hidden_dim,
                                                kernel_init = self.kernel_init,
                                                use_bias=True)
        
        self.second_feedforward_dense = nn.Dense(self.hidden_dim,
                                                 kernel_init = self.kernel_init,
                                                 use_bias=True)
    
    
    
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
                                                         use_bias=self.use_bias)
        
    def __call__(self, 
                 datamat, 
                 padding_mask, 
                 sow_intermediates:bool, 
                 training:bool):  
        batch_size = datamat.shape[0]
        max_len = datamat.shape[1]
        
        ####################
        ### attention part #
        ####################
        skip = datamat
        
        ### Norm
        datamat = self.norm(datamat,
                            mask = padding_mask)
        
        # manually mask again, because layernorm leaves NaNs
        datamat = jnp.where( padding_mask,
                            datamat,
                            0)
        
        if sow_intermediates:
            label = f'{self.name}/after first {self.norm_type}Norm'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
        
        
        ### make masks
        # padding mask is: (B,1,L,L)
        attn_padding_mask = expand_padding_mask(padding_mask)
        
        # causal mask is: (B,1,L,L)
        # (1,1,max_position_embeddings,max_position_embeddings) -> 
        #   (B,1,max_position_embeddings,max_position_embeddings) -> 
        #   (B,1,L,L)
        if self.causal:
            #(B,1,max_position_embeddings,max_position_embeddings)
            causal_mask = jnp.broadcast_to( self.causal_mask, 
                                            ( (batch_size,) + 
                                             self.causal_mask.shape[1:]
                                             )
                                            )
            causal_mask = causal_mask[:, :, :max_len, :max_len] #(B,1,L,L)
            
            attention_mask = nn.combine_masks(attn_padding_mask, 
                                              causal_mask,
                                              dtype=bool)
        
        elif not self.causal:
            attention_mask = attn_padding_mask.astype(bool)
        
        
        ### self-attention
        datamat = self.self_attn(inputs_q = datamat, 
                                 mask=attention_mask, 
                                 deterministic=not training,
                                 sow_weights=self.output_attn_weights)
        
        
        ### dropout and residual add
        # dropout
        datamat = self.dropout_layer(datamat,
                                     deterministic = not training)
        
        # add
        datamat = skip + datamat
        
        if sow_intermediates:
            label = f'{self.name}/after self-attention half'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
        
        
        ######################
        ### feedforward part #
        ######################
        skip = datamat
        
        
        ### Norm
        datamat = self.norm(datamat,
                            mask = padding_mask)
        
        # manually mask again, because layernorm leaves NaNs
        datamat = jnp.where( padding_mask,
                            datamat,
                            0)
        
        if sow_intermediates:
            label = f'{self.name}/after second {self.norm_type}Norm'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
        
        
        ### feedforward: dense -> relu -> dense
        datamat = self.first_feedforward_dense(datamat)
        
        datamat = self.act(datamat)
        if sow_intermediates:
            label = f'{self.name}/in feedforward, after {self.act_type}'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
        
        datamat = self.second_feedforward_dense(datamat)
        
        
        ### dropout and residual add
        # dropout
        datamat = self.dropout_layer(datamat,
                                     deterministic = not training)
        
        # add
        datamat = skip + datamat
        
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
                 datamat, 
                 padding_mask, 
                 sow_intermediates:bool, 
                 training:bool):  
        datamat = PositionalEncoding( hidden_dim = self.config['hidden_dim'],
                                      max_len = self.config.get('max_len',3000) )(x = datamat)
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
                                                      max_position_embeddings = self.max_len,
                                                      use_bias = self.use_bias,
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
                 datamat, 
                 padding_mask, 
                 sow_intermediates:bool, 
                 training:bool):
        # !!! over-write defaults
        self.act_type = 'gelu'
        self.act = nn.gelu
        
        batch_size = datamat.shape[0]
        max_len = datamat.shape[1]
        
        ####################
        ### attention part #
        ####################
        skip = datamat
        
        ### make masks
        # padding mask is: (B,1,L,L)
        padding_mask = expand_padding_mask(padding_mask)
        
        # causal mask is: (B,1,L,L)
        # (1,1,max_position_embeddings,max_position_embeddings) -> 
        #   (B,1,max_position_embeddings,max_position_embeddings) -> 
        #   (B,1,L,L)
        if self.causal:
            #(B,1,max_position_embeddings,max_position_embeddings)
            causal_mask = jnp.broadcast_to( self.causal_mask, 
                                            ( (batch_size,) + 
                                              self.causal_mask.shape[1:] 
                                              )
                                            )
            causal_mask = causal_mask[:, :, :max_len, :max_len] #(B,1,L,L)
            
            attention_mask = nn.combine_masks(padding_mask, 
                                              causal_mask,
                                              dtype=bool)
        
        elif not self.causal:
            attention_mask = padding_mask.astype(bool)
        
        
        ### self-attention
        datamat = self.self_attn(inputs_q = datamat, 
                                 mask=attention_mask, 
                                 deterministic=not training,
                                 sow_weights=self.output_attn_weights)
        
        
        ### dropout and residual add
        # dropout
        datamat = self.dropout_layer(datamat,
                                     deterministic = not training)
        
        # add
        datamat = skip + datamat
        
        if sow_intermediates:
            label = f'{self.name}/after self-attention half'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
        
        
        ##################
        ### First Norm   #
        ##################
        datamat = self.norm(datamat,
                            mask = padding_mask)
        
        # manually mask again, because layernorm leaves NaNs
        datamat = jnp.where( padding_mask,
                            datamat,
                            0)
        
        if sow_intermediates:
            label = f'{self.name}/after first {self.norm_type}Norm'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
            
        
        ######################
        ### feedforward part #
        ######################
        skip = datamat
        
        ### feedforward: dense -> relu -> dense
        datamat = self.first_feedforward_dense(datamat)
        
        datamat = self.nn.gelu(datamat)
        if sow_intermediates:
            label = f'{self.name}/in feedforward, after gelu'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
        
        datamat = self.second_feedforward_dense(datamat)
        
        
        ### dropout and residual add
        # dropout
        datamat = self.dropout_layer(datamat,
                                     deterministic = not training)
        
        # add
        datamat = skip + datamat
        
        if sow_intermediates:
            label = f'{self.name}/after feedforward half'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars']) 
            del label
        
        
        #################
        ### Second Norm #
        #################
        # don't need to record intermediates; I'll do that in main embedder
        datamat = self.norm(datamat, 
                            mask = padding_mask)
        
        # manually mask again, because layernorm leaves NaNs
        datamat = jnp.where( padding_mask,
                            datamat,
                            0)
        
        return datamat
        
        
            