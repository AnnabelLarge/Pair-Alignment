#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:44:26 2025

@author: annabel

About:
=======
Take concatenated outputs from both sequence embedders and postprocess for 
downstream blocks that create logits


classes available:
==================
1.) Placeholder (ignore outputs from sequence embedders)
2.) SelectMask (one-hot encode amino acids from training path)
3.) FeedforwardPostproc 
    norm -> dense -> act -> dropout

"""
from flax import linen as nn
import jax
import jax.numpy as jnp

from models.BaseClasses import ModuleBase
from models.neural_hmm_predict.model_functions import (process_datamat_lst)


class Placeholder(ModuleBase):
    """
    to ignore embeddings entirely, use this 
        (useful when you're reading log-probabilities from files')
    """
    config: None
    name: str
    
    @nn.compact
    def __call__(self, 
                 *args,
                 **kwargs):
        """
        placeholder method; returns None
        
        
        B: batch size
        L_align: length of alignment
        
        Arguments
        ----------
        padding_mask : ArrayLike, (B, L_align)
        
        Returns
        --------
        None
        """
        
        return None
    
    
class SelectMask(ModuleBase):
    config: dict
    name: str
    
    def setup(self):
        self.use_anc_emb = self.config['use_anc_emb']
        self.use_desc_emb = self.config['use_desc_emb']
        self.use_prev_align_info = self.config['use_prev_align_info']
    
    def __call__(self,
                 datamat_lst: list,
                 padding_mask: jnp.array,
                 *args,
                 **kwargs):
        """
        B: batch size
        L_align: length of alignment
        
        Arguments
        ----------
        datamat_lst : list[ArrayLike, ArrayLike]
            ancestor embedding, descendant embedding (in that order); each are
            (B, L_align, H_in)
        
        Returns
        --------
        datamat : ArrayLike, (B, L_align, n*H)
            concatenated/masked data
            > n=1, if only using ancestor embedding OR descendant embedding
            > n=2, if using both embeddings
            
        """
        # select (potentially concat) inputs
        datamat = process_datamat_lst(datamat_lst,
                                      padding_mask,
                                      self.use_anc_emb,
                                      self.use_desc_emb,
                                      self.use_prev_align_info)
        
        return datamat
        

class FeedforwardPostproc(SelectMask):
    """
    apply this blocks as many times as specified by layer_sizes: 
        [norm -> dense -> activation -> dropout]
    """
    config: dict
    name: str
    
    def setup(self):
        """
        B = batch size
        L_align = alignment length
        A = alphabet size
        
        Flax Module Parameters
        -----------------------
        [fill in later]
        
        """
        ### read config
        # required
        self.use_anc_emb = self.config['use_anc_emb']
        self.use_desc_emb = self.config['use_desc_emb']
        self.use_prev_align_info = self.config['use_prev_align_info']
        self.layer_sizes = self.config['layer_sizes']
        
        # optional
        self.normalize_inputs = self.config.get("normalize_inputs", True)
        self.dropout = self.config.get("dropout", 0.0)
        
        
        ### set up parameterized layers of the MLP
        dense_layers = []
        norm_layers = []
        
        # first layer: instance norm
        if self.normalize_inputs:
            norm_layers.append( nn.LayerNorm( reduction_axes= -1, 
                                              feature_axes=-1,
                                              name = f'{self.name}/instance norm 0') )
        
        # first layer: identity function (if not normalizing inputs)
        elif not self.normalize_inputs:
            norm_layers.append( lambda x: x )
        
        dense_layers.append( nn.Dense(features = self.layer_sizes[0], 
                                      use_bias = use_bias, 
                                      kernel_init = nn.initializers.lecun_normal(),
                                      name=f'{self.name}/feedforward layer 0') )
        
        # subsequent normalization and dense layers
        for i, hid_dim in enumerate(self.layer_sizes[1:]):
            layer_idx = i + 1
            norm_layers.append( nn.LayerNorm( reduction_axes= -1, 
                                              feature_axes=-1,
                                              name = f'{self.name}/instance norm {layer_idx}') )
            dense_layers.append( nn.Dense(features = hid_dim, 
                                          use_bias = use_bias, 
                                          kernel_init = nn.initializers.lecun_normal(),
                                          name=f'{self.name}/feedforward layer {layer_idx}') )
            
        self.dense_layers = dense_layers
        self.norm_layers = norm_layers
        self.act= nn.silu 
    
    @nn.compact
    def __call__(self, 
                 datamat_lst: list,
                 padding_mask: jnp.array,
                 training: bool, 
                 sow_intermediates: bool=False):
        """
        B: batch size
        L_align: length of alignment
        H_in, H_out: size of embedding dimension in/out of this block
        
        Arguments
        ----------
        datamat_lst : list[ArrayLike, ArrayLike, ArrayLike]
            ancestor embedding, descendant embedding, previous alignment state 
            (in that order); embeddings are (B, L_align, H_in), and previous
            alignment state is (B, L_align, 5)
        
        padding_mask : ArrayLike, (B, L_align)
            location of padding in alignment
        
        training : bool
            are you in training or not (affects dropout behavior)
        
        sow_intermediates : bool
            switch for tensorboard logging
        
        Returns
        --------
        datamat : ArrayLike, (B, L_align, H_out)
            concatenated and post-processed data
            > n=1, if only using ancestor embedding OR descendant embedding
            > n=2, if using both embeddings
        """
        # anc_embeddings: (B, L, H)
        # desc_embeddings: (B, L, H)
        # prev_align_one_hot_vec: (B, L, 5)
        anc_embeddings, desc_embeddings, prev_align_one_hot_vec = datamat_lst
        del datamat_lst
        
        
        ### First block
        # select (potentially concat) the ancestor and descendant embeddings
        emb_lst = [anc_embeddings, desc_embeddings]
        embeddings_datamat = process_datamat_lst(datamat_lst = emb_lst,
                                                 padding_mask = padding_mask,
                                                 use_anc_emb = self.use_anc_emb,
                                                 use_desc_emb = self.use_desc_emb,
                                                 use_prev_align_info = False) # (B, L, n*H)
        del emb_lst
        
        # 1.1) record distribution into block
        if sow_intermediates:
            label = (f'{self.name}/'+
                     f'final feedforward layer 0/'+
                     f'embeddings before block')
            self.sow_histograms_scalars(mat = embeddings_datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
            
        # 1.2) possibly instance norm and mask padding tokens
        embeddings_datamat = self.norm_layers[0](embeddings_datamat) #(B, L, n*H)
        
        if self.normalize_inputs:
            embeddings_datamat = self._mask_padding_tokens( x=embeddings_datamat,
                                                            mask=padding_mask ) #(B, L, n*H)
        
            if sow_intermediates:
                label = (f'{self.name}/'+
                         f'final feedforward layer 0/'+
                         f'embeddings after instance normalization')
                self.sow_histograms_scalars(mat = embeddings_datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
        
        # 1.3) possibly concatenate normalized sequence embeddings to 
        #      previous alignment state; this preserves signal from one-hot 
        #      encoding
        if self.use_prev_align_info:
            datamat = jnp.concatenate( [embeddings_datamat, prev_align_one_hot_vec],
                                       axis = -1 )  #(B, L, n*H + 5)
        elif not self.use_prev_align_info:
            datamat = embeddings_datamat #(B, L, n*H)
        
        del embeddings_datamat
        
        # 1.4) dense layer
        datamat = self.dense_layers[0](datamat) #(B, L, hid_dim[0] )
        
        if sow_intermediates:
            label = (f'{self.name}/'+
                     f'final feedforward layer 0/'+
                     'after dense')
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
        
        # 1.5) activation
        datamat = self.act(datamat) #(B, L, hid_dim[0] )
        
        if sow_intermediates:
            label = (f'{self.name}/'+
                     f'final feedforward layer 0/'+
                     f'after silu')
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
        
        # 1.6) dropout
        datamat = nn.Dropout(rate = self.dropout)(datamat,
                                            deterministic = not training)  #(B, L, layer_sizes[0] )
         
        if sow_intermediates and (self.dropout != 0):
            label = (f'{self.name}/'+
                     f'final feedforward layer 0/'+
                     'after dropout')
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
        
        
        ### Subsequent blocks: norm -> dense -> activation -> dropout
        for i in range( len(self.layer_sizes[1:]) ):
            layer_idx = i + 1
             
            # 1.) norm (plus recording to tensorboard)
            datamat = self.norm_layers[layer_idx](datamat) #(B, L, layer_sizes[layer_idx-1])
            datamat = self._mask_padding_tokens( x=datamat,
                                                 mask=padding_mask ) #(B, L, layer_sizes[layer_idx]-1)
            
            if sow_intermediates:
                label = (f'{self.name}/'+
                         f'final feedforward layer {layer_idx}/'+
                         f'after instance norm')
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
                
            # 2.) dense
            datamat = self.dense_layers[layer_idx](datamat) #(B, L, layer_sizes[layer_idx])
            
            if sow_intermediates:
                label = (f'{self.name}/'+
                         f'final feedforward layer {layer_idx}/'+
                         'after dense')
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
            
            # 3.) activation
            datamat = self.act(datamat) #(B, L, layer_sizes[layer_idx])
            
            if sow_intermediates:
                label = (f'{self.name}/'+
                         f'final feedforward layer {layer_idx}/'+
                         f'after silu')
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
            
            # 4.) dropout
            datamat = nn.Dropout(rate = self.dropout)(datamat,
                                                deterministic = not training) #(B, L, layer_sizes[layer_idx])
            
            if sow_intermediates and (self.dropout != 0):
                label = (f'{self.name}/'+
                         f'final feedforward layer {layer_idx}/'+
                         'after dropout')
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
        
            
        ### mask out padding tokens one last time
        datamat = self._mask_padding_tokens( x=datamat,
                                             mask=padding_mask ) #(B, L, layer_sizes[-1])
        return datamat  #(B, L, layer_sizes[-1])
    
    
    def _mask_padding_tokens( self,
                             x: jnp.array, 
                             mask: jnp.array ):
        expanded_mask = jnp.broadcast_to( mask[...,None], x.shape ) 
        return jnp.multiply(expanded_mask, x)
        
    
