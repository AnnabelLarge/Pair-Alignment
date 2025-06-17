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
3.) FeedforwardToEvoparams 
    norm -> dense -> act -> dropout
4.) ConvToEvoparams
    norm -> conv -> mask -> act -> dropout

"""
from flax import linen as nn
import jax
import jax.numpy as jnp

from models.model_utils.BaseClasses import ModuleBase
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
                 padding_mask,
                 *args,
                 **kwargs):
        """
        placeholder method; just returns the padding mask
        
        
        B: batch size
        L_align: length of alignment
        
        Arguments
        ----------
        padding_mask : ArrayLike, (B, L_align)
        
        Returns
        --------
        None
        
        padding_mask: ArrayLike, (B, L_align)
            location of padding in alignment
        """
        
        return (None, padding_mask)
    
    
class SelectMask(ModuleBase):
    config: dict
    name: str
    
    @nn.compact
    def __call__(self,
                 datamat_lst: list,
                 padding_mask: jnp.array,
                 **kwargs):
        """
        B: batch size
        L_align: length of alignment
        
        Arguments
        ----------
        datamat_lst : list[ArrayLike, ArrayLike]
            ancestor embedding, descendant embedding (in that order); each are
            (B, L_align, H_in)
        
        padding_mask : ArrayLike, (B, L_align)
            location of padding in alignment
        
        Returns
        --------
        datamat : ArrayLike, (B, L_align, n*H)
            concatenated/masked data
            > n=1, if only using ancestor embedding OR descendant embedding
            > n=2, if using both embeddings
            
        padding_mask: ArrayLike, (B, L_align)
            location of padding in alignment
        """
        # unpack config file
        use_anc_emb = self.config['use_anc_emb']
        use_desc_emb = self.config['use_desc_emb']
        
        # select (potentially concat) and mask padding
        datamat, padding_mask = process_datamat_lst(datamat_lst,
                                                    padding_mask,
                                                    use_anc_emb,
                                                    use_desc_emb)
        
        return datamat, padding_mask[...,0]
        

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
        self.layer_sizes = self.config['layer_sizes']
        
        # optional
        self.dropout = self.config.get("dropout", 0.0)
        
        
        ### set up layers
        dense_layers = []
        norm_layers = []
        for layer_idx, hid_dim in enumerate(self.layer_sizes):
            norm_layers.append( nn.LayerNorm( reduction_axes= -1, 
                                              feature_axes=-1,
                                              name = f'{self.name}/instance norm {layer_idx}') )
            
            dense_layers.append( nn.Dense(features = hid_dim, 
                                          use_bias = True, 
                                          kernel_init = nn.initializers.lecun_normal(),
                                          name=f'{self.name}/feedforward layer {layer_idx}')(datamat) )
            
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
        datamat_lst : list[ArrayLike, ArrayLike]
            ancestor embedding, descendant embedding (in that order); each are
            (B, L_align, H_in)
        
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
            
        padding_mask: ArrayLike, (B, L_align)
            location of padding in alignment
        """
        
        ### select (potentially concat) and mask padding
        # datamat: (B, L, n*H)
        # full_concat_masking_mat: (B, L, n*H)
        datamat, full_concat_masking_mat = process_datamat_lst(datamat_lst,
                                                               padding_mask,
                                                               self.use_anc_emb,
                                                               self.use_desc_emb)
        del datamat_lst
        
        
        ### norm -> dense -> activation -> dropout
        for layer_idx, hid_dim in enumerate(self.layer_sizes):
            # 1.) record distribution into block
            if sow_intermediates:
                label = (f'{self.name}/'+
                         f'final feedforward layer {layer_idx}/'+
                         f'before block')
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
                
            # 2.) norm (plus recording to tensorboard)
            #     shouldn't need masking after this
            concat_masking_mask = full_concat_masking_mat[...,:datamat.shape[-1]]
            datamat = self.norm_layers[layer_idx](datamat, 
                                                  mask=concat_masking_mask)
            datamat = jnp.where(concat_masking_mask,
                                datamat,
                                0)
            del concat_masking_mask
            
            if sow_intermediates:
                label = (f'{self.name}/'+
                         f'final feedforward layer {layer_idx}/'+
                         f'after instance norm')
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
                
            # 3.) dense
            datamat = self.dense_layers[layer_idx](datamat)
            
            if sow_intermediates:
                label = (f'{self.name}/'+
                         f'final feedforward layer {layer_idx}/'+
                         'after dense')
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
            
            # 4.) activation
            datamat = self.act(datamat)
            
            if sow_intermediates:
                label = (f'{self.name}/'+
                         f'final feedforward layer {layer_idx}/'+
                         f'after silu')
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
            
            # 5.) dropout
            datamat = nn.Dropout(rate = self.dropout)(datamat,
                                                deterministic = not training)
            
            if sow_intermediates and (self.dropout != 0):
                label = (f'{self.name}/'+
                         f'final feedforward layer {layer_idx}/'+
                         'after dropout')
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
        
        return datamat, padding_mask
