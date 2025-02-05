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
# jumping jax and leaping flax
from flax import linen as nn
import jax
import jax.numpy as jnp

from models.model_utils.BaseClasses import ModuleBase


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
        return (None, padding_mask)
    
class SelectMask(ModuleBase):
    config: dict
    name: str
    
    @nn.compact
    def __call__(self,
                 datamat_lst: list,
                 padding_mask: jnp.array,
                 **kwargs):
        # unpack config file
        use_anc_emb = self.config['use_anc_emb']
        use_desc_emb = self.config['use_desc_emb']
        
        # select (potentially concat) and mask padding
        datamat, padding_mask = self.process_datamat_lst(datamat_lst,
                                                        padding_mask,
                                                        use_anc_emb,
                                                        use_desc_emb)
        
        return datamat, padding_mask
        
    
    def process_datamat_lst(self,
                        datamat_lst: list,
                        padding_mask: jnp.array,
                        use_anc_emb: bool,
                        use_desc_emb: bool):
        """
        select which embedding, then mask out padding tokens
        """
        if use_anc_emb and use_desc_emb:
            datamat = jnp.concatenate( datamat_lst, axis = -1 )
        
        elif use_anc_emb and not use_desc_emb:
            datamat = datamat_lst[0]
        
        elif not use_anc_emb and use_desc_emb:
            datamat = datamat_lst[1]
        
        new_shape = (padding_mask.shape[0],
                     padding_mask.shape[1],
                     datamat.shape[2])
        masking_mat = jnp.broadcast_to(padding_mask[:,:,None], new_shape)
        del new_shape
        
        datamat = jnp.multiply(datamat, masking_mat)
        return datamat, masking_mat
    


class FeedforwardToEvoparams(SelectMask):
    """
    inherit process_datamat_lst() from SelectMask
    
    apply this blocks as many times as specified by layer_sizes: 
        [norm -> dense -> activation -> dropout]
    
    """
    config: dict
    name: str
    
    def setup(self):
        # !!! hard set
        self.norm_type = 'layer'
        self.norm = nn.LayerNorm( reduction_axes= -1, 
                                  feature_axes=-1 )
        self.act_type = 'silu'
        self.act= nn.silu
        self.kernel_init = nn.initializers.lecun_normal()
        
        
        # required
        self.use_anc_emb = self.config['use_anc_emb']
        self.use_desc_emb = self.config['use_desc_emb']
        self.layer_sizes = self.config['layer_sizes']
        
        # optional
        self.dropout = self.config.get("dropout", 0.0)
        
        
    @nn.compact
    def __call__(self, 
                 datamat_lst: list,
                 padding_mask: jnp.array,
                 training: bool, 
                 sow_intermediates: bool=False):
        
        ### select (potentially concat) and mask padding
        datamat, concat_masking_mat = self.process_datamat_lst(datamat_lst,
                                                               padding_mask,
                                                               self.use_anc_emb,
                                                               self.use_desc_emb)
        del datamat_lst, padding_mask
        
        
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
            datamat = self.norm(datamat, mask=concat_masking_mat)
            del concat_masking_mat
            
            if sow_intermediates:
                label = (f'{self.name}/'+
                         f'final feedforward layer {layer_idx}/'+
                         f'after {self.norm_type}')
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
                
            # 3.) dense
            datamat = nn.Dense(features = hid_dim, 
                         use_bias = True, 
                         kernel_init = self.kernel_init,
                         name=f'{self.name}/feedforward layer {layer_idx}')(datamat)
            
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
                         f'after {self.act_type}')
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
        
        return datamat, concat_masking_mat
    

class ConvToEvoparams(SelectMask):
    """
    (not used yet)
    
    inherit process_datamat_lst() from SelectMask
    
    apply this blocks as many times as specified by layer_sizes: 
        [norm -> conv -> activation -> dropout]
    
    """
    config: dict
    name: str
    
    def setup(self):
        # !!! hard set
        self.norm_type = 'layer'
        self.norm = nn.LayerNorm( reduction_axes= -1, 
                                  feature_axes=-1 )
        self.act_type = 'silu'
        self.act= nn.silu
        self.kernel_init = nn.initializers.lecun_normal()
        
        
        # required
        self.use_anc_emb = self.config['use_anc_emb']
        self.use_desc_emb = self.config['use_desc_emb']
        self.hidden_size_lst = self.config['hidden_size_lst']
        self.kern_size_lst = self.config['kern_size_lst']
        
        assert len(self.hidden_size_lst) == len(self.kern_size_lst)
        
        # optional
        self.dropout = self.config.get("dropout", 0.0)
        
        
    @nn.compact
    def __call__(self, 
                 datamat_lst: list,
                 padding_mask: jnp.array,
                 training: bool, 
                 sow_intermediates: bool=False):
        
        ### select (potentially concat) and mask padding
        datamat, concat_masking_mat = self.process_datamat_lst(datamat_lst,
                                                               padding_mask,
                                                               self.use_anc_emb,
                                                               self.use_desc_emb)
        del datamat_lst, padding_mask
        
        
        ### repeat: norm -> conv -> activation -> dropout
        for layer_idx in range( len(self.kern_size_lst) ):
            # 1.) record distribution into block
            if sow_intermediates:
                label = (f'{self.name}/'+
                         f'final conv layer {layer_idx}/'+
                         f'before block')
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
                
            # 2.) norm (plus recording to tensorboard)
            datamat = self.norm(datamat, mask=concat_masking_mat)
            
            if sow_intermediates:
                label = (f'{self.name}/'+
                         f'final conv layer {layer_idx}/'+
                         f'after {self.norm_type}')
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
                    
            # 3.) convolution+masking, record to tensorboard
            kernel_size = self.kern_size_lst[layer_idx]
            hidden_dim = self.hidden_size_lst[layer_idx]
            
            datamat = nn.Conv(features = hidden_dim, 
                              kernel_size = kernel_size,
                              kernel_init = self.kernel_init,
                              padding='CAUSAL',
                              name=f'{self.name}/final conv layer {layer_idx}')(datamat)
            datamat = jnp.multiply(datamat, concat_masking_mat)
            
            if sow_intermediates:
                label = (f'{self.name}/'+
                         f'final conv layer {layer_idx}/'+
                         f'after conv')
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
            
            # 4.) optional activation (plus recording to tensorboard)
            datamat = self.act(datamat)
            
            if sow_intermediates:
                label = (f'{self.name}/'+
                         'final conv layer {layer_idx}/'+
                         'after {self.act_type}')
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
            
            # 5.) dropout (plus recording to tensorboard)
            datamat = nn.Dropout(rate = self.dropout)(datamat,
                                                      deterministic = not training)
            
            if sow_intermediates and (self.dropout != 0):
                label = (f'{self.name}/'+
                         'final conv layer {layer_idx}/'+
                         'after block')
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
        
        return datamat, concat_masking_mat
    