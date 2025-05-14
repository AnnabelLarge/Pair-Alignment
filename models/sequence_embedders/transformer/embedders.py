#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 17:05:47 2023

@author: annabel_large

ABOUT:
======
The embedding trunk for both ancestor and descendant sequence, using:
    Transformer-based architecture

"""
# general python
import logging
from typing import Any, Callable, Sequence, Union, Tuple
from dataclasses import field

# jax, flax (and wax, oh my!)
from flax import linen as nn
import jax
import jax.numpy as jnp

# custom imports
from models.BaseClasses import SeqEmbBase


class TransfSeqEmb(SeqEmbBase):
    """
    Use a variant of a Transformer:
        - Pre-norm with sinusoidal embedding
        - Pre-norm with rotary embedding
        - TAPE transformer
    
    
    init with:
    ==========
    initial_embed_module (callable): module for initial projection to hidden dim
    first_block_module (callable): first Transf block
    subsequent_block_module (callable): subsequent Transf blocks, if desired
    causal (bool): true if working with the descendant sequence; false otherwise
    config (dict): config to pass to each subsequent module
    name (str): "ANCESTOR EMBEDDER" or "DESCENDANT EMBEDDER"
    
    
    config will have:
    =================
    num_blocks (int): how many transformer blocks to use
    num_heads (int): how many attention heads per block
    hidden_dim (int): length of the embedded vector
    dropout (float = 0.0): dropout rate in transformer block
    padding_idx (int = 0): padding token
    base_alphabet_size (int = 23): <pad>, <bos>, <eos>, then all alphabet 
                                  (20 for amino acids, 4 for DNA)
    
    
    call arguments are:
    ===================
    datamat: matrix of sequences (B, L)
    training: controls dropout behavior
    sow_intermediates: if you want to capture intermediates for debugging
    
    
    outputs:
    ========
    datamat (altered matrix): position-specific encodings for all 
                             sequences (B, L, H)
    
    """
    initial_embed_module: callable
    first_block_module: callable
    subsequent_block_module: callable
    causal: bool
    config: dict
    name: str
    
    def setup(self):
        num_blocks = self.config['num_blocks']
        
        # first module projects (B,L) -> (B,L,H)
        name = f'{self.name} 0/initial embed'
        self.initial_embed = self.initial_embed_module(config = self.config,
                                                       causal = self.causal,
                                                       name = name)
        del name
        
        # second module does the first sequence embedding: (B,L,H) -> (B,L,H)
        name = f'{self.name} 1/Transf Block 0'
        self.first_block = self.first_block_module(config = self.config,
                                                   causal = self.causal,
                                                   name = name)
        del name
        
        # may have additional blocks: (B,L,H) -> (B,L,H)
        subsequent_blocks = []
        for i in range(num_blocks-1):
            layer_idx = i + 2
            block_idx = i + 1
            name = f'{self.name} {layer_idx}/Transf Block {block_idx}'
            l = self.subsequent_block_module(config = self.config,
                                         causal = self.causal,
                                         name = name)
            del name
            subsequent_blocks.append(l)
        self.subsequent_blocks = subsequent_blocks
    
    
    def __call__(self, 
                 datamat, 
                 sow_intermediates: bool, 
                 training: bool):
        ### initial embedding: (B,L) -> (B,L,H)
        datamat, padding_mask = self.initial_embed(datamat, training)
        
        ### first transformer block: (B, L, H) -> (B, L, H)
        datamat = self.first_block(datamat = datamat,
                                   padding_mask = padding_mask,
                                   sow_intermediates = sow_intermediates,
                                   training = training)
        
        if sow_intermediates:
            label = f'{self.name} 1/Transf Block 0/after block'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
        
        
        ### apply successive blocks; these start at layernum=2, Transf Block 1
        # (B, L, H) -> (B, L, H)
        for i,block in enumerate(self.subsequent_blocks):
            layer_idx = i+2
            block_idx = i+1
            datamat = block(datamat = datamat,
                            padding_mask = padding_mask,
                            sow_intermediates = sow_intermediates,
                            training = training)
            
            if sow_intermediates:
                label = (f'{self.name} {layer_idx}/'+
                         f'Transf Block {block_idx}/'+
                         f'after block')
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
        
        # output is (B, L, H)
        return datamat
    
    
    def extract_attn_weights(self, interms_dict):
        """
        comb through attention weights dictionary and simplify it
        """
        attn_weights_dict = dict()
        for layername, layer_dict in interms_dict['intermediates'].items():
            for k, v_dict in layer_dict.items():
                if 'attention_weights' in v_dict.keys():
                    new_key = layername + '/' + k
                    attn_weights_dict[new_key] = v_dict['attention_weights'][0]
        return attn_weights_dict

    
    def apply_seq_embedder_in_eval(self, 
                                   seqs, 
                                   final_trainstate, 
                                   sow_outputs, 
                                   extra_args_for_eval):
        output_attn_weights = extra_args_for_eval.get('output_attn_weights', 
                                                      False)
        
        if output_attn_weights:
            mut_lst = ['intermediates']
        else:
            mut_lst = []
        
        # embed the sequence
        out = final_trainstate.apply_fn(variables = final_trainstate.params,
                                        datamat = seqs,
                                        training = False,
                                        sow_intermediates = sow_outputs,
                                        mutable = mut_lst +['histograms','scalars'] if sow_outputs else mut_lst)
        out_embeddings, out_aux_dict = out
        del out
        
        # pack up all the auxilary data 
        metrics_dict_name = f'{self.embedding_which}_layer_metrics'
        aux_data = {metrics_dict_name: {'histograms': out_aux_dict.get( 'histograms', 
                                                                        dict() ),
                                        'scalars': out_aux_dict.get( 'scalars', 
                                                                     dict() )
                                        }
                    }
        
        # if you ever use batch norm in ancestor sequence embedder, need 
        #  to replace this whole method and extract batch_stats from out_aux_dict
        if self.embedding_which == 'anc':
            aux_data['anc_aux'] = None
        
        # attention weights will be here, if you want to return them
        if output_attn_weights:
            attn_weights_dict_name = f'{self.embedding_which}_attn_weights'
            aux_data[attn_weights_dict_name] = self.extract_attn_weights(out_aux_dict)
        
        return (out_embeddings, aux_data)
    