#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 17:05:47 2023

@author: annabel_large

ABOUT:
======
The embedding trunk for both ancestor and descendant sequence, using:
    LSTM
    
"""
from typing import Callable

from flax import linen as nn
import jax
import jax.numpy as jnp

from models.BaseClasses import SeqEmbBase


class LSTMSeqEmb(SeqEmbBase):
    """
    init with:
    ==========
    initial_embed_module (callable): module for initial projection to hidden dim
    first_block_module (callable): first LSTM block
    subsequent_block_module (callable): subsequent LSTM blocks, if desired
    config (dict): config to pass to each subsequent module
    name (str): "ANCESTOR EMBEDDER" or "DESCENDANT EMBEDDER"
    
    
    config will have:
    =================
    hidden_dim (int): length of the embedded vector
    
    padding_idx (int = 0): padding token
    base_alphabet_size (int = 23): <pad>, <bos>, <eos>, then all alphabet 
                                  (20 for amino acids, 4 for DNA)
    dropout (float = 0.0): dropout rate
    
    
    call arguments are:
    ===================
    datamat: matrix of sequences (B, L)
    training: controls behavior of intermediate dropout layers
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
        # !!! hard-code this
        self.return_final_carry = False
        
        
        ### unpack config
        n_layers = self.config["n_layers"]
        self.padding_idx = self.config.get("seq_padding_idx", 0)
        
        
        ### setup layers
        # first module projects (B,L) -> (B,L,H)
        name = f'{self.name} 0/initial embed'
        self.initial_embed = self.initial_embed_module(config = self.config,
                                                  causal = self.causal,
                                                  name = name)
        del name
        
        # second module does the first sequence embedding: (B,L,H) -> (B,L,H)
        # note: LSTM modules don't take "causal" argument
        name = f'{self.name} 1/LSTM Block 0'
        self.first_block = self.first_block_module(config = self.config,
                                              name = name)
        del name
        
        # may have additional blocks: (B,L,H) -> (B,L,H)
        subsequent_blocks = []
        for i in range(n_layers-1):
            layer_idx = i + 2
            block_idx = i + 1
            name = f'{self.name} {layer_idx}/LSTM Block {block_idx}'
            l = self.subsequent_block_module(config = self.config,
                                         name = name)
            subsequent_blocks.append(l)
        self.subsequent_blocks = subsequent_blocks
    
    
    def __call__(self, 
                 datamat, 
                 sow_intermediates: bool, 
                 training: bool):
        ### get the sequence lengths in this batch
        # (B,L) -> (B,)
        datalens = jnp.where( datamat != self.padding_idx, 1, 0 ).sum( axis=1 )
        
        
        ### initial embedding: (B,L) -> (B,L,H)
        datamat, padding_mask = self.initial_embed(datamat)
        
        
        ### first LSTM: (B, L, H) -> (B, L, H)
        out_carry, datamat = self.first_block(datamat = datamat,
                                              datalens = datalens,
                                              training = training,
                                              carry = None)
        
        # sow values (as long as this isn't the last block)
        if sow_intermediates and len(self.subsequent_blocks) > 0:
            label = f'{self.name} 1/LSTM Block 0/after block'
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
            
            self.write_carry_wrapper(causal = self.causal, 
                                     carry_tuple = out_carry,
                                     prefix = f'{self.name} 1/LSTM Block 0')
        
        
        ### apply successive blocks; these start at layernum=2, LSTM Block 1
        # (B, L, H) -> (B, L, H)
        for i,block in enumerate(self.subsequent_blocks):
            layer_idx = i+2
            block_idx = i+1
            out_carry, datamat = block(datamat = datamat,
                                       datalens = datalens,
                                       training = training,
                                       carry = None)
            
            # sow values
            if sow_intermediates:
                label = (f'{self.name} {layer_idx}/'+
                         f'LSTM Block {block_idx}/'+
                         f'after block')
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
                
                prefix = f'{self.name} {layer_idx}/LSTM Block {block_idx}'
                self.write_carry_wrapper(causal = self.causal, 
                                         carry_tuple = out_carry,
                                         prefix = prefix)
                del prefix
                
                    
        ### return the carry from the final LSTM layer, if you want
        if self.return_final_carry:
            return (out_carry, datamat)
        
        else:
            return (None, datamat)
    
    
    def write_carry(self, 
                    carry_tuple, 
                    prefix, 
                    layer_name):
        cell_state, hidden_state = carry_tuple
        
        self.sow_histograms_scalars(mat = cell_state, 
                                    label = f'{prefix}/{layer_name} cell state', 
                                    which=['scalars'])
        
        self.sow_histograms_scalars(mat = hidden_state, 
                                    label = f'{prefix}/{layer_name} hidden state', 
                                    which=['scalars'])
    
    
    def write_carry_wrapper(self, 
                            causal, 
                            carry_tuple, 
                            prefix):
        """
        helper to sow the carry for LSTMs
        
        carry is ( c, f ) if uni-directional
        carry is ( (c_fw, h_fw),  (c_rv, h_rv) ) if bidirectional
        """
        if causal:
            fw_out_carry, rv_out_carry = out_carry
            
            self.write_carry(carry_tuple = fw_out_carry, 
                             prefix = prefix,
                             layer_name = f'FORW')
            
            self.write_carry(carry_tuple = rv_out_carry, 
                             prefix = prefix,
                             layer_name = f'REV')
        
        elif not causal:
            self.write_carry(carry_tuple = out_carry,
                             prefix = prefix,
                             layer_name = '')

    def apply_seq_embedder_in_training(self, **kwargs):
        # unpack kwargs
        seqs = kwargs['seqs']
        rng_key = kwargs['rng_key']
        params_for_apply = kwargs['params_for_apply']
        seq_emb_trainstate = kwargs['seq_emb_trainstate']
        sow_outputs = kwargs['sow_outputs']
        
        # embed the sequence
        (out_carry, out_embeddings), out_aux_dict = seq_emb_trainstate.apply_fn(variables = params_for_apply,
                                                                   datamat = seqs,
                                                                   training = True,
                                                                   sow_intermediates = sow_outputs,
                                                                   mutable = ['histograms','scalars'] if sow_outputs else [],
                                                                   rngs={'dropout': rng_key})
        
        # pack up all the auxilary data 
        metrics_dict_name = f'{self.embedding_which}_layer_metrics'
        carry_name = f'{self.embedding_which}_carry'
        aux_data = {metrics_dict_name: {'histograms': out_aux_dict.get( 'histograms', dict() ),
                                        'scalars': out_aux_dict.get( 'scalars', dict() )
                                        },
                    carry_name: out_carry
                    }
        
        # if you ever use batch norm in ancestor sequence embedder, need 
        #  to replace this whole method and extract batch_stats from out_aux_dict
        if self.embedding_which == 'anc':
            aux_data['anc_aux'] = None
        
        return (out_embeddings, aux_data)
    
    def apply_seq_embedder_in_eval(self,
                                   seqs,
                                   final_trainstate,
                                   sow_outputs,
                                   **kwargs):
        # embed the ancestor seq
        (out_carry, out_embeddings), out_aux_dict = final_trainstate.apply_fn(variables = final_trainstate.params,
                                                                 datamat = seqs,
                                                                 training = False,
                                                                 sow_intermediates = sow_outputs,
                                                                 mutable = ['histograms','scalars'] if sow_outputs else [])
        
        # pack up all the auxilary data 
        metrics_dict_name = f'{self.embedding_which}_layer_metrics'
        carry_name = f'{self.embedding_which}_carry'
        aux_data = {metrics_dict_name: {'histograms': out_aux_dict.get( 'histograms', dict() ),
                                        'scalars': out_aux_dict.get( 'scalars', dict() )
                                        },
                    carry_name: out_carry
                    }
        
        # if you ever use batch norm in ancestor sequence embedder, need 
        #  to replace this whole method and extract batch_stats from out_aux_dict
        if self.embedding_which == 'anc':
            aux_data['anc_aux'] = None
        
        return (out_embeddings, aux_data)