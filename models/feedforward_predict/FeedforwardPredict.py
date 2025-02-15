#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 02:31:21 2023

@author: annabel_large

ABOUT:
======
Use these projection layers when generating possible descendant + alignment,
  conditioned on an ancestor sequence

full amino acid alphabet is 44 tokens:
  - <pad>: 0
  - <bos>: 1
  - <eos>: 2
  - 20 AAs + match: 3-22
  - 20 AAs + insert: 23 - 42
  - gap: 43


full DNA alphabet would be 12 tokens:
  - <pad>: 0
  - <bos>: 1
  - <eos>: 2
  - 4 nucls + match: 3-6
  - 4 nucls + insert: 7-10
  - gap: 11

  
"""
# general python
import logging
from typing import Any, Callable, Sequence, Union, Tuple
from dataclasses import field

# jumping jax and leaping flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax

from models.model_utils.BaseClasses import ModuleBase


class FeedforwardPredict(ModuleBase):
    """
    process concatenated feature vectors into output alphabet;
    
    repeat [norm -> dense -> activation -> dropout] blocks
      > norm in first block can be different from norm in other blocks
      > norm can be None (in which case, no normalization applied)
      > could have no blocks at all
    
    then, ues final dense layer to project from last block's hidden dimension
      to full_alphabet_size
    """
    config: dict
    name: str
    
    def setup(self):
        ### unpack config
        self.layer_sizes = self.config['layer_sizes']
        self.normalize_inputs = self.config["normalize_inputs"]
        self.dropout = self.config.get("dropout", 0.0)
        
        #!!! hardcode these options for now; change as desired
        self.act_type = 'silu'
        self.act= nn.silu
        self.kernel_init = nn.initializers.lecun_normal()
        self.norm = nn.LayerNorm(reduction_axes = -1, feature_axes = -1)
        self.output_size = self.config.get("full_alphabet_size", 44)
        self.use_bias = True
            
    @nn.compact
    def __call__(self, 
                 datamat_lst, 
                 padding_mask, 
                 training: bool, 
                 sow_intermediates: bool=False,
                 **kwargs):
        ### concatenate embeddings along last axis; order will be: 
        ###   (anc_embs, desc_embs, previous alignment state)
        ###   shape will be: ( B, length_for_scan, 2H (+6, if 
        ###   providing previous state) )
        datamat = jnp.concatenate(datamat_lst, axis=-1)
            
        
        ### mask out padding tokens before passing to any blocks
        new_shape = (padding_mask.shape[0],
                     padding_mask.shape[1],
                     datamat.shape[2])
        masking_mat = jnp.broadcast_to(padding_mask[...,None], new_shape)
        del new_shape
        
        datamat = jnp.multiply(datamat, masking_mat)
        
        
        ### initial norm
        if self.normalize_inputs:
            datamat = self.norm(datamat)
            
            if sow_intermediates:
                label = f'{self.name}/after initial norm'
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
            
        
        ### first (dense -> relu -> dropout)
        if len(self.layer_sizes) > 0:
            # dense
            datamat = nn.Dense(features = self.layer_sizes[0], 
                         use_bias = self.use_bias, 
                         kernel_init = self.kernel_init,
                         name=f'{self.name}/feedforward layer 0')(datamat)
            
            # activation
            datamat = self.act(datamat)
            if sow_intermediates:
                label = (f'{self.name}/'+
                         f'feedforward layer 0/'+
                         f'after {self.act_type}')
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
            
            # dropout
            datamat = nn.Dropout(rate = self.dropout)(datamat,
                                                deterministic = not training)
            if sow_intermediates:
                label = f'{self.name}/feedforward layer 0/after block'
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
        
        
        ### subsequent layers leading up to final projection 
        ###   (norm -> dense -> relu -> dropout)
        for i, hid_dim in enumerate(self.layer_sizes[1:]):
            layer_idx = i + 1
            
            # norm
            datamat = self.norm(datamat)
            
            if sow_intermediates:
                label = (f'{self.name}/'+
                         f'feedforward layer {layer_idx}/'+
                         f'after norm')
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
                
            # dense
            name = f'{self.name}/feedforward layer {layer_idx}'
            datamat = nn.Dense(features = hid_dim, 
                         use_bias = self.use_bias, 
                         kernel_init = self.kernel_init,
                         name=name)(datamat)
            del name
            
            # activation
            datamat = self.act(datamat)
            
            if sow_intermediates:
                label = (f'{self.name}/'+
                         f'feedforward layer {layer_idx}/'+
                         f'after {self.act_type}')
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
                
            
            # dropout
            datamat = nn.Dropout(rate = self.dropout)(datamat,
                                                deterministic = not training)
            if sow_intermediates:
                label = (f'{self.name}/'+
                         f'feedforward layer {layer_idx}/'+
                         f'after block')
                self.sow_histograms_scalars(mat = datamat, 
                                            label = label, 
                                            which=['scalars'])
                del label
          
            
        ### final projection to probabilities
        ###   ( B, length_for_scan, output_size)
        final_logits = nn.Dense(features = self.output_size,
                                use_bias = self.use_bias, 
                                kernel_init = self.kernel_init,
                                name='Project to Logits')(datamat)
        
        return {'FPO_final_logits': final_logits}
    

    def neg_loglike_in_scan_fn(self, 
                               forward_pass_outputs, 
                               true_out,
                               seq_padding_idx: int = 0):
        """
        Cross-entropy loss per position, using a chunk over alignment length
        true_out is (B, L): the alignment-augmented descendant
        
        return sum(-logP)
        """
        ### unpack forward pass outputs 
        ###   here, this only only returns logits: 
        ###   (B, length_for_scan, output_size)
        final_logits = forward_pass_outputs['FPO_final_logits']
        
        
        ### make padding mask
        output_padding_mask = jnp.where(true_out != seq_padding_idx,
                                        True,
                                        False)
        
        
        ### evaluate CrossEnt loss, using the implementation from optax
        # (B, L, full_alphabet_size) -> (B, L)
        CE_loss = optax.softmax_cross_entropy_with_integer_labels(logits = final_logits, 
                                                                  labels = true_out)
        
        
        ### mask out the positions corresponding to padding characters; they
        ###  shouldn't contribute to loss calculations
        #  (B, L)
        neg_logP_perSamp_perPos = jnp.multiply(CE_loss, output_padding_mask)
        
        
        ### output sum
        # (B, L) -> (B,)
        sum_neg_logP = jnp.sum(neg_logP_perSamp_perPos, axis=1)
        intermeds_to_stack = {'neg_logP_perSamp_perPos': neg_logP_perSamp_perPos, #(B, L)
                              }
        return (sum_neg_logP, intermeds_to_stack)
        
    
    def compile_metrics_in_scan(self,
                                forward_pass_outputs, 
                                seq_padding_idx = 0):
        # forward pass only returns logits
        final_logits = forward_pass_outputs['FPO_final_logits']
        del forward_pass_outputs
        
        pred_outputs = jnp.argmax(nn.softmax(final_logits,-1), -1) #(B, L)
        
        metrics_in_scan = {'pred_outputs': pred_outputs}
        return metrics_in_scan
    
    
    def evaluate_loss_after_scan(self, 
                                 scan_fn_outputs,
                                 length_for_normalization,
                                 seq_padding_idx: int = 0,
                                 **kwargs):
        # unpack; remove dummy time from sum_neg_logP
        # (T=1, B) -> (B, )
        sum_neg_logP, scan_intermediates = scan_fn_outputs
        sum_neg_logP = sum_neg_logP[0,:]
        del scan_fn_outputs
        
        # normalize by chosen length
        neg_logP_length_normed = jnp.divide( sum_neg_logP, 
                                             length_for_normalization )
        
        
        # final loss is mean ( (1/L)(-logP) ); one vaue
        loss = jnp.mean(neg_logP_length_normed)
        
        # auxilary dictionary to output
        aux_dict = {'sum_neg_logP': sum_neg_logP,
                    'neg_logP_length_normed': neg_logP_length_normed}
        
        # add scan intermediates, which will be of shape 
        # (num_scan_iters, ..., chunk_len, ...)
        aux_dict = {**aux_dict, **scan_intermediates}
        
        return loss, aux_dict

        
    def compile_metrics(self, 
                        true_out, 
                        loss,
                        loss_fn_dict,
                        seq_padding_idx = 0,
                        out_alph_size = 43):
        """
        metrics include:
            - accuracy + confusion matrix
            - perplexity
            - exponentiated cross entropy
        
        (though, technically accuracy and confusion matrix don't
           make a lot of sense for this problem? Ian didn't like it,
           but keep it anyways)
        
        this is now done OUTSIDE jax.value_and_grad call, but is still 
          jit-compiled
        """
        ##################
        ### adjust sizes #
        ##################
        ### pred_outputs will be (num_scan_iters, B, chunk_len)
        pred_outputs = loss_fn_dict['pred_outputs']
        
        # make this compatible even if you're not scanning
        if len(pred_outputs.shape) == 3:
            # need to transpose and reshape to (B, L) (not ideal because big 
            #   intermediates, but worry about this later)
            num_scan_iters, batch_size, chunk_len = pred_outputs.shape
            seq_len = num_scan_iters * chunk_len
            
            # (B, num_scan_iters, chunk_len)
            pred_outputs = jnp.transpose(pred_outputs, (1,0,2))
            pred_outputs = jnp.reshape(pred_outputs, (batch_size, seq_len))
        
        
        ####################
        ### accuracy block #
        ####################
        # (B,)
        matches_per_samp = jnp.sum( jnp.where( (pred_outputs == true_out) & 
                                                (true_out != seq_padding_idx),
                                              True,
                                              False
                                              ),
                                    axis = 1)
        
        # regardless of how you choose to normalize inputs, accuracy needs
        #   to be divided by alignment length
        align_lens = jnp.where(true_out != seq_padding_idx,
                               1,
                               0).sum(axis=1)
        
        acc_perSamp = jnp.divide( matches_per_samp, align_lens ) #(B,)
        
        
        ### confusion matrix
        def confusion_matrix(true_idx, 
                             pred_idx):
            # true at dim0, pred at dim1
            cm = jnp.zeros( (out_alph_size, out_alph_size) )
            indices = (true_idx, pred_idx)
            cm = cm.at[indices].add(1)
            return cm
            
        # vmap this over the batch
        vmapped_fn = jax.vmap(confusion_matrix,
                              in_axes = (0, 0))
        
        cm_perSamp = vmapped_fn(true_out, 
                                pred_outputs)
        
        
        ########################
        ### Perplexity and ECE #
        ########################
        # perplexity per sample
        neg_logP_length_normed = loss_fn_dict['neg_logP_length_normed']
        perplexity_perSamp = jnp.exp(neg_logP_length_normed) #(B,)
        
        # exponentiated cross entropy
        ece = jnp.exp(loss)
        
        
        ########################
        ### Return all metrics #
        ########################
        out_dict = {'perplexity_perSamp': perplexity_perSamp,
                    'ece': ece,
                    'acc_perSamp': acc_perSamp,
                    'cm_perSamp': cm_perSamp}
        return out_dict
    
    