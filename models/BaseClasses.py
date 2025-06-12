#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:52:59 2024

@author: annabel_large


about:
=======

ModuleBase: gives each model the sow_histograms_scalars and summary_stats 
            helpers, for tensorboard writing

neuralTKFModuleBase: adds functions for automatically applying key 
                     activations: bound_sigmoid and log_softmax

SeqEmbBase: inherits ModuleBase and adds extra helpers for sequence embedding
            applying encoder and decoder in training/eval; the following 
            models will need newer versions (and why):
                - LSTM (uses "datalens" in argument list)
                - Transformer (handle "output attn weights" argument)
                - if you ever want to implement BatchNorm, rage quit and 
                  migrate to flax.NNX

"""
from typing import Callable

from flax import linen as nn
import jax
import jax.numpy as jnp
import optax

from models.neural_hmm_predict.model_functions import bound_sigmoid


class ModuleBase(nn.Module):
    def sow_histograms_scalars(self, mat, label, which=['hists','scalars']):
        """
        helper to sow intermediate values
        """
        if 'hists' in which:
            # full histograms
            self.sow("histograms",
                     label,
                     mat,
                     reduce_fn = lambda a, b: b)
        
        if 'scalars' in which:
            # scalars
            out_dict = self.summary_stats(mat=mat, key_prefix=label)
            for key, val in out_dict.items():
                self.sow("scalars",
                         key,
                         val,
                         reduce_fn = lambda a, b: b)
    
    def summary_stats(self, mat, key_prefix):
        if mat.size == 1:
            out_dict = {f'{key_prefix}': jnp.squeeze(mat)}
        
        else:
            perc_zeros = (mat==0).sum() / mat.size
            mean_without_zeros = mat.sum() / (mat!=0).sum()
            
            out_dict = {f'{key_prefix}/MAX': mat.max(),
                        f'{key_prefix}/MIN': mat.min(),
                        f'{key_prefix}/MEAN': mat.mean(),
                        f'{key_prefix}/VAR': mat.var(),
                        f'{key_prefix}/MEAN-WITHOUT-ZEROS': mean_without_zeros,
                        f'{key_prefix}/PERC-ZEROS': perc_zeros}
        
        return out_dict
    

class neuralTKFModuleBase(ModuleBase):
    def _maybe_sow(self,
                   vals,
                   lab,
                   sow_intermediates):
        if sow_intermediates:
            self.sow_histograms_scalars(mat= vals, 
                                        label=lab, 
                                        which='scalars')
            del lab
    
    def apply_bound_sigmoid_activation(self,
                                       logits,
                                       min_val,
                                       max_val,
                                       param_name,
                                       sow_intermediates):
        # record 
        self._maybe_sow( vals=logits,
                         lab=f'{self.name}/logits for {param_name}',
                         sow_intermediates = sow_intermediates )
        
        # get parameters
        out = bound_sigmoid(logits, min_val, max_val) 
        
        # record again
        self._maybe_sow( vals=out,
                         lab=f'{self.name}/{param_name}',
                         sow_intermediates = sow_intermediates )
        
        return out 
    
    def apply_log_softmax_activation(self,
                                     logits,
                                     param_name,
                                     sow_intermediates: bool):
        # record
        self._maybe_sow( vals=logits,
                         lab=f'{self.name}/logits for {param_name}',
                         sow_intermediates = sow_intermediates )
                        
        # get params
        out = nn.log_softmax( logits, axis = -1 )

        # record again
        self._maybe_sow( vals=out,
                         lab=f'{self.name}/{param_name}',
                         sow_intermediates = sow_intermediates )
        
        return out


class SeqEmbBase(ModuleBase):
    def apply_seq_embedder_in_training(self,
                                       seqs,
                                       rng_key,
                                       params_for_apply,
                                       seq_emb_trainstate,
                                       sow_outputs):
        # embed the sequence
        out_embeddings, out_aux_dict = seq_emb_trainstate.apply_fn(variables = params_for_apply,
                                                                   datamat = seqs,
                                                                   training = True,
                                                                   sow_intermediates = sow_outputs,
                                                                   mutable = ['histograms','scalars'] if sow_outputs else [],
                                                                   rngs={'dropout': rng_key})
        
        # pack up all the auxilary data
        metrics_dict_name = f'{self.embedding_which}_layer_metrics' 
        aux_data = {metrics_dict_name: {'histograms': out_aux_dict.get( 'histograms', dict() ),
                                        'scalars': out_aux_dict.get( 'scalars', dict() )
                                        }
                    }
        
        # if you ever use batch norm in ancestor sequence embedder, need 
        #  to replace this whole method and extract batch_stats from out_aux_dict
        if self.embedding_which == 'anc':
            aux_data['anc_aux'] = None
        
        return (out_embeddings, aux_data)
    
    
    def update_seq_embedder_tstate(self, 
                                   tstate,
                                   new_opt_state,
                                   optim_updates):
        """
        If you apply batch norm ever, you'll need a new one of these
        """
        new_params = optax.apply_updates(tstate.params, 
                                         optim_updates)
        
        
        new_tstate = tstate.replace(params = new_params,
                                    opt_state = new_opt_state)
        
        return new_tstate
    
    
    
    def apply_seq_embedder_in_eval(self,
                                   seqs,
                                   final_trainstate,
                                   sow_outputs,
                                   **kwargs):
        # embed the sequence
        out_embeddings, out_aux_dict = final_trainstate.apply_fn(variables = final_trainstate.params,
                                                                 datamat = seqs,
                                                                 training = False,
                                                                 sow_intermediates = sow_outputs,
                                                                 mutable = ['histograms','scalars'] if sow_outputs else [])
        
        # pack up all the auxilary data 
        metrics_dict_name = f'{self.embedding_which}_layer_metrics'
        aux_data = {metrics_dict_name: {'histograms': out_aux_dict.get( 'histograms', dict() ),
                                        'scalars': out_aux_dict.get( 'scalars', dict() )
                                        }
                    }
        
        # if you ever use batch norm in ancestor sequence embedder, need 
        #  to replace this whole method and extract batch_stats from out_aux_dict
        if self.embedding_which == 'anc':
            aux_data['anc_aux'] = None
        
        return (out_embeddings, aux_data)
