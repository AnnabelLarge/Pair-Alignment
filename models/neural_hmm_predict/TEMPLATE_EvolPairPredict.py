#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:04:24 2024

@author: annabel

Use sequence embeddings and previous alignment position to (potentially)
calculate:
    - exchangeabilities matrix
    - equilibrium distribution
    - transition probabilities

Then do all the normal HMM math, including:
    - calculating logprob matrices per time t
    - marginalizing over t
    
"""
# jumping jax and leaping flax
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm as matrix_exp
from jax.scipy.special import logsumexp
import optax

from models.modeling_utils.BaseClasses import ModuleBase
from models.EvolPairPredict.scoring_fns import (score_transitions,
                                                score_substitutions,
                                                score_insertions)
from utils.logsumexp_utils import logsumexp_with_padding


class EvolPairPredict(ModuleBase):
    process_embeds_for_exchang_module: callable # concatted feats -> new feats
    exchang_module: callable # new feats -> logits -> params
    
    process_embeds_for_equilibr_module: callable
    equilibr_module: callable
    
    process_embeds_for_indels_module: callable
    indels_module: callable
    
    emit_match_logprobs_module: callable
    emit_ins_logprobs_module: callable
    transits_logprobs_module: callable
    
    config: dict
    name: str
    
    def setup(self):
        ### unpack config
        # for this module
        self.emission_alphabet_size = self.config['emission_alphabet_size']
        self.t_grid_step = self.config.get('t_grid_step', 1)
        self.eos_as_match = self.config['indels_config']['eos_as_match']
        
        # sub-configs for blocks
        exchang_config = self.config['exchang_config']
        equilibr_config = self.config['equilibr_config']
        indels_config = self.config['indels_config']
        
        
        ### parts to generate evolutionary parameters
        #  for exchangeabilities, chi
        self.process_embeds_for_exchang = self.process_embeds_for_exchang_module(config = exchang_config,
                                                                                 name = f'{self.name}/embeds_to_exchageabilities')
        self.get_exchang = self.exchang_module(config = exchang_config,
                                               name = f'{self.name}/get_exchangeabilities')
        
        # for equilibrium distribution
        self.process_embeds_for_equilibr = self.process_embeds_for_equilibr_module(config = equilibr_config,
                                                                                   name = f'{self.name}/embeds_to_equilibriums')
        self.get_equilibr = self.equilibr_module(config = equilibr_config,
                                                 name = f'{self.name}/get_equilibriums')
        
        # for indel parameters
        self.process_embeds_for_indel_params = self.process_embeds_for_indels_module(config = indels_config,
                                                                               name = f'{self.name}/embeds_to_indel_params')
        self.get_indel_params = self.indels_module(config = indels_config,
                                                   name = f'{self.name}/get_indel_params')
        
        
        ### parts to generate logprob transitions, emissions
        # logprob( emissions at match sites )
        joint_config = {**exchang_config, **equilibr_config}
        self.logprob_emit_match_block = self.emit_match_logprobs_module(config = joint_config,
                                                                        name = f'{self.name}/logprob_match_emissions')
        
        # logprob( emissions at insert sites )
        self.logprob_emit_ins_block = self.emit_ins_logprobs_module(config = equilibr_config,
                                                                    name = f'{self.name}/logprob_ins_emissions')
        
        # logprob( transitions )
        self.logprob_trans_block = self.transits_logprobs_module(config = indels_config,
                                                                 name = f'{self.name}/logprob_transits')
        
    def __call__(self, 
                 datamat_lst, 
                 padding_mask, 
                 t_array,
                 training: bool, 
                 sow_intermediates: bool=False,
                 **kwargs):
        """
        hidden reps -> 
            norm -> 
            optional feedforward steps ->
            (exchangeabilities, equilibrium, indel params) ->
            logprob matrices
          
        """
        ######################################################
        ### get evolutionary parameters (or logits for them) #
        ######################################################
        # exchangeabilitity matrices 
        # (B, length_for_scan, H) -> (B, length_for_scan, emission_alphabet_size, emission_alphabet_size), OR
        # return (1, 1, emission_alphabet_size, emission_alphabet_size) matrix for all sites
        exchangeability_matrices = self.generate_evolutionary_parameters(embeds_to_datamat = self.process_embeds_for_exchang,
                                                                         datamat_to_evolparams = self.get_exchang,
                                                                         datamat_lst = datamat_lst,
                                                                         padding_mask = padding_mask,
                                                                         training = training, 
                                                                         sow_intermediates = sow_intermediates)
        
        # logits for equilibrium distribution
        # (B, length_for_scan, H) -> (B, length_for_scan, emission_alphabet_size), OR
        # return (1, 1, emission_alphabet_size) matrix for all sites
        equilibr_logits = self.generate_evolutionary_parameters(embeds_to_datamat = self.process_embeds_for_equilibr,
                                                                datamat_to_evolparams = self.get_equilibr,
                                                                datamat_lst = datamat_lst,
                                                                padding_mask = padding_mask,
                                                                training = training, 
                                                                sow_intermediates = sow_intermediates)
        
        # logits for indel params (indel rate, extension probability)
        # (B, length_for_scan, H) -> (B, length_for_scan, num_indel_params ), OR
        # return  (1, 1, num_indel_params ) matrix for all sites
        # number of parameters are 2 for lambda = mu, 3 otherwise
        indel_param_logits = self.generate_evolutionary_parameters(embeds_to_datamat = self.process_embeds_for_indel_params,
                                                                   datamat_to_evolparams = self.get_indel_params,
                                                                   datamat_lst = datamat_lst,
                                                                   padding_mask = padding_mask,
                                                                   training = training, 
                                                                   sow_intermediates = sow_intermediates)
        
        #############################################
        ### generate probabilites, log probabilites #
        #############################################
        # probability AND logprob(emissions at insert sites)
        # (B, length_for_scan, emission_alphabet_size), OR
        # (1, length_for_scan, emission_alphabet_size)
        prob_emit_ins, logprob_emit_ins = self.logprob_emit_ins_block(equilibr_logits=equilibr_logits,
                                                                      sow_intermediates=sow_intermediates)
        
        # logprob(emissions at match sites)
        # (T, B, length_for_scan, emission_alphabet_size, emission_alphabet_size), OR
        # (T, 1, 1, emission_alphabet_size, emission_alphabet_size)
        T = t_array.shape[0]
        B = datamat_lst[0].shape[0]
        L = datamat_lst[0].shape[1]
        
        
        final_shape = (T,
                       B,
                       L,
                       self.emission_alphabet_size, #alph
                       self.emission_alphabet_size) #alph
        logprob_emit_match, subst_rate_matrix = self.logprob_emit_match_block(final_shape=final_shape,
                                                                              exchangeability_matrices=exchangeability_matrices,
                                                                              equilibr_distrib=prob_emit_ins,
                                                                              t_array=t_array,
                                                                              sow_intermediates=sow_intermediates)
        
        # logprob(transitions)
        # (T, B, length_for_scan, 3, 3), OR
        # (T, 1, 1, 3, 3)
        out = self.logprob_trans_block(indel_param_logits=indel_param_logits,
                                                    t_array=t_array,
                                                    sow_intermediates=sow_intermediates)
        logprob_transits, transits_intermeds_dict = out
        del out

        out_dict = {'FPO_logprob_emit_match':logprob_emit_match, 
                    'FPO_logprob_emit_ins':logprob_emit_ins, 
                    'FPO_logprob_transits':logprob_transits, 
                    'FPO_exchangeabilities':exchangeability_matrices,
                    'FPO_subst_rate_matrix':subst_rate_matrix,
                    'eos_as_match': self.eos_as_match, #kind of a bad hack, but I need this later and Flax won't let me retrieve it outside of apply()
                    't_grid_step': self.t_grid_step} #kind of a bad hack, but I need this later and Flax won't let me retrieve it outside of apply()
        
        for key, val in transits_intermeds_dict.items():
            out_dict[key] = val
        return out_dict
    
    
    def get_length_for_normalization(self,
                                     true_out,
                                     norm_loss_by,
                                     seq_padding_idx = 0,
                                     gap_tok = 43):
        """
        using true_out, figure out the length to normalize by
          - true_out will be (B, L, 2)
          - this won't include <bos>
        """
        length = jnp.where(true_out != seq_padding_idx, 
                           True, 
                           False).sum(axis=1)[:,0]
        
        if norm_loss_by == 'desc_len':
            num_gaps = jnp.where(true_out[:,:,1] == gap_tok, 
                                 True, 
                                 False).sum(axis=1)
            length = length - num_gaps
        
        return length
    
    
    def neg_loglike_in_scan_fn(self, 
                              forward_pass_outputs, 
                              true_out,
                              alignment_state,
                              seq_padding_idx = 0,
                              **kwargs):
        """
        loss of ONE alignment path, given by alignment_state
        
        maybe need a different function for Forward algorithm?
        """
        ### unpack inputs
        logprob_emit_match = forward_pass_outputs['FPO_logprob_emit_match']
        logprob_emit_ins = forward_pass_outputs['FPO_logprob_emit_ins']
        logprob_transits = forward_pass_outputs['FPO_logprob_transits']
        
        # hacked my way into getting this value (see above)
        eos_as_match = forward_pass_outputs['eos_as_match']
        
        
        ### score transitions: (T, B, length_for_scan)
        # all positions involving padding tokens will have 
        #   transitions_scores = 0
        # alignment_state is (B, length_for_scan, 2), where-
        # (dim2=0): previous position's state
        # (dim2=1): current position's state (current position will never be <bos>)
        transitions_scores = score_transitions(alignment_state = alignment_state,
                                               trans_mat = logprob_transits, 
                                               eos_as_match = eos_as_match, 
                                               padding_idx=seq_padding_idx)
        
        
        # ### score emissions (T, B, length_for_scan)
        # # three cases: M, I, and D (which has no emission score)
        # # first: is the position a match or insert?
        # # second: decide between the two functions to run for match and insert
        emissions_scores = jnp.where( (alignment_state[:, :, 1] == 3) |  (alignment_state[:, :, 1] == 4),
                                     
                                      jnp.where(alignment_state[:, :, 1] == 3,
                                                
                                                # if match, use substitution_scoring
                                                score_substitutions(true_out = true_out,
                                                                    subs_mat = logprob_emit_match,
                                                                    token_offset = 3),
                                                
                                                # if insert, use insertions_scoring
                                                score_insertions(true_out = true_out,
                                                                 ins_vec = logprob_emit_ins,
                                                                 token_offset = 3)
                                                
                                                ),
                                        
                                      # otherwise, position does not have emissions logprob
                                      0)
        
        ### final logprob(sequences)
        # add logprobs together: (T, B, L)
        logprob_perSamp_perPos_perTime = transitions_scores + emissions_scores
        
        # take the SUM down the length to get logprob_perSamp_perTime; padding
        #   tokens should have score of 0; don't normalize by length yet: (T, B)
        logprob_perSamp_perTime = logprob_perSamp_perPos_perTime.sum(axis=2)
        
        
        # intermediates to return
        intermeds = {'t_grid_step': forward_pass_outputs['t_grid_step']}
        return logprob_perSamp_perTime, intermeds
        
    
    
    def compile_metrics_in_scan(self,
                                forward_pass_outputs, 
                                true_out, 
                                seq_padding_idx = 0,
                                **kwargs):
        """
        no metrics that depend on forward pass outputs
        """
        return dict()
    
    
    def evaluate_loss_after_scan(self, 
                                 scan_fn_outputs,
                                 t_array,
                                 length_for_normalization,
                                 seq_padding_idx = 0,
                                 **kwargs):
        # unpack
        logprob_perSamp_perTime, scan_intermeds = scan_fn_outputs
        
        
        ### if using multiple timepoints, need to logsumexp across them
        if t_array.shape[0] != 1:
            t_grid_step = scan_intermeds['t_grid_step'][0]
            del scan_fn_outputs, scan_intermeds
            
            # logsumexp with marginalization constant
            #   t_array is (T, B)
            #   marg_const_perTime is (T,B)
            #   logprob_perSamp_perTime is (T,B)
            #   logP_perSamp_perTime_withConst will be (T, B)
            #   after marginalization, logP_perSamp_raw will be (B, )
            marg_const_perTime = jnp.log(t_array) - ( t_array / (t_grid_step-1) ) 
            logP_perSamp_perTime_withConst = ( logprob_perSamp_perTime + 
                                                marg_const_perTime )
            logP_perSamp_raw = logsumexp(logP_perSamp_perTime_withConst, axis=0)
        
        # otherwise, just take the first timepoint result
        else:
            logP_perSamp_raw = logprob_perSamp_perTime[0,:] 
            
            
        # normalize by alignment length after all of this: (B,)
        logprob_perSamp = jnp.divide( logP_perSamp_raw, 
                                      length_for_normalization )
        
        
        ### final loss is -average across all samples; one value
        loss = -jnp.mean(logprob_perSamp)
        
        # names are weird, but this is leftover from FeedforwardPredict
        intermediate_vals = { 'sum_neg_logP': -logP_perSamp_raw,
                              'neg_logP_length_normed': -logprob_perSamp}
        
        return loss, intermediate_vals
        
    
    
    def compile_metrics(self,  
                        loss,
                        loss_fn_dict,
                        seq_padding_idx = 0,
                        **kwargs):
        # perplexity per sample
        neg_logP_length_normed = loss_fn_dict['neg_logP_length_normed']
        perplexity_perSamp = jnp.exp(neg_logP_length_normed) #(B,)
        
        # exponentiated cross entropy
        ece = jnp.exp(loss)
        
        # Return all metrics
        out_dict = {'perplexity_perSamp': perplexity_perSamp,
                    'ece': ece}
        
        return out_dict
    
    
    def generate_evolutionary_parameters(self, 
                                          embeds_to_datamat,
                                          datamat_to_evolparams,
                                          datamat_lst,
                                          padding_mask,
                                          training: bool, 
                                          sow_intermediates: bool=False):
        # hidden reps -> datamat
        datamat, padding_mask = embeds_to_datamat(datamat_lst = datamat_lst,
                                                  padding_mask = padding_mask,
                                                  training = training,
                                                  sow_intermediates = sow_intermediates)
        
        # datamat -> evoparam logits
        logits = datamat_to_evolparams(datamat = datamat,
                                       padding_mask = padding_mask,
                                       training = training,
                                       sow_intermediates = sow_intermediates)
        
        return logits
    
    
    # def encode_states(self, 
    #                   arr):
    #     """
    #     *** HARD CODED FOR PROTEINS!!! ***
        
    #     pad: 0
    #     <bos>: 1 [START state in a pairHMM]
    #     <eos>: 2 [END state in a pairHMM]
    #     match: 3
    #     insert: 4
    #     delete: 5
    #     """
    #     bos_idx = 1
    #     eos_idx = 2
    #     gap_idx = 43
    #     seq_padding_idx = 0
        
        
    #     # find bos and eos positions; (B, L)
    #     bos = jnp.where(arr == bos_idx, 1, 0)[:,:,0]
    #     eos = jnp.where(arr == eos_idx, 2, 0)[:,:,0]
        
    #     # find match pos; (B, L)
    #     tmp = jnp.where( (arr >= 3) & (arr <= 22), 1, 0 ).sum(axis=2) 
    #     matches = jnp.where(tmp == 2, 3, 0)
        
    #     # find ins pos i.e. where ancestor is gap; (B,L)
    #     ins = jnp.where(arr[:,:,0] == gap_idx, 4, 0)
        
    #     # find del pos i.e. where descendant is gap; (B,L)
    #     dels = jnp.where(arr[:,:,1] == gap_idx, 5, 0)
        
    #     # categorical encoding
    #     alignment = bos + eos + matches + ins + dels
        
    #     return alignment
    
  