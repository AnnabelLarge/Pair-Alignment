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

from models.model_utils.BaseClasses import ModuleBase
from models.neural_hmm_predict.model_parts.scoring_fns import (score_transitions,
                                                               score_substitutions,
                                                               score_indels)
from utils.logsumexp_utils import logsumexp_with_padding


class NeuralHmmBase(ModuleBase):
    process_embeds_for_exchang_module: callable # concatted feats -> new feats
    exchang_module: callable # new feats -> logits -> params
    
    process_embeds_for_equilibr_module: callable
    equilibr_module: callable
    
    process_embeds_for_lam_mu_module: callable
    lam_mu_module: callable
    
    process_embeds_for_r_module: callable
    r_extend_module: callable
    
    emit_match_logprobs_module: callable
    transits_logprobs_module: callable
    
    config: dict
    name: str
    
    def setup(self):
        ### unpack config
        # for this module
        self.emission_alphabet_size = self.config['emission_alphabet_size']
        
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
        
        # for equilibrium distribution (also technically produces logprob)
        self.process_embeds_for_equilibr = self.process_embeds_for_equilibr_module(config = equilibr_config,
                                                                                   name = f'{self.name}/embeds_to_equilibriums')
        self.get_equilibr = self.equilibr_module(config = equilibr_config,
                                                 name = f'{self.name}/get_equilibriums')
        
        # for lambda, mu
        self.process_embeds_for_lam_mu = self.process_embeds_for_lam_mu_module(config = indels_config,
                                                                               name = f'{self.name}/embeds_to_tkf_lam_mu')
        self.get_lam_mu = self.lam_mu_module(config = indels_config,
                                                   name = f'{self.name}/get_lam_mu')
        
        # for r (extension probability in TKF92)
        self.process_embeds_for_r = self.process_embeds_for_r_module(config = indels_config,
                                                                     name = f'{self.name}/embeds_to_tkf_ext_prob')
        self.get_r_extend = self.r_extend_module(config = indels_config,
                                                   name = f'{self.name}/get_r')
        
        
        ### parts to generate logprob transitions, emissions
        # logprob( emissions at match sites )
        joint_config = {**exchang_config, **equilibr_config}
        self.logprob_emit_match_block = self.emit_match_logprobs_module(config = joint_config,
                                                                        name = f'{self.name}/logprob_match_emissions')
        
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
        ###################################
        ### get evolutionary parameters   #
        ###################################
        # exchangeabilitity matrices 
        # (B, length_for_scan, H) -> (B, length_for_scan, emission_alphabet_size, emission_alphabet_size), OR
        # return (1, 1, emission_alphabet_size, emission_alphabet_size) matrix for all sites
        exchangeability_matrices = self.generate_evolutionary_parameters( embeds_to_datamat = self.process_embeds_for_exchang,
                                                                          datamat_to_evolparams = self.get_exchang,
                                                                          datamat_lst = datamat_lst,
                                                                          padding_mask = padding_mask,
                                                                          training = training, 
                                                                          sow_intermediates = sow_intermediates )
        
        # logits for indel rates (lambda, mu)
        # (B, length_for_scan, H) -> (B, length_for_scan, num_indel_params ), OR
        # return  (1, 1, 2 ) matrix for all sites
        out = self.generate_evolutionary_parameters( embeds_to_datamat = self.process_embeds_for_lam_mu,
                                                     datamat_to_evolparams = self.get_lam_mu,
                                                     datamat_lst = datamat_lst,
                                                     padding_mask = padding_mask,
                                                     training = training, 
                                                     sow_intermediates = sow_intermediates )
        if len(out) == 1:
            lam_mu = out
            use_approx = False
        else:
            lam_mu, use_approx = out
        
        # logits for extension probability (r)
        # (B, length_for_scan, H) -> (B, length_for_scan ), OR
        # return (1, 1 ) matrix for all sites, OR
        # return None (for TKF91)
        r_extend = self.generate_evolutionary_parameters( embeds_to_datamat = self.process_embeds_for_r,
                                                          datamat_to_evolparams = self.get_r_extend,
                                                          datamat_lst = datamat_lst,
                                                          padding_mask = padding_mask,
                                                          training = training, 
                                                          sow_intermediates = sow_intermediates )
        
        
        
        #############################################
        ### generate probabilites, log probabilites #
        #############################################
        # logprob(emissions at insert sites)
        # (B, length_for_scan, H) -> (B, length_for_scan, emission_alphabet_size), OR
        # return (1, 1, emission_alphabet_size) matrix for all sites
        logprob_emit_indel = self.generate_evolutionary_parameters(embeds_to_datamat = self.process_embeds_for_equilibr,
                                                                   datamat_to_evolparams = self.get_equilibr,
                                                                   datamat_lst = datamat_lst,
                                                                   padding_mask = padding_mask,
                                                                   training = training, 
                                                                   sow_intermediates = sow_intermediates)
        
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
        out = self.logprob_emit_match_block(final_shape=final_shape,
                                            exchangeability_matrices=exchangeability_matrices,
                                            log_equilibr_distrib=logprob_emit_indel,
                                            t_array=t_array,
                                            sow_intermediates=sow_intermediates)
        logprob_emit_match, subst_rate_matrix = out
        del out
        
        # logprob(transitions)
        # (T, B, length_for_scan, 4, 4), OR
        # (T, 1, 1, 4, 4)
        out = self.logprob_trans_block(lam_mu = lam_mu,
                                       r_extend = r_extend,
                                       use_approx = use_approx,
                                       t_array = t_array,
                                       sow_intermediates = sow_intermediates)
        logprob_transits, transits_intermeds_dict = out
        del out
        
        out_dict = {'FPO_logprob_emit_match':logprob_emit_match, 
                    'FPO_logprob_emit_indel':logprob_emit_indel, 
                    'FPO_logprob_transits':logprob_transits, 
                    'FPO_exchangeabilities':exchangeability_matrices,
                    'FPO_subst_rate_matrix':subst_rate_matrix} 
        
        for key, val in transits_intermeds_dict.items():
            out_dict[key] = val

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
        out = datamat_to_evolparams(datamat = datamat,
                                       padding_mask = padding_mask,
                                       training = training,
                                       sow_intermediates = sow_intermediates)
        
        return out
    
    
    ###########################################################################
    ### evaluating parts of the loss in the scan chunk   ######################
    ###########################################################################
    def neg_loglike_in_scan_fn(self, 
                              forward_pass_outputs, 
                              true_out,
                              loss_type,
                              seq_padding_idx: int = 0):
        """
        loss of ONE alignment path, given by alignment_state
        
        true_out is (B, L, 4):
            - dim2 = 0: gapped anc at currrent position b
            - dim2 = 1: gapped desc at currrent position b
            - dim2 = 2: from state (0-6) "prev_state", a
            - dim2 = 3: to_state (0-6) "curr_state", b
        """
        ### unpack inputs
        logprob_emit_match = forward_pass_outputs['FPO_logprob_emit_match']
        logprob_emit_indel = forward_pass_outputs['FPO_logprob_emit_indel']
        logprob_transits = forward_pass_outputs['FPO_logprob_transits']
        
        alignment_path = true_out[...,2:]
        curr_state = true_out[...,3]
        residues_in_alignment = true_out[...,:2]
        
        # get dims
        T = logprob_emit_match.shape[0]
        B = logprob_emit_match.shape[1]
        L = logprob_emit_match.shape[2]
        
        
        ### score transitions: (T, B, length_for_scan)
        tr = score_transitions(alignment_state = alignment_path,
                               logprob_trans_mat = logprob_transits, 
                               padding_idx=seq_padding_idx)
        
        
        # ### score emissions (T, B, length_for_scan)
        # # three cases: M, I, and D (which has no emission score)
        # # first: is the position a match or insert?
        # # second: decide between the two functions to run for match and insert
        e = jnp.zeros( (T,B,L) )
        e = e + jnp.where( curr_state == 1,
                           score_substitutions( true_out = residues_in_alignment,
                                                logprob_subs_mat = logprob_emit_match,
                                                token_offset = 3 ),
                           0
                           )
        
        # insert: score with descendant tok
        e = e + jnp.where( curr_state == 2,
                           score_indels(true_out = residues_in_alignment,
                                        logprob_scoring_vec = logprob_emit_indel,
                                        which_seq = 'desc',
                                        token_offset = 3),
                           0
                           )
        
        if loss_type == 'joint':
            # deletions: score with anc tok
            e = e + jnp.where( curr_state == 3,
                               score_indels(true_out = residues_in_alignment,
                                            logprob_scoring_vec = logprob_emit_indel,
                                            which_seq = 'anc',
                                            token_offset = 3),
                               0
                               )
        
       
        ### final logprob(sequences)
        # add logprobs together: (T, B, L)
        logprob_perSamp_perPos_perTime = tr + e
        
        # take the SUM down the length to get logprob_perSamp_perTime; padding
        #   tokens should have score of 0; don't normalize by length yet: (T, B)
        logprob_perSamp_perTime = logprob_perSamp_perPos_perTime.sum(axis=2)
        
        
        # intermediates to stack during scan fn
        intermeds_to_stack = {}
        return logprob_perSamp_perTime, intermeds_to_stack
        
    
    ###########################################################################
    ### get final loss for the whole sequence, from scan chunks   #############
    ###########################################################################
    def evaluate_loss_after_scan(self, 
                                 scan_fn_outputs,
                                 length_for_normalization,
                                 t_array,
                                 exponential_dist_param,
                                 seq_padding_idx: int = 0):
        # unpack
        logprob_perSamp_perTime, scan_intermeds = scan_fn_outputs
        
        ### if using multiple timepoints, need to logsumexp across them
        if t_array.shape[0] != 1:
            # logP(t_k) = exponential distribution
            logP_time = ( jnp.log(exponential_dist_param) - 
                          jnp.log(exponential_dist_param) * t_array )
            log_t_grid = jnp.log( t_array[1:] - t_array[:-1] )
            
            # kind of a hack, but repeat the last time array value
            log_t_grid = jnp.concatenate( [log_t_grid, log_t_grid[-1] ], axis=0)
            
            logP_perSamp_perTime_withConst = ( logprob_perSamp_perTime +
                                               logP_time +
                                               log_t_grid )
            
            logP_perSamp_raw = logsumexp(logP_perSamp_perTime_withConst, axis=0)
        
        # otherwise, just take the first timepoint result
        else:
            logP_perSamp_raw = logprob_perSamp_perTime[0,...] 
            
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
    
    
    
    