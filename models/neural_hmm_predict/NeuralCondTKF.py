#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 17:11:33 2025

@author: annabel
"""
import sys

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm as matrix_exp
from jax.scipy.special import logsumexp
import optax

from models.BaseClasses import ModuleBase
from models.neural_hmm_predict.scoring_fns import (score_f81_substitutions_marg_over_times,
                                                   score_f81_substitutions_t_per_samp,
                                                   score_gtr_substitutions,
                                                   score_indels,
                                                   score_transitions)
from models.neural_hmm_predict.emission_models import (EqulFromFile,
                                                       F81FromFile,
                                                       GTRFromFile,
                                                       GTRGlobalExchGlobalRateMult,
                                                       GTRGlobalExchLocalRateMult,
                                                       GTRLocalExchLocalRateMult,
                                                       GlobalEqul,
                                                       GlobalF81,
                                                       LocalEqul,
                                                       LocalF81)
from models.neural_hmm_predict.transition_models import (GlobalTKF91,
                                                         LocalTKF91,
                                                         TKF92GlobalRateGlobalFragSize,
                                                         TKF92GlobalRateLocalFragSize,
                                                         TKF92LocalRateLocalFragSize,
                                                         GlobalTKF91FromFile,
                                                         GlobalTKF92FromFile)

from models.neural_shared.postprocessing_models import (FeedforwardPostproc,
                                                       SelectMask)


class NeuralCondTKF(ModuleBase):
    """
    [fill in later]
    """
    config: dict
    name: str
    
    def setup(self):
        """
        Attributes from pred_config:
        -----------------------------
        self.times_from : str
        self.subst_model_type : str
        self.exponential_dist_param : float
        self.rate_multiplier_regularization_rate : float
        self.transitions_postproc_config : dict
        self.emissions_postproc_config : dict
        
        Attributes created after model initialization:
        -----------------------------------------------
        self.postproc_equl
        self.equl_module
        
        self.postproc_subs
        self.subs_module
        
        self.postproc_trans
        self.trans_module
        
        decide the post-processing architecture to combine 
        position-specific embeddings with:
        ---------------------------------------------------
            - config['emissions_postproc_module']
              - provide config for this called config['emissions_postproc_config']
            - config['transitions_postproc_module']
              - provide config for this called config['transitions_postproc_config']
        
        entries in config['global_or_local']:
        -------------------------------------
        equl_dist : str
        rate_mult : str
        (if gtr) exch : str
        tkf_rates : str
        (if tkf92) tkf92_frag_size : str
        """
        ###################
        ### read config   #
        ###################
        # required
        self.times_from = self.config['times_from']
        self.subst_model_type = self.config['subst_model_type'].lower()
        self.indel_model_type = self.config['indel_model_type'].lower()
        times_from = self.config['times_from'].lower()
        global_or_local_dict = self.config['global_or_local']
        
        # optional
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1.)
        self.rate_multiplier_regularization_rate = self.config.get('rate_mult_regular_rate', 0.0)
        self.transitions_postproc_config = self.config.get('transitions_postproc_config', dict() )
        self.emissions_postproc_config = self.config.get('emissions_postproc_config', dict() )
        self.transitions_postproc_model_type = self.config.get('transitions_postproc_model_type', None)
        emissions_postproc_model_type = self.config.get('emissions_postproc_model_type', None)
        
        # handle time
        if times_from =='t_per_sample':
            self.unique_time_per_sample = True
        
        elif times_from in ['geometric','t_array_from_file']:
            self.unique_time_per_sample = False
        
        
        ######################################################################
        ### pick architecture for preprocessing features, for local params   #
        ######################################################################
        postproc_module_registry = {'selectmask': SelectMask,
                                    'feedforward': FeedforwardPostproc,
                                    None: lambda *args, **kwargs: lambda *args, **kwargs: None}
        
        transitions_postproc_module = postproc_module_registry[self.transitions_postproc_model_type]
        emissions_postproc_module = postproc_module_registry[emissions_postproc_model_type]
        
        
        ###########################################
        ### module for equilibrium distribution   #
        ###########################################
        # postprocess the concatenated embeddings
        self.postproc_equl = emissions_postproc_module(config = self.emissions_postproc_config,
                                           name = f'{self.name}/{emissions_postproc_model_type}_to_equl_module')
        
        # get equilibrium distribution
        if global_or_local_dict['equl_dist'].lower() == 'global':
            self.equl_module = GlobalEqul(config = self.emissions_postproc_config,
                                          name = f'{self.name}/get_equl')
        
        elif global_or_local_dict['equl_dist'].lower() == 'local':
            self.equl_module = LocalEqul(config = self.emissions_postproc_config,
                                          name = f'{self.name}/get_equl')
        
        
        ########################################
        ### module for logprob substitutions   #
        ########################################
        # postprocess the concatenated embeddings
        self.postproc_subs =  emissions_postproc_module(config = self.emissions_postproc_config,
                                            name = f'{self.name}/{emissions_postproc_model_type}_to_subs_module')
        
        if self.subst_model_type == 'f81':
            if global_or_local_dict['rate_mult'].lower() == 'global':
                self.subs_module = GlobalF81(config = self.emissions_postproc_config,
                                              name = f'{self.name}/get_subs')
            
            elif global_or_local_dict['rate_mult'].lower() == 'local':
                self.subs_module = LocalF81(config = self.emissions_postproc_config,
                                            name = f'{self.name}/get_subs')

                
        elif self.subst_model_type == 'gtr':
            # combination of global/local values to test
            exch_global = global_or_local_dict['exch'].lower() == 'global'
            rate_global = global_or_local_dict['rate_mult'].lower() == 'global'
            exch_local = global_or_local_dict['exch'].lower() == 'local'
            rate_local = global_or_local_dict['rate_mult'].lower() == 'local'
            
            if exch_global and rate_global:
                self.subs_module = GTRGlobalExchGlobalRateMult(config = self.emissions_postproc_config,
                                            name = f'{self.name}/get_subs')
            
            elif exch_global and rate_local:
                self.subs_module = GTRGlobalExchLocalRateMult(config = self.emissions_postproc_config,
                                            name = f'{self.name}/get_subs')
            
            elif exch_local and rate_local:
                self.subs_module = GTRLocalExchLocalRateMult(config = self.emissions_postproc_config,
                                            name = f'{self.name}/get_subs')
            
            # weird case that I'm not testing yet
            elif exch_local and rate_global:
                raise NotImplementedError
            
            
        ###########################################
        ### module for transition probabilities   #
        ###########################################
        # postprocess the concatenated embeddings
        self.postproc_trans =  transitions_postproc_module(config = self.transitions_postproc_config,
                                             name = f'{self.name}/{self.transitions_postproc_model_type}_to_trans_module')
        
        if self.indel_model_type == 'tkf91':
            if global_or_local_dict['tkf_rates'].lower() == 'global':
                self.trans_module = GlobalTKF91(config = self.transitions_postproc_config,
                                            name = f'{self.name}/get_trans')
            
            elif global_or_local_dict['tkf_rates'].lower() == 'local':
                self.trans_module = LocalTKF91(config = self.transitions_postproc_config,
                                            name = f'{self.name}/get_trans')
                
        
        elif self.indel_model_type == 'tkf92':
            # combination of global/local values to test
            indel_rates_global = global_or_local_dict['tkf_rates'].lower() == 'global'
            frag_size_global = global_or_local_dict['tkf92_frag_size'].lower() == 'global'
            indel_rates_local = global_or_local_dict['tkf_rates'].lower() == 'local'
            frag_size_local = global_or_local_dict['tkf92_frag_size'].lower() == 'local'
            
            if indel_rates_global and frag_size_global:
                self.trans_module = TKF92GlobalRateGlobalFragSize(config = self.transitions_postproc_config,
                                                        name = f'{self.name}/get_trans')
                
            elif indel_rates_global and frag_size_local:
                    self.trans_module = TKF92GlobalRateLocalFragSize(config = self.transitions_postproc_config,
                                                            name = f'{self.name}/get_trans')
                
            elif indel_rates_local and frag_size_local:
                self.trans_module = TKF92LocalRateLocalFragSize(config = self.transitions_postproc_config,
                                                        name = f'{self.name}/get_trans')
            
            # weird case that I'm not testing yet
            elif indel_rates_local and frag_size_global:
                raise NotImplementedError
            
    def __call__(self, 
                 datamat_lst: list[jnp.array], 
                 padding_mask: jnp.array,  #(B, L)
                 t_array: jnp.array, #(B,) or (T,)
                 training: bool, 
                 sow_intermediates: bool=False,
                 *args,
                 **kwargs):
        """
        unlike pairHMM implementation, this ONLY generates scoring matrices
        """
        # elements of datamat_lst are:
        # anc_embeddings: (B, L, H)
        # desc_embeddings: (B, L, H)
        # prev_align_one_hot_vec: (B, L, 5)

        ### equilibrium distribution; used to score emissions from indel sites
        # don't feed times here; they get incorporated during scoring only!
        equl_feats = self.postproc_equl(anc_emb = datamat_lst[0],
                                        desc_causal_emb = datamat_lst[1],
                                        prev_align_one_hot_vec = datamat_lst[2],
                                        padding_mask = padding_mask,
                                        training = training,
                                        sow_intermediates = sow_intermediates,
                                        t_array = None)  #(B, L, H_out)
        
        logprob_emit_indel = self.equl_module(datamat = equl_feats,
                                              sow_intermediates = sow_intermediates) #(B, L, A)
        
        
        ### substitution model; used to score emissions from match sites
        # don't feed times here; they get incorporated during scoring only!
        sub_feats = self.postproc_subs(anc_emb = datamat_lst[0],
                                        desc_causal_emb = datamat_lst[1],
                                        prev_align_one_hot_vec = datamat_lst[2],
                                        padding_mask = padding_mask,
                                        training = training,
                                        sow_intermediates = sow_intermediates,
                                        t_array = None)  #(B, L, H_out)
        
        # logprob_emit_match is either (T, B, L, A, A) or (B, L, A, A)
        # subs_model_params is a dictionary of parameters; see module for more details
        logprob_emit_match, subs_model_params = self.subs_module(datamat = sub_feats,
                                                                 padding_mask = padding_mask,
                                                                 log_equl = logprob_emit_indel,
                                                                 t_array = t_array,
                                                                 unique_time_per_sample = self.unique_time_per_sample,
                                                                 sow_intermediates = sow_intermediates)
        
        
        ### transition model; used to score markovian alignment path
        # don't feed times here; they get incorporated during scoring only!
        trans_feats = self.postproc_trans(anc_emb = datamat_lst[0],
                                        desc_causal_emb = datamat_lst[1],
                                        prev_align_one_hot_vec = datamat_lst[2],
                                        padding_mask = padding_mask,
                                        training = training,
                                        sow_intermediates = sow_intermediates,
                                        t_array = None)  #(B, L, H_out)
        
        out = self.trans_module(datamat = trans_feats,
                                t_array = t_array,
                                unique_time_per_sample = self.unique_time_per_sample,
                                sow_intermediates = sow_intermediates) 
        
        # logprob_transits is either (T, B, L, S, S) or (B, L, S, S)
        # approx_flags_dict and indel_model_params are dictionaroes; 
        # see module for more details
        logprob_transits, approx_flags_dict, indel_model_params = out
        del out
        
        # out dictionary
        out_dict = {'logprob_emit_match': logprob_emit_match,  # (T, B, L, A, A) or (B, L, A, A)
                    'logprob_emit_indel': logprob_emit_indel,  #(B, L, A)
                    'logprob_transits': logprob_transits, # (T, B, L, S, S) or (B, L, S, S)
                    'approx_flags_dict': approx_flags_dict, #dict
                    'subs_model_params': subs_model_params, #dict
                    'indel_model_params': indel_model_params}  #dict
        
        
        ### correction to conditional logprob, if tkf92
        ### if alignment begins with S -> ins, then this class will
        ###   see ancestor transition path as em -> em -> ... instead 
        ###   of S -> em -> ...; that is, it will omit an (S->em) 
        ###   transition and add an extra (em -> em) transition.
        
        # if tkf91, no corrections needed
        if self.indel_model_type == 'tkf91':
            corr_shape = (1,1) if not self.unique_time_per_sample else (1,)
            out_dict['corr'] = jnp.zeros( corr_shape ) #(T,B) or (B,)
            
        # if tkf92, include correction factor for starting with s->ins transition, 
        # and ending with ins->e
        elif self.indel_model_type == 'tkf92':
            lam = indel_model_params['lambda'] #(B, L) or (1,1)
            mu = indel_model_params['mu'] #(B, L) or (1,1)
            r_extend = indel_model_params['r_extend'] #(B, L) or (1,1)
            
            corr = ( lam / mu ) / ( r_extend + (1-r_extend)*(lam/mu) )
            log_corr = jnp.log(corr)
            
            out_dict['corr'] = log_corr #(B, L) or (1,1)
        return out_dict
        
    
    def neg_loglike_in_scan_fn(self, 
                              logprob_emit_match: jnp.array, 
                              logprob_emit_indel: jnp.array,
                              logprob_transits: jnp.array,
                              rate_multiplier: jnp.array,
                              corr: jnp.array,
                              true_out: jnp.array,
                              gap_idx: int=43,
                              padding_idx: int=0,
                              start_idx: int=1,
                              end_idx: int=2,
                              return_result_before_sum: bool=False,
                              return_transit_emit: bool=False,
                              *args,
                              **kwargs):
        """
        loss of alignment path, given by alignment_state
        
        true_out is (B, L, 4):
            - dim2 = 0: gapped anc at currrent position b
            - dim2 = 1: gapped desc at currrent position b
            - dim2 = 2: from state (0-6) "prev_state", a
            - dim2 = 3: to_state (0-6) "curr_state", b
        """
        B = logprob_emit_indel.shape[0]
        L = logprob_emit_indel.shape[1]
        
        # unpack inputs
        staggered_alignment_state = true_out[...,2:] #(B, length_for_scan, 2)
        curr_state = true_out[...,3] #(B,length_for_scan)
        true_alignment_without_start = true_out[...,:2] #(B, length_for_scan, 2)
        
        
        ### score transitions
        # if unique_time_per_sample, tr is (B, length_for_scan)
        # elif not unique_time_per_sample, tr is (T, B, length_for_scan)
        tr = score_transitions(staggered_alignment_state = staggered_alignment_state,
                               logprob_trans_mat = logprob_transits, 
                               unique_time_per_sample = self.unique_time_per_sample,
                               padding_idx=padding_idx)
        
        # extra correction factors for S -> I transition
        s_i_corr_mask = jnp.all( staggered_alignment_state == jnp.array([4, 2]), axis=-1 ) #(B, length_for_scan)
        
        if len(tr.shape) == 3:
            s_i_corr_mask = jnp.broadcast_to( s_i_corr_mask[None,...], tr.shape ) #(T, B, length_for_scan)
            s_i_correction = jnp.broadcast_to( corr[None,...], tr.shape ) #(T, B, length_for_scan)
        
        elif len(tr.shape) == 2:
            s_i_correction = jnp.broadcast_to( corr, tr.shape ) #(B, length_for_scan)
            
        # make corrections selectively, per sample and per position
        tr = jnp.where( s_i_corr_mask,
                        tr - s_i_correction,
                        tr ) #(T, B, length_for_scan) or (B, length_for_scan)
        
        
        
        ### score emissions
        # if unique_time_per_sample, e is (B, length_for_scan)
        # elif not unique_time_per_sample, e is (T, B, length_for_scan)
        if self.unique_time_per_sample:
            e = jnp.zeros( (B,L) )
        elif not self.unique_time_per_sample:
            T = logprob_emit_match.shape[0]
            e = jnp.zeros( (T,B,L) )
        
        # match positions: decide function
        if (self.subst_model_type == 'f81') and self.unique_time_per_sample:
            score_substitutions = score_f81_substitutions_t_per_samp
            
        elif (self.subst_model_type == 'f81') and not self.unique_time_per_sample:
            score_substitutions = score_f81_substitutions_marg_over_times
            
        elif self.subst_model_type == 'gtr':
            score_substitutions = score_gtr_substitutions
            
        # match positions: score
        e = e + jnp.where( curr_state == 1,
                           score_substitutions( true_alignment_without_start = true_alignment_without_start,
                                                logprob_scoring_mat = logprob_emit_match,
                                                unique_time_per_sample = self.unique_time_per_sample,
                                                gap_idx = gap_idx,
                                                padding_idx = padding_idx,
                                                start_idx = start_idx,
                                                end_idx = end_idx),
                           0 )
        
        # insert positions: score with descendant tok
        e = e + jnp.where( curr_state == 2,
                           score_indels(true_alignment_without_start = true_alignment_without_start,
                                        logprob_scoring_vec = logprob_emit_indel,
                                        which_seq = 'desc',
                                        gap_idx = gap_idx,
                                        padding_idx = padding_idx,
                                        start_idx = start_idx,
                                        end_idx = end_idx),
                           0 )
        # conditional logprob, so don't score "emissions" from ancestor tokens
        
        ### final logprob(sequences)
        # if unique_time_per_sample, logprob_perSamp_perPos_perTime is (B, length_for_scan)
        # elif not unique_time_per_sample, logprob_perSamp_perPos_perTime is (T, B, length_for_scan)
        logprob_perSamp_perPos_perTime = tr + e
        
        if return_result_before_sum:
            # if unique_time_per_sample, logprob_perSamp_perPos_perTime is (B, length_for_scan)
            # elif not unique_time_per_sample, logprob_perSamp_perPos_perTime is (T, B, length_for_scan)
            out = {'logprob_perSamp_perPos_perTime': logprob_perSamp_perPos_perTime} #(T, B, length_for_scan) or (B, length_for_scan)
        
        elif not return_result_before_sum:
            # take the SUM down the length to get logprob_perSamp_perTime:
            # if unique_time_per_sample, logprob_perSamp_perTime is (B)
            # elif not unique_time_per_sample, logprob_perSamp_perTime is (T, B)
            logprob_perSamp_perTime = logprob_perSamp_perPos_perTime.sum(axis=-1) #(T,B) or (B,)
            out = {'logprob_perSamp_perTime': logprob_perSamp_perTime} #(T,B) or (B,)
        
        # accumulate sum of rate multipliers (for later regularization)
        out['rate_multiplier_sum'] = jnp.multiply( rate_multiplier, (curr_state != 0) ).sum() #float
        out['total_seen_toks'] = (curr_state != 0).sum() #float
        
        # possible return intermediate scores (for debugging)
        if return_transit_emit:
            out['tr'] = tr
            out['e'] = e
        
        return out
        
    
    def evaluate_loss_after_scan(self, 
                                 loss_dict: dict,
                                 length_for_normalization_for_reporting: jnp.array,
                                 t_array: jnp.array,
                                 padding_idx: int = 0,
                                 *args,
                                 **kwargs):
        """
        postprocessing after accumulating logprobs in a scan function
        """
        ### handle time, if needed
        logprob_perSamp_perTime = loss_dict['logprob_perSamp_perTime'] #(B,) or (T, B)

        # marginalize over time grid
        if not self.unique_time_per_sample:
            if t_array.shape[0] > 1:
                # logP(t_k) = exponential distribution
                logP_time = ( jnp.log(self.exponential_dist_param) - 
                              (self.exponential_dist_param * t_array) ) #(T,)
                log_t_grid = jnp.log( t_array[1:] - t_array[:-1] ) #(T-1,)
                
                # kind of a hack, but repeat the last time array value
                log_t_grid = jnp.concatenate( [log_t_grid, log_t_grid[-1][None] ], axis=0)  #(T,)
                
                logP_perSamp_perTime_withConst = ( logprob_perSamp_perTime +
                                                   logP_time[:,None] +
                                                   log_t_grid[:,None] ) #(T,B)
                
                logprob_perSamp = logsumexp(logP_perSamp_perTime_withConst, axis=0) #(B,)
            
            elif t_array.shape[0] == 1:
                logprob_perSamp = logprob_perSamp_perTime[0,...]
        
        # otherwise, just rename the variable
        elif self.unique_time_per_sample:
            logprob_perSamp = logprob_perSamp_perTime
        
        
        ### calculate loss, possibly regularize
        loss = -jnp.mean(logprob_perSamp)
        
        # regularization: encourage mean rate multiplier to be close to 1
        rate_multiplier_sum = loss_dict['rate_multiplier_sum'] #float
        total_seen_toks = loss_dict['total_seen_toks'] #float
        mean_rate_mult = rate_multiplier_sum / total_seen_toks #float
        loss = loss + self.rate_multiplier_regularization_rate * jnp.square( 1 - mean_rate_mult )
        
        
        ### collect loss and intermediate values
        intermediate_vals = { 'sum_neg_logP': -logprob_perSamp,
                              'neg_logP_length_normed': -logprob_perSamp/length_for_normalization_for_reporting,
                              'mean_rate_mult': mean_rate_mult}
        
        return loss, intermediate_vals
    
    def get_perplexity_per_sample(self,
                                  loss_fn_dict):
        neg_logP_length_normed = loss_fn_dict['neg_logP_length_normed']
        perplexity_perSamp = jnp.exp(neg_logP_length_normed) #(B,)
        return perplexity_perSamp
    

class NeuralCondTKFLoadAll(NeuralCondTKF):
    """
    Replicate a simple tkf model using the neural codebase
    
    [fill in later]
    """
    config: dict
    name: str
    
    def setup(self):
        """
        Attributes from pred_config:
        -----------------------------
        self.times_from
        self.subst_model_type
        self.exponential_dist_param
        
        Attributes created after model initialization:
        -----------------------------------------------
        self.postproc_equl
        self.equl_module
        
        self.postproc_subs
        self.subs_module
        
        self.postproc_trans
        self.trans_module
        
        entries in config['global_or_local']:
        -------------------------------------
        equl_dist
        rate_mult
        (if gtr) exch
        tkf_rates
        (if tkf92) tkf92_frag_size
        
        entries in config['use_which_emb']:
        -----------------------------------------
        postproc_equl
        postproc_subs
        postproc_trans
        
        """
        ###################
        ### read config   #
        ###################
        # required
        self.subst_model_type = self.config['subst_model_type'].lower()
        times_from = self.config['times_from'].lower()
        self.indel_model_type = self.config['indel_model_type'].lower()
        
        # optional
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        self.emissions_postproc_config = self.config.get('emissions_postproc_config', dict() )
        self.transitions_postproc_config = self.config.get('transitions_postproc_config', dict() )
        
        # handle time
        if times_from =='t_per_sample':
            self.unique_time_per_sample = True
        
        elif times_from in ['geometric','t_array_from_file']:
            self.unique_time_per_sample = False
        
        
        ###########################################
        ### module for equilibrium distribution   #
        ###########################################
        self.postproc_equl = lambda *args, **kwargs: None
        self.equl_module = EqulFromFile(config = self.emissions_postproc_config,
                                        name = f'{self.name}/get_equl')
        
        ########################################
        ### module for logprob substitutions   #
        ########################################
        self.postproc_subs = lambda *args, **kwargs: None
        
        if self.subst_model_type == 'f81':
            self.subs_module = F81FromFile(config = self.emissions_postproc_config,
                                           name = f'{self.name}/get_subs')
            
        elif self.subst_model_type == 'gtr':
            self.subs_module = GTRFromFile(config = self.emissions_postproc_config,
                                           name = f'{self.name}/get_subs')
            
        ###########################################
        ### module for transition probabilities   #
        ###########################################
        self.postproc_trans = lambda *args, **kwargs: None
        
        if self.indel_model_type == 'tkf91':
            self.trans_module = GlobalTKF91FromFile(config = self.transitions_postproc_config,
                                                    name = f'{self.name}/get_trans')
         
        elif self.indel_model_type == 'tkf92':
            self.trans_module = GlobalTKF92FromFile(config = self.transitions_postproc_config,
                                                    name = f'{self.name}/get_trans')
            
            
            