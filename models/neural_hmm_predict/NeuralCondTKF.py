#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 17:11:33 2025

@author: annabel
"""
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
from models.neural_hmm_predict.postprocessing_models import (FeedforwardPostproc,
                                                             Placeholder,
                                                             SelectMask)


class NeuralCondTKF(ModuleBase):
    """
    [fill in later]
    """
    config: dict
    name: str
    
    def setup(self):
        """
        Attributes:
        ------------
        self.times_from
        self.subst_model_type
        self.exponential_dist_param : float
        
        self.preproc_equl
        self.equl_module
        
        self.preproc_subs
        self.subs_module
        
        self.preproc_trans
        self.trans_module
        
        entries in config['global_or_local']:
        -------------------------------------
        equl_dist : str
        rate_mult : str
        (if gtr) exch : str
        tkf_rates : str
        (if tkf92) tkf92_frag_size : str
        
        entries in config['use_which_emb']:
        -----------------------------------------
        (these are tuples of booleans; first is whether or not to use the 
         ancestor embedding, second refers to descendant embedding)
        preproc_equl : (bool, bool)
        preproc_subs : (bool, bool)
        preproc_trans : (bool, bool)
        
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
        use_which_emb_dict = self.config['use_which_emb']
        
        # optional
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1.)
        preproc_model_type = self.config.get('preproc_model_type', 'selectmask').lower()
        
        # handle time
        if times_from =='t_per_sample':
            self.unique_time_per_sample = True
        
        elif times_from in ['geometric','t_array_from_file']:
            self.unique_time_per_sample = False
        
        
        ######################################################################
        ### pick architecture for preprocessing features, for local params   #
        ######################################################################
        preproc_module_registry = {'selectmask': SelectMask,
                                   'feedforward': FeedforwardPostproc}
        preproc_module = preproc_module_registry[preproc_model_type]
        
        
        ###########################################
        ### module for equilibrium distribution   #
        ###########################################
        if global_or_local_dict['equl_dist'].lower() == 'global':
            self.preproc_equl = lambda *args, **kwargs: None
            self.equl_module = GlobalEqul(config = self.config,
                                          name = f'{self.name}/get_equl')
        
        elif global_or_local_dict['equl_dist'].lower() == 'local':
            use_anc_emb, use_desc_emb = use_which_emb_dict['preproc_equl']
            self.preproc_equl = preproc_module(config = self.config,
                                               use_anc_emb = use_anc_emb,
                                               use_desc_emb = use_desc_emb,
                                               name = f'{self.name}/{preproc_model_type}_to_equl_module')
            self.equl_module = LocalEqul(config = self.config,
                                          name = f'{self.name}/get_equl')
        
        
        ########################################
        ### module for logprob substitutions   #
        ########################################
        if self.subst_model_type == 'f81':
            if global_or_local_dict['rate_mult'].lower() == 'global':
                self.preproc_subs = lambda *args, **kwargs: None
                self.subs_module = GlobalF81(config = self.config,
                                              name = f'{self.name}/get_subs')
            
            elif global_or_local_dict['rate_mult'].lower() == 'local':
                use_anc_emb, use_desc_emb = use_which_emb_dict['preproc_subs']
                self.preproc_subs =  preproc_module(config = self.config,
                                                    use_anc_emb = use_anc_emb,
                                                    use_desc_emb = use_desc_emb,
                                                    name = f'{self.name}/{preproc_model_type}_to_subs_module')
                self.subs_module = LocalF81(config = self.config,
                                            name = f'{self.name}/get_subs')

                
        elif self.subst_model_type == 'gtr':
            # combination of global/local values to test
            exch_global = global_or_local_dict['exch'].lower() == 'global'
            rate_global = global_or_local_dict['rate_mult'].lower() == 'global'
            exch_local = global_or_local_dict['exch'].lower() == 'local'
            rate_local = global_or_local_dict['rate_mult'].lower() == 'local'
            
            if exch_global and rate_global:
                self.preproc_subs = lambda *args, **kwargs: None
                self.subs_module = GTRGlobalExchGlobalRateMult(config = self.config,
                                            name = f'{self.name}/get_subs')
            
            else:
                use_anc_emb, use_desc_emb = use_which_emb_dict['preproc_subs']
                self.preproc_subs = preproc_module(config = self.config,
                                                   use_anc_emb = use_anc_emb,
                                                   use_desc_emb = use_desc_emb,
                                                   name = f'{self.name}/{preproc_model_type}_to_subs_module')
            
                if exch_global and rate_local:
                    self.subs_module = GTRGlobalExchLocalRateMult(config = self.config,
                                                name = f'{self.name}/get_subs')
                
                elif exch_local and rate_local:
                    self.subs_module = GTRLocalExchLocalRateMult(config = self.config,
                                                name = f'{self.name}/get_subs')
                
                # weird case that I'm not testing yet
                elif exch_local and rate_global:
                    raise NotImplementedError
            
            
        ###########################################
        ### module for transition probabilities   #
        ###########################################
        if self.indel_model_type == 'tkf91':
            
            if global_or_local_dict['tkf_rates'].lower() == 'global':
                self.preproc_trans = lambda *args, **kwargs: None
                self.trans_module = GlobalTKF91(config = self.config,
                                            name = f'{self.name}/get_trans')
            
            elif global_or_local_dict['tkf_rates'].lower() == 'local':
                use_anc_emb, use_desc_emb = use_which_emb_dict['preproc_trans']
                self.preproc_trans =  preproc_module(config = self.config,
                                                     use_anc_emb = use_anc_emb,
                                                     use_desc_emb = use_desc_emb,
                                                     name = f'{self.name}/{preproc_model_type}_to_trans_module')
                self.trans_module = LocalTKF91(config = self.config,
                                            name = f'{self.name}/get_trans')
                
        
        elif self.indel_model_type == 'tkf92':
            # combination of global/local values to test
            indel_rates_global = global_or_local_dict['tkf_rates'].lower() == 'global'
            frag_size_global = global_or_local_dict['tkf92_frag_size'].lower() == 'global'
            indel_rates_local = global_or_local_dict['tkf_rates'].lower() == 'local'
            frag_size_local = global_or_local_dict['tkf92_frag_size'].lower() == 'local'
            
            if indel_rates_global and frag_size_global:
                self.preproc_trans = lambda *args, **kwargs: None
                self.trans_module = TKF92GlobalRateGlobalFragSize(config = self.config,
                                                        name = f'{self.name}/get_trans')
                
            else:
                use_anc_emb, use_desc_emb = use_which_emb_dict['preproc_trans']
                self.preproc_trans =  preproc_module(config = self.config,
                                                     use_anc_emb = use_anc_emb,
                                                     use_desc_emb = use_desc_emb,
                                                     name = f'{self.name}/{preproc_model_type}_to_trans_module')
                
                if indel_rates_global and frag_size_local:
                    self.trans_module = TKF92GlobalRateLocalFragSize(config = self.config,
                                                            name = f'{self.name}/get_trans')
                
                elif indel_rates_local and frag_size_local:
                    self.trans_module = TKF92LocalRateLocalFragSize(config = self.config,
                                                            name = f'{self.name}/get_trans')
                
                # weird case that I'm not testing yet
                elif indel_rates_local and frag_size_global:
                    raise NotImplementedError
            
    def __call__(self, 
                 datamat_lst: list[jnp.array], 
                 padding_mask: jnp.array, 
                 t_array: jnp.array,
                 training: bool, 
                 sow_intermediates: bool=False,
                 **kwargs):
        """
        unlike pairHMM implementation, this ONLY generates scoring matrices
        """
        # equilibrium distribution; used to score emissions from indel sites
        equl_feats = self.preproc_equl(datamat_lst = datamat_lst,
                                       padding_mask = padding_mask,
                                       training = training,
                                       sow_intermediates = sow_intermediates)
        
        logprob_emit_indel = self.equl_module(datamat = equl_feats,
                                              sow_intermediates = sow_intermediates)
        
        # substitution model; used to score emissions from match sites
        sub_feats = self.preproc_subs(datamat_lst = datamat_lst,
                                      padding_mask = padding_mask,
                                      training = training,
                                      sow_intermediates = sow_intermediates)
        
        logprob_emit_match, subs_model_params = self.subs_module(datamat = sub_feats,
                                                                log_equl = logprob_emit_indel,
                                                                t_array = t_array,
                                                                unique_time_per_sample = self.unique_time_per_sample,
                                                                sow_intermediates = sow_intermediates)
        
        # transition model; used to score markovian alignment path
        trans_feats = self.preproc_trans(datamat_lst = datamat_lst,
                                         padding_mask = padding_mask,
                                         training = training,
                                         sow_intermediates = sow_intermediates)
        
        out = self.trans_module(datamat = trans_feats,
                                t_array = t_array,
                                unique_time_per_sample = self.unique_time_per_sample,
                                sow_intermediates = sow_intermediates) 
        logprob_transits, approx_flags_dict, indel_model_params = out
        del out
        
        out_dict = {'logprob_emit_match': logprob_emit_match, 
                    'logprob_emit_indel': logprob_emit_indel, 
                    'logprob_transits': logprob_transits,
                    'approx_flags_dict': approx_flags_dict,
                    'subs_model_params': subs_model_params,
                    'indel_model_params': indel_model_params}
        
        ### correction to conditional logprob
        # if tkf91, no corrections needed
        if self.indel_model_type == 'tkf91':
            out_dict['corr'] = ( jnp.zeros( indel_model_params['lambda'].shape ), #(T,B) or (B,)
                                 jnp.zeros( indel_model_params['lambda'].shape ) ) #(T,B) or (B,)
            
        # if tkf92, include correction factor for starting with s->ins transition, 
        # and ending with ins->e
        elif self.indel_model_type == 'tkf92':
            lam = indel_model_params['lambda'] #(T,B) or (B,)
            mu = indel_model_params['mu'] #(T,B) or (B,)
            r_extend = indel_model_params['r_extend'] #(T,B) or (B,)
            out_dict['corr'] = ( jnp.log(mu/lam), #(T,B) or (B,)
                                 jnp.log( r_extend + (1-r_extend)*(lam/mu) ) ) #(T,B) or (B,)
        
        return out_dict
        
    
    def neg_loglike_in_scan_fn(self, 
                              logprob_emit_match: jnp.array, 
                              logprob_emit_indel: jnp.array,
                              logprob_transits: jnp.array,
                              corr: tuple[jnp.array, jnp.array],
                              true_out: jnp.array,
                              gap_idx: int=43,
                              padding_idx: int=0,
                              start_idx: int=1,
                              end_idx: int=2):
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
        staggered_alignment_state = true_out[...,2:] #(B, L_align-1, 2)
        curr_state = true_out[...,3] #(B,L_align-1)
        true_alignment_without_start = true_out[...,:2] #(B, L_align-1, 2)
        
        
        ### score transitions
        # if unique_time_per_sample, tr is (B, length_for_scan)
        # elif not unique_time_per_sample, tr is (T, B, length_for_scan)
        tr = score_transitions(staggered_alignment_state = staggered_alignment_state,
                               logprob_trans_mat = logprob_transits, 
                               unique_time_per_sample = self.unique_time_per_sample,
                               padding_idx=padding_idx)
        
        
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
            
        # match positiosn: score
        e = e + jnp.where( curr_state == 1,
                           score_substitutions( true_alignment_without_start = true_alignment_without_start,
                                                logprob_scoring_mat = logprob_emit_match,
                                                unique_time_per_sample = self.unique_time_per_sample,
                                                gap_idx = gap_idx,
                                                padding_idx = padding_idx,
                                                start_idx = start_idx,
                                                end_idx = end_idx),
                           0
                           )
        
        # insert positions: score with descendant tok
        e = e + jnp.where( curr_state == 2,
                           score_indels(true_alignment_without_start = true_alignment_without_start,
                                        logprob_scoring_vec = logprob_emit_indel,
                                        which_seq = 'desc',
                                        gap_idx = gap_idx,
                                        padding_idx = padding_idx,
                                        start_idx = start_idx,
                                        end_idx = end_idx),
                           0
                           )
        # conditional logprob, so don't score "emissions" from ancestor tokens
        
        ### final logprob(sequences)
        # if unique_time_per_sample, logprob_perSamp_perPos_perTime is (B, length_for_scan)
        # elif not unique_time_per_sample, logprob_perSamp_perPos_perTime is (T, B, length_for_scan)
        logprob_perSamp_perPos_perTime = tr + e
        
        # take the SUM down the length to get logprob_perSamp_perTime:
        # if unique_time_per_sample, logprob_perSamp_perTime is (B)
        # elif not unique_time_per_sample, logprob_perSamp_perTime is (T, B)
        logprob_perSamp_perTime = logprob_perSamp_perPos_perTime.sum(axis=-1)
        
        # extra correction factors, if needed
        include_s_i_corr_mask = jnp.all(staggered_alignment_state[:,0,:] == jnp.array([4, 2]), axis=-1) #(B,)
        include_i_e_corr_mask = jnp.any( jnp.all(staggered_alignment_state == jnp.array([2, 5]), axis=-1), axis=-1 ) #(B,)

        if len(logprob_perSamp_perTime.shape) == 2:
            include_s_i_corr_mask = jnp.broadcast_to(include_s_i_corr_mask[None,:],
                                                      logprob_perSamp_perTime.shape) #(T,B)
            include_i_e_corr_mask = jnp.broadcast_to(include_i_e_corr_mask[None,:],
                                                      logprob_perSamp_perTime.shape) #(T,B)
        
        logprob_perSamp_perTime = jnp.where( include_s_i_corr_mask,
                                             logprob_perSamp_perTime + corr[0],
                                             logprob_perSamp_perTime ) #(T,B) or (B,)
        
        logprob_perSamp_perTime = jnp.where( include_i_e_corr_mask,
                                             logprob_perSamp_perTime + corr[1],
                                             logprob_perSamp_perTime ) #(T,B) or (B,)
        
        return logprob_perSamp_perTime
        
    
    def evaluate_loss_after_scan(self, 
                                 logprob_perSamp_perTime,
                                 length_for_normalization,
                                 t_array,
                                 padding_idx: int = 0):
        """
        postprocessing after accumulating logprobs in a scan function
        """
        ### handle time, if needed
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
        
        # collect loss and intermediate values
        loss = -jnp.mean(logprob_perSamp)
        intermediate_vals = { 'sum_neg_logP': -logprob_perSamp,
                              'neg_logP_length_normed': -logprob_perSamp/length_for_normalization}
        
        return loss, intermediate_vals
    
    def get_perplexity_per_sample(self,
                                  loss_fn_dict):
        neg_logP_length_normed = loss_fn_dict['neg_logP_length_normed']
        perplexity_perSamp = jnp.exp(neg_logP_length_normed) #(B,)
        return perplexity_perSamp
    
    def get_ece(self,
                loss):
        return jnp.exp(loss)



class NeuralCondTKFLoadAll(NeuralCondTKF):
    """
    Replicate a simple tkf model using the neural codebase
    
    I think I only need a unique setup; everything else should work out
    
    [fill in later]
    """
    config: dict
    name: str
    
    def setup(self):
        """
        Attributes:
        ------------
        self.times_from
        self.subst_model_type
        self.norm_loss_by
        self.norm_loss_by_length
        self.exponential_dist_param
        
        self.preproc_equl
        self.equl_module
        
        self.preproc_subs
        self.subs_module
        
        self.preproc_trans
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
        preproc_equl
        preproc_subs
        preproc_trans
        
        """
        ###################
        ### read config   #
        ###################
        # required
        self.subst_model_type = self.config['subst_model_type'].lower()
        times_from = self.config['times_from'].lower()
        indel_model_type = self.config['indel_model_type'].lower()
        
        # optional
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        preproc_model_type = self.config.get('preproc_model_type', 'selectmask')
        
        # handle time
        if times_from =='t_per_sample':
            self.unique_time_per_sample = True
        
        elif times_from in ['geometric','t_array_from_file']:
            self.unique_time_per_sample = False
        
        
        ###########################################
        ### module for equilibrium distribution   #
        ###########################################
        self.preproc_equl = lambda *args, **kwargs: None
        self.equl_module = EqulFromFile(config = self.config,
                                        name = f'{self.name}/get_equl')
        
        ########################################
        ### module for logprob substitutions   #
        ########################################
        self.preproc_subs = lambda *args, **kwargs: None
        
        if self.subst_model_type == 'f81':
            self.subs_module = F81FromFile(config = self.config,
                                           name = f'{self.name}/get_subs')
            
        elif self.subst_model_type == 'gtr':
            self.subs_module = GTRFromFile(config = self.config,
                                           name = f'{self.name}/get_subs')
            
        ###########################################
        ### module for transition probabilities   #
        ###########################################
        self.preproc_trans = lambda *args, **kwargs: None
        if self.indel_model_type == 'tkf91':
            self.trans_module = GlobalTKF91FromFile(config = self.config,
                                                    name = f'{self.name}/get_trans')
         
        elif self.indel_model_type == 'tkf92':
            self.trans_module = GlobalTKF92FromFile(config = self.config,
                                                    name = f'{self.name}/get_trans')
            
            
            