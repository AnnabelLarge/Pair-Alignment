#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 04:33:00 2025

@author: annabel

models:
=======
FragAndSiteClasses
FragAndSiteClassesLoadAll

"""
import pickle
import numpy as np

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax.scipy.special import logsumexp

from models.BaseClasses import ModuleBase
from models.simple_site_class_predict.emission_models import (EqulDistLogprobsFromCounts,
                                                              EqulDistLogprobsPerClass,
                                                              EqulDistLogprobsFromFile,
                                                              GTRLogprobs,
                                                              GTRLogprobsFromFile,
                                                              SiteClassLogprobs,
                                                              SiteClassLogprobsFromFile,
                                                              HKY85Logprobs,
                                                              HKY85LogprobsFromFile,
                                                              F81Logprobs,
                                                              F81LogprobsFromFile)
from models.simple_site_class_predict.transition_models import (TKF92TransitionLogprobs,
                                                                TKF92TransitionLogprobsFromFile)
from models.simple_site_class_predict.model_functions import (bound_sigmoid,
                                                              safe_log,
                                                              cond_logprob_emit_at_match_per_mixture,
                                                              joint_logprob_emit_at_match_per_mixture,
                                                              scale_rate_multipliers,
                                                              joint_only_forward,
                                                              all_loglikes_forward,
                                                              marginalize_over_times,
                                                              write_matrix_to_npy,
                                                              maybe_write_matrix_to_ascii)


class FragAndSiteClasses(ModuleBase):
    """
    pairHMM that finds joint loglikelihood of alignments, P(Anc, Desc, Align),
      given different hidden fragment classes; each discrete site class has
      its own equilibrium distribution, rate multiplier, and tkf extension 
      probability
    
    if using one time per sample, and wanting to QUANTIZE the time, need a 
        different model
    
    
    Initialize with
    ----------------
    config : dict
        config['num_mixtures'] :  int
            number of emission site classes
        
        config['subst_model_type'] : {gtr, hky85}
            which substitution model
        
        config['norm_loss_by'] :  {desc_len, align_len}, optional
            what length to normalize loglikelihood by
            Default is 'desc_len'
        
        config['exponential_dist_param'] : int, optional
            There is an exponential prior over time; this provides the
            parameter for this during marginalization over times
            Default is 1
        
        config['times_from'] : {geometric, t_array_from_file, t_per_sample}
        
    name : str
        class name, for flax
    
    Main methods here
    -----------------
    setup
    
    __call__
        unpack batch and calculate logP(anc, desc, align)
    
    calculate_all_loglikes
        calculate logP(anc, desc, align), logP(anc), logP(desc), and
        logP(desc, align | anc)
    
    write_params
        write parameters to files
    
    return_bound_sigmoid_limits
        after initializing model, get the limits for bound_sigmoid activations
    
    
    Other methods
    --------------
    _init_rate_matrix_module
        decide what function to use for rate matrix
    
    _joint_logprob_align
        calculate logP(anc, desc, align)
    
    
    Methods inherited from models.model_utils.BaseClasses.ModuleBase
    -----------------------------------------------------------------
    sow_histograms_scalars
        for tensorboard logging
    """
    config: dict
    name: str
    
    def setup(self):
        ###################
        ### read config   #
        ###################
        # required
        num_mixtures = self.config['num_mixtures']
        self.indp_rate_mults = self.config['indp_rate_mults']
        self.subst_model_type = self.config['subst_model_type']
        self.times_from = self.config['times_from']
        
        # optional
        self.norm_reported_loss_by = self.config.get('norm_reported_loss_by', 'desc_len')
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        self.gap_tok = self.config.get('gap_tok', 43)
        
        
        ###############################################################
        ### modules for probability of being in latent site classes,  #
        ### probability of having a particular subsitution rate       #
        ### rate multiplier, and the rate multipliers themselves      #
        ###############################################################
        # Latent site class probabilities
        self.site_class_probability_module = SiteClassLogprobs(config = self.config,
                                                  name = f'get site class probabilities')
        
        # substitution rate multipliers
        if not self.indp_rate_mults:
            self.rate_mult_module = RateMultipliersPerClass(config = self.config,
                                                      name = f'get rate multipliers')
        
        elif self.indp_rate_mults:
            self.rate_mult_module = IndpRateMultipliers(config = self.config,
                                                      name = f'get rate multipliers')
        
        
        ###########################################
        ### module for equilibrium distribution   #
        ###########################################
        if num_mixtures == 1:
            self.equl_dist_module = EqulDistLogprobsFromCounts(config = self.config,
                                                       name = f'get equilibrium')
        elif num_mixtures > 1:
            self.equl_dist_module = EqulDistLogprobsPerClass(config = self.config,
                                                     name = f'get equilibrium')
        
        
        ################################
        ### module for logprob subst   #
        ################################
        if self.subst_model_type == 'gtr':
            self.logprob_subst_module = GTRLogprobs( config = self.config,
                                                  name = f'gtr subst. model' )
            
        elif self.subst_model_type == 'f81':
            self.logprob_subst_module = F81Logprobs( config = self.config,
                                                     name = f'f81 subst. model' )

        elif self.subst_model_type == 'hky85':
            self.logprob_subst_module = HKY85Logprobs( config = self.config,
                                                    name = f'hky85 subst. model' )
            # this only works with DNA
            assert self.config['emission_alphabet_size'] == 4
            
        ###########################################
        ### module for transition probabilities   #
        ###########################################        
        # has to be tkf92
        self.transitions_module = TKF92TransitionLogprobs(config = self.config,
                                                 name = f'tkf92 indel model')
    
    def __call__(self,
                 batch,
                 t_array,
                 sow_intermediates: bool):
        """
        Use this during active model training
        
        returns:
            - loss: average across the batch, based on joint log-likelihood
                    
            - aux_dict: has the following keys and values
              1.) 'joint_neg_logP': sum down the length
              2.) 'joint_neg_logP_length_normed': sum down the length,  
                  normalized by desired length (set by self.norm_reported_loss_by)
              3.) align_length_for_normalization : desired length for reporting
              3.) whether or not you used approximation formula for TKF indel model
        """
        aligned_inputs = batch[0]
        
        # which times to use for scoring matrices
        if self.times_from =='t_per_sample':
            times_for_matrices = batch[1] #(B,)
            unique_time_per_sample = True
        
        elif self.times_from in ['geometric','t_array_from_file']:
            times_for_matrices = t_array #(T,)
            unique_time_per_sample = False
        
        # scoring matrices
        # 
        # scoring_matrices_dict has the following keys (when return_intermeds is False)
        #   logprob_emit_at_indel, (C, A)
        #   joint_logprob_emit_at_match, (T, C, A, A)
        #   all_transit_matrices, dict, with joint transit matrix being (T, C, C, S, S) or (B, C, C, S, S)
        #   used_approx, dict
        scoring_matrices_dict = self._get_scoring_matrices(t_array=times_for_matrices,
                                                           sow_intermediates=sow_intermediates,
                                                           return_intermeds=False)
        
        
        ### calculate joint loglike using 1D forward algorithm over latent site 
        ###   classes
        logprob_emit_at_indel = scoring_matrices_dict['logprob_emit_at_indel'] #(C, A)
        joint_logprob_emit_at_match = scoring_matrices_dict['joint_logprob_emit_at_match'] #(T, C, A, A) or (B, C, A, A)
        joint_logprob_transit =  scoring_matrices_dict['all_transit_matrices']['joint'] #(T, C, S, S) or (B, C, S, S)
        joint_logprob_perSamp_maybePerTime = joint_only_forward(aligned_inputs = aligned_inputs,
                                                 joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                                 logprob_emit_at_indel = logprob_emit_at_indel,
                                                 joint_logprob_transit = joint_logprob_transit,
                                                 unique_time_per_sample = unique_time_per_sample,
                                                 return_all_intermeds = False)  #(T, B)  or (B,)
        
        ### marginalize over times where needed
        if (not unique_time_per_sample) and (t_array.shape[0] > 1):
            joint_neg_logP = -marginalize_over_times(logprob_perSamp_perTime = joint_logprob_perSamp_maybePerTime,
                                                      exponential_dist_param = self.exponential_dist_param,
                                                      t_array = times_for_matrices) #(B,)
             
        elif (not unique_time_per_sample) and (t_array.shape[0] == 1):
            joint_neg_logP = -joint_logprob_perSamp_maybePerTime[0,:] #(B,)
        
        elif unique_time_per_sample:
            joint_neg_logP = -joint_logprob_perSamp_maybePerTime #(B,)
            
            
        ### for REPORTING ONLY (not the objective function), normalize by length
        if self.norm_reported_loss_by == 'desc_len':
            # where descendant is not pad or gap
            banned_toks = [0,1,2,self.gap_tok]
            
        elif self.norm_reported_loss_by == 'align_len':
            # where descendant is not pad (but could be gap)
            banned_toks = [0,1,2]
        
        mask = ~jnp.isin( aligned_inputs[...,1], banned_toks)
        length_for_normalization = mask.sum(axis=1)
        joint_neg_logP_length_normed = joint_neg_logP / length_for_normalization
        del mask
        
        
        ### compile final outputs
        # aux dict
        aux_dict = {'joint_neg_logP': joint_neg_logP,
                    'joint_neg_logP_length_normed': joint_neg_logP_length_normed,
                    'align_length_for_normalization': length_for_normalization,
                    'used_approx': scoring_matrices_dict['used_approx']}
        
        # final loss
        loss = jnp.mean( aux_dict['joint_neg_logP'] )
        
        return loss, aux_dict
    
    
    def calculate_all_loglikes(self,
                               batch,
                               t_array):
        """
        Use this during final eval
        
        returns all four loglikelihoods in a dictionary:
        
        1.) 'joint_neg_logP': P(anc, desc, align)
        2.) 'joint_neg_logP_length_normed': P(anc, desc, align), normalized 
            by desired length (set by self.norm_by)
        3.) 'anc_neg_logP': P(anc)
        4.) 'anc_neg_logP_length_normed': P(anc), normalized by ancestor 
            length
        5.) 'desc_neg_logP': P(desc)
        6.) 'desc_neg_logP_length_normed': P(desc), normalized by descendant 
            length
        7.) 'cond_neg_logP': P(desc, align | anc)
        8.) 'cond_neg_logP_length_normed': P(desc, align | anc), normalized 
            by desired length (set by self.norm_by)
        
        Calculate joint and sequence marginals in one jax.lax.scan operation
        """
        aligned_inputs = batch[0]
        
        # which times to use for scoring matrices
        if self.times_from =='t_per_sample':
            times_for_matrices = batch[1] #(B,)
            unique_time_per_sample = True
        
        elif self.times_from in ['geometric','t_array_from_file']:
            times_for_matrices = t_array #(T,)
            unique_time_per_sample = False
            
        # get lengths, not including <bos> and <eos>
        align_len = ~jnp.isin( aligned_inputs[...,0], [0,1,2] )
        anc_len = ~jnp.isin( aligned_inputs[...,0], [0,1,2,self.gap_tok] )
        desc_len = ~jnp.isin( aligned_inputs[...,1], [0,1,2,self.gap_tok] )
        align_len = align_len.sum(axis=1)
        anc_len = anc_len.sum(axis=1)
        desc_len = desc_len.sum(axis=1)
        
        # score matrices
        # 
        # scoring_matrices_dict has the following keys (when return_intermeds is False)
        #   logprob_emit_at_indel, (C, A)
        #   joint_logprob_emit_at_match, (T, C, A, A)
        #   all_transit_matrices, dict, with joint transit matrix being (T, C, C, S, S) or (B, C, C, S, S)
        #   used_approx, dict
        scoring_matrices_dict = self._get_scoring_matrices( t_array=times_for_matrices,
                                                            sow_intermediates=False,
                                                            return_intermeds=False )
        
        
        ### get all log-likelihoods
        logprob_emit_at_indel = scoring_matrices_dict['logprob_emit_at_indel']
        joint_logprob_emit_at_match = scoring_matrices_dict['joint_logprob_emit_at_match']
        all_transit_matrices =  scoring_matrices_dict['all_transit_matrices']
        out = all_loglikes_forward( aligned_inputs = aligned_inputs,
                                    joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                    logprob_emit_at_indel = logprob_emit_at_indel,
                                    all_transit_matrices = all_transit_matrices,
                                    unique_time_per_sample = unique_time_per_sample)
        
        # for joint loglike: marginalize over times where needed
        if (not unique_time_per_sample) and (t_array.shape[0] > 1):
            joint_logprob_perSamp_maybePerTime = out['joint_neg_logP']  #(T,B)
            overwrite_joint_neg_logP = -marginalize_over_times(logprob_perSamp_perTime = -joint_logprob_perSamp_maybePerTime,
                                                     exponential_dist_param = self.exponential_dist_param,
                                                     t_array = times_for_matrices) #(B,)
            out['joint_neg_logP'] = overwrite_joint_neg_logP #(B,)
             
        elif (not unique_time_per_sample) and (t_array.shape[0] == 1):
            out['joint_neg_logP'] = out['joint_neg_logP'][0,:] #(B,)
        
        
        ### conditional comes from joint / anc
        cond_neg_logP = - (-out['joint_neg_logP'] - -out['anc_neg_logP'])
        
        
        ### for REPORTING ONLY (not the objective function), normalize by length
        anc_neg_logP_length_normed = out['anc_neg_logP'] / anc_len
        desc_neg_logP_length_normed = out['desc_neg_logP'] / desc_len
        
        if self.norm_reported_loss_by == 'desc_len':
            joint_neg_logP_length_normed = out['joint_neg_logP'] / desc_len
            cond_neg_logP_length_normed = cond_neg_logP / desc_len
        
        elif self.norm_reported_loss_by == 'align_len':
            joint_neg_logP_length_normed = out['joint_neg_logP'] / align_len
            cond_neg_logP_length_normed = cond_neg_logP / align_len
        
        to_add = { 'joint_neg_logP_length_normed': joint_neg_logP_length_normed,
                   'anc_neg_logP_length_normed': anc_neg_logP_length_normed,
                   'desc_neg_logP_length_normed': desc_neg_logP_length_normed,
                   'cond_neg_logP': cond_neg_logP,
                   'cond_neg_logP_length_normed': cond_neg_logP_length_normed,
                   'used_approx': scoring_matrices_dict['used_approx']
                }
        
        return {**out, **to_add}
    
    
    def write_params(self,
                     t_array,
                     out_folder: str,
                     prefix: str,
                     write_time_static_objs: bool):
        
        #########################################################
        ### only write once: activations_times_used text file   #
        #########################################################
        if write_time_static_objs:
            with open(f'{out_folder}/activations_and_times_used.tsv','w') as g:
                if not self.config['load_all']:
                    act = self.rate_mult_module.rate_mult_activation
                    g.write(f'activation for rate multipliers: {act}\n')
                    g.write(f'activation for exchangeabiliites: bound_sigmoid\n')
                
                if self.times_from in ['geometric','t_array_from_file']:
                    g.write(f't_array for all samples; possible marginalized over them\n')
                    g.write(f'{t_array}')
                    g.write('\n')
                
                elif self.times_from == 't_per_sample':
                    g.write(f'one branch length for each sample; times used for {prefix}:\n')
                    g.write(f'{t_array}')
                    g.write('\n')
        
        ###################################
        ### always write: Full matrices   #
        ###################################
        out = self._get_scoring_matrices(t_array=t_array,
                                        sow_intermediates=False,
                                        return_intermeds=True)
        
        # final conditional and joint prob of match (before and after LSE over rate multipliers)
        for key in ['joint_logprob_emit_at_match',
                    'cond_subst_logprobs_per_mixture',
                    'joint_subst_logprobs_per_mixture']:
            mat = np.exp ( out[key] ) 
            new_key = f'{prefix}_{key}'.replace('log','')
            write_matrix_to_npy( out_folder, mat, new_key )
            maybe_write_matrix_to_ascii( out_folder, mat, new_key )
            del mat, new_key
            
        # transition matrix
        for loss_type in ['joint','conditional','marginal']:
            mat = np.exp(out['all_transit_matrices'][loss_type]) 
            new_key = f'{prefix}_{loss_type}_prob_transit_matrix'
            write_matrix_to_npy( out_folder, mat, new_key )
            maybe_write_matrix_to_ascii( out_folder, mat, new_key )
            del mat, new_key
            
        
        #####################################################################
        ### only write once: parameters, things that don't depend on time   #
        #####################################################################
        if write_time_static_objs:
            ###############################
            ### these are always returned #
            ###############################
            ### substitution rate matrix
            rate_matrix = out['rate_matrix'] #(C, A, A) or None
            if rate_matrix is not None:
                key = f'{prefix}_rate_matrix'
                write_matrix_to_npy( out_folder, rate_matrix, key )
                del key

                for c in range(rate_matrix.shape[0]):
                    mat_to_save = rate_matrix[c,...]
                    key = f'{prefix}_class-{c}_rate_matrix'
                    maybe_write_matrix_to_ascii( out_folder, mat_to_save, key )
                    del mat_to_save, key
            

            ###########################################
            ### these are returned if params were fit #
            ### (i.e. did not load these from files)  #
            ###########################################
            if not self.config['load_all']:
                ### logprob_emit_at_indel
                mat = np.exp( out['logprob_emit_at_indel'] ) #(C, A)
                new_key = f'{prefix}_logprob_emit_at_indel'.replace('log','')
                write_matrix_to_npy( out_folder, mat, new_key )
                maybe_write_matrix_to_ascii( out_folder, mat, new_key )
                del mat, new_key
                
                
                ### site class probs (if num_mixtures > 1)
                if self.config['num_mixtures'] > 1:
                    class_probs = nn.softmax(self.site_class_probability_module.class_logits) #(C,)
                    key = f'{prefix}_class_probs'
                    write_matrix_to_npy( out_folder, class_probs, key )
                    maybe_write_matrix_to_ascii( out_folder, class_probs, key )
                    del key
                    
            
                ### rate multipliers 
                # P(K|C) or P(K), if not 1
                if not self.rate_mult_module.prob_rate_mult_is_one:
                    rate_mult_probs = nn.softmax(self.rate_mult_module.rate_mult_prob_logits, axis=-1) #(C,K) or (K,)
                    key = f'{prefix}_rate_mult_probs'
                    write_matrix_to_npy( out_folder, rate_mult_probs, key )
                    maybe_write_matrix_to_ascii( out_folder, rate_mult_probs, key )
                    del key
            
                # \rho_{c,k} or \rho_k
                if not self.rate_mult_module.use_unit_rate_mult:
                    rate_multipliers = self.rate_mult_module.rate_multiplier_activation( self.rate_mult_module.rate_mult_logits ) #(C,K) or (K,)
                    key = f'{prefix}_rate_multipliers'
                    write_matrix_to_npy( out_folder, rate_multipliers, key )
                    maybe_write_matrix_to_ascii( out_folder, rate_multipliers, key )
                    del key
                    
                
                ### exchangeabilities, if gtr or hky85
                exchangeabilities = out['exchangeabilities'] #(A, A) or None
                
                if exchangeabilities is not None:
                    if self.subst_model_type == 'gtr':
                        key = f'{prefix}_gtr-exchangeabilities'
                        write_matrix_to_npy( out_folder, exchangeabilities, key )
                        maybe_write_matrix_to_ascii( out_folder, exchangeabilities, key )
                        del key
                        
                    elif self.subst_model_type == 'hky85':
                        ti = exchangeabilities[0, 2]
                        tv = exchangeabilities[0, 1]
                        arr = np.array( [ti, tv] )
                        key = f'{prefix}_hky85_ti_tv'
                        write_matrix_to_npy( out_folder, arr, key )
                        
                        with open(f'{out_folder}/ASCII_{prefix}_hky85_ti_tv.txt','w') as g:
                            g.write(f'transition rate, ti: {ti}\n')
                            g.write(f'transition rate, tv: {tv}')
                        del key, arr
                    
                    
                ####################################################
                ### extract transition paramaters, intermediates   # 
                ### needed for final scoring matrices              #
                ### (also does not depend on time)                 #
                ####################################################
                # lambda and mu
                # also record if you used any tkf approximations
                mu_min_val = self.transitions_module.mu_min_val #float
                mu_max_val = self.transitions_module.mu_max_val #float
                offs_min_val = self.transitions_module.offs_min_val #float
                offs_max_val = self.transitions_module.offs_max_val #float
                mu_offset_logits = self.transitions_module.tkf_mu_offset_logits #(2,)
            
                mu = bound_sigmoid(x = mu_offset_logits[0],
                                   min_val = mu_min_val,
                                   max_val = mu_max_val).item() #float
                
                offset = bound_sigmoid(x = mu_offset_logits[1],
                                         min_val = offs_min_val,
                                         max_val = offs_max_val).item() #float
                lam = mu * (1 - offset)  #float
                
                with open(f'{out_folder}/ASCII_tkf92_indel_params.txt','w') as g:
                    g.write(f'insert rate, lambda: {lam}\n')
                    g.write(f'deletion rate, mu: {mu}\n')
                    g.write(f'offset: {offset}\n\n')
                
                out_dict = {'lambda': np.array(lam), # shape=()
                            'mu': np.array(mu), # shape=()
                            'offset': np.array(offset)} # shape=()
                                
                # tkf92 r_ext param
                r_extend_min_val = self.transitions_module.r_extend_min_val
                r_extend_max_val = self.transitions_module.r_extend_max_val
                r_extend_logits = self.transitions_module.r_extend_logits #(C)
                
                r_extend = bound_sigmoid(x = r_extend_logits,
                                         min_val = r_extend_min_val,
                                         max_val = r_extend_max_val) #(C)
                
                mean_indel_lengths = 1 / (1 - r_extend) #(C)
                
                with open(f'{out_folder}/ASCII_tkf92_indel_params.txt','a') as g:
                    g.write(f'extension prob, r: ')
                    [g.write(f'{elem}\t') for elem in r_extend]
                    g.write('\n')
                    g.write(f'mean indel length: ')
                    [g.write(f'{elem}\t') for elem in mean_indel_lengths]
                    g.write('\n')
                
                out_dict['r_extend'] = r_extend #(C,)
            
                with open(f'{out_folder}/PARAMS-DICT_tkf92_indel_params.pkl','wb') as g:
                    pickle.dump(out_dict, g)
                del out_dict
                
        
    # don't think I need this anymore
    # def return_bound_sigmoid_limits(self):
    #     ### rate_matrix_module
    #     # exchangeabilities
    #     exchange_min_val = self.rate_matrix_module.exchange_min_val
    #     exchange_max_val = self.rate_matrix_module.exchange_max_val
    #     params_range = { "exchange_min_val": exchange_min_val,
    #                      "exchange_max_val": exchange_max_val }
        
    #     #rate multiplier
    #     if self.rate_mult_activation == 'bound_sigmoid':
    #         rate_mult_min_val = self.rate_matrix_module.rate_mult_min_val
    #         rate_mult_max_val = self.rate_matrix_module.rate_mult_max_val
    #         to_add = {"rate_mult_min_val": rate_mult_min_val,
    #                   "rate_mult_max_val": rate_mult_max_val}
    #         params_range = {**params_range, **to_add}
        
        
    #     ### transitions_module
    #     # insert rate lambda
    #     mu_min_val = self.transitions_module.mu_min_val
    #     mu_max_val = self.transitions_module.mu_max_val
        
    #     # offset (for deletion rate mu)
    #     offs_min_val = self.transitions_module.offs_min_val
    #     offs_max_val = self.transitions_module.offs_max_val
        
    #     # r extension probability
    #     r_extend_min_val = self.transitions_module.r_extend_min_val
    #     r_extend_max_val = self.transitions_module.r_extend_max_val
        
    #     to_add = {"mu_min_val": mu_min_val,
    #               "mu_max_val": mu_max_val,
    #               "offs_min_val": offs_min_val,
    #               "offs_max_val": offs_max_val,
    #               "r_extend_min_val": r_extend_min_val,
    #               "r_extend_max_val": r_extend_max_val}
            
    #     params_range = {**params_range, **to_add} 
        
    #     return params_range


    def _get_scoring_matrices( self,
                               t_array,
                               sow_intermediates: bool,
                               return_intermeds: bool = False):
        # Probability of each site class; is one, if no site clases
        log_class_probs = self.site_class_probability_module( sow_intermediates = sow_intermediates ) #(C,)
        
        # Substitution rate multipliers
        # both are (C, K)
        log_rate_mult_probs, rate_multipliers = self.rate_mult_module(sow_intermediates = sow_intermediates,
                                                                      log_class_probs = log_class_probs) #(C,K)
        
        
        ######################################################
        ### build log-transformed equilibrium distribution   #
        ### use this to score emissions from indels sites    #
        ######################################################
        log_equl_dist_per_mixture = self.equl_dist_module( sow_intermediates = sow_intermediates )  #(C, A)
        
        
        ####################################################
        ### build substitution log-probability matrix      #
        ### use this to score emissions from match sites   #
        ####################################################
        # to get joint logprob:
        # 1.) generate (C, K) different rate multipliers
        # 2.) using all these rate multipliers, get (C, K, A, A) different 
        #     rate matrices
        # 3.) multiply be time, matrix exponential, then multiply by P(anc) 
        #     to get P(x,y|c,k,t), a (T, C, K, A, A) matrix of substitution 
        #     probabilities at every time, site class, and rate class
        #
        # 4.) generate P(k|c) matrix (C, K)
        # 5.) multiply by raw P(x,y|c,k,t) rate matrices (T, )
        # 6.) sum_k P(k|c) P(x,y|c,k,t) = P(x,y|c,t); this is now ready to be
        #     multiplied by sites class P(c) in forward algorithm
        
        # cond_logprobs_per_mixture is (T, C, K, A, A) or (B, C, K, A, A)
        # subst_module_intermeds is a dictionary of intermediates
        out = self.logprob_subst_module( logprob_equl = log_equl_dist_per_mixture,
                                         rate_multipliers = rate_multipliers,
                                         t_array = t_array,
                                         sow_intermediates = sow_intermediates,
                                         return_cond = True,
                                         return_intermeds = True )        
        cond_subst_logprobs_per_mixture, subst_module_intermeds = out
        del out
        
        # get the joint probability; (T, C, K, A, A) or (B, C, K, A, A)
        joint_subst_logprobs_per_mixture = joint_logprob_emit_at_match_per_mixture( cond_logprob_emit_at_match_per_mixture = cond_subst_logprobs_per_mixture,
                                                                              log_equl_dist_per_mixture = log_equl_dist_per_mixture ) # (T, C, K, A, A) or (B, C, K, A, A)
        
        # marginalize over k possible rate matrices; this is now ready to be
        #   multiplied by site class P(c) in forward algorithm
        log_rate_mult_probs = log_rate_mult_probs[None, :, :, None, None] #(1, C, K, 1, 1)
        
        # P(x,y,k|c,t) = P(x,y|c,k,t) * P(k|c)
        # logP(x,y,k|c,t) = logP(x,y|c,k,t) + logP(k|c)
        subst_logprobs_rescaled = log_rate_mult_probs + joint_subst_logprobs_per_mixture #(T, C, K, A, A) or (B, C, K, A, A)
        
        # P(x,y|c,t) = \sum_k P(x,y,k|c,t)
        # logP(x,y|c,t) = \LSE_k logP(x,y,k|c,t)
        joint_logprob_emit_at_match = logsumexp( subst_logprobs_rescaled, axis=2 ) #(T, C, A, A) or (B, C, A, A)
        

        ####################################################
        ### build transition log-probability matrix        #
        ####################################################
        # all_transit_matrices['joint']: (T, C, C, S, S) or (B, C, C, S, S)
        # all_transit_matrices['conditional']: (T, C, C, S, S) or (B, C, C, S, S)
        # all_transit_matrices['marginal']: (C, C, 2, 2)
        all_transit_matrices, used_approx = self.transitions_module( t_array = t_array,
                                                                     log_class_probs = log_class_probs,
                                                                     sow_intermediates = sow_intermediates )
        
        out_dict = {'logprob_emit_at_indel': log_equl_dist_per_mixture, #(C, A)
                    'joint_logprob_emit_at_match': joint_logprob_emit_at_match, #(T, C, A, A) or (B, C, A, A)
                    'all_transit_matrices': all_transit_matrices, #dict
                    'used_approx': used_approx} #dict
        
        if return_intermeds:
            to_add = {'rate_matrix': subst_module_intermeds.get('rate_matrix',None), #(C,A,A) or None
                      'exchangeabilities': subst_module_intermeds.get('exchangeabilities',None), #(A,A) or None
                      'cond_subst_logprobs_per_mixture': cond_subst_logprobs_per_mixture, #(T, C, K, A, A) or (B, C, K, A, A)
                      'joint_subst_logprobs_per_mixture': joint_subst_logprobs_per_mixture} #(T, C, K, A, A) or (B, C, K, A, A)
            out_dict = {**out_dict, **to_add}
        
        return out_dict

class FragAndSiteClassesLoadAll(FragAndSiteClasses):
    """
    same as  FragAndSiteClasses, but load all parameters from files (excluding time, 
        exponential distribution parameter)
    
    
    Initialize with
    ----------------
    config : dict
        config['num_mixtures'] :  int
            number of emission site classes
        
        config['subst_model_type'] : {gtr, hky85}
            which substitution model
        
        config['norm_loss_by'] :  {desc_len, align_len}, optional
            what length to normalize loglikelihood by
            Default is 'desc_len'
        
        config['exponential_dist_param'] : int, optional
            There is an exponential prior over time; this provides the
            parameter for this during marginalization over times
            Default is 1
        
        config['times_from'] : {geometric, t_array_from_file, t_per_sample}
        
        config['filenames'] : files of parameters to load
        
    name : str
        class name, for flax
    
    Main methods here
    -----------------
    setup
    
    
    Inherited from FragAndSiteClasses
    ----------------------------------
    __call__
        unpack batch and calculate logP(anc, desc, align)
    
    calculate_all_loglikes
        calculate logP(anc, desc, align), logP(anc), logP(desc), and
        logP(desc, align | anc)
    
    write_params
        write parameters to files
    
    return_bound_sigmoid_limits
        after initializing model, get the limits for bound_sigmoid activations
    
    _init_rate_matrix_module
        decide what function to use for rate matrix
    
    _joint_logprob_align
        calculate logP(anc, desc, align)
    
    
    Methods inherited from models.model_utils.BaseClasses.ModuleBase
    -----------------------------------------------------------------
    sow_histograms_scalars
        for tensorboard logging
    """
    config: dict
    name: str
    
    def setup(self):
        ###################
        ### read config   #
        ###################
        # required
        num_mixtures = self.config['num_mixtures']
        self.indp_rate_mults = self.config['indp_rate_mults']
        self.subst_model_type = self.config['subst_model_type']
        self.times_from = self.config['times_from']
        
        # optional
        self.norm_reported_loss_by = self.config.get('norm_reported_loss_by', 'desc_len')
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        self.gap_tok = self.config.get('gap_tok', 43)
        
        
        ###############################################################
        ### modules for probability of being in latent site classes,  #
        ### probability of having a particular subsitution rate       #
        ### rate multiplier, and the rate multipliers themselves      #
        ###############################################################
        # Latent site class probabilities
        self.site_class_probability_module = SiteClassLogprobsFromFile(config = self.config,
                                                  name = f'get site class probabilities')
        
        # substitution rate multipliers
        if not self.indp_rate_mults:
            self.rate_mult_module = RateMultipliersPerClassFromFile(config = self.config,
                                                      name = f'get rate multipliers')
        
        elif self.indp_rate_mults:
            self.rate_mult_module = IndpRateMultipliersFromFile(config = self.config,
                                                      name = f'get rate multipliers')
        
        
        ###########################################
        ### module for equilibrium distribution   #
        ###########################################
        self.equl_dist_module = EqulDistLogprobsFromFile(config = self.config,
                                                       name = f'get equilibrium')
        
        
        ################################
        ### module for logprob subst   #
        ################################
        if self.subst_model_type == 'gtr':
            self.logprob_subst_module = GTRLogprobsFromFile( config = self.config,
                                                  name = f'gtr subst. model' )
            
        elif self.subst_model_type == 'f81':
            self.logprob_subst_module = F81LogprobsFromFile( config = self.config,
                                                     name = f'f81 subst. model' )

        elif self.subst_model_type == 'hky85':
            self.logprob_subst_module = HKY85LogprobsFromFile( config = self.config,
                                                    name = f'hky85 subst. model' )
            # this only works with DNA
            assert self.config['emission_alphabet_size'] == 4
            
        ###########################################
        ### module for transition probabilities   #
        ###########################################        
        # has to be tkf92
        self.transitions_module = TKF92TransitionLogprobsFromFile(config = self.config,
                                                 name = f'tkf92 indel model')
        