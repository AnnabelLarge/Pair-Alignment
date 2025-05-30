#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 04:33:00 2025

@author: annabel

models:
=======
MarkovFrags
MarkovFragsLoadAll
MarkovFragsHKY85
MarkovFragsHKY85LoadAll

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
                                                              GTRRateMat,
                                                              GTRRateMatFromFile,
                                                              SiteClassLogprobs,
                                                              SiteClassLogprobsFromFile,
                                                              HKY85RateMat,
                                                              HKY85RateMatFromFile)
from models.simple_site_class_predict.transition_models import (TKF92TransitionLogprobs,
                                                                TKF92TransitionLogprobsFromFile)
from models.simple_site_class_predict.model_functions import (bound_sigmoid,
                                                              safe_log,
                                                              get_cond_logprob_emit_at_match_per_class,
                                                              get_joint_logprob_emit_at_match_per_class,
                                                              scale_rate_multipliers,
                                                              joint_only_forward,
                                                              all_loglikes_forward,
                                                              marginalize_over_times)


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
        self.subst_model_type = self.config['subst_model_type']
        self.times_from = self.config['times_from']
        
        # optional
        self.norm_loss_by = self.config.get('norm_loss_by', 'desc_len') # this is for reporting
        self.norm_loss_by_length = self.config.get('norm_loss_by_length', False) # this affects the objective DURING training
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        self.gap_tok = self.config.get('gap_tok', 43)
        
        
        ###########################################
        ### module for equilibrium distribution   #
        ###########################################
        if num_mixtures == 1:
            self.indel_prob_module = EqulDistLogprobsFromCounts(config = self.config,
                                                       name = f'get equilibrium')
        elif num_mixtures > 1:
            self.indel_prob_module = EqulDistLogprobsPerClass(config = self.config,
                                                     name = f'get equilibrium')
        
        
        ###########################################
        ### module for substitution rate matrix   #
        ###########################################
        if self.subst_model_type.lower() == 'gtr':
            self.rate_matrix_module = GTRRateMat( config = self.config,
                                                  name = f'get rate matrix' )
            
        elif self.subst_model_type.lower() == 'hky85':
            self.rate_matrix_module = HKY85RateMat( config = self.config,
                                                    name = f'get rate matrix' )
        
        
        ##############################################################
        ### module for probability of being in latent site classes   #
        ##############################################################
        self.site_class_probability_module = SiteClassLogprobs(config = self.config,
                                                  name = f'get site class probabilities')
        
        
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
                  normalized by desired length (set by self.norm_loss_by)
              3.) whether or not you used approximation formula for TKF indel model
        """
        aligned_inputs = batch[0]
        
        # which times to use for scoring matrices
        if self.times_from =='t_per_sample':
            times_for_matrices = batch[1] #(B,)
            unique_time_per_branch = True
        
        elif self.times_from in ['geometric','t_array_from_file']:
            times_for_matrices = t_array #(T,)
            unique_time_per_branch = False
        
        # scoring matrices
        scoring_matrices_dict = self._get_scoring_matrices(t_array=times_for_matrices,
                                                           sow_intermediates=sow_intermediates)
        
        
        ### calculate joint loglike using 1D forward algorithm over latent site 
        ###   classes
        logprob_emit_at_indel = scoring_matrices_dict['logprob_emit_at_indel']
        joint_logprob_emit_at_match = scoring_matrices_dict['joint_logprob_emit_at_match']
        joint_logprob_transit =  scoring_matrices_dict['all_transit_matrices']['joint']
        forward_intermeds = joint_only_forward(aligned_inputs = aligned_inputs,
                                                 joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                                 logprob_emit_at_indel = logprob_emit_at_indel,
                                                 joint_logprob_transit = joint_logprob_transit,
                                                 unique_time_per_branch = unique_time_per_branch)
        joint_logprob_perSamp_maybePerTime = logsumexp(forward_intermeds[-1,...], 
                                                       axis = 1 if not unique_time_per_branch else 0) #(T, B) or (B,)
        
        
        ### marginalize over times where needed
        if (not unique_time_per_branch) and (t_array.shape[0] > 1):
            joint_neg_logP = -marginalize_over_times(logprob_perSamp_perTime = joint_logprob_perSamp_maybePerTime,
                                                     exponential_dist_param = self.exponential_dist_param,
                                                     t_array = times_for_matrices) #(B,)
             
        elif (not unique_time_per_branch) and (t_array.shape[0] == 1):
            joint_neg_logP = -joint_logprob_perSamp_maybePerTime[0,:] #(B,)
        
        elif unique_time_per_branch:
            joint_neg_logP = -joint_logprob_perSamp_maybePerTime #(B,)
            
            
        ### for REPORTING ONLY (not the objective function), normalize by length
        if self.norm_loss_by == 'desc_len':
            # where descendant is not pad or gap
            banned_toks = jnp.array([0,1,2,self.gap_tok])
            
        elif self.norm_loss_by == 'align_len':
            # where descendant is not pad (but could be gap)
            banned_toks = jnp.array([0,1,2])
        
        mask = ~jnp.isin( aligned_inputs[...,1], banned_toks)
        length_for_normalization = mask.sum(axis=1)
        joint_neg_logP_length_normed = joint_neg_logP / length_for_normalization
        del mask
        
        
        ### compile final outputs
        # aux dict
        aux_dict = {'joint_neg_logP': joint_neg_logP,
                    'joint_neg_logP_length_normed': joint_neg_logP_length_normed,
                    'used_approx': scoring_matrices_dict['used_approx']}
        
        # final loss
        if self.norm_loss_by_length:
            loss = jnp.mean( aux_dict['joint_neg_logP_length_normed'] )
        
        elif not self.norm_loss_by_length:
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
            unique_time_per_branch = True
        
        elif self.times_from in ['geometric','t_array_from_file']:
            times_for_matrices = t_array #(T,)
            unique_time_per_branch = False
            
        # get lengths, not including <bos> and <eos>
        align_len = ~jnp.isin( aligned_inputs[...,0], jnp.array([0,1,2]) )
        anc_len = ~jnp.isin( aligned_inputs[...,0], jnp.array([0,1,2,self.gap_tok]) )
        desc_len = ~jnp.isin( aligned_inputs[...,1], jnp.array([0,1,2,self.gap_tok]) )
        align_len = align_len.sum(axis=1)
        anc_len = anc_len.sum(axis=1)
        desc_len = desc_len.sum(axis=1)
        
        # score matrices
        scoring_matrices_dict = self._get_scoring_matrices( t_array=t_array,
                                                            sow_intermediates=False )
        
        
        ### get all log-likelihoods
        logprob_emit_at_indel = scoring_matrices_dict['logprob_emit_at_indel']
        joint_logprob_emit_at_match = scoring_matrices_dict['joint_logprob_emit_at_match']
        all_transit_matrices =  scoring_matrices_dict['all_transit_matrices']
        out = all_loglikes_forward( aligned_inputs = aligned_inputs,
                                    joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                    logprob_emit_at_indel = logprob_emit_at_indel,
                                    all_transit_matrices = all_transit_matrices,
                                    unique_time_per_branch = unique_time_per_branch)
        
        # for joint loglike: marginalize over times where needed
        if (not unique_time_per_branch) and (t_array.shape[0] > 1):
            joint_logprob_perSamp_maybePerTime = out['joint_neg_logP']  #(T,B)
            overwrite_joint_neg_logP = -marginalize_over_times(logprob_perSamp_perTime = -joint_logprob_perSamp_maybePerTime,
                                                     exponential_dist_param = self.exponential_dist_param,
                                                     t_array = times_for_matrices) #(B,)
            out['joint_neg_logP'] = overwrite_joint_neg_logP #(B,)
             
        elif (not unique_time_per_branch) and (t_array.shape[0] == 1):
            out['joint_neg_logP'] = out['joint_neg_logP'][0,:] #(B,)
        
        
        ### conditional comes from joint / anc
        cond_neg_logP = - (-out['joint_neg_logP'] - -out['anc_neg_logP'])
        
        
        ### for REPORTING ONLY (not the objective function), normalize by length
        anc_neg_logP_length_normed = out['anc_neg_logP'] / anc_len
        desc_neg_logP_length_normed = out['desc_neg_logP'] / desc_len
        
        if self.norm_loss_by == 'desc_len':
            joint_neg_logP_length_normed = out['joint_neg_logP'] / desc_len
            cond_neg_logP_length_normed = cond_neg_logP / desc_len
        
        elif self.norm_loss_by == 'align_len':
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
        
        if write_time_static_objs:
            with open(f'{out_folder}/activations_and_times_used.tsv','w') as g:
                act = self.rate_matrix_module.rate_mult_activation
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
        
        #####################
        ### Full matrices   #
        #####################
        out = self._get_scoring_matrices(t_array=t_array,
                                        sow_intermediates=False)
        
        ### these depend on time
        # rho * Q * t
        to_expm = np.squeeze( out['to_expm'] )
        
        with open(f'{out_folder}/{prefix}_to_expm.npy', 'wb') as g:
            np.save(g, to_expm)
        
        if len(to_expm.shape) <= 2:
            np.savetxt( f'{out_folder}/{prefix}_ASCII_to_expm.tsv', 
                        to_expm, 
                        fmt = '%.4f',
                        delimiter= '\t' )
        
        del to_expm, g
    
        # final joint prob of match (after LSE over classes)
        mat = np.exp(out['joint_logprob_emit_at_match'])
        new_key = 'joint_logprob_emit_at_match'.replace('logprob','prob')
        
        with open(f'{out_folder}/{prefix}_{new_key}.npy', 'wb') as g:
            np.save(g, mat)
        
        mat = np.squeeze(mat)
        if len(mat.shape) <= 2:
            np.savetxt( f'{out_folder}/{prefix}_ASCII_{new_key}.tsv', 
                        np.array(mat), 
                        fmt = '%.4f',
                        delimiter= '\t' )
        
        del new_key, mat, g
    
        # transition matrices, or P(emit) for geometrically distributed model
        for key, mat in out['all_transit_matrices'].items():
            mat = np.exp(mat)
            new_key = key.replace('logprob','prob')
            
            with open(f'{out_folder}/{prefix}_{new_key}_transit_matrix.npy', 'wb') as g:
                np.save(g, mat)
            
        
        ### these do not; only write once 
        if write_time_static_objs:
            # equilibrium distribution 
            mat = np.exp(out['logprob_emit_at_indel'])
            new_key = 'logprob_emit_at_indel'.replace('logprob','prob')
            
            with open(f'{out_folder}/{prefix}_{new_key}.npy', 'wb') as g:
                np.save(g, mat)
            
            mat = np.squeeze(mat)
            np.savetxt( f'{out_folder}/{prefix}_ASCII_{new_key}.tsv', 
                        mat, 
                        fmt = '%.4f',
                        delimiter= '\t' )
                
            del new_key, mat, g
        
            # emission from match sites
            # rho * Q
            scaled_rate_mat_per_class = out['rate_mat_times_rho']
            for c in range(scaled_rate_mat_per_class.shape[0]):
                mat_to_save = scaled_rate_mat_per_class[c,...]
                
                with open(f'{out_folder}/{prefix}_class-{c}_rate_matrix_times_rho.npy', 'wb') as g:
                    np.save(g, mat_to_save)
                
                np.savetxt( f'{out_folder}/{prefix}_ASCII_class-{c}_rate_matrix_times_rho.tsv', 
                            np.array(mat_to_save), 
                            fmt = '%.4f',
                            delimiter= '\t' )
                
                del mat_to_save, g
        
        
            ###################################################
            ### extract emissions paramaters, intermediates   # 
            ### needed for final scoring matrices             #
            ###################################################
            ### site class probs
            if 'class_logits' in dir(self.site_class_probability_module):
                class_probs = nn.softmax(self.site_class_probability_module.class_logits)
                with open(f'{out_folder}/PARAMS_class_probs.txt','w') as g:
                    [g.write(f'{elem.item()}\n') for elem in class_probs]
            
            
            ### exchangeabilities
            if 'exchangeabilities_logits_vec' in dir(self.rate_matrix_module):
                exch_logits = self.rate_matrix_module.exchangeabilities_logits_vec
                exchangeabilities = self.rate_matrix_module.exchange_activation( exch_logits )
                
                if self.subst_model_type.lower() == 'gtr':
                    np.savetxt( f'{out_folder}/PARAMS_exchangeabilities.tsv', 
                                np.array(exchangeabilities), 
                                fmt = '%.4f',
                                delimiter= '\t' )
                    
                    with open(f'{out_folder}/PARAMS_exchangeabilities.npy','wb') as g:
                        jnp.save(g, exchangeabilities)
                
                elif self.subst_model_type.lower() == 'hky85':
                    with open(f'{out_folder}/PARAMS_HKY85RateMat_model.txt','w') as g:
                        g.write(f'transition rate, ti: {exchangeabilities[1]}\n')
                        g.write(f'transition rate, tv: {exchangeabilities[0]}')
                    
                    
            ### rate multipliers
            if 'rate_mult_logits' in dir(self.rate_matrix_module):
                norm_rate_mults = self.rate_matrix_module.norm_rate_mults
                rate_mult_logits = self.rate_matrix_module.rate_mult_logits
                rate_mult = self.rate_matrix_module.rate_multiplier_activation( rate_mult_logits )
                
                if norm_rate_mults:
                    rate_mult = scale_rate_multipliers( unnormed_rate_multipliers = rate_mult,
                                            log_class_probs = jnp.log(class_probs) )
    
                with open(f'{out_folder}/PARAMS_rate_multipliers.txt','w') as g:
                    [g.write(f'{elem.item()}\n') for elem in rate_mult]
            
            
            ### equilibrium distribution
            if 'logits' in dir(self.indel_prob_module):
                equl_logits = self.indel_prob_module.logits
                equl_dist = nn.softmax( equl_logits, axis=1 )
                
                np.savetxt( f'{out_folder}/PARAMS_equilibriums.tsv', 
                            np.array(equl_dist), 
                            fmt = '%.4f',
                            delimiter= '\t' )
                
                with open(f'{out_folder}/PARAMS-ARR_equilibriums.npy','wb') as g:
                    jnp.save(g, equl_dist)
                    
                    
                    
            ####################################################
            ### extract transition paramaters, intermediates   # 
            ### needed for final scoring matrices              #
            ### (also does not depend on time)                 #
            ####################################################
            # always write lambda and mu
            # also record if you used any tkf approximations
            if 'tkf_lam_mu_logits' in dir(self.transitions_module):
                lam_min_val = self.transitions_module.lam_min_val
                lam_max_val = self.transitions_module.lam_max_val
                offs_min_val = self.transitions_module.offs_min_val
                offs_max_val = self.transitions_module.offs_max_val
                lam_mu_logits = self.transitions_module.tkf_lam_mu_logits
            
                lam = bound_sigmoid(x = lam_mu_logits[0],
                                      min_val = lam_min_val,
                                      max_val = lam_max_val)
                
                offset = bound_sigmoid(x = lam_mu_logits[1],
                                         min_val = offs_min_val,
                                         max_val = offs_max_val)
                mu = lam / ( 1 -  offset) 
                
                r_extend_min_val = self.transitions_module.r_extend_min_val
                r_extend_max_val = self.transitions_module.r_extend_max_val
                r_extend_logits = self.transitions_module.r_extend_logits
                
                r_extend = bound_sigmoid(x = r_extend_logits,
                                           min_val = r_extend_min_val,
                                           max_val = r_extend_max_val)
                
                mean_indel_lengths = 1 / (1 - r_extend)
                
                with open(f'{out_folder}/PARAMS_TKF92_indel_params.txt','w') as g:
                    g.write(f'insert rate, lambda: {lam}\n')
                    g.write(f'deletion rate, mu: {mu}\n')
                    g.write(f'extension prob, r: ')
                    [g.write(f'{elem}\t') for elem in r_extend]
                    g.write('\n')
                    g.write(f'mean indel length: ')
                    [g.write(f'{elem}\t') for elem in mean_indel_lengths]
                    g.write('\n')
        
    def return_bound_sigmoid_limits(self):
        ### rate_matrix_module
        # exchangeabilities
        exchange_min_val = self.rate_matrix_module.exchange_min_val
        exchange_max_val = self.rate_matrix_module.exchange_max_val
        params_range = { "exchange_min_val": exchange_min_val,
                         "exchange_max_val": exchange_max_val }
        
        #rate multiplier
        if self.rate_mult_activation == 'bound_sigmoid':
            rate_mult_min_val = self.rate_matrix_module.rate_mult_min_val
            rate_mult_max_val = self.rate_matrix_module.rate_mult_max_val
            to_add = {"rate_mult_min_val": rate_mult_min_val,
                      "rate_mult_max_val": rate_mult_max_val}
            params_range = {**params_range, **to_add}
        
        
        ### transitions_module
        # insert rate lambda
        lam_min_val = self.transitions_module.lam_min_val
        lam_max_val = self.transitions_module.lam_max_val
        
        # offset (for deletion rate mu)
        offs_min_val = self.transitions_module.offs_min_val
        offs_max_val = self.transitions_module.offs_max_val
        
        # r extension probability
        r_extend_min_val = self.transitions_module.r_extend_min_val
        r_extend_max_val = self.transitions_module.r_extend_max_val
        
        to_add = {"lam_min_val": lam_min_val,
                  "lam_max_val": lam_max_val,
                  "offs_min_val": offs_min_val,
                  "offs_max_val": offs_max_val,
                  "r_extend_min_val": r_extend_min_val,
                  "r_extend_max_val": r_extend_max_val}
            
        params_range = {**params_range, **to_add} 
        
        return params_range


    def _get_scoring_matrices( self,
                               t_array,
                               sow_intermediates: bool):
        # Probability of each site class; is one, if no site clases
        log_class_probs = self.site_class_probability_module( sow_intermediates = sow_intermediates ) #(C,)
        
        
        ######################################################
        ### build log-transformed equilibrium distribution   #
        ### use this to score emissions from indels sites    #
        ######################################################
        logprob_emit_at_indel = self.indel_prob_module( sow_intermediates = sow_intermediates )  #(C, A)
        
        
        ####################################################
        ### build substitution log-probability matrix      #
        ### use this to score emissions from match sites   #
        ####################################################
        # rho * Q
        scaled_rate_mat_per_class = self.rate_matrix_module( logprob_equl = logprob_emit_at_indel,
                                                             log_class_probs = log_class_probs,
                                                             sow_intermediates = sow_intermediates ) #(C, A, A)
        
        # conditional probability
        # cond_logprob_emit_at_match is (T, C, A, A) or (B, C, A, A)
        # to_expm is (T, C, A, A) or (B, C, A, A)
        out = get_cond_logprob_emit_at_match_per_class(t_array = t_array,
                                                        scaled_rate_mat_per_class = scaled_rate_mat_per_class)
        cond_logprob_emit_at_match, to_expm = out 
        del out
        
        # joint probability
        joint_logprob_emit_at_match = get_joint_logprob_emit_at_match_per_class( cond_logprob_emit_at_match_per_class = cond_logprob_emit_at_match,
                                                            log_equl_dist_per_class = logprob_emit_at_indel) #(T, C, A, A) or (B, C, A, A)
        

        ####################################################
        ### build transition log-probability matrix        #
        ####################################################
        # all_transit_matrices['joint']: (T, C, C, S, S) or (B, C, C, S, S)
        # all_transit_matrices['conditional']: (T, C, C, S, S) or (B, C, C, S, S)
        # all_transit_matrices['marginal']: (C, C, 2, 2)
        all_transit_matrices, used_approx = self.transitions_module( t_array = t_array,
                                                                     log_class_probs = log_class_probs,
                                                                     sow_intermediates = sow_intermediates )
        
        out_dict = {'logprob_emit_at_indel': logprob_emit_at_indel, #(C, A)
                    'joint_logprob_emit_at_match': joint_logprob_emit_at_match, #(T, C, A, A)
                    'all_transit_matrices': all_transit_matrices, #dict
                    'rate_mat_times_rho': scaled_rate_mat_per_class, #(C, A, A)
                    'to_expm': to_expm, #(T, C, A, A)
                    'cond_logprob_emit_at_match': cond_logprob_emit_at_match, #(T, C, A, A)
                    'used_approx': used_approx} #dict
        
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
        self.subst_model_type = self.config['subst_model_type']
        self.times_from = self.config['times_from']
        
        # optional
        self.norm_loss_by = self.config.get('norm_loss_by', 'desc_len') # this is for reporting
        self.norm_loss_by_length = self.config.get('norm_loss_by_length', False) # doesn't really matter, but keep anyways
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        self.gap_tok = self.config.get('gap_tok', 43)
        
        # update config file
        self.config['num_tkf_fragment_classes'] = num_mixtures
        
        
        ###########################################
        ### module for equilibrium distribution   #
        ###########################################
        self.indel_prob_module = EqulDistLogprobsFromFile(config = self.config,
                                                          name = f'get equilibrium')
        
        
        ###########################################
        ### module for substitution rate matrix   #
        ###########################################
        if self.subst_model_type.lower() == 'gtr':
            self.rate_matrix_module = GTRRateMatFromFile( config = self.config,
                                                  name = f'get rate matrix' )
            
        elif self.subst_model_type.lower() == 'hky85':
            self.rate_matrix_module = HKY85RateMatFromFile( config = self.config,
                                                    name = f'get rate matrix' )
        
        
        ##############################################################
        ### module for probability of being in latent site classes   #
        ##############################################################
        self.site_class_probability_module = SiteClassLogprobsFromFile(config = self.config,
                                                  name = f'get site class probabilities')
        
        
        ###########################################
        ### module for transition probabilities   #
        ###########################################        
        # has to be tkf92
        self.transitions_module = TKF92TransitionLogprobsFromFile(config = self.config,
                                                 name = f'tkf92 indel model')