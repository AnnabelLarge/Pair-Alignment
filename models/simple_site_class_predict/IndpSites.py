#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 04:33:00 2025

@author: annabel

"""
import numpy as np
import pickle

# jumping jax and leaping flax
from flax import linen as nn
import jax
from jax._src.typing import Array, ArrayLike
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
from models.simple_site_class_predict.transition_models import (TKF91TransitionLogprobs,
                                                                TKF92TransitionLogprobs,
                                                                TKF91TransitionLogprobsFromFile,
                                                                TKF92TransitionLogprobsFromFile,
                                                                GeomLenTransitionLogprobs,
                                                                GeomLenTransitionLogprobsFromFile)
from models.simple_site_class_predict.model_functions import (bound_sigmoid,
                                                              safe_log,
                                                              scale_rate_multipliers,
                                                              get_cond_logprob_emit_at_match_per_class,
                                                              get_joint_logprob_emit_at_match_per_class,
                                                              lse_over_match_logprobs_per_class,
                                                              lse_over_equl_logprobs_per_class,
                                                              joint_prob_from_counts,
                                                              anc_marginal_probs_from_counts,
                                                              desc_marginal_probs_from_counts)


class IndpSites(ModuleBase):
    """
    pairHMM that finds joint loglikelihood of alignments, P(Anc, Desc, Align)
    
    if using one time per sample, and wanting to QUANTIZE the time, need a 
        different model
    
    
    Initialize with
    ----------------
    config : dict
        config['num_mixtures'] :  int
            number of emission site classes
        
        config['subst_model_type'] : {gtr, hky85}
            which substitution model
        
        config['indel_model_type'] : {tkf91, tkf92, None}
            which indel model, if any
            
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
        self.subst_model_type = self.config['subst_model_type']
        self.indel_model_type = self.config['indel_model_type']
        self.times_from = self.config['times_from']
        num_mixtures = self.config['num_mixtures']
        
        # optional
        self.norm_loss_by = self.config.get('norm_loss_by', 'desc_len') # this is for reporting
        self.norm_loss_by_length = self.config.get('norm_loss_by_length', False) # this is the objective during training
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        
        
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
        if self.indel_model_type is None:
            self.transitions_module = GeomLenTransitionLogprobs(config = self.config,
                                                     name = f'geom seq lengths model')
            
        elif self.indel_model_type.lower() == 'tkf91':
            self.transitions_module = TKF91TransitionLogprobs(config = self.config,
                                                     name = f'tkf91 indel model')
        
        elif self.indel_model_type.lower() == 'tkf92':
            self.transitions_module = TKF92TransitionLogprobs(config = self.config,
                                                     name = f'tkf92 indel model')
        
        
    def __call__(self,
                 batch,
                 t_array,
                 sow_intermediates: bool,
                 whole_dset_grad_desc: bool=False):
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
        # which times to use for scoring matrices
        if self.times_from =='t_per_sample':
            times_for_matrices = batch[4] #(B,)
        
        elif self.times_from in ['geometric','t_array_from_file']:
            times_for_matrices = t_array #(T,)

        # get the scoring matrices needed
        scoring_matrices_dict = self._get_scoring_matrices(t_array=times_for_matrices,
                                        sow_intermediates=sow_intermediates)
        
        # calculate loglikelihoods; provide both batch and t_array, just in case
        # time marginalization hidden in joint_prob_from_counts function
        aux_dict = joint_prob_from_counts( batch = batch,
                                           times_from = self.times_from,
                                           score_indels = False if self.indel_model_type is None else True,
                                           scoring_matrices_dict = scoring_matrices_dict,
                                           t_array = t_array,
                                           exponential_dist_param = self.exponential_dist_param,
                                           norm_loss_by = self.norm_loss_by )
        aux_dict['used_approx'] = scoring_matrices_dict['used_approx']
        

        # if doing stochastic gradient descent, take the average over the batch
        # if doing gradient descent with whole dataset, only use the sum
        reduction = jnp.mean if not whole_dset_grad_desc else jnp.sum

        if self.norm_loss_by_length:
            loss = reduction( aux_dict['joint_neg_logP_length_normed'] )
        
        elif not self.norm_loss_by_length:
            loss = reduction( aux_dict['joint_neg_logP'] )
            
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
        9.) whether or not you used approximation formula for TKF indel model
        """
        # which times to use for scoring matrices
        if self.times_from =='t_per_sample':
            times_for_matrices = batch[4] #(B,)
        
        elif self.times_from in ['geometric','t_array_from_file']:
            times_for_matrices = t_array #(T,)

        # get the scoring matrices needed
        scoring_matrices_dict = self._get_scoring_matrices(t_array=times_for_matrices,
                                        sow_intermediates=False)
        
        #########################
        ### joint probability   #
        #########################
        # time marginalization hidden in joint_prob_from_counts function
        aux_dict = joint_prob_from_counts( batch = batch,
                                           times_from = self.times_from,
                                           score_indels = False if self.indel_model_type is None else True,
                                           scoring_matrices_dict = scoring_matrices_dict,
                                           t_array = t_array,
                                           exponential_dist_param = self.exponential_dist_param,
                                           norm_loss_by = self.norm_loss_by )
        aux_dict['used_approx'] = scoring_matrices_dict['used_approx']
        
        
        #####################################
        ### ancestor marginal probability   #
        #####################################
        to_add = anc_marginal_probs_from_counts( batch = batch,
                                            score_indels = False if self.indel_model_type is None else True,
                                            scoring_matrices_dict = scoring_matrices_dict )
        
        aux_dict = {**aux_dict, **to_add}
        del to_add
        
        
        #######################################
        ### descendant marginal probability   #
        #######################################
        to_add = desc_marginal_probs_from_counts( batch = batch,
                                            score_indels = False if self.indel_model_type is None else True,
                                            scoring_matrices_dict = scoring_matrices_dict )
        
        aux_dict = {**aux_dict, **to_add}
        del to_add
        
        
        #####################################################
        ### calculate conditional from joint and marginal   #
        #####################################################
        cond_neg_logP = -( -aux_dict['joint_neg_logP'] - -aux_dict['anc_neg_logP'] )
        length_for_normalization = aux_dict['align_length_for_normalization']
        cond_neg_logP_length_normed = cond_neg_logP / length_for_normalization
            
        aux_dict['cond_neg_logP'] = cond_neg_logP
        aux_dict['cond_neg_logP_length_normed'] = cond_neg_logP_length_normed
        
        return aux_dict
        
    
    def _get_scoring_matrices(self,
                             t_array,
                             sow_intermediates: bool):
        # Probability of each site class; is one, if no site clases
        log_class_probs = self.site_class_probability_module(sow_intermediates = sow_intermediates) #(C,)
        
        
        ######################################################
        ### build log-transformed equilibrium distribution   #
        ### use this to score emissions from indels sites    #
        ######################################################
        log_equl_dist_per_class = self.indel_prob_module(sow_intermediates = sow_intermediates) # (C, A)
        logprob_emit_at_indel = lse_over_equl_logprobs_per_class( log_class_probs = log_class_probs,
                                                                  log_equl_dist_per_class = log_equl_dist_per_class) #(A,)
        
        
        ####################################################
        ### build substitution log-probability matrix      #
        ### use this to score emissions from match sites   #
        ####################################################
        # rho * Q
        scaled_rate_mat_per_class = self.rate_matrix_module(logprob_equl = log_equl_dist_per_class,
                                                            log_class_probs = log_class_probs,
                                                            sow_intermediates = sow_intermediates) #(C, A, A)
        
        # conditional probability
        # cond_logprob_emit_at_match_per_class is (T, C, A, A)
        # to_expm is (T, C, A, A)
        out = get_cond_logprob_emit_at_match_per_class(t_array = t_array,
                                                        scaled_rate_mat_per_class = scaled_rate_mat_per_class)
        cond_logprob_emit_at_match_per_class, to_expm = out 
        del out
        
        # joint probability
        joint_logprob_emit_at_match_per_class = get_joint_logprob_emit_at_match_per_class( cond_logprob_emit_at_match_per_class = cond_logprob_emit_at_match_per_class,
                                                                        log_equl_dist_per_class = log_equl_dist_per_class) #(T, C, A, A)
        joint_logprob_emit_at_match = lse_over_match_logprobs_per_class(log_class_probs = log_class_probs,
                                               joint_logprob_emit_at_match_per_class = joint_logprob_emit_at_match_per_class) #(T, A, A)
        
        
        ####################################################
        ### build transition log-probability matrix        #
        ####################################################
        if self.indel_model_type == 'tkf91':
            # all_transit_matrices['joint']: (T, C, C, S, S) or (B, C, C, S, S)
            # all_transit_matrices['conditional']: (T, C, C, S, S) or (B, C, C, S, S)
            # all_transit_matrices['marginal']: (C, C, 2, 2)
            # used_approx is a dictionary of boolean arrays
            all_transit_matrices, used_approx = self.transitions_module(t_array = t_array,
                                                           sow_intermediates = sow_intermediates) 
        
        elif self.indel_model_type == 'tkf92':
            # all_transit_matrices['joint']: (T, C, C, S, S) or (B, C, C, S, S)
            # all_transit_matrices['conditional']: (T, C, C, S, S) or (B, C, C, S, S)
            # all_transit_matrices['marginal']: (C, C, 2, 2)
            # used_approx is a dictionary of boolean arrays
            all_transit_matrices, used_approx = self.transitions_module(t_array = t_array,
                                                           log_class_probs = jnp.array([0.]),
                                                           sow_intermediates = sow_intermediates)
            
            # C=1, so remove intermediate dims
            all_transit_matrices['joint'] = all_transit_matrices['joint'][:,0,0,...] # (T, S, S)
            all_transit_matrices['conditional'] = all_transit_matrices['conditional'][:,0,0,...] # (T, S, S)
            all_transit_matrices['marginal'] = all_transit_matrices['marginal'][0,0,...] # (T, S, S)
        
        elif self.indel_model_type is None:
            # all_transit_matrices['joint']: (2, 1)
            # all_transit_matrices['conditional']: (2, 1)
            # all_transit_matrices['marginal']: (2, 1)
            # used_approx is None
            all_transit_matrices, used_approx = self.transitions_module(sow_intermediates = sow_intermediates)
            
        out_dict = {'logprob_emit_at_indel': logprob_emit_at_indel, #(A,)
                    'joint_logprob_emit_at_match': joint_logprob_emit_at_match, #(T,A,A)
                    'all_transit_matrices': all_transit_matrices, #dict
                    'rate_mat_times_rho': scaled_rate_mat_per_class, #(C,A,A)
                    'to_expm': to_expm, #(T,C,A,A)
                    'cond_logprob_emit_at_match': cond_logprob_emit_at_match_per_class, #(T,C,A,A)
                    'used_approx': used_approx} #dict
        
        return out_dict
    
    
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
        if self.indel_model_type is not None:
            for key, mat in out['all_transit_matrices'].items():
                mat = np.exp(mat)
                new_key = key.replace('logprob','prob')
                
                with open(f'{out_folder}/{prefix}_{new_key}_transit_matrix.npy', 'wb') as g:
                    np.save(g, mat)
                
                mat = np.squeeze(mat)
                if len(mat.shape) <= 2:
                    np.savetxt( f'{out_folder}/{prefix}_ASCII_{new_key}_transit_matrix.tsv', 
                                np.array(mat), 
                                fmt = '%.4f',
                                delimiter= '\t' )
                
                del key, mat, g
        
        
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
            ### under geometric length (only scoring subs)
            if self.indel_model_type is None:
                geom_p_emit = nn.sigmoid(self.transitions_module.p_emit_logit) #(1,)
                with open(f'{out_folder}/PARAMS_geom_seq_len.txt','w') as g:
                    g.write(f'P(emit): {geom_p_emit}\n')
                    g.write(f'1-P(emit): {1 - geom_p_emit}\n')
                    
            ### for TKF models
            elif self.indel_model_type.lower() in ['tkf91', 'tkf92']:
                # always write lambda and mu
                # also record if you used any tkf approximations
                if 'tkf_mu_offset_logits' in dir(self.transitions_module):
                    mu_min_val = self.transitions_module.mu_min_val
                    mu_max_val = self.transitions_module.mu_max_val
                    offs_min_val = self.transitions_module.offs_min_val
                    offs_max_val = self.transitions_module.offs_max_val
                    mu_offset_logits = self.transitions_module.tkf_mu_offset_logits
                
                    mu = bound_sigmoid(x = mu_offset_logits[0],
                                       min_val = mu_min_val,
                                       max_val = mu_max_val)
                    
                    offset = bound_sigmoid(x = mu_offset_logits[1],
                                             min_val = offs_min_val,
                                             max_val = offs_max_val)
                    lam = mu * (1 - offset) 
                    
                    with open(f'{out_folder}/PARAMS_{self.indel_model_type}_indel_params.txt','w') as g:
                        g.write(f'insert rate, lambda: {lam}\n')
                        g.write(f'deletion rate, mu: {mu}\n')
                        g.write(f'offset: {offset}\n\n')
                    
                    out_dict = {'lambda': lam,
                                'mu': mu,
                                'offset': offset}
                                
                # if tkf92, have extra r_ext param
                if self.indel_model_type == 'tkf92':
                    r_extend_min_val = self.transitions_module.r_extend_min_val
                    r_extend_max_val = self.transitions_module.r_extend_max_val
                    r_extend_logits = self.transitions_module.r_extend_logits
                    
                    r_extend = bound_sigmoid(x = r_extend_logits,
                                               min_val = r_extend_min_val,
                                               max_val = r_extend_max_val)
                    
                    mean_indel_lengths = 1 / (1 - r_extend)
                    
                    with open(f'{out_folder}/PARAMS_{self.indel_model_type}_indel_params.txt','a') as g:
                        g.write(f'extension prob, r: ')
                        [g.write(f'{elem}\t') for elem in r_extend]
                        g.write('\n')
                        g.write(f'mean indel length: ')
                        [g.write(f'{elem}\t') for elem in mean_indel_lengths]
                        g.write('\n')
                    
                    out_dict['r_extend'] = r_extend
                
                with open(f'{out_folder}/{self.indel_model_type}_indel_params.pkl','wb') as g:
                    pickle.dump(out_dict)
                del out_dict

        
    def return_bound_sigmoid_limits(self):
        ### rate_matrix_module
        # exchangeabilities
        exchange_min_val = self.rate_matrix_module.exchange_min_val
        exchange_max_val = self.rate_matrix_module.exchange_max_val
        params_range = { "exchange_min_val": exchange_min_val,
                         "exchange_max_val": exchange_max_val}
        
        #rate multiplier
        if self.rate_mult_activation == 'bound_sigmoid':
            rate_mult_min_val = self.rate_matrix_module.rate_mult_min_val
            rate_mult_max_val = self.rate_matrix_module.rate_mult_max_val
            to_add = {"rate_mult_min_val": rate_mult_min_val,
                      "rate_mult_max_val": rate_mult_max_val}
            params_range = {**params_range, **to_add}
        
        
        ### transitions_module
        if self.indel_model_type is not None:
            # delete rate mu
            mu_min_val = self.transitions_module.mu_min_val
            mu_max_val = self.transitions_module.mu_max_val
            
            # offset (for deletion rate mu)
            offs_min_val = self.transitions_module.offs_min_val
            offs_max_val = self.transitions_module.offs_max_val
            
            to_add = {"mu_min_val": mu_min_val,
                      "mu_max_val": mu_max_val,
                      "offs_min_val": offs_min_val,
                      "offs_max_val": offs_max_val}
            
            if self.indel_model_type.lower() == 'tkf92':
                # r extension probability
                r_extend_min_val = self.transitions_module.r_extend_min_val
                r_extend_max_val = self.transitions_module.r_extend_max_val
        
                to_add['r_extend_min_val'] = r_extend_min_val
                to_add['r_extend_max_val'] = r_extend_max_val
            
            params_range = {**params_range, **to_add} 
        
        return params_range


class IndpSitesLoadAll(IndpSites):
    """
    like IndpSites, but load all parameters to use (excluding time, 
        exponential distribution parameter)
    
    
    Initialize with
    ----------------
    config : dict
        config['num_mixtures'] :  int
            number of emission site classes
            
        config['subst_model_type'] : {gtr, hky85}
            which substitution model
        
        config['indel_model_type'] : {tkf91, tkf92, None}
            which indel model, if any
            
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
    
    write_params
        write parameters to files
        
    Inherited from IndpSites
    -------------------------
    __call__
        unpack batch and calculate logP(anc, desc, align)
    
    calculate_all_loglikes
        calculate logP(anc, desc, align), logP(anc), logP(desc), and
        logP(desc, align | anc)
    
    
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
        self.subst_model_type = self.config['subst_model_type']
        self.indel_model_type = self.config['indel_model_type']
        self.times_from = self.config['times_from']
        self.norm_loss_by_length = self.config['norm_loss_by_length']
        num_mixtures = self.config['num_mixtures']
        
        # optional
        self.norm_loss_by = self.config.get('norm_loss_by', 'desc_len')
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        
        
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
        if self.indel_model_type is None:
            self.transitions_module = GeomLenTransitionLogprobsFromFile(config = self.config,
                                                     name = f'geom seq lengths model')
            
        elif self.indel_model_type.lower() == 'tkf91':
            self.transitions_module = TKF91TransitionLogprobsFromFile(config = self.config,
                                                     name = f'tkf91 indel model')
        
        elif self.indel_model_type.lower() == 'tkf92':
            self.transitions_module = TKF92TransitionLogprobsFromFile(config = self.config,
                                                     name = f'tkf92 indel model')
    
    def write_params(self,
                     t_array,
                     out_folder: str,
                     prefix: str,
                     write_time_static_objs: bool):
        if write_time_static_objs:
            with open(f'{out_folder}/activations_and_times_used.tsv','w') as g:
                if 'rate_mult_activation' in dir(self.rate_matrix_module):
                    act = self.rate_matrix_module.rate_mult_activation
                else:
                    act = 'N/A'
                
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
        if self.indel_model_type is not None:
            for key, mat in out['all_transit_matrices'].items():
                mat = np.exp(mat)
                new_key = key.replace('logprob','prob')
                
                with open(f'{out_folder}/{prefix}_{new_key}_transit_matrix.npy', 'wb') as g:
                    np.save(g, mat)
                
                mat = np.squeeze(mat)
                if len(mat.shape) <= 2:
                    np.savetxt( f'{out_folder}/{prefix}_ASCII_{new_key}_transit_matrix.tsv', 
                                np.array(mat), 
                                fmt = '%.4f',
                                delimiter= '\t' )
                
                del key, mat, g
        
        
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