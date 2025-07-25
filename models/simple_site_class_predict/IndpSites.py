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
                                                              GTRLogprobs,
                                                              GTRLogprobsFromFile,
                                                              SiteClassLogprobs,
                                                              SiteClassLogprobsFromFile,
                                                              RateMultipliersPerClass,
                                                              IndpRateMultipliers,
                                                              RateMultipliersPerClassFromFile,
                                                              IndpRateMultipliersFromFile,
                                                              HKY85Logprobs,
                                                              HKY85LogprobsFromFile,
                                                              F81Logprobs,
                                                              F81LogprobsFromFile)
from models.simple_site_class_predict.transition_models import (TKF91TransitionLogprobs,
                                                                TKF92TransitionLogprobs,
                                                                TKF91TransitionLogprobsFromFile,
                                                                TKF92TransitionLogprobsFromFile,
                                                                GeomLenTransitionLogprobs,
                                                                GeomLenTransitionLogprobsFromFile)
from models.simple_site_class_predict.model_functions import (bound_sigmoid,
                                                              safe_log,
                                                              joint_logprob_emit_at_match_per_mixture,
                                                              lse_over_match_logprobs_per_mixture,
                                                              lse_over_equl_logprobs_per_mixture,
                                                              joint_prob_from_counts,
                                                              anc_marginal_probs_from_counts,
                                                              desc_marginal_probs_from_counts,
                                                              write_matrix_to_npy,
                                                              maybe_write_matrix_to_ascii)


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
        
        config['indp_rate_mults'] :  bool
            if true, then rate multipliers are independent from latent
            site classes; P(k|c) = P(k) and \rho_{c,k} = \rho{k}
        
        config['subst_model_type'] : {gtr, hky85, f81}
            which substitution model
        
        config['indel_model_type'] : {tkf91, tkf92, None}
            which indel model, if any
            
        config['norm_reported_loss_by'] :  {desc_len, align_len}, optional
            what length to normalize loglikelihood by, when reporting values
            this does NOT affect the objective function!!!
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
        self.subst_model_type = self.config['subst_model_type'].lower()
        indel_model_type = self.config['indel_model_type']
        self.indel_model_type = indel_model_type.lower() if indel_model_type is not None else None
        self.times_from = self.config['times_from'].lower()
        
        # optional
        self.norm_reported_loss_by = self.config.get('norm_reported_loss_by', 'desc_len')
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        
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
        if self.indel_model_type is None:
            self.transitions_module = GeomLenTransitionLogprobs(config = self.config,
                                                     name = f'geom seq lengths model')
            
        elif self.indel_model_type == 'tkf91':
            self.transitions_module = TKF91TransitionLogprobs(config = self.config,
                                                     name = f'tkf91 indel model')
        
        elif self.indel_model_type == 'tkf92':
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
                  normalized by desired length (set by self.norm_reported_loss_by)
              3.) whether or not you used approximation formula for TKF indel model
        """
        # which times to use for scoring matrices
        if self.times_from =='t_per_sample':
            times_for_matrices = batch[4] #(B,)
        
        elif self.times_from in ['geometric','t_array_from_file']:
            times_for_matrices = t_array #(T,)

        # get the scoring matrices needed
        # 
        # scoring_matrices_dict has the following keys (when return_intermeds is False)
        #   logprob_emit_at_indel, (A, )
        #   joint_logprob_emit_at_match, (T, A, A)
        #   all_transit_matrices, dict, with joint transit matrix being (T, S, S)
        #   used_approx, dict
        scoring_matrices_dict = self._get_scoring_matrices(t_array=times_for_matrices,
                                        sow_intermediates=sow_intermediates,
                                        return_intermeds = False)
        
        # calculate loglikelihoods; provide both batch and t_array, just in case
        # time marginalization hidden in joint_prob_from_counts function
        #
        # aux_dict has the following keys (when return_intermeds is False)
        #   joint_neg_logP (B)
        #   joint_neg_logP_length_normed (B)
        #   align_length_for_normalization (B,)
        aux_dict = joint_prob_from_counts( batch = batch,
                                           times_from = self.times_from,
                                           score_indels = False if self.indel_model_type is None else True,
                                           scoring_matrices_dict = scoring_matrices_dict,
                                           t_array = t_array,
                                           exponential_dist_param = self.exponential_dist_param,
                                           norm_reported_loss_by = self.norm_reported_loss_by,
                                           return_intermeds = False )
        aux_dict['used_approx'] = scoring_matrices_dict['used_approx']

        # if doing stochastic gradient descent, take the average over the batch
        # if doing gradient descent with whole dataset, only use the sum
        reduction = jnp.mean if not whole_dset_grad_desc else jnp.sum
        loss = reduction( aux_dict['joint_neg_logP'] )
        
        return loss, aux_dict
    
    
    def calculate_all_loglikes(self,
                               batch: tuple,
                               t_array: jnp.array,
                               return_intermeds: bool=False):
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
    
        if returning intermediates, include extra things:
            10.) rate matrix, (C, K)
            11.) exchangeabilities, (A, A)
            12.) cond_logprob_emit_at_match, (T, A, A) or (B, A, A)
            12.) joint_transit_score, (B,)
            12.) joint_emission_score, (B,)
        """
        # which times to use for scoring matrices
        if self.times_from =='t_per_sample':
            times_for_matrices = batch[4] #(B,)
        
        elif self.times_from in ['geometric','t_array_from_file']:
            times_for_matrices = t_array #(T,)

        # get the scoring matrices needed
        # 
        # scoring_matrices_dict has the following keys (when return_intermeds is False)
        #   logprob_emit_at_indel, (A, )
        #   joint_logprob_emit_at_match, (T, A, A)
        #   all_transit_matrices, dict, with joint transit matrix being (T, S, S)
        #   used_approx, dict
        scoring_matrices_dict = self._get_scoring_matrices(t_array=times_for_matrices,
                                        sow_intermediates=False,
                                        return_intermeds=False)
        
        #########################
        ### joint probability   #
        #########################
        # time marginalization hidden in joint_prob_from_counts function
        #
        # aux_dict has the following keys 
        #   joint_neg_logP (B)
        #   joint_neg_logP_length_normed (B)
        #   align_length_for_normalization (B,)
        #
        # (extra keys when return_intermeds is True)
        #   joint_transit_score, (B,)
        #   joint_emission_score, (B,)
        aux_dict = joint_prob_from_counts( batch = batch,
                                           times_from = self.times_from,
                                           score_indels = False if self.indel_model_type is None else True,
                                           scoring_matrices_dict = scoring_matrices_dict,
                                           t_array = t_array,
                                           exponential_dist_param = self.exponential_dist_param,
                                           norm_reported_loss_by = self.norm_reported_loss_by,
                                           return_intermeds=return_intermeds )
        aux_dict['used_approx'] = scoring_matrices_dict['used_approx']
        
        
        #####################################
        ### ancestor marginal probability   #
        #####################################
        to_add = anc_marginal_probs_from_counts( batch = batch,
                                            score_indels = False if self.indel_model_type is None else True,
                                            scoring_matrices_dict = scoring_matrices_dict,
                                            return_intermeds=return_intermeds  )
        
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
        
        
        #############################
        ### calculate conditional   #
        #############################
        ### just dividing joint by anc is good enough
        cond_neg_logP = -( -aux_dict['joint_neg_logP'] - -aux_dict['anc_neg_logP'] )
        length_for_normalization = aux_dict['align_length_for_normalization']
        cond_neg_logP_length_normed = cond_neg_logP / length_for_normalization
            
        aux_dict['cond_neg_logP'] = cond_neg_logP
        aux_dict['cond_neg_logP_length_normed'] = cond_neg_logP_length_normed
        
        
        # ## uncomment to explicitly calculate this
        # # recalculate this, to add cond logprob at match to the scoring matrices
        # scoring_matrices_dict_again = self._get_scoring_matrices(t_array=times_for_matrices,
        #                                 sow_intermediates=False,
        #                                 return_intermeds=True)
        # to_add = cond_prob_from_counts( batch = batch,
        #                                 times_from = self.times_from,
        #                                 score_indels = False if self.indel_model_type is None else True,
        #                                 scoring_matrices_dict = scoring_matrices_dict_again,
        #                                 t_array = t_array,
        #                                 exponential_dist_param = self.exponential_dist_param,
        #                                 norm_reported_loss_by = self.norm_reported_loss_by,
        #                                 return_intermeds=False )
        # aux_dict = {**aux_dict, **to_add}
        return aux_dict
        
    
    def _get_scoring_matrices(self,
                             t_array,
                             sow_intermediates: bool,
                             return_intermeds: bool):
        # Probability of each site class; is one, if no site clases
        log_class_probs = self.site_class_probability_module(sow_intermediates = sow_intermediates) #(C,)
        
        # Substitution rate multipliers
        # both are (C, K)
        log_rate_mult_probs, rate_multipliers = self.rate_mult_module(sow_intermediates = sow_intermediates,
                                                                      log_class_probs = log_class_probs) 
        
        
        ######################################################
        ### build log-transformed equilibrium distribution   #
        ### use this to score emissions from indels sites    #
        ######################################################
        log_equl_dist_per_mixture = self.equl_dist_module(sow_intermediates = sow_intermediates) # (C, A)
        
        # P(x) = \sum_c P(c) * P(x|c)
        logprob_emit_at_indel = lse_over_equl_logprobs_per_mixture( log_class_probs = log_class_probs,
                                                                    log_equl_dist_per_mixture = log_equl_dist_per_mixture) #(A,)
        
        
        ####################################################
        ### build substitution log-probability matrix      #
        ### use this to score emissions from match sites   #
        ####################################################
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
        
        # get the joint probability
        joint_subst_logprobs_per_mixture = joint_logprob_emit_at_match_per_mixture( cond_logprob_emit_at_match_per_mixture = cond_subst_logprobs_per_mixture,
                                                                              log_equl_dist_per_mixture = log_equl_dist_per_mixture ) # (T, C, K, A, A) or (B, C, K, A, A)
        
        
        # marginalize over c classes and k possible rate multipliers
        joint_logprob_emit_at_match = lse_over_match_logprobs_per_mixture(log_class_probs = log_class_probs,
                                                                          log_rate_mult_probs = log_rate_mult_probs,
                                                                          logprob_emit_at_match_per_mixture = joint_subst_logprobs_per_mixture) #(T, A, A) or (B, A, A)
        cond_logprob_emit_at_match = lse_over_match_logprobs_per_mixture(log_class_probs = log_class_probs,
                                                                         log_rate_mult_probs = log_rate_mult_probs,
                                                                         logprob_emit_at_match_per_mixture = cond_subst_logprobs_per_mixture) #(T, A, A) or (B, A, A)
        
        
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
                    'used_approx': used_approx} #dict
        
        if return_intermeds:
            to_add = {'rate_matrix': subst_module_intermeds.get('rate_matrix',None), #(C,A,A) or None
                      'exchangeabilities': subst_module_intermeds.get('exchangeabilities',None), #(A,A) or None
                      'cond_logprob_emit_at_match': cond_logprob_emit_at_match, #(T, A, A)
                      'cond_subst_logprobs_per_mixture': cond_subst_logprobs_per_mixture, # (T, C, K, A, A)
                      'joint_subst_logprobs_per_mixture': joint_subst_logprobs_per_mixture, # (T, C, K, A, A) 
                      'log_equl_dist_per_mixture': log_equl_dist_per_mixture, #(C, A)
                      'log_class_probs': log_class_probs, #(C,)
                      'log_rate_mult_probs': log_rate_mult_probs } #(K,C)
            out_dict = {**out_dict, **to_add}
        
        return out_dict
    
    def write_params(self,
                     t_array,
                     out_folder: str,
                     prefix: str,
                     write_time_static_objs: bool):
        #########################################################
        ### only write once: activations_times_used text file   #
        #########################################################
        if write_time_static_objs:
            with open(f'{out_folder}/activations_times_used.tsv','w') as g:
                if not self.config['load_all']:
                    g.write(f'activation for rate multipliers: bound_sigmoid\n')
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
        
        # final conditional and joint prob of match (before and after LSE over classes)
        for loss_type in ['joint', 'cond']:
            for suffix in ['logprob_emit_at_match', 'subst_logprobs_per_mixture']:
                mat = np.exp (out[f'{loss_type}_{suffix}'] ) 
                new_key = f'{prefix}_{loss_type}_{suffix}'.replace('log','')
                write_matrix_to_npy( out_folder, mat, new_key )
                maybe_write_matrix_to_ascii( out_folder, mat, new_key )
                del mat, new_key
                
        # transition matrix: joint transition matrix
        mat = np.exp(out['all_transit_matrices']['joint']) #(T, A, A)
        key = f'{prefix}_joint_prob_transit_matrix'
        write_matrix_to_npy( out_folder, mat, key )
        maybe_write_matrix_to_ascii( out_folder, mat, key )
        del mat, key
        
        # transition matrix: conditional and marginal transition matrices
        if self.indel_model_type is not None:
            for loss_type in ['conditional','marginal']:
                mat = np.exp(out['all_transit_matrices'][loss_type]) 
                key = f'{prefix}_{loss_type}_prob_transit_matrix'
                write_matrix_to_npy( out_folder, mat, key )
                maybe_write_matrix_to_ascii( out_folder, mat, key )
                del mat, key
                
        
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
                ### logprob_emit_at_indel (AFTER marginalizing over classes)
                mat = np.exp( out['logprob_emit_at_indel'] ) #(A,)
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
                        
                        
                ### equilibrium distribution (BEFORE marginalizing over site clases)
                if 'logits' in dir(self.equl_dist_module):
                    equl_dist = nn.softmax( self.equl_dist_module.logits, axis=1 ) #(C, A)
                    key = f'{prefix}_equilibriums-per-class'
                    write_matrix_to_npy( out_folder, equl_dist, key )
                    maybe_write_matrix_to_ascii( out_folder, equl_dist, key )
                    del key
                    
                ####################################################
                ### extract transition paramaters, intermediates   # 
                ### needed for final scoring matrices              #
                ### (also does not depend on time)                 #
                ####################################################
                ### under geometric length (only scoring subs)
                if self.indel_model_type is None:
                    geom_p_emit = nn.sigmoid(self.transitions_module.p_emit_logit).item() #(1,)
                    arr = np.array( [geom_p_emit, 1 - geom_p_emit] )
                    key = f'{prefix}_geom_seq_len'
                    write_matrix_to_npy( out_folder, arr, key )
                    del key, arr
                    
                    with open(f'{out_folder}/ASCII_{prefix}_geom_seq_len.txt','w') as g:
                        g.write(f'P(emit): {geom_p_emit}\n')
                        g.write(f'1-P(emit): {1 - geom_p_emit}\n')
                        
                        
                ### for TKF models
                elif self.indel_model_type in ['tkf91', 'tkf92']:
                    # always write lambda and mu
                    # also record if you used any tkf approximations
                    if 'tkf_mu_offset_logits' in dir(self.transitions_module):
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
                        
                        with open(f'{out_folder}/ASCII_{prefix}_{self.indel_model_type}_indel_params.txt','w') as g:
                            g.write(f'insert rate, lambda: {lam}\n')
                            g.write(f'deletion rate, mu: {mu}\n')
                            g.write(f'offset: {offset}\n\n')
                        
                        out_dict = {'lambda': np.array(lam), # shape=()
                                    'mu': np.array(mu), # shape=()
                                    'offset': np.array(offset)} # shape=()
                                    
                    # if tkf92, have extra r_ext param
                    if self.indel_model_type == 'tkf92':
                        r_extend_min_val = self.transitions_module.r_extend_min_val
                        r_extend_max_val = self.transitions_module.r_extend_max_val
                        r_extend_logits = self.transitions_module.r_extend_logits #(C)
                        
                        r_extend = bound_sigmoid(x = r_extend_logits,
                                                 min_val = r_extend_min_val,
                                                 max_val = r_extend_max_val) #(C)
                        
                        mean_indel_lengths = 1 / (1 - r_extend) #(C)
                        
                        with open(f'{out_folder}/ASCII_{prefix}_{self.indel_model_type}_indel_params.txt','a') as g:
                            g.write(f'extension prob, r: ')
                            [g.write(f'{elem}\t') for elem in r_extend]
                            g.write('\n')
                            g.write(f'mean indel length: ')
                            [g.write(f'{elem}\t') for elem in mean_indel_lengths]
                            g.write('\n')
                        
                        out_dict['r_extend'] = r_extend #(C,)
                    
                    with open(f'{out_folder}/PARAMS-DICT_{prefix}_{self.indel_model_type}_indel_params.pkl','wb') as g:
                        pickle.dump(out_dict, g)
                    del out_dict


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
            
        config['norm_reported_loss_by'] :  {desc_len, align_len}, optional
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
        
    Inherited from IndpSites
    -------------------------
    __call__
        unpack batch and calculate logP(anc, desc, align)
    
    write_params
        write parameters to files

    calculate_all_loglikes
        calculate logP(anc, desc, align), logP(anc), logP(desc), and
        logP(desc, align | anc)
    
    
    return_bound_sigmoid_limits
        after initializing model, get the limits for bound_sigmoid activations
    
    
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
        self.subst_model_type = self.config['subst_model_type'].lower()
        indel_model_type = self.config['indel_model_type']
        self.indel_model_type = indel_model_type.lower() if indel_model_type is not None else None
        self.times_from = self.config['times_from'].lower()
        
        # optional
        self.norm_reported_loss_by = self.config.get('norm_reported_loss_by', 'desc_len')
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        
        
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
        
        ###########################################
        ### module for substitution rate matrix   #
        ###########################################
        if self.subst_model_type == 'gtr':
            self.logprob_subst_module = GTRLogprobsFromFile( config = self.config,
                                                  name = f'gtr subst. model' )
            
        elif self.subst_model_type == 'f81':
            self.logprob_subst_module = F81LogprobsFromFile( config = self.config,
                                                     name = f'f81 subst. model' )

        elif self.subst_model_type == 'hky85':
            self.logprob_subst_module = HKY85LogprobsFromFile( config = self.config,
                                                    name = f'hky85 subst. model' )
            
        
        ###########################################
        ### module for transition probabilities   #
        ###########################################        
        if self.indel_model_type is None:
            self.transitions_module = GeomLenTransitionLogprobsFromFile(config = self.config,
                                                     name = f'geom seq lengths model')
            
        elif self.indel_model_type == 'tkf91':
            self.transitions_module = TKF91TransitionLogprobsFromFile(config = self.config,
                                                     name = f'tkf91 indel model')
        
        elif self.indel_model_type == 'tkf92':
            self.transitions_module = TKF92TransitionLogprobsFromFile(config = self.config,
                                                     name = f'tkf92 indel model')
                   