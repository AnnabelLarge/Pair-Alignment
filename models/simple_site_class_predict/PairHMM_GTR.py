#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 18:51:35 2025

@author: annabel
"""
import numpy as np
import pickle

# jumping jax and leaping flax
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax.scipy.special import logsumexp

from models.model_utils.BaseClasses import ModuleBase
from models.simple_site_class_predict.emission_models import (LogEqulVecFromCounts,
                                                              LogEqulVecPerClass,
                                                              LogEqulVecFromFile,
                                                              RateMatFromFile,
                                                              RateMatFitBoth,
                                                              SiteClassLogprobs,
                                                              SiteClassLogprobsFromFile,
                                                              HKY85,
                                                              HKY85FromFile)
from models.simple_site_class_predict.transition_models import (TKF91TransitionLogprobs,
                                                                TKF92TransitionLogprobs,
                                                                TKF91TransitionLogprobsFromFile,
                                                                TKF92TransitionLogprobsFromFile)
from utils.pairhmm_helpers import (bounded_sigmoid,
                                   safe_log)

class GTRPairHMM(ModuleBase):
    """
    don't score indel sites
    
     
    main methods:
    =============
        - setup    
        
        - __call__: calculate loss based on joint prob P(anc, desc, align);
                   use this during training; is jit compatible
        
        - calculate_all_loglikes: calculate joint prob P(anc, desc, align),
                   conditional prob P(desc, align | anc), and both marginals
                   P(desc) and P(anc); use this during final eval; is also
                   jit compatible

    other helpers:
    ==============
        - write_params: write the parameters to files
    

    internal methods:
    ==================
        - _get_scoring_matrices
        - _joint_logprob_align
        - _marginalize_over_times
    """
    config: dict
    name: str
    
    def setup(self):
        num_emit_site_classes = self.config['num_emit_site_classes']
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        
        ### how to score emissions from indel sites
        if num_emit_site_classes == 1:
            self.indel_prob_module = LogEqulVecFromCounts(config = self.config,
                                                        name = f'get equilibrium')
        elif num_emit_site_classes > 1:
            self.indel_prob_module = LogEqulVecPerClass(config = self.config,
                                                      name = f'get equilibrium')
        
        
        ### rate matrix to score emissions from match sites
        out = self._init_rate_matrix_module(self.config)
        self.rate_matrix_module, self.subst_model_type = out
        del out
        
        
        ### now need probabilities for rate classes themselves
        self.site_class_probability_module = SiteClassLogprobs(config = self.config,
                                                  name = f'get site class probabilities')
        
    
    def __call__(self,
                 batch,
                 t_array,
                 sow_intermediates: bool):
        """
        Use this during active model training
        
        returns:
            - loss: average across the batch, based on length-normalized
                    joint log-likelihood
                    
            - aux_dict: has the following keys and values
              1.) 'joint_neg_logP': sum down the length
              2.) 'joint_neg_logP_length_normed': sum down the length,  
                  normalized by desired length (set by self.norm_by)
        """
        # get scoring matrices for joint probability
        out = self._get_scoring_matrices(t_array=t_array,
                                        sow_intermediates=sow_intermediates)
        
        logprob_emit_at_indel = out['logprob_emit_at_indel']
        joint_logprob_emit_at_match = out['joint_logprob_emit_at_match']
        del out
        
        # calculate scores
        aux_dict = self._joint_logprob_align( batch=batch,
                                             t_array=t_array,
                                             joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                             sow_intermediates=sow_intermediates )
        
        loss = jnp.mean( aux_dict['joint_neg_logP_length_normed'] )
        aux_dict['used_tkf_beta_approx'] = False
        
        return loss, aux_dict
    
    
    def calculate_all_loglikes(self,
                               batch,
                               t_array,
                               sow_intermediates: bool):
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
        """
        
        ####################################
        ### prep matrices, unpack values   #
        ####################################
        # unpack batch: (B, ...)
        subCounts = batch[0] #(B, 20, 20)
        
        # get scoring matrices for joint and marginal probabilities 
        #   (conditional comes from dividing joint by marginal)
        out = self._get_scoring_matrices(t_array=t_array,
                                        sow_intermediates=sow_intermediates)
                                        
        logprob_emit_at_indel = out['logprob_emit_at_indel']
        joint_logprob_emit_at_match = out['joint_logprob_emit_at_match']
        del out
        
        # get all lengths
        align_len = subCounts.sum(axis=(-2, -1))
        anc_len = align_len
        desc_len = align_len
        
        
        #########################
        ### joint probability   #
        #########################
        out = self._joint_logprob_align( batch=batch,
                                        t_array=t_array,
                                        joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                        sow_intermediates=sow_intermediates )
        out['used_tkf_beta_approx'] = False
        
        
        #####################################
        ### ancestor marginal probability   #
        #####################################
        # emissions from match row sums 
        anc_emitCounts = subCounts.sum(axis=2)
        anc_marg_emit_score = jnp.einsum('i,bi->b',
                                         logprob_emit_at_indel,
                                         anc_emitCounts)
        
        anc_neg_logP = -anc_marg_emit_score
        anc_neg_logP_length_normed = anc_neg_logP / anc_len
        
        out['anc_neg_logP'] = anc_neg_logP
        out['anc_neg_logP_length_normed'] = anc_neg_logP_length_normed
        
        
        #######################################
        ### descendant marginal probability   #
        #######################################
        # emissions from match column sums
        desc_emitCounts = subCounts.sum(axis=1) 
        desc_marg_emit_score = jnp.einsum('i,bi->b',
                                         logprob_emit_at_indel,
                                         desc_emitCounts)
        
        desc_neg_logP = -desc_marg_emit_score
        desc_neg_logP_length_normed = desc_neg_logP / desc_len
            
        out['desc_neg_logP'] = desc_neg_logP
        out['desc_neg_logP_length_normed'] = desc_neg_logP_length_normed
                
        
        #####################################################
        ### calculate conditional from joint and marginal   #
        #####################################################
        cond_neg_logP = -( -out['joint_neg_logP'] - -anc_neg_logP )
        cond_neg_logP_length_normed = cond_neg_logP / align_len 
        
        out['cond_neg_logP'] = cond_neg_logP
        out['cond_neg_logP_length_normed'] = cond_neg_logP_length_normed
        
        return out
    
    
    def write_params(self,
                     t_array,
                     out_folder: str):
        with open(f'{out_folder}/activations_used.tsv','w') as g:
            act = self.rate_matrix_module.rate_mult_activation
            g.write(f'activation for rate multipliers: {act}\n')
            g.write(f'activation for exchangeabiliites: bound_sigmoid\n')
        
        
        ##########################
        ### the final matrices   #
        ##########################  
        out = self._get_scoring_matrices(t_array=t_array,
                                        sow_intermediates=False)
        
        
        rate_mat_times_rho_per_class = out['rate_mat_times_rho_per_class']
        for c in range(rate_mat_times_rho_per_class.shape[0]):
            mat_to_save = rate_mat_times_rho_per_class[c,...]
            
            with open(f'{out_folder}/class-{c}_rate_matrix_times_rho.npy', 'wb') as g:
                np.save(g, mat_to_save)
            
            np.savetxt( f'{out_folder}/ASCII_class-{c}_rate_matrix_times_rho.tsv', 
                        np.array(mat_to_save), 
                        fmt = '%.4f',
                        delimiter= '\t' )
            
            del mat_to_save, g
            

        # matrix that you apply expm() to
        to_expm = np.squeeze( out['to_expm'] )
        
        with open(f'{out_folder}/to_expm.npy', 'wb') as g:
            np.save(g, to_expm)
        
        if len(to_expm.shape) <= 2:
            np.savetxt( f'{out_folder}/ASCII_to_expm.tsv', 
                        to_expm, 
                        fmt = '%.4f',
                        delimiter= '\t' )
        
        del to_expm, g
        
        # other emission matrices; exponentiate them first
        for key in ['logprob_emit_at_indel', 
                    'joint_logprob_emit_at_match']:
            mat = np.exp(out[key])
            new_key = key.replace('logprob','prob')
            
            with open(f'{out_folder}/{new_key}.npy', 'wb') as g:
                np.save(g, mat)
            
            mat = np.squeeze(mat)
            if len(mat.shape) <= 2:
                np.savetxt( f'{out_folder}/ASCII_{new_key}.tsv', 
                            np.array(mat), 
                            fmt = '%.4f',
                            delimiter= '\t' )
            
            del key, mat, g
            
        
        ###############
        ### extract   #
        ###############
        ### site class probs
        if 'class_logits' in dir(self.site_class_probability_module):
            class_probs = nn.softmax(self.site_class_probability_module.class_logits)
            with open(f'{out_folder}/PARAMS_class_probs.txt','w') as g:
                [g.write(f'{elem.item()}\n') for elem in class_probs]
        
        
        ### emissions
        # exchangeabilities
        if 'exchangeabilities_logits_vec' in dir(self.rate_matrix_module):
            exch_logits = self.rate_matrix_module.exchangeabilities_logits_vec
            exchangeabilities = self.rate_matrix_module.exchange_activation( exch_logits )
            
            if self.subst_model_type == 'GTR':
                np.savetxt( f'{out_folder}/PARAMS_exchangeabilities.tsv', 
                            np.array(exchangeabilities), 
                            fmt = '%.4f',
                            delimiter= '\t' )
                
                with open(f'{out_folder}/PARAMS_exchangeabilities.npy','wb') as g:
                    jnp.save(g, exchangeabilities)
            
            elif self.subst_model_type == 'HKY85':
                with open(f'{out_folder}/PARAMS_HKY85_model.txt','w') as g:
                    g.write(f'transition rate, ti: {exchangeabilities[1]}')
                    g.write(f'transition rate, tv: {exchangeabilities[0]}')
                
        # emissions: rate multipliers
        if 'rate_mult_logits' in dir(self.rate_matrix_module):
            rate_mult_logits = self.rate_matrix_module.rate_mult_logits
            rate_mult = self.rate_matrix_module.rate_multiplier_activation( rate_mult_logits )

            with open(f'{out_folder}/PARAMS_rate_multipliers.txt','w') as g:
                [g.write(f'{elem.item()}\n') for elem in rate_mult]
        
        # emissions: equilibrium distribution
        if 'logits' in dir(self.indel_prob_module):
            equl_logits = self.indel_prob_module.logits
            equl_dist = nn.softmax( equl_logits, axis=1 )
            
            np.savetxt( f'{out_folder}/PARAMS_equilibriums.tsv', 
                        np.array(equl_dist), 
                        fmt = '%.4f',
                        delimiter= '\t' )
            
            with open(f'{out_folder}/PARAMS-ARR_equilibriums.npy','wb') as g:
                jnp.save(g, equl_dist)
                
    
    def _init_rate_matrix_module(self, config):
        mod = RateMatFitBoth( config = self.config,
                               name = f'get rate matrix' )
        return mod, 'GTR'
        
    def _get_scoring_matrices(self,
                             t_array,
                             sow_intermediates: bool):
        """
        TODO: if using one time per sample (i.e. the time from FastTree), then
        you'll need to unpack time from batch; will have to initialize a new
        time array with shape (T, B) instead of (T,)
        """
        ### build logprob emissions at match
        # first, get log(equilibrium distribution): (C, alph)
        log_equl_dist_per_class = self.indel_prob_module(sow_intermediates = sow_intermediates)
        
        # get normalized rate matrix times rate multiplier, per each class
        # (C, alph, alph)
        rate_mat_times_rho_per_class = self.rate_matrix_module(logprob_equl = log_equl_dist_per_class,
                                                                sow_intermediates = sow_intermediates)
        
        # build logprob matrix at every class
        # time: (T,)
        # output: (T, C, from_alph, to_alph)
        to_expm = jnp.multiply( rate_mat_times_rho_per_class[None,...],
                                t_array[:, None,None,None,] )
        cond_prob_emit_at_match_per_class = expm(to_expm)
        cond_logprob_emit_at_match_per_class = safe_log( cond_prob_emit_at_match_per_class )
        joint_logprob_emit_at_match_per_class = ( cond_logprob_emit_at_match_per_class + 
                                                  log_equl_dist_per_class[None,:,:,None] )
        del cond_logprob_emit_at_match_per_class
        
        # apply weighting; LSE across classes: (T, alph, alph)
        log_class_probs = self.site_class_probability_module(sow_intermediates = sow_intermediates) #(C,)
        weighted_joint_logprob_emit_at_match = ( log_class_probs[None,:,None,None] + 
                                                 joint_logprob_emit_at_match_per_class )
        joint_logprob_emit_at_match = logsumexp( weighted_joint_logprob_emit_at_match, axis=1 )
        
        
        ### build logprob emissions at indels
        to_logsumexp = log_equl_dist_per_class + log_class_probs[:, None] #(C, alph)
        logprob_emit_at_indel = logsumexp( to_logsumexp, axis=0 )
        
        
        out_dict = {'logprob_emit_at_indel': logprob_emit_at_indel,
                    'joint_logprob_emit_at_match': joint_logprob_emit_at_match,
                    'cond_logprob_emit_at_match': cond_prob_emit_at_match_per_class,
                    'rate_mat_times_rho_per_class': rate_mat_times_rho_per_class,
                    'to_expm': to_expm}
        
        return out_dict
    
    
    def _joint_logprob_align( self,
                             batch,
                             t_array,
                             joint_logprob_emit_at_match,
                             sow_intermediates: bool ):
        # unpack batch: (B, ...)
        subCounts = batch[0] #(B, 20, 20)
        
        
        #######################################
        ### score emissions and transitions   #
        #######################################
        # matches; (T, B)
        match_emit_score = jnp.einsum('tij,bij->tb',
                                      joint_logprob_emit_at_match, 
                                      subCounts)
        
        joint_logprob_perSamp_perTime = match_emit_score
        
        # marginalize over times; (B,)
        if t_array.shape[0] > 1:
            joint_neg_logP = -self._marginalize_over_times(logprob_perSamp_perTime = joint_logprob_perSamp_perTime,
                                        exponential_dist_param = self.exponential_dist_param,
                                        t_array = t_array,
                                        sow_intermediates = sow_intermediates)
        else:
            joint_neg_logP = -joint_logprob_perSamp_perTime[0,:]
        
        # normalize (don't include <bos> or <eos>)
        length_for_normalization = subCounts.sum(axis=(-2, -1))
        joint_neg_logP_length_normed = joint_neg_logP / length_for_normalization
        
        return {'joint_neg_logP': joint_neg_logP,
                'joint_neg_logP_length_normed': joint_neg_logP_length_normed}
    
    
    def _marginalize_over_times(self,
                               logprob_perSamp_perTime,
                               exponential_dist_param,
                               t_array,
                               sow_intermediates: bool):
        ### constants to add (multiply by)
        # logP(t_k) = exponential distribution
        logP_time = ( jnp.log(exponential_dist_param) - 
                      (exponential_dist_param * t_array) )
        log_t_grid = jnp.log( t_array[1:] - t_array[:-1] )
        
        # kind of a hack, but repeat the last time array value
        log_t_grid = jnp.concatenate( [log_t_grid, log_t_grid[-1][None] ], axis=0)
        
        
        ### add in log space, multiply in probability space; logsumexp
        logP_perSamp_perTime_withConst = ( logprob_perSamp_perTime +
                                           logP_time[:,None] +
                                           log_t_grid[:,None] )
        
        if sow_intermediates:
            lab = f'{self.name}/time_marginalization/before logsumexp'
            self.sow_histograms_scalars(mat= logP_perSamp_perTime_withConst, 
                                        label=lab, 
                                        which='scalars')
            del lab
        
        
        logP_perSamp_raw = logsumexp(logP_perSamp_perTime_withConst, axis=0)
        
        if sow_intermediates:
            lab = f'{self.name}/time_marginalization/after logsumexp'
            self.sow_histograms_scalars(mat= logP_perSamp_raw, 
                                        label=lab, 
                                        which='scalars')
            del lab
        
        return logP_perSamp_raw
        
    
    def _return_bound_sigmoid_limits(self):
        ### rate_matrix_module
        # exchangeabilities
        exchange_min_val = self.rate_matrix_module.exchange_min_val
        exchange_max_val = self.rate_matrix_module.exchange_max_val
        
        #rate multiplier
        if self.rate_mult_activation == 'bound_sigmoid':
            rate_mult_min_val = self.rate_matrix_module.rate_mult_min_val
            rate_mult_max_val = self.rate_matrix_module.rate_mult_max_val
        
        params_range = { "exchange_min_val": exchange_min_val,
                         "exchange_max_val": exchange_max_val,
                         "rate_mult_min_val": rate_mult_min_val,
                         "rate_mult_max_val": rate_mult_max_val
                         }
        
        return params_range



class GTRPairHMMLoadAll(GTRPairHMM):
    """
    same as GTRPairHMM, but load values (i.e. no free parameters)
    
    inherits everything except setup and write_params 
    
    """
    config: dict
    name: str
    
    def setup(self):
        num_emit_site_classes = self.config['num_emit_site_classes']
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        
        ### how to score emissions from indel sites
        self.indel_prob_module = LogEqulVecFromFile(config = self.config,
                                                   name = f'get equilibrium')
        
        
        ### rate matrix to score emissions from match sites
        out = self._init_rate_matrix_module(self.config)
        self.rate_matrix_module, self.subst_model_type = out
        del out
        
        
        ### probability of site classes
        self.site_class_probability_module = SiteClassLogprobsFromFile(config = self.config,
                                                 name = f'get site class probabilities')
        
            
    def write_params(self,
                     t_array,
                     out_folder: str):
        ##########################
        ### the final matrices   #
        ##########################  
        out = self._get_scoring_matrices(t_array=t_array,
                                        sow_intermediates=False)
        
        
        rate_mat_times_rho_per_class = out['rate_mat_times_rho_per_class']
        for c in range(rate_mat_times_rho_per_class.shape[0]):
            mat_to_save = rate_mat_times_rho_per_class[c,...]
            
            with open(f'{out_folder}/class-{c}_rate_matrix_times_rho.npy', 'wb') as g:
                np.save(g, mat_to_save)
            
            np.savetxt( f'{out_folder}/ASCII_class-{c}_rate_matrix_times_rho.tsv', 
                        np.array(mat_to_save), 
                        fmt = '%.4f',
                        delimiter= '\t' )
            
            del mat_to_save, g
            

        # matrix that you apply expm() to
        to_expm = np.squeeze( out['to_expm'] )
        
        with open(f'{out_folder}/to_expm.npy', 'wb') as g:
            np.save(g, to_expm)
        
        if len(to_expm.shape) <= 2:
            np.savetxt( f'{out_folder}/ASCII_to_expm.tsv', 
                        to_expm, 
                        fmt = '%.4f',
                        delimiter= '\t' )
        
        del to_expm, g
        
        # other emission matrices; exponentiate them first
        for key in ['logprob_emit_at_indel', 
                    'joint_logprob_emit_at_match']:
            mat = np.exp(out[key])
            new_key = key.replace('logprob','prob')
            
            with open(f'{out_folder}/{new_key}.npy', 'wb') as g:
                np.save(g, mat)
            
            mat = np.squeeze(mat)
            if len(mat.shape) <= 2:
                np.savetxt( f'{out_folder}/ASCII_{new_key}.tsv', 
                            np.array(mat), 
                            fmt = '%.4f',
                            delimiter= '\t' )
            
            del key, mat, g
            
    
    def _init_rate_matrix_module(self, config):
        mod = RateMatFromFile( config = self.config,
                               name = f'get rate matrix' )
        return mod, 'GTR'