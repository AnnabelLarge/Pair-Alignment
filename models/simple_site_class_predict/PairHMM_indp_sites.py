#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 04:33:00 2025

@author: annabel


models here:
============
'IndpPairHMMFitBoth',
 'IndpPairHMMFitRateMult',
 'IndpPairHMMLoadAll',


main methods for all models:
============================
- setup (self-explanatory)

- __call__: calculate loss based on joint prob P(anc, desc, align);
           use this during training; is jit compatible

- calculate_all_loglikes: calculate joint prob P(anc, desc, align),
           conditional prob P(desc, align | anc), and both marginals
           P(desc) and P(anc); use this during final eval; is also
           jit compatible
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
                                                       LG08RateMatFromFile,
                                                       LG08RateMatFitRateMult,
                                                       LG08RateMatFitBoth,
                                                       SiteClassLogprobs,
                                                       SiteClassLogprobsFromFile)
from models.simple_site_class_predict.transition_models import (TKF91TransitionLogprobs,
                                                                TKF92TransitionLogprobs,
                                                                TKF91TransitionLogprobsFromFile,
                                                                TKF92TransitionLogprobsFromFile)
from utils.pairhmm_helpers import (bounded_sigmoid,
                                   safe_log)

class IndpPairHMMFitBoth(ModuleBase):
    """
    uses LG08RateMatFitBoth for susbtitution model; i.e. load LG08 
       exchangeabilities as initial values
     
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
        self.indel_model_type = self.config['indel_model_type']
        self.norm_loss_by = self.config['norm_loss_by']
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        
        ### how to score emissions from indel sites
        if num_emit_site_classes == 1:
            self.indel_prob_module = LogEqulVecFromCounts(config = self.config,
                                                       name = f'get equilibrium')
        elif num_emit_site_classes > 1:
            self.indel_prob_module = LogEqulVecPerClass(config = self.config,
                                                     name = f'get equilibrium')
        
        
        ### rate matrix to score emissions from match sites
        self.rate_matrix_module = LG08RateMatFitBoth(config = self.config,
                                                 name = f'get rate matrix')
        
        
        ### now need probabilities for rate classes themselves
        self.site_class_probability_module = SiteClassLogprobs(config = self.config,
                                                  name = f'get site class probabilities')
        
        
        ### TKF91 or TKF92
        if self.indel_model_type == 'tkf91':
            self.transitions_module = TKF91TransitionLogprobs(config = self.config,
                                                     name = f'tkf91 indel model')
        
        elif self.indel_model_type == 'tkf92':
            self.transitions_module = TKF92TransitionLogprobs(config = self.config,
                                                     name = f'tkf92 indel model')
    
    
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
        joint_transit_mat = out['all_transit_matrices']['joint']
        del out
        
        # calculate scores
        aux_dict = self._joint_logprob_align( batch=batch,
                                             t_array=t_array,
                                             logprob_emit_at_indel=logprob_emit_at_indel,
                                             joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                             joint_transit_mat=joint_transit_mat )
        
        loss = jnp.mean( aux_dict['joint_neg_logP_length_normed'] )
        
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
        insCounts = batch[1] #(B, 20)
        delCounts = batch[2]
        transCounts = batch[3] #(B, 4)
        
        # get scoring matrices for joint and marginal probabilities 
        #   (conditional comes from dividing joint by marginal)
        out = self._get_scoring_matrices(t_array=t_array,
                                        sow_intermediates=sow_intermediates)
        
        logprob_emit_at_indel = out['logprob_emit_at_indel']
        joint_logprob_emit_at_match = out['joint_logprob_emit_at_match']
        joint_transit_mat = out['all_transit_matrices']['joint']
        marginal_transit_mat = out['all_transit_matrices']['marginal'] 
        del out
        
        # get all lengths
        anc_len = ( subCounts.sum(axis=(-2, -1)) + 
            delCounts.sum(axis=(-1))
            ) 

        desc_len = ( subCounts.sum(axis=(-2, -1)) + 
                     insCounts.sum(axis=(-1))
                     )
        
        align_len = ( subCounts.sum(axis=(-2, -1)) + 
                      insCounts.sum(axis=(-1)) + 
                      delCounts.sum(axis=(-1))
                      )
        
        
        #########################
        ### joint probability   #
        #########################
        out = self._joint_logprob_align( batch=batch,
                                        t_array=t_array,
                                        logprob_emit_at_indel=logprob_emit_at_indel,
                                        joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                        joint_transit_mat=joint_transit_mat )
        
        
        #####################################
        ### ancestor marginal probability   #
        #####################################
        # emissions from match row sums and del positions
        anc_emitCounts = subCounts.sum(axis=2) + delCounts
        anc_marg_emit_score = jnp.einsum('i,bi->b',
                                         logprob_emit_at_indel,
                                         anc_emitCounts)
        
        # use only transitions that end with match (0) and del (2)
        anc_emit_to_emit = ( transCounts[...,0].sum( (-1) ) + 
                             transCounts[...,2].sum( (-1) ) ) - 1
        anc_transCounts = jnp.stack( [jnp.stack( [anc_emit_to_emit, 
                                                  jnp.ones(anc_emit_to_emit.shape[0])], 
                                                axis=-1 ),
                                      jnp.stack( [jnp.ones(anc_emit_to_emit.shape[0]), 
                                                  jnp.zeros(anc_emit_to_emit.shape[0])], 
                                                axis=-1 )],
                                      axis = -2 )
        anc_marg_transit_score = jnp.einsum( 'mn,bmn->b', 
                                             marginal_transit_mat, 
                                             anc_transCounts )
        anc_neg_logP = -(anc_marg_emit_score + anc_marg_transit_score)
        anc_neg_logP_length_normed = anc_neg_logP / anc_len
        
        out['anc_neg_logP'] = anc_neg_logP
        out['anc_neg_logP_length_normed'] = anc_neg_logP_length_normed
        
        
        #######################################
        ### descendant marginal probability   #
        #######################################
        # emissions from match column sums and ins positions
        desc_emitCounts = subCounts.sum(axis=1) + insCounts
        desc_marg_emit_score = jnp.einsum('i,bi->b',
                                         logprob_emit_at_indel,
                                         desc_emitCounts)
        
        # use only transitions that end with match (0) and ins (1)
        desc_emit_to_emit = ( transCounts[...,0].sum( (-1) ) + 
                              transCounts[...,1].sum( (-1) ) ) - 1
        desc_transCounts = jnp.stack( [jnp.stack( [desc_emit_to_emit, 
                                                  jnp.ones(desc_emit_to_emit.shape[0])], 
                                                axis=-1 ),
                                      jnp.stack( [jnp.ones(desc_emit_to_emit.shape[0]), 
                                                  jnp.zeros(desc_emit_to_emit.shape[0])], 
                                                axis=-1 )],
                                      axis = -2 )
        desc_marg_transit_score = jnp.einsum( 'mn,bmn->b', 
                                             marginal_transit_mat, 
                                             desc_transCounts )
        desc_neg_logP = -(desc_marg_emit_score + desc_marg_transit_score)
        desc_neg_logP_length_normed = desc_neg_logP / desc_len
            
        out['desc_neg_logP'] = desc_neg_logP
        out['desc_neg_logP_length_normed'] = desc_neg_logP_length_normed
                
        
        #####################################################
        ### calculate conditional from joint and marginal   #
        #####################################################
        cond_neg_logP = -( -out['joint_neg_logP'] - -anc_neg_logP )
        
        if self.norm_loss_by == 'desc_len':
            cond_neg_logP_length_normed = cond_neg_logP / desc_len
            
        elif self.norm_loss_by == 'align_len':
            cond_neg_logP_length_normed = cond_neg_logP / align_len 
        
        out['cond_neg_logP'] = cond_neg_logP
        out['cond_neg_logP_length_normed'] = cond_neg_logP_length_normed
        
        return out
    
    
    def write_params(self,
                     pred_config,
                     tstate,
                     out_folder: str):
        params_dict = tstate.params['params']
        
        
        ##################################################
        ### use default values, if ranges aren't found   #
        ##################################################
        with open(f'{out_folder}/ranges_used.tsv','w') as g:
            g.write('Ranges used to convert params (if values were not provided, noted below)\n\n')
            
        def read_pred_config(key, default_tup):
            if key not in pred_config.keys():
                with open(f'{out_folder}/ranges_used.tsv','a') as g:
                    g.write(f'{key}: {default_tup} [NOT PROVIDED; USED DEFAULT VALUE]\n')                
                return default_tup
            
            else:
                with open(f'{out_folder}/ranges_used.tsv','a') as g:
                    g.write(f'{key}: {pred_config[key]}\n')
                return pred_config[key]
        
        
        out = read_pred_config( 'exchange_range', (1e-4, 10) )
        exchange_min_val, exchange_max_val = out
        del out
        
        out = read_pred_config( 'rate_mult_range', (0.01, 10) )
        rate_mult_min_val, rate_mult_max_val = out
        del out
        
        out = read_pred_config( 'lambda_range', (pred_config['tkf_err'], 3) )
        lam_min_val, lam_max_val = out
        del out
         
        out = read_pred_config( 'offset_range', (pred_config['tkf_err'], 0.333) )
        offs_min_val, offs_max_val = out
        del out
        
        
        
        ###############
        ### extract   #
        ###############
        ### site class probs
        if 'get site class probabilities' in params_dict.keys():
            class_logits = params_dict['get site class probabilities']['class_logits']
            class_probs = nn.softmax(class_logits)
            with open(f'{out_folder}/PARAMS_class_probs.txt','w') as g:
                [g.write(f'{elem.item()}\n') for elem in class_probs]
                
                
        ### emissions
        if 'get rate matrix' in params_dict.keys():
            
            if 'exchangeabilities' in params_dict['get rate matrix']:
                exch_logits = params_dict['get rate matrix']['exchangeabilities']
                exchangeabilities = bounded_sigmoid(x = exch_logits, 
                                                    min_val = exchange_min_val,
                                                    max_val = exchange_max_val)
                
                np.savetxt( f'{out_folder}/PARAMS_exchangeabilities.tsv', 
                            np.array(exchangeabilities), 
                            fmt = '%.4f',
                            delimiter= '\t' )
                
                with open(f'{out_folder}/PARAMS_exchangeabilities.npy','wb') as g:
                    jnp.save(g, exchangeabilities)
                
            if 'rate_multipliers' in params_dict['get rate matrix']:
                rate_mult_logits = params_dict['get rate matrix']['rate_multipliers']
                rate_mult = bounded_sigmoid(x = rate_mult_logits, 
                                            min_val = rate_mult_min_val,
                                            max_val = rate_mult_max_val)
    
                with open(f'{out_folder}/PARAMS_rate_multipliers.txt','w') as g:
                    [g.write(f'{elem.item()}\n') for elem in rate_mult]
        
        if 'get equilibrium' in params_dict.keys():
            equl_logits = params_dict['get equilibrium']['Equilibrium distr.']
            equl_dist = nn.softmax( equl_logits, axis=1 )
            
            np.savetxt( f'{out_folder}/PARAMS_equilibriums.tsv', 
                        np.array(equl_dist), 
                        fmt = '%.4f',
                        delimiter= '\t' )
            
            with open(f'{out_folder}/PARAMS-ARR_equilibriums.npy','wb') as g:
                jnp.save(g, equl_dist)
                
        ### transitions
        # tkf91
        if 'tkf91 indel model' in params_dict.keys():
            lam_mu_logits = params_dict['tkf91 indel model']['TKF91 lambda, mu']
            
            lam = bounded_sigmoid(x = lam_mu_logits[0],
                                  min_val = lam_min_val,
                                  max_val = lam_max_val)
            
            offset = bounded_sigmoid(x = lam_mu_logits[1],
                                     min_val = offs_min_val,
                                     max_val = offs_max_val)
            mu = lam / ( 1 -  offset) 
            
            with open(f'{out_folder}/PARAMS_tkf91_indel_params.txt','w') as g:
                g.write(f'insert rate, lambda: {lam}\n')
                g.write(f'deletion rate, mu: {mu}\n')
            
        # tkf92
        elif 'tkf92 indel model' in params_dict.keys():
            # also need range for r values
            out = read_pred_config( 'r_range', (pred_config['tkf_err'], 0.8) )
            r_extend_min_val, r_extend_max_val = out
            del out
        
            lam_mu_logits = params_dict['tkf92 indel model']['TKF92 lambda, mu']
        
            lam = bounded_sigmoid(x = lam_mu_logits[0],
                                  min_val = lam_min_val,
                                  max_val = lam_max_val)
            
            offset = bounded_sigmoid(x = lam_mu_logits[1],
                                     min_val = offs_min_val,
                                     max_val = offs_max_val)
            mu = lam / ( 1 -  offset) 
            
            r_extend_logits = params_dict['tkf92 indel model']['TKF92 r extension prob']
            r_extend = bounded_sigmoid(x = r_extend_logits,
                                       min_val = r_extend_min_val,
                                       max_val = r_extend_max_val)
            
            mean_indel_lengths = 1 / (1 - r_extend)
            
            with open(f'{out_folder}/PARAMS_tkf92_indel_params.txt','w') as g:
                g.write(f'insert rate, lambda: {lam}\n')
                g.write(f'deletion rate, mu: {mu}\n')
                g.write(f'extension prob, r: ')
                [g.write(f'{elem}\t') for elem in r_extend]
                g.write('\n')
                g.write(f'mean indel length: ')
                [g.write(f'{elem}\t') for elem in mean_indel_lengths]
                g.write('\n')
    
    
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
        
        
        ### logprob transitions is more straightforward (only one)
        # (T,4,4)
        if self.indel_model_type == 'tkf91':
            all_transit_matrices = self.transitions_module(t_array = t_array,
                                                           sow_intermediates = sow_intermediates)
        
        elif self.indel_model_type == 'tkf92':
            all_transit_matrices = self.transitions_module(t_array = t_array,
                                                           class_probs = jnp.array([1.]),
                                                           sow_intermediates = sow_intermediates)
            all_transit_matrices['joint'] = all_transit_matrices['joint'][:,0,0,...]
            all_transit_matrices['conditional'] = all_transit_matrices['conditional'][:,0,0,...]
            all_transit_matrices['marginal'] = all_transit_matrices['marginal'][0,0,...]
            
        out_dict = {'logprob_emit_at_indel': logprob_emit_at_indel,
                    'joint_logprob_emit_at_match': joint_logprob_emit_at_match,
                    'all_transit_matrices': all_transit_matrices}
        
        return out_dict
    
    
    def _joint_logprob_align( self,
                             batch,
                             t_array,
                             logprob_emit_at_indel,
                             joint_logprob_emit_at_match,
                             joint_transit_mat ):
        # unpack batch: (B, ...)
        subCounts = batch[0] #(B, 20, 20)
        insCounts = batch[1] #(B, 20)
        delCounts = batch[2]
        transCounts = batch[3] #(B, 4)
        
        
        #######################################
        ### score emissions and transitions   #
        #######################################
        # matches; (T, B)
        match_emit_score = jnp.einsum('tij,bij->tb',
                                      joint_logprob_emit_at_match, 
                                      subCounts)
        # inserts; (B,)
        ins_emit_score = jnp.einsum('i,bi->b',
                                    logprob_emit_at_indel, 
                                    insCounts)
        # deletions; (B,)
        del_emit_score = jnp.einsum('i,bi->b',
                                    logprob_emit_at_indel, 
                                    delCounts)
        
        # transitions; (T,B)
        joint_transit_score = jnp.einsum('tmn,bmn->tb', 
                                         joint_transit_mat, 
                                         transCounts)
        
        # final score is logprob transitions + logprob emissions
        # (T,B)
        joint_logprob_perSamp_perTime = (match_emit_score + 
                                         ins_emit_score[None,:] +
                                         del_emit_score[None,:] +
                                         joint_transit_score)
        
        # marginalize over times; (B,)
        if t_array.shape[0] > 1:
            joint_neg_logP = -self._marginalize_over_times(logprob_perSamp_perTime = joint_logprob_perSamp_perTime,
                                        exponential_dist_param = self.exponential_dist_param,
                                        t_array = t_array)
        else:
            joint_neg_logP = -joint_logprob_perSamp_perTime[0,:]
        
        # normalize (don't include <bos> or <eos>)
        if self.norm_loss_by == 'desc_len':
            length_for_normalization = ( subCounts.sum(axis=(-2, -1)) + 
                                         insCounts.sum(axis=(-1))
                                         )
        
        elif self.norm_loss_by == 'align_len':
            length_for_normalization = ( subCounts.sum(axis=(-2, -1)) + 
                                         insCounts.sum(axis=(-1)) + 
                                         delCounts.sum(axis=(-1))
                                         ) 
        
        joint_neg_logP_length_normed = joint_neg_logP / length_for_normalization
        
        return {'joint_neg_logP': joint_neg_logP,
                'joint_neg_logP_length_normed': joint_neg_logP_length_normed}
    
    
    def _marginalize_over_times(self,
                               logprob_perSamp_perTime,
                               exponential_dist_param,
                               t_array):
        # logP(t_k) = exponential distribution
        logP_time = ( jnp.log(exponential_dist_param) - 
                      jnp.log(exponential_dist_param) * t_array )
        log_t_grid = jnp.log( t_array[1:] - t_array[:-1] )
        
        # kind of a hack, but repeat the last time array value
        log_t_grid = jnp.concatenate( [log_t_grid, log_t_grid[-1][None] ], axis=0)
        
        logP_perSamp_perTime_withConst = ( logprob_perSamp_perTime +
                                           logP_time[:,None] +
                                           log_t_grid[:,None] )
        
        logP_perSamp_raw = logsumexp(logP_perSamp_perTime_withConst, axis=0)
        
        return logP_perSamp_raw
    
    
class IndpPairHMMFitRateMult(IndpPairHMMFitBoth):
    """
    same as IndpPairHMMFitBoth, but now keep LG08 exchangeabilites
    
    inherits everything except setup
    """
    config: dict
    name: str
    
    def setup(self):
        num_emit_site_classes = self.config['num_emit_site_classes']
        self.indel_model_type = self.config['indel_model_type']
        self.norm_loss_by = self.config['norm_loss_by']
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        
        ### how to score emissions from indel sites
        if num_emit_site_classes == 1:
            self.indel_prob_module = LogEqulVecFromCounts(config = self.config,
                                                       name = f'get equilibrium')
        elif num_emit_site_classes > 1:
            self.indel_prob_module = LogEqulVecPerClass(config = self.config,
                                                     name = f'get equilibrium')
        
        
        ### rate matrix to score emissions from match sites
        self.rate_matrix_module = LG08RateMatFitRateMult(config = self.config,
                                                 name = f'get rate matrix')
        
        
        ### now need probabilities for rate classes themselves
        self.site_class_probability_module = SiteClassLogprobs(config = self.config,
                                                  name = f'get site class probabilities')
        
        
        ### TKF91 or TKF92
        if self.indel_model_type == 'tkf91':
            self.transitions_module = TKF91TransitionLogprobs(config = self.config,
                                                     name = f'tkf91 indel model')
        
        elif self.indel_model_type == 'tkf92':
            self.transitions_module = TKF92TransitionLogprobs(config = self.config,
                                                     name = f'tkf92 indel model')


class IndpPairHMMLoadAll(IndpPairHMMFitBoth):
    """
    same as IndpPairHMMFitBoth, but load values (i.e. no free parameters)
    
    inherits everything except setup and write_params (which is 
        overwritten with placeholder method)
    
    files must exist:
        equl_file
        tkf_params_file
    """
    config: dict
    name: str
    
    def setup(self):
        num_emit_site_classes = self.config['num_emit_site_classes']
        self.indel_model_type = self.config['indel_model_type']
        self.norm_loss_by = self.config['norm_loss_by']
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        
        ### how to score emissions from indel sites
        self.indel_prob_module = LogEqulVecFromFile(config = self.config,
                                                   name = f'get equilibrium')
        
        
        ### rate matrix to score emissions from match sites
        self.rate_matrix_module = LG08RateMatFromFile(config = self.config,
                                                 name = f'get rate matrix')
        
        
        ### probability of site classes
        self.site_class_probability_module = SiteClassLogprobsFromFile(config = self.config,
                                                 name = f'get site class probabilities')
        
        
        ### TKF91 or TKF92
        ### make sure you're loading from a model file here
        if self.indel_model_type == 'tkf91':
            self.transitions_module = TKF91TransitionLogprobsFromFile(config = self.config,
                                                     name = f'tkf91 indel model')
        
        elif self.indel_model_type == 'tkf92':
            self.transitions_module = TKF92TransitionLogprobsFromFile(config = self.config,
                                                     name = f'tkf92 indel model')
            
    def write_params(self, **kwargs):
        pass

