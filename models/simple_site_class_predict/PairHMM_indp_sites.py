#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 04:33:00 2025

@author: annabel
"""
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
from models.simple_site_class_predict.transition_models import (CondTKF91TransitionLogprobs, 
                                                         JointTKF91TransitionLogprobs,
                                                         CondTKF92TransitionLogprobs, 
                                                         JointTKF92TransitionLogprobs,
                                                         JointTKF91TransitionLogprobsFromFile,
                                                         JointTKF92TransitionLogprobsFromFile)

def bounded_sigmoid(x, min_val, max_val):
    return min_val + (max_val - min_val) / (1 + jnp.exp(-x))

def safe_log(x):
    return jnp.log( jnp.where( x>0, 
                               x, 
                               jnp.finfo('float32').smallest_normal ) )
   

class CondPairHMM(ModuleBase):
    config: dict
    name: str
    
    def setup(self):
        num_emit_site_classes = self.config['num_emit_site_classes']
        indel_model_type = self.config['indel_model_type']
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
        
        
        ### TKF91 or TKF92
        if indel_model_type == 'tkf91':
            self.transitions_module = CondTKF91TransitionLogprobs(config = self.config,
                                                     name = f'tkf91 indel model')
        
        elif indel_model_type == 'tkf92':
            self.transitions_module = CondTKF92TransitionLogprobs(config = self.config,
                                                     name = f'tkf92 indel model')
    
    def __call__(self,
                 batch,
                 t_array,
                 sow_intermediates: bool):
        # unpack batch: (B, ...)
        subCounts = batch[0] #(B, 20, 20)
        insCounts = batch[1] #(B, 20)
        transCounts = batch[3] #(B, 4)
        
        # TODO: if using one time per sample (i.e. the time from FastTree), then
        # you'll need to unpack time from batch; will have to initialize a new
        # time array with shape (T, B) instead of (T,)
        
        ### get logprob matrices
        logprob_emit_at_indel = self.indel_prob_module()
        rate_mat_times_rho = self.rate_matrix_module(logprob_equl = logprob_emit_at_indel,
                                                     sow_intermediates = sow_intermediates)
        
        # rate_mat_times_rho: (C, alph, alph)
        # time: (T,)
        # output: (T, C, alph, alph)
        to_expm = jnp.multiply( rate_mat_times_rho[None,...],
                                t_array[:, None,None,None,] )
        
        prob_emit_at_match = expm(to_expm)
        logprob_emit_at_match = safe_log( prob_emit_at_match )
        logprob_emit_at_match = self.apply_weighting(logprob_emit_at_indel = logprob_emit_at_indel,
                                                     logprob_emit_at_match = logprob_emit_at_match)
        
        # (T,4,4)
        logprob_transit = self.transitions_module(t_array = t_array,
                                                  sow_intermediates = sow_intermediates)
        
        
        ### score
        # (T, B)
        match_emit_score = jnp.einsum('tcij,bij->tb',
                                      logprob_emit_at_match, 
                                      subCounts)
        # (B,)
        ins_emit_score = jnp.einsum('ci,bi->b',
                                    logprob_emit_at_indel, 
                                    insCounts)
        
        # T,B)
        transit_score = jnp.einsum('tcmn,bmn->tb', 
                                   logprob_transit, 
                                   transCounts)
        
        # final score is (T,B)
        logprob_perSamp_perTime = (match_emit_score + 
                                   ins_emit_score[None,:] +
                                   transit_score)
        
        ### marginalize over times
        # (B,)
        if t_array.shape[0] > 1:
            sum_neg_logP = self.marginalize_over_times(logprob_perSamp_perTime = logprob_perSamp_perTime,
                                        exponential_dist_param = self.exponential_dist_param,
                                        t_array = t_array)
        else:
            sum_neg_logP = logprob_perSamp_perTime[0,:]
        
        
        ### normalize
        if self.norm_loss_by == 'desc_len':
            length_for_normalization = ( subCounts.sum(axis=(-2, -1)) + 
                                         insCounts.sum(axis=(-1))
                                         )
            length_for_normalization += 1 #for <eos>
        
        elif self.norm_loss_by == 'align_len':
            length_for_normalization = transCounts.sum(axis=(-2, -1)) 
        
        logprob_perSamp_length_normed = sum_neg_logP / length_for_normalization
        loss = -jnp.mean(logprob_perSamp_length_normed)
        
        out = {'loss': loss,
               'sum_neg_logP': sum_neg_logP,
               'neg_logP_length_normed': logprob_perSamp_length_normed}
        
        return out
        
    def apply_weighting(self,
                        logprob_emit_at_indel,
                        logprob_emit_at_match): 
        """
        weights determined by ancestor frequencies
        """
        # (C, alph)
        equl_prob = jnp.exp( logprob_emit_at_indel )
        
        # weight each by pi(c|x); broadcast across rows
        sum_per_class = equl_prob.sum(axis=0) #(alph)
        weight = (equl_prob / sum_per_class[None,:]) #(C, alph)
        log_weight = safe_log(weight)
        weighted_logprob_mat = log_weight[None,:,:,None] + logprob_emit_at_match
        return weighted_logprob_mat
    
    def marginalize_over_times(self,
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
    
    
    def write_params(self,
                     pred_config,
                     tstate,
                     out_folder: str):
        params_dict = tstate.params['params']
        
        ##################################################
        ### use default values, if ranges aren't found   #
        ##################################################
        out  = pred_config.get( 'exchange_range', (1e-4, 10) )
        exchange_min_val, exchange_max_val = out
        del out
        
        out  = pred_config.get( 'rate_mult_range', (0.01, 10) )
        rate_mult_min_val, rate_mult_max_val = out
        del out
        
        out = pred_config.get( 'lambda_range', (pred_config['tkf_err'], 3) )
        lam_min_val, lam_max_val = out
        del out
         
        out = pred_config.get( 'offset_range', (pred_config['tkf_err'], 0.333) )
        offs_min_val, offs_max_val = out
        del out
        
        out = pred_config.get( 'r_range', (pred_config['tkf_err'], 0.8) )
        r_extend_min_val, r_extend_max_val = out
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
                
                with open(f'{out_folder}/PARAMS_exchangeabilities.npy','wb') as g:
                    jnp.save(g, exchangeabilities)
                
            if 'rate_multipliers' in params_dict['get rate matrix']:
                rate_mult_logits = params_dict['get rate matrix']['rate_multipliers']
                rate_mult = bounded_sigmoid(x = rate_mult_logits, 
                                            min_val = rate_mult_min_val,
                                            max_val = rate_mult_max_val)
    
                with open(f'{out_folder}/PARAMS_rate_multipliers.txt','w') as g:
                    [g.write(f'{elem.item()}\n') for elem in rate_mult]
                
                
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
                g.write(f'mean indel legnth: ')
                [g.write(f'{elem}\t') for elem in mean_indel_lengths]
                g.write('\n')
        
        
    
class JointPairHMMFitBoth(CondPairHMM):
    """
    inherit marginalize_over_times, and write_params from CondPairHMM
    
    """
    config: dict
    name: str
    
    def setup(self):
        num_emit_site_classes = self.config['num_emit_site_classes']
        indel_model_type = self.config['indel_model_type']
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
        if indel_model_type == 'tkf91':
            self.transitions_module = JointTKF91TransitionLogprobs(config = self.config,
                                                     name = f'tkf91 indel model')
        
        elif indel_model_type == 'tkf92':
            self.transitions_module = JointTKF92TransitionLogprobs(config = self.config,
                                                     name = f'tkf92 indel model')
    
    def __call__(self,
                 batch,
                 t_array,
                 sow_intermediates: bool):
        # unpack batch: (B, ...)
        subCounts = batch[0] #(B, 20, 20)
        insCounts = batch[1] #(B, 20)
        delCounts = batch[2]
        transCounts = batch[3] #(B, 4)
        
        # TODO: if using one time per sample (i.e. the time from FastTree), then
        # you'll need to unpack time from batch; will have to initialize a new
        # time array with shape (T, B) instead of (T,)
        
        ### get logprob matrices
        logprob_emit_at_indel = self.indel_prob_module()
        rate_mat_times_rho = self.rate_matrix_module(logprob_equl = logprob_emit_at_indel,
                                                     sow_intermediates = sow_intermediates)
        
        # rate_mat_times_rho: (C, alph, alph)
        # time: (T,)
        # output: (T, C, alph, alph)
        to_expm = jnp.multiply( rate_mat_times_rho[None,...],
                                t_array[:, None,None,None,] )
        cond_prob_emit_at_match = expm(to_expm)
        cond_logprob_emit_at_match = safe_log( cond_prob_emit_at_match )
        joint_logprob_emit_at_match = cond_logprob_emit_at_match + logprob_emit_at_indel[None,:,:,None]
        joint_logprob_emit_at_match = self.apply_weighting(joint_logprob_per_class = joint_logprob_emit_at_match)
        
        # (T,4,4)
        logprob_transit = self.transitions_module(t_array = t_array,
                                                  sow_intermediates = sow_intermediates)
        

        ### score
        # (T, B)
        match_emit_score = jnp.einsum('tcij,bij->tb',
                                      joint_logprob_emit_at_match, 
                                      subCounts)
        # (B,)
        ins_emit_score = jnp.einsum('ci,bi->b',
                                    logprob_emit_at_indel, 
                                    insCounts)
        # (B,)
        del_emit_score = jnp.einsum('ci,bi->b',
                                    logprob_emit_at_indel, 
                                    delCounts)
        
        # T,B)
        transit_score = jnp.einsum('tcmn,bmn->tb', 
                                   logprob_transit, 
                                   transCounts)
        
        # final score is (T,B)
        logprob_perSamp_perTime = (match_emit_score + 
                                   ins_emit_score[None,:] +
                                   del_emit_score[None,:] +
                                   transit_score)
        
        ### marginalize over times
        # (B,)
        if t_array.shape[0] > 1:
            sum_neg_logP = self.marginalize_over_times(logprob_perSamp_perTime = logprob_perSamp_perTime,
                                        exponential_dist_param = self.exponential_dist_param,
                                        t_array = t_array)
        else:
            sum_neg_logP = logprob_perSamp_perTime[0,:]
        
        
        ### normalize
        if self.norm_loss_by == 'desc_len':
            length_for_normalization = ( subCounts.sum(axis=(-2, -1)) + 
                                         insCounts.sum(axis=(-1))
                                         )
            length_for_normalization += 1 #for <eos>
        
        elif self.norm_loss_by == 'align_len':
            length_for_normalization = transCounts.sum(axis=(-2, -1)) 
        
        logprob_perSamp_length_normed = sum_neg_logP / length_for_normalization
        loss = -jnp.mean(logprob_perSamp_length_normed)
        
        out = {'loss': loss,
                    'sum_neg_logP': sum_neg_logP,
                    'neg_logP_length_normed': logprob_perSamp_length_normed}
        
        return out
    
    
    def apply_weighting(self,
                        joint_logprob_per_class):
        log_weight = self.site_class_probability_module()
        log_weight = log_weight[None,:,None,None]
        return log_weight + joint_logprob_per_class


class JointPairHMMFitRateMult(JointPairHMMFitBoth):
    """
    inherit marginalize_over_times, and write_params from CondPairHMM
    
    inherit call from JointPairHMMFitBoth
    """
    config: dict
    name: str
    
    def setup(self):
        num_emit_site_classes = self.config['num_emit_site_classes']
        indel_model_type = self.config['indel_model_type']
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
        if indel_model_type == 'tkf91':
            self.transitions_module = JointTKF91TransitionLogprobs(config = self.config,
                                                     name = f'tkf91 indel model')
        
        elif indel_model_type == 'tkf92':
            self.transitions_module = JointTKF92TransitionLogprobs(config = self.config,
                                                     name = f'tkf92 indel model')



    
class JointPairHMMLoadAll(JointPairHMMFitBoth):
    """
    same as JointPairHMM, but load values (i.e. no free parameters)
    
    files must exist:
        equl_file
        tkf_params_file
    """
    config: dict
    name: str
    
    def setup(self):
        num_emit_site_classes = self.config['num_emit_site_classes']
        indel_model_type = self.config['indel_model_type']
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
        if indel_model_type == 'tkf91':
            self.transitions_module = JointTKF91TransitionLogprobsFromFile(config = self.config,
                                                     name = f'tkf91 indel model')
        
        elif indel_model_type == 'tkf92':
            self.transitions_module = JointTKF92TransitionLogprobsFromFile(config = self.config,
                                                     name = f'tkf92 indel model')
            
    def write_params(self, **kwargs):
        pass
    
            