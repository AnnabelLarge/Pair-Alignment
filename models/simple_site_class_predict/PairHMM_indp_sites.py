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

from models.model_utils.BaseClasses import ModuleBase
from models.simple_site_class_predict.emission_models import (EqulVecFromCounts,
                                                       EqulVecPerClass,
                                                       LG08RateMat,
                                                       PerClassRateMat)
from models.simple_site_class_predict.transition_models import (CondTKF91TransitionLogprobs, 
                                                         JointTKF91TransitionLogprobs,
                                                         CondTKF92TransitionLogprobs, 
                                                         JointTKF92TransitionLogprobs)


class CondPairHMM(ModuleBase):
    config: dict
    name: str
    
    def setup(self):
        num_emit_site_classes = self.config['num_emit_site_classes']
        indel_model = self.config['indel_model']
        self.norm_loss_by = self.config['norm_loss_by']
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        
        ### how to score emissions from indel sites
        if num_emit_site_classes == 1:
            self.indel_prob_module = EqulVecFromCounts(config = self.config,
                                                       name = f'get equilibrium')
        elif num_emit_site_classes > 1:
            self.indel_prob_module = EqulVecPerClass(config = self.config,
                                                     name = f'get equilibrium')
        
        
        ### rate matrix to score emissions from match sites
        self.rate_matrix_module = PerClassRateMat(config = self.config,
                                                 name = f'get rate matrix')
        
        
        ### TKF91 or TKF92
        if indel_model == 'tkf91':
            self.transitions_module = CondTKF91TransitionLogprobs(config = self.config,
                                                     name = f'tkf91 indel model')
        
        elif indel_model == 'tkf92':
            self.transitions_module = CondTKF92TransitionLogprobs(config = self.config,
                                                     name = f'tkf92 indel model')
    
    def __call__(self,
                 batch,
                 sow_intermediates: bool):
        # unpack batch: (B, ...)
        subCounts = batch[0] #(B, 20, 20)
        insCounts = batch[1] #(B, 20)
        transCounts = batch[3] #(B, 4)
        t_array = batch[4]
        
        ### get logprob matrices
        logprob_emit_at_indel = self.indel_prob_module()
        
        rate_mat_times_rho = self.rate_matrix_module(logprob_equl = logprob_emit_at_indel,
                                                     sow_intermediates = sow_intermediates)
        
        # rate_mat_times_rho: (C, alph, alph)
        # time: (T,)
        # output: (T, C, alph, alph)
        to_expm = jnp.einsum('cij,t->tcij', 
                             rate_mat_times_rho[None, ...],
                             t_array[..., None,None,None])
        logprob_emit_at_match = expm(to_expm)
        logprob_emit_at_match = self.weight_by_equilibrium(logprob_emit_at_indel = logprob_emit_at_indel,
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
                                   ins_emit_score +
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
            length_for_normalization = ( subCounts.sum(axis=(1,2)) + 
                                         insCounts.sum(axis=(1,2))
                                         )
            length_for_normalization += 1 #for <eos>
        
        elif self.norm_loss_by == 'align_len':
            length_for_normalization = transCounts.sum(axis=(1,2)) 
        
        logprob_perSamp_length_normed = sum_neg_logP / length_for_normalization
        loss = -jnp.mean(logprob_perSamp_length_normed)
        
        aux_dict = {'sum_neg_logP': sum_neg_logP,
                    'neg_logP_length_normed': logprob_perSamp_length_normed}
        
        return loss, aux_dict
        
        
    def weight_by_equilibrium(self,
                              logprob_emit_at_indel,
                              logprob_emit_at_match): 
        # (C, alph, alph)
        equl_prob = jnp.exp( logprob_emit_at_indel )
        
        # weight each by pi(c|x)
        sum_per_class = equl_prob.sum(axis=0)
        weight = (equl_prob / sum_per_class[None,:])
        log_weight = safe_log( weight )
        
        # add weights; return (C, alph, alph) matrix
        weighted_logprob_mat = log_weight + logprob_emit_at_match
        return weighted_logprob_mat
    
    def marginalize_over_times(self,
                               logprob_perSamp_perTime,
                               exponential_dist_param)
                               t_array):
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
        
        return logP_perSamp_raw
        
    
class JointPairHMM(CondPairHMM):
    """
    inherit setup(), weight_by_equilibrium, and marginalize_over_times
      from CondPairHMM
    """
    config: dict
    name: str
    
    def setup(self):
        num_emit_site_classes = self.config['num_emit_site_classes']
        indel_model = self.config['indel_model']
        self.norm_loss_by = self.config['norm_loss_by']
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        
        ### how to score emissions from indel sites
        if num_emit_site_classes == 1:
            self.indel_prob_module = EqulVecFromCounts(config = self.config,
                                                       name = f'get equilibrium')
        elif num_emit_site_classes > 1:
            self.indel_prob_module = EqulVecPerClass(config = self.config,
                                                     name = f'get equilibrium')
        
        
        ### rate matrix to score emissions from match sites
        self.rate_matrix_module = PerClassRateMat(config = self.config,
                                                 name = f'get rate matrix')
        
        
        ### TKF91 or TKF92
        if indel_model == 'tkf91':
            self.transitions_module = JointTKF91TransitionLogprobs(config = self.config,
                                                     name = f'tkf91 indel model')
        
        elif indel_model == 'tkf92':
            self.transitions_module = JointTKF92TransitionLogprobs(config = self.config,
                                                     name = f'tkf92 indel model')
    
    def __call__(self,
                 batch,
                 sow_intermediates: bool):
        # unpack batch: (B, ...)
        subCounts = batch[0] #(B, 20, 20)
        insCounts = batch[1] #(B, 20)
        delCounts = batch[2]
        transCounts = batch[3] #(B, 4)
        t_array = batch[4] #(T,)
        
        ### get logprob matrices
        logprob_emit_at_indel = self.indel_prob_module()
        
        rate_mat_times_rho = self.rate_matrix_module(logprob_equl = logprob_emit_at_indel,
                                                     sow_intermediates = sow_intermediates)
        
        # rate_mat_times_rho: (C, alph, alph)
        # time: (T,)
        # output: (T, C, alph, alph)
        to_expm = jnp.einsum('cij,t->tcij', 
                             rate_mat_times_rho[None, ...],
                             t_array[..., None,None,None])
        cond_logprob_emit_at_match = expm(to_expm)
        joint_logprob_emit_at_match = cond_logprob_emit_at_match + equl_logprob[None,...,None]
        joint_logprob_emit_at_match = self.weight_by_equilibrium(logprob_emit_at_indel = logprob_emit_at_indel,
                                                           logprob_emit_at_match = joint_logprob_emit_at_match)
        
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
                                   ins_emit_score +
                                   del_emit_score +
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
            length_for_normalization = ( subCounts.sum(axis=(1,2)) + 
                                         insCounts.sum(axis=(1,2))
                                         )
            length_for_normalization += 1 #for <eos>
        
        elif self.norm_loss_by == 'align_len':
            length_for_normalization = transCounts.sum(axis=(1,2)) 
        
        logprob_perSamp_length_normed = sum_neg_logP / length_for_normalization
        loss = -jnp.mean(logprob_perSamp_length_normed)
        
        aux_dict = {'sum_neg_logP': sum_neg_logP,
                    'neg_logP_length_normed': logprob_perSamp_length_normed}
        
        return loss, aux_dict


class JointPairHMMLoadAll(JointPairHMM):
    """
    same as JointPairHMM, but load values (i.e. no free parameters)
    """
    config: dict
    name: str
    
    def setup(self):
        num_emit_site_classes = 1 #CHANGE HERE: force this to be 1
        indel_model = self.config['indel_model']
        self.norm_loss_by = self.config['norm_loss_by']
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        
        ### how to score emissions from indel sites
        ### CHANGE HERE: force this to be from counts
        self.indel_prob_module = EqulVecFromCounts(config = self.config,
                                                   name = f'get equilibrium')
        
        
        ### rate matrix to score emissions from match sites
        self.rate_matrix_module = LG08RateMat(config = self.config,
                                                 name = f'get rate matrix')
        
        
        ### TKF91 or TKF92
        ### make sure you're loading from a model file here
        if indel_model == 'tkf91':
            self.transitions_module = JointTKF91TransitionLogprobs(config = self.config,
                                                     name = f'tkf91 indel model')
        
        elif indel_model == 'tkf92':
            self.transitions_module = JointTKF92TransitionLogprobs(config = self.config,
                                                     name = f'tkf92 indel model')

    




