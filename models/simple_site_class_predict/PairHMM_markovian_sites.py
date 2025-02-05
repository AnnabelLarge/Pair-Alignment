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
from models.simple_site_class_predict.transition_models import JointTKF92TransitionLogprobs



class MarkovSitesJointPairHMM(ModuleBase):
    config: dict
    name: str
    
    def setup(self):
        num_site_classes = self.config['num_emit_site_classes']
        indel_model = self.config['indel_model']
        self.norm_loss_by = self.config['norm_loss_by']
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        
        ### how to score emissions from indel sites
        if num_site_classes == 1:
            self.indel_prob_module = EqulVecFromCounts(config = self.config,
                                                       name = f'get equilibrium')
        elif num_site_classes > 1:
            self.indel_prob_module = EqulVecPerClass(config = self.config,
                                                     name = f'get equilibrium')
        
        # probability of being at a particular class
        self.curr_class_logits = self.param('Site class logits',
                                            nn.initializers.glorot_uniform(),
                                            (num_site_classes,),
                                            jnp.float32)
    
    
        ### rate matrix to score emissions from match sites
        self.rate_matrix_module = PerClassRateMat(config = self.config,
                                                 name = f'get rate matrix')
        
        
        ### Has to be TKF92 joint
        self.transitions_module = JointTKF92TransitionLogprobs(config = self.config,
                                                     name = f'tkf92 indel model')
    
    def __call__(self,
                 aligned_inputs,
                 t_array,
                 sow_intermediates: bool):
        finalpred_sow_outputs = interms_for_tboard['finalpred_sow_outputs']

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
        del to_expm
        
        # (T,C,4,4)
        logprob_transit = self.transitions_module(t_array = t_array,
                                                  sow_intermediates = sow_intermediates)
        
        
        ######################################
        ### initialize with <start> -> any   #
        ######################################
        aligned_inputs = jnp.where( aligned_inputs != 5,
                                    aligned_inputs,
                                    4)
        
        prev_state = aligned_inputs[:,0,2] # B,
        curr_state = aligned_inputs[:,1,2] # B,
        anc_tok =    aligned_inputs[:,1,0] # B,
        desc_tok =   aligned_inputs[:,1,1] # B,
        
        ### emissions
        # match
        e = e + jnp.where( curr_state == 1,
                           logprob_emit_at_match[:,:,anc_toks-3, desc_toks-3],
                           0 )
        # ins (score descendant)
        e = e + jnp.where( curr_state == 2,
                           logprob_emit_at_indel[:,desc_toks-3],
                           0 )
        # del (score ancestor)
        e = e + jnp.where( curr_state == 3,
                           logprob_emit_at_indel[:,anc_toks-3],
                           0 )
        
        ### transitions
        tmp = jnp.take_along_axis(arr = logprob_transit, 
                                  indices = prev_state[None, None, :, None]-1, 
                                  axis=2)
        
        tr = jnp.take_along_axis( arr = tmp,
                                  indices = curr_state[None, None, :, None]-1,
                                  axis = 3)
        tr = tr[...,0]
        
        init_carry = {'alpha': (tr + e),
                      'state': curr_state}
        
        
        #######################################################
        ### scan donwn length dimension to end of alignment   #
        #######################################################
        def scan_fn(carry_dict, index):
            prev_alpha = carry_dict['alpha']
            prev_state = carry_dict['state']
            
            prev_state = aligned_inputs[:,index-1,2]
            curr_state = aligned_inputs[:,  index,2]
            anc_toks =   aligned_inputs[:,  index,0]
            desc_toks =  aligned_inputs[:,  index,1]
            
            ### emissions
            e = jnp.zeros( (T, C, B,) )
            e = e + jnp.where( curr_state == 1,
                               logprob_emit_at_match[:,:,anc_toks-3, desc_toks-3],
                               0 )
            e = e + jnp.where( curr_state == 2,
                               logprob_emit_at_indel[:,desc_toks-3],
                               0 )
            e = e + jnp.where( curr_state == 3,
                               logprob_emit_at_indel[:,anc_toks-3],
                               0 )
            
            ### transition probabilities
            tmp = jnp.take_along_axis(arr = logprob_transit, 
                                      indices = prev_state[None, None, :, None]-1, 
                                      axis=2)
            
            tr = jnp.take_along_axis( arr = tmp,
                                      indices = curr_state[None, None, :, None]-1,
                                      axis = 3)

            def main_body(in_carry):
                # (T, C_from, A, A), (C_to) -> (T, C_from, C_to, A)
                tr_per_class = tr[:, :, None, :] + log_class_probs[None, None, :, None]
                
                # like dot product with C_from, C_to
                # output is T, C, B
                return e + logsumexp(in_carry[:, :, None, :] + tr_per_class, 
                                     axis=1)
            
            def end(in_carry):
                # output is T, C, B
                return tr + in_carry
            
            ### alpha update, in log space ONLY if curr_state is not pad
            new_alpha = jnp.where(curr_state != 0,
                                  jnp.where( curr_state != 4,
                                             main_body(prev_alpha),
                                             end(prev_alpha) ),
                                  prev_alpha )
                                     
            new_carry_dict = {'alpha': new_alpha,
                              'state': curr_state}
            
            return (new_carry_dict, None)
        
        ### end scan function; need to loop over L ,the second dim
        idx_arr = jnp.array( [i for i in range(2, aligned_inputs.shape[1])] )
        scan_out,_ = jax.lax.scan( f = scan_fn,
                                   init = init_carry,
                                   xs = idx_arr,
                                   length = idx_arr.shape[0] )
        
        # T, B
        logprob_perSamp_perTime = logsumexp(scan_out['alpha'], axis=1)
        
        
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
            # don't count <bos>
            cond1 = (aligned_inputs[...,0] != 0).sum(axis=1)
            cond2 = (aligned_inputs[...,0] != 43).sum(axis=1)
            length_for_normalization = (cond1 & cond2).sum(axis=1) - 1 
        
        elif self.norm_loss_by == 'align_len':
            # don't count <bos>
            length_for_normalization = (aligned_inputs[...,0] != 0).sum(axis=1) - 1
        
        logprob_perSamp_length_normed = sum_neg_logP / length_for_normalization
        loss = -jnp.mean(logprob_perSamp_length_normed)
        
        aux_dict = {'sum_neg_logP': sum_neg_logP,
                    'neg_logP_length_normed': logprob_perSamp_length_normed}
        
        return loss, aux_dict
    
    
    def marginalize_over_times(self,
                               logprob_perSamp_perTime,
                               exponential_dist_param,
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
        
        
        
class JointPairHMMLoadAll(MarkovSitesJointPairHMM):
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
        
        
        ### transitions modele
        self.transitions_module = JointTKF92TransitionLogprobs(config = self.config,
                                                     name = f'tkf92 indel model')

    