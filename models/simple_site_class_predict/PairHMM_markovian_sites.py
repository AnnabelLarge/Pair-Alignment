#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 04:33:00 2025

@author: annabel
"""
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
                                                       LG08RateMat,
                                                       PerClassRateMat,
                                                       SiteClassLogprobs,
                                                       SiteClassLogprobsFromFile)
from models.simple_site_class_predict.transition_models import (JointTKF92TransitionLogprobs,
                                                        JointTKF92TransitionLogprobsFromFile)


def bounded_sigmoid(x, min_val, max_val):
    return min_val + (max_val - min_val) / (1 + jnp.exp(-x))
    
    
class MarkovSitesJointPairHMM(ModuleBase):
    config: dict
    name: str
    
    def setup(self):
        self.num_site_classes = self.config['num_emit_site_classes']
        self.norm_loss_by = self.config['norm_loss_by']
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        
        ### how to score emissions from indel sites
        if self.num_site_classes == 1:
            self.indel_prob_module = LogEqulVecFromCounts(config = self.config,
                                                       name = f'get equilibrium')
        elif self.num_site_classes > 1:
            self.indel_prob_module = LogEqulVecPerClass(config = self.config,
                                                     name = f'get equilibrium')
        
        
        ### probability of site classes
        self.class_logprobs_module = SiteClassLogprobs(config = self.config,
                                                       name = 'class_logits')
    
    
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
        T = t_array.shape[0]
        B = aligned_inputs.shape[0]
        L = aligned_inputs.shape[1]
        C = self.num_site_classes
        
        ############################
        ### get logprob matrices   #
        ############################
        ### emissions from indels
        logprob_emit_at_indel = self.indel_prob_module()
        
        
        ### emissions from match sites
        # this is already rho * chi * pi
        rate_mat_times_rho = self.rate_matrix_module(logprob_equl = logprob_emit_at_indel,
                                                     sow_intermediates = sow_intermediates)
        
        # rate_mat_times_rho: (C, alph, alph)
        # time: (T,)
        # output: (T, C, alph, alph)
        to_expm = jnp.multiply( rate_mat_times_rho[None, ...],
                                t_array[..., None,None,None] )
        cond_prob_emit_at_match = expm(to_expm)
        cond_logprob_emit_at_match = jnp.where( cond_prob_emit_at_match>0,
                                           jnp.log(cond_prob_emit_at_match),
                                           jnp.log(jnp.finfo('float32').smallest_normal) )
        
        joint_logprob_emit_at_match = cond_logprob_emit_at_match + logprob_emit_at_indel[None,:,:,None]
        del to_expm
        
        
        ### transition logprobs
        # (T,C,4,4)
        logprob_transit = self.transitions_module(t_array = t_array,
                                                  sow_intermediates = sow_intermediates)
        
        
        ### probability of being in any particular class
        log_class_probs = self.class_logprobs_module()
        
        
        ######################################
        ### initialize with <start> -> any   #
        ######################################
        prev_state = aligned_inputs[:,0,2] # B,
        curr_state = aligned_inputs[:,1,2] # B,
        anc_toks =    aligned_inputs[:,1,0] # B,
        desc_toks =   aligned_inputs[:,1,1] # B,
        
        # for easier indexing: code <eos> as 4
        curr_state = jnp.where( curr_state != 5, curr_state, 4)
        
        
        ### emissions
        e = jnp.zeros( (T, C, B,) )

        # match
        e = e + jnp.where( curr_state == 1,
                           joint_logprob_emit_at_match[:,:,anc_toks-3, desc_toks-3],
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
        tr = tr[...,0] + log_class_probs[None, :, None]
        
        init_carry = {'alpha': (tr + e),
                      'state': curr_state}
        
        
        ######################################################
        ### scan down length dimension to end of alignment   #
        ######################################################
        def scan_fn(carry_dict, curr_index):
            prev_alpha = carry_dict['alpha']
            prev_state = carry_dict['state']
            
            prev_state = aligned_inputs[:,curr_index-1,2]
            curr_state = aligned_inputs[:,  curr_index,2]
            anc_toks =   aligned_inputs[:,  curr_index,0]
            desc_toks =  aligned_inputs[:,  curr_index,1]
            
            # for easier indexing: code <eos> as 4
            curr_state = jnp.where( curr_state != 5, curr_state, 4)
            
            
            ### emissions
            e = jnp.zeros( (T, C, B,) )
            e = e + jnp.where( curr_state == 1,
                               joint_logprob_emit_at_match[:,:,anc_toks-3, desc_toks-3],
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
            
            tr = tr[...,0]
            del tmp

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
        
        ### end scan function definition, use scan
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
            predicate = (aligned_inputs[...,0] !=0) & (aligned_inputs[...,0] !=43)
            length_for_normalization = predicate.sum(axis=1) - 1
        
        elif self.norm_loss_by == 'align_len':
            # don't count <bos>
            length_for_normalization = (aligned_inputs[...,0] != 0).sum(axis=1) - 1
        
        logprob_perSamp_length_normed = sum_neg_logP / length_for_normalization
        loss = -jnp.mean(logprob_perSamp_length_normed)
        
        out = {'loss': loss,
               'sum_neg_logP': sum_neg_logP,
               'neg_logP_length_normed': logprob_perSamp_length_normed}
        
        return out
    
    
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
        class_logits = params_dict['get site class probabilities']['class_logits']
        class_probs = nn.log_softmax(class_logits)
        with open(f'{out_folder}/PARAMS_class_probs.txt','w') as g:
            [g.write(f'{elem.item()}') for elem in class_probs]
        
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
                    [g.write(f'{elem.item()}') for elem in rate_multipliers]
                
                
        ### transitions
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
        
        

class JointPairHMMLoadAll(MarkovSitesJointPairHMM):
    """
    same as JointPairHMM, but load values (i.e. no free parameters)
    
    files must exist:
        rate_multiplier_file
        equl_file
        logprob of classes file
        tkf_params_file
    """
    config: dict
    name: str
    
    def setup(self):
        self.num_site_classes = self.config['num_emit_site_classes']
        self.norm_loss_by = self.config['norm_loss_by']
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        
        ### how to score emissions from indel sites
        self.indel_prob_module = LogEqulVecFromFile(config = self.config,
                                                   name = f'get equilibrium')
        
        
        ### rate matrix to score emissions from match sites
        self.rate_matrix_module = LG08RateMat(config = self.config,
                                                 name = f'get rate matrix')
        
        ### probability of site classes
        self.class_logprobs_module = SiteClassLogprobsFromFile(config = self.config,
                                                               name = 'class_logits')
        
        ### transitions modele
        self.transitions_module = JointTKF92TransitionLogprobsFromFile(config = self.config,
                                                     name = f'tkf92 indel model')
    
    def write_params(self, **kwargs):
        pass

    