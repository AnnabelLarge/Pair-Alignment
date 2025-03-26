#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 04:33:00 2025

@author: annabel

models:
=======
MarkovPairHMM
MarkovPairHMMLoadAll

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
                                                       LG08RateMatFromFile,
                                                       LG08RateMatFitBoth,
                                                       SiteClassLogprobs,
                                                       SiteClassLogprobsFromFile)
from models.simple_site_class_predict.transition_models import (TKF92TransitionLogprobs,
                                                        TKF92TransitionLogprobsFromFile)
from utils.pairhmm_helpers import (bounded_sigmoid,
                                   safe_log)


################################################
### helpers only for pairhmm_markovian_sites   #
################################################
def log_dot_bigger(log_vec, log_mat):
    broadcasted_sum = log_vec[:, :, None, :] + log_mat
    out = jnp.where( broadcasted_sum.sum() < 0,
                     logsumexp(broadcasted_sum, axis=1),
                     jnp.zeros(log_vec.shape)
                     )
    return out

# class DebugNansInfs:
#     def __enter__(self):
#         jax.config.update("jax_debug_nans", True)
#         jax.config.update("jax_debug_infs", True)

#     def __exit__(self, exc_type, exc_value, traceback):
#         jax.config.update("jax_debug_nans", False)
#         jax.config.update("jax_debug_infs", False)

    
class MarkovPairHMM(ModuleBase):
    """
    THIS ASSUMES GAP TOKEN IS 43!!!
    
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
        - _joint_emit_scores
        - _marginalize_over_times
        
    """
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
        # init with values from LG08
        self.rate_matrix_module = LG08RateMatFitBoth(config = self.config,
                                                 name = f'get rate matrix')
        
        
        ### Has to be TKF92 joint
        self.transitions_module = TKF92TransitionLogprobs(config = self.config,
                                                          name = f'tkf92 indel model')
    
    def __call__(self,
                 aligned_inputs,
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
        L_align = aligned_inputs.shape[1]
        
        out = self._get_scoring_matrices( t_array=t_array,
                                          sow_intermediates=sow_intermediates )
        
        logprob_emit_at_indel = out['logprob_emit_at_indel']
        joint_logprob_emit_at_match = out['joint_logprob_emit_at_match']
        joint_transit = out['all_transit_matrices']['joint']
        del out
        
        ######################################################
        ### initialize with <start> -> any (curr pos is 1)   #
        ######################################################
        # emissions; (T, C_curr, B)
        init_emission_logprob = self._joint_emit_scores( aligned_inputs=aligned_inputs,
                                                         pos=1,
                                                         joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                                         logprob_emit_at_indel=logprob_emit_at_indel )
        
        # transitions; assume there's never start -> end; (T, C_curr, B)
        # joint_transit is (T, C_prev, C_curr, S_prev, S_curr)
        # initial state is 4 (<start>); take the last row
        # use C_prev=0 for start class (but it doesn't matter, because the 
        # transition probability is the same for all C_prev)
        curr_state = aligned_inputs[:, 1, 2] # B
        init_trans_logprob = joint_transit[:, 0, :, -1, curr_state-1]
        
        # carry value; (T, C_curr, B)
        init_alpha = init_emission_logprob + init_trans_logprob
        del init_emission_logprob, curr_state, init_trans_logprob
        
        
        ######################################################
        ### scan down length dimension to end of alignment   #
        ######################################################
        def scan_fn(prev_alpha, pos):
            ### unpack
            anc_toks =   aligned_inputs[:,   pos, 0]
            desc_toks =  aligned_inputs[:,   pos, 1]

            prev_state = aligned_inputs[:, pos-1, 2]
            curr_state = aligned_inputs[:,   pos, 2]
            curr_state = jnp.where( curr_state!=5, curr_state, 4 )
            
            
            ### emissions
            e = self._joint_emit_scores( aligned_inputs=aligned_inputs,
                                         pos=pos,
                                         joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                         logprob_emit_at_indel=logprob_emit_at_indel )
            
            
            ### transition probabilities
            def main_body(in_carry):
                # like dot product with C_prev, C_curr
                tr_per_class = joint_transit[..., prev_state-1, curr_state-1]                
                return e + logsumexp(in_carry[:, :, None, :] + tr_per_class, axis=1)
            
            def end(in_carry):
                # if end, then curr_state = -1 (<end>)
                tr_per_class = joint_transit[..., prev_state-1, -1]
                return tr_per_class + in_carry
            
            
            ### alpha update, in log space ONLY if curr_state is not pad
            new_alpha = jnp.where(curr_state != 0,
                                  jnp.where( curr_state != 4,
                                              main_body(prev_alpha),
                                              end(prev_alpha) ),
                                  prev_alpha )
            
            return (new_alpha, None)
        
        ### end scan function definition, use scan
        idx_arr = jnp.array( [ i for i in range(2, L_align) ] )
        final_alpha, _ = jax.lax.scan( f = scan_fn,
                                       init = init_alpha,
                                       xs = idx_arr,
                                       length = idx_arr.shape[0] )
        
        # (T, C_curr, B) -> (T, B)
        joint_logprob_perSamp_perTime = logsumexp(final_alpha, axis=1)
        
        
        ### marginalize over times
        # (B,)
        if t_array.shape[0] > 1:
            joint_neg_logP = -self._marginalize_over_times(logprob_perSamp_perTime = joint_logprob_perSamp_perTime,
                                        exponential_dist_param = self.exponential_dist_param,
                                        t_array = t_array)
        else:
            joint_neg_logP = -joint_logprob_perSamp_perTime[0,:]
        
        
        ### normalize
        if self.norm_loss_by == 'desc_len':
            # where descendant is not pad or gap
            # ASSUME gap token is 43
            mask = (aligned_inputs[...,1] !=0) & (aligned_inputs[...,1] !=43)

        elif self.norm_loss_by == 'align_len':
            # where descendant is not pad (but could be gap)
            mask = (aligned_inputs[...,1] != 0).sum(axis=1)
        
        # don't count <bos> or <eos>, so always subtract 2
        length_for_normalization = mask.sum(axis=1) - 2
        joint_neg_logP_length_normed = joint_neg_logP / length_for_normalization
        loss = jnp.mean(joint_neg_logP_length_normed)
        
        aux_dict = {'joint_neg_logP': joint_neg_logP,
                    'joint_neg_logP_length_normed': joint_neg_logP_length_normed}
        
        return loss, aux_dict
    
    
    def calculate_all_loglikes(self,
                               aligned_inputs,
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
        
        Calculate joint and sequence marginals in one jax.lax.scan operation
        """
        # get lengths; subtract two to remove <bos> and <eos>
        align_len = (aligned_inputs[...,0] != 0).sum(axis=1) - 2
        anc_len = ( (aligned_inputs[...,0] !=0) & (aligned_inputs[...,0] !=43) ).sum(axis=1) - 2
        desc_len = ( (aligned_inputs[...,1] !=0) & (aligned_inputs[...,1] !=43) ).sum(axis=1) - 2
        
        # get score matrices
        out = self._get_scoring_matrices( t_array=t_array,
                                          sow_intermediates=sow_intermediates )
        
        logprob_emit_at_indel = out['logprob_emit_at_indel']
        joint_logprob_emit_at_match = out['joint_logprob_emit_at_match']
        joint_transit = out['all_transit_matrices']['joint']
        marginal_transit = out['all_transit_matrices']['marginal'] 
        del out
        
        ######################################################
        ### initialize with <start> -> any (curr pos is 1)   #
        ######################################################
        # anc and desc toks may or may not be gaps...
        anc_toks =   aligned_inputs[:, 1, 0]
        desc_toks =  aligned_inputs[:, 1, 1]
        curr_state = aligned_inputs[:, 1, 2]
        
        ### P(anc, desc, align)
        # emissions; (T, C_curr, B)
        init_joint_e = self._joint_emit_scores( aligned_inputs=aligned_inputs,
                                                pos=1,
                                                joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                                logprob_emit_at_indel=logprob_emit_at_indel )
        
        # transitions; assume there's never start -> end; (T, C_curr, B)
        # joint_transit is (T, C_prev, C_curr, S_prev, S_curr)
        # initial state is 4 (<start>); take the last row
        # use C_prev=0 for start class (but it doesn't matter, because the 
        # transition probability is the same for all C_prev)
        init_joint_tr = joint_transit[:, 0, :, -1, curr_state-1]
        
        # carry value; (T, C_curr, B)
        init_joint_alpha = init_joint_e + init_joint_tr
        del init_joint_e, init_joint_tr
        
        
        ### P(anc); (C_curr, B)
        # emissions; only valid if current position is match or delete
        anc_mask = (curr_state == 1) | (curr_state == 3)
        init_anc_e = logprob_emit_at_indel[:, anc_toks - 3] * anc_mask 
        
        # transitions
        # marginal_transit is (C_prev, C_curr, S_prev, S_curr), where:
        #   (S_prev=0, S_curr=0) is emit->emit
        #   (S_prev=1, S_curr=0) is <s>->emit
        #   (S_prev=0, S_curr=1) is emit-><e>
        # use C_prev=0 for start class (but it doesn't matter, because the 
        # transition probability is the same for all C_prev)
        init_anc_tr = marginal_transit[0,:,1,0][...,None] * anc_mask
        
        # things to remember are:
        #   alpha: for forward algo
        #   md_seen: used to remember if <s> -> emit has been used yet
        #   (there could be gaps in between <s> and first emission)
        init_anc_alpha = init_anc_e + init_anc_tr
        md_seen = anc_mask
        del init_anc_e, init_anc_tr, anc_mask
        
        
        ### P(desc); (C, B)
        # emissions; only valid if current position is match or ins
        desc_mask = (curr_state == 1) | (curr_state == 2)
        init_desc_e = logprob_emit_at_indel[:, desc_toks - 3] * desc_mask 
        
        # transitions
        init_desc_tr = marginal_transit[0,:,1,0][...,None] * desc_mask
        
        # things to remember are:
        #   alpha: for forward algo
        #   mi_seen: used to remember if <s> -> emit has been used yet
        #   (there could be gaps in between <s> and first emission)
        init_desc_alpha = init_desc_e + init_desc_tr
        mi_seen = desc_mask
        del init_desc_e, init_desc_tr, desc_mask, curr_state
        
        init_dict = {'joint_alpha': init_joint_alpha,
                     'anc_alpha': init_anc_alpha,
                     'desc_alpha': init_desc_alpha,
                     'md_seen': md_seen,
                     'mi_seen': mi_seen}
        
        
        ######################################################
        ### scan down length dimension to end of alignment   #
        ######################################################
        def scan_fn(carry_dict, pos):
            ### unpack 
            # carry dict
            prev_joint_alpha = carry_dict['joint_alpha']
            prev_anc_alpha = carry_dict['anc_alpha']
            prev_desc_alpha = carry_dict['desc_alpha']
            prev_md_seen = carry_dict['md_seen']
            prev_mi_seen = carry_dict['mi_seen']
            
            # batch
            anc_toks =   aligned_inputs[:,   pos, 0]
            desc_toks =  aligned_inputs[:,   pos, 1]
            prev_state = aligned_inputs[:, pos-1, 2]
            curr_state = aligned_inputs[:,   pos, 2]
            curr_state = jnp.where( curr_state!=5, curr_state, 4 )
            
            
            ### emissions
            joint_e = self._joint_emit_scores( aligned_inputs=aligned_inputs,
                                               pos=pos,
                                               joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                               logprob_emit_at_indel=logprob_emit_at_indel )
            
            anc_mask = (curr_state == 1) | (curr_state == 3)
            anc_e = logprob_emit_at_indel[:, anc_toks - 3] * anc_mask 

            desc_mask = (curr_state == 1) | (curr_state == 2)
            desc_e = logprob_emit_at_indel[:, desc_toks - 3] * desc_mask 
            
            
            ### flags needed for transitions
            # first_emission_flag: is the current position <s> -> emit?
            # continued_emission_flag: is the current postion emit -> emit?
            # need these because gaps happen in between single sequence 
            #   emissions...
            first_anc_emission_flag = (~prev_md_seen) & anc_mask
            continued_anc_emission_flag = prev_md_seen & anc_mask
            first_desc_emission_flag = (~prev_mi_seen) & desc_mask
            continued_desc_emission_flag = (prev_mi_seen) & desc_mask
            
            
            ### transition probabilities
            def main_body(joint_carry, anc_carry, desc_carry):
                # P(anc, desc, align)
                joint_tr_per_class = joint_transit[..., prev_state-1, curr_state-1]                
                joint_out = e + logsumexp(joint_carry[:, :, None, :] + joint_tr_per_class, axis=1)
                
                # P(anc)
                anc_first_tr = marginal_transit[0,:,1,0][...,None]
                anc_cont_tr = log_dot_bigger( log_vec = anc_carry[None,...],
                                              log_mat = marginal_transit[...,0,0][None,...,None])[0,...]
                anc_tr = ( anc_cont_tr * continued_anc_emission_flag +
                           anc_first_tr * first_anc_emission_flag )
                anc_out = anc_e + anc_tr
                
                # P(desc)
                desc_first_tr = marginal_transit[0,:,1,0][...,None]
                desc_cont_tr = log_dot_bigger( log_vec = desc_carry[None,...],
                                               log_mat = marginal_transit[...,0,0][None,...,None])[0,...]
                desc_tr = ( desc_cont_tr * continued_desc_emission_flag +
                            desc_first_tr * first_desc_emission_flag )
                desc_out = desc_e + desc_tr
                
                return (joint_out, anc_out, desc_out)
            
            def end(joint_carry, anc_carry, desc_carry):
                # note for all: if end, then curr_state = -1 (<end>)
                # P(anc, desc, align)
                joint_tr_per_class = joint_transit[..., prev_state-1, -1]
                joint_out = joint_tr_per_class + joint_carry
                
                # P(anc)
                final_anc_tr = marginal_transit[:,-1,0,1]
                final_anc_tr = jnp.broadcast_to( final_anc_tr[:,None], anc_carry.shape )
                anc_out = anc_carry + final_anc_tr
                
                # P(desc)
                final_desc_tr = marginal_transit[:,-1,0,1]
                final_desc_tr = jnp.broadcast_to( final_desc_tr[:,None], desc_carry.shape )
                desc_out = desc_carry + final_desc_tr
                
                return (joint_out, anc_out, desc_out)
            
            
            ### alpha updates, in log space 
            continued_out = main_body( prev_joint_alpha, 
                                       prev_anc_alpha, 
                                       prev_desc_alpha )
            end_out = end( prev_joint_alpha, 
                           prev_anc_alpha, 
                           prev_desc_alpha )
            
            # joint: update ONLY if curr_state is not pad
            new_joint_alpha = jnp.where( curr_state != 0,
                                         jnp.where( curr_state != 4,
                                                    continued_out[0],
                                                    end_out[0] ),
                                         prev_joint_alpha )
            
            # anc marginal; update ONLY if curr_state is not pad or ins
            new_anc_alpha = jnp.where( (curr_state != 0) & (curr_state != 2),
                                         jnp.where( curr_state != 4,
                                                    continued_out[1],
                                                    end_out[1] ),
                                         prev_anc_alpha )
            
            # desc margianl; update ONLY if curr_state is not pad or del
            new_desc_alpha = jnp.where( (curr_state != 0) & (curr_state != 3),
                                        jnp.where( curr_state != 4,
                                                   continued_out[2],
                                                   end_out[2] ),
                                        prev_desc_alpha )
            
            
            out_dict = { 'joint_alpha': new_joint_alpha,
                         'anc_alpha': new_anc_alpha,
                         'desc_alpha': new_desc_alpha,
                         'md_seen': (first_anc_emission_flag + md_seen).astype(bool),
                         'mi_seen': (first_desc_emission_flag + mi_seen).astype(bool) }
            
            return (out_dict, None)
    
        ### scan over remaining length
        idx_arr = jnp.array( [i for i in range(2, L)] )
        out_dict,_ = jax.lax.scan( f = scan_fn,
                                   init = init_dict,
                                   xs = idx_arr,
                                   length = idx_arr.shape[0] )
        
        # (T,C,B) -> (T, B)
        joint_logprob_perSamp_perTime = logsumexp( out_dict['joint_alpha'], axis=1 )
        
        # (C,B) -> (B,)
        anc_neg_logP = -logsumexp( out_dict['anc_alpha'], axis=0 )
        desc_neg_logP = -logsumexp( out_dict['desc_alpha'], axis=0 )
        
        
        ### marginalize over times, where needed
        # (B,)
        if t_array.shape[0] > 1:
            joint_neg_logP = -self._marginalize_over_times(logprob_perSamp_perTime = joint_logprob_perSamp_perTime,
                                        exponential_dist_param = self.exponential_dist_param,
                                        t_array = t_array)
        else:
            joint_neg_logP = -joint_logprob_perSamp_perTime[0,:]
        
        
        ### conditional comes from joint / anc
        cond_neg_logP = - (-joint_neg_logP - -anc_neg_logP)
        
        
        ### normalize all
        anc_neg_logP_length_normed = anc_neg_logP / anc_len
        desc_neg_logP_length_normed = desc_neg_logP / desc_len
        
        if self.norm_loss_by == 'desc_len':
            joint_neg_logP_length_normed = joint_neg_logP / desc_len
            cond_neg_logP_length_normed = cond_neg_logP / desc_len
        
        elif self.norm_loss_by == 'align_len':
            joint_neg_logP_length_normed = joint_neg_logP / align_len
            cond_neg_logP_length_normed = cond_neg_logP / align_len
        
        
        out = { 'joint_neg_logP': joint_neg_logP,
                'joint_neg_logP_length_normed': joint_neg_logP_length_normed,
                'anc_neg_logP': anc_neg_logP,
                'anc_neg_logP_length_normed': anc_neg_logP_length_normed,
                'desc_neg_logP': desc_neg_logP,
                'desc_neg_logP_length_normed': desc_neg_logP_length_normed,
                'cond_neg_logP': cond_neg_logP,
                'cond_neg_logP_length_normed': cond_neg_logP_length_normed
                }
        
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
                
        
    def _get_scoring_matrices( self,
                               t_array,
                               sow_intermediates: bool):
        ### emissions from indels
        logprob_emit_at_indel = self.indel_prob_module( sow_intermediates = sow_intermediates )
        
        
        ### emissions from match sites
        # get normalized rate matrix times rate multiplier, per each class
        # (C, alph, alph)
        rate_mat_times_rho = self.rate_matrix_module( logprob_equl = logprob_emit_at_indel,
                                                      sow_intermediates = sow_intermediates )
        
        # rate_mat_times_rho: (C, alph, alph)
        # time: (T,)
        # output: (T, C, alph, alph)
        to_expm = jnp.multiply( rate_mat_times_rho[None, ...],
                                t_array[..., None,None,None] )
        cond_prob_emit_at_match = expm( to_expm )
        cond_logprob_emit_at_match = safe_log( cond_prob_emit_at_match )
        joint_logprob_emit_at_match = cond_logprob_emit_at_match + logprob_emit_at_indel[None,:,:,None]
        del to_expm, cond_logprob_emit_at_match
        
        
        ### probability of being in any particular class
        log_class_probs = self.class_logprobs_module( sow_intermediates = sow_intermediates )


        ### transition logprobs
        # (T,C,C,4,4)
        all_transit_matrices = self.transitions_module( t_array = t_array,
                                                        class_probs = jnp.exp( log_class_probs ),
                                                        sow_intermediates = sow_intermediates )
        
        out_dict = {'logprob_emit_at_indel': logprob_emit_at_indel,
                    'joint_logprob_emit_at_match': joint_logprob_emit_at_match,
                    'all_transit_matrices': all_transit_matrices}
        
        return out_dict
    
    
    def _joint_emit_scores( self,
                            aligned_inputs,
                            pos,
                            joint_logprob_emit_at_match,
                            logprob_emit_at_indel ):
        T = joint_logprob_emit_at_match.shape[0]
        B = aligned_inputs.shape[0]
        L = aligned_inputs.shape[1]
        C = self.num_site_classes
        
        # unpack
        anc_toks = aligned_inputs[:,pos,0]
        desc_toks = aligned_inputs[:,pos,1]
        curr_state = aligned_inputs[:,pos,2]
        
        # get all possible scores
        joint_emit_if_match = joint_logprob_emit_at_match[:, :, anc_toks - 3, desc_toks - 3]
        emit_if_indel_desc = logprob_emit_at_indel[:, desc_toks - 3]
        emit_if_indel_anc = logprob_emit_at_indel[:, anc_toks - 3]
        
        # stack all; (3, T, C, B)
        emit_if_indel_desc = jnp.broadcast_to(emit_if_indel_desc[None, :, :], (T, C, B))
        emit_if_indel_anc = jnp.broadcast_to(emit_if_indel_anc[None, :, :], (T, C, B))
        joint_emissions = jnp.stack([joint_emit_if_match, 
                                     emit_if_indel_desc, 
                                     emit_if_indel_anc], axis=0)

        # expand current state for take_along_axis operation
        curr_state_expanded = jnp.broadcast_to( curr_state[None, None, None, :]-1, 
                                                (1, T, C, B) )

        # gather, remove temporary leading axis
        joint_e = jnp.take_along_axis( joint_emissions, 
                                       curr_state_expanded,
                                       axis=0 )[0, ...]
        
        return joint_e
    
    
    def _marginalize_over_times(self,
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
    
    
class MarkovPairHMMLoadAll(MarkovPairHMM):
    """
    same as MarkovPairHMM, but load values (i.e. no free parameters)
    
    only replace setup and write_params (replace with placeholder function)
    
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
        self.rate_matrix_module = LG08RateMatFromFile(config = self.config,
                                                 name = f'get rate matrix')
        
        ### probability of site classes
        self.class_logprobs_module = SiteClassLogprobsFromFile(config = self.config,
                                                               name = 'class_logits')
        
        ### transitions modele
        self.transitions_module = TKF92TransitionLogprobsFromFile(config = self.config,
                                                     name = f'tkf92 indel model')
    
    def write_params(self, **kwargs):
        pass







### TODO: uncomment and update only if flax models still return nan gradients
# class WithForLoopMarkovSitesJointPairHMM(MarkovSitesJointPairHMM):
#     """
#     same as MarkovSitesJointPairHMM, but replace scan with for loop
#     """
#     config: dict
#     name: str
    
#     def __call__(self,
#                  aligned_inputs,
#                  t_array,
#                  sow_intermediates: bool):
#         T = t_array.shape[0]
#         B = aligned_inputs.shape[0]
#         L = aligned_inputs.shape[1]
#         C = self.num_site_classes
        
#         ############################
#         ### get logprob matrices   #
#         ############################
#         ### emissions from indels
#         logprob_emit_at_indel = self.indel_prob_module( sow_intermediates = sow_intermediates )
        
        
#         ### emissions from match sites
#         # this is already rho * chi * pi
#         rate_mat_times_rho = self.rate_matrix_module(logprob_equl = logprob_emit_at_indel,
#                                                      sow_intermediates = sow_intermediates)
        
#         # rate_mat_times_rho: (C, alph, alph)
#         # time: (T,)
#         # output: (T, C, alph, alph)
#         to_expm = jnp.multiply( rate_mat_times_rho[None, ...],
#                                 t_array[..., None,None,None] )
#         cond_prob_emit_at_match = expm(to_expm)
#         cond_logprob_emit_at_match = safe_log (cond_prob_emit_at_match )
#         joint_logprob_emit_at_match = cond_logprob_emit_at_match + logprob_emit_at_indel[None,:,:,None]
#         del to_expm
        
        
#         ### transition logprobs
#         # (T,C,4,4)
#         logprob_transit = self.transitions_module(t_array = t_array,
#                                                   sow_intermediates = sow_intermediates)
        
        
        
#         ### probability of being in any particular class
#         log_class_probs = self.class_logprobs_module(sow_intermediates = sow_intermediates)
        
        
#         ######################################
#         ### initialize with <start> -> any   #
#         ######################################
#         prev_state = aligned_inputs[:,0,2] # B,
#         curr_state = aligned_inputs[:,1,2] # B,
#         anc_toks =    aligned_inputs[:,1,0] # B,
#         desc_toks =   aligned_inputs[:,1,1] # B,
        
#         # for easier indexing: code <eos> as 4
#         curr_state = jnp.where( curr_state != 5, curr_state, 4)
        
        
#         ### emissions
#         e = jnp.zeros( (T, C, B,) )

#         # match
#         e = e + jnp.where( curr_state == 1,
#                            joint_logprob_emit_at_match[:,:,anc_toks-3, desc_toks-3],
#                            0 )
#         # ins (score descendant)
#         e = e + jnp.where( curr_state == 2,
#                            logprob_emit_at_indel[:,desc_toks-3],
#                            0 )
#         # del (score ancestor)
#         e = e + jnp.where( curr_state == 3,
#                            logprob_emit_at_indel[:,anc_toks-3],
#                            0 )
        
        
#         ### transitions
#         tmp = jnp.take_along_axis(arr = logprob_transit, 
#                                   indices = prev_state[None, None, :, None]-1, 
#                                   axis=2)
        
#         tr = jnp.take_along_axis( arr = tmp,
#                                   indices = curr_state[None, None, :, None]-1,
#                                   axis = 3)
#         tr = tr[...,0] + log_class_probs[None, :, None]
        
#         # init_carry = {'alpha': (tr + e),
#         #               'state': curr_state}
        
        
#         ##########################################################
#         ### for loop over length dimension to end of alignment   #
#         ##########################################################
#         # idx_arr = jnp.array( [i for i in range(2, aligned_inputs.shape[1])] )
        
#         alpha = tr + e
#         for pos in range(2, aligned_inputs.shape[1]):
#             prev_state = aligned_inputs[:,pos-1,2]
#             curr_state = aligned_inputs[:,  pos,2]
#             anc_toks =   aligned_inputs[:,  pos,0]
#             desc_toks =  aligned_inputs[:,  pos,1]
            
#             # for easier indexing: code <eos> as 4
#             curr_state = jnp.where( curr_state != 5, curr_state, 4)
            
            
#             ### emissions
#             e = jnp.zeros( (T, C, B,) )
#             e = e + jnp.where( curr_state == 1,
#                                 joint_logprob_emit_at_match[:,:,anc_toks-3, desc_toks-3],
#                                 0 )
#             e = e + jnp.where( curr_state == 2,
#                                 logprob_emit_at_indel[:,desc_toks-3],
#                                 0 )
#             e = e + jnp.where( curr_state == 3,
#                                 logprob_emit_at_indel[:,anc_toks-3],
#                                 0 )
            
            
#             ### transition probabilities
#             tmp = jnp.take_along_axis(arr = logprob_transit, 
#                                       indices = prev_state[None, None, :, None]-1, 
#                                       axis=2)
            
#             tr = jnp.take_along_axis( arr = tmp,
#                                       indices = curr_state[None, None, :, None]-1,
#                                       axis = 3)
            
#             tr = tr[...,0]
#             del tmp

#             def main_body(in_carry):
#                 # (T, C_prev, A, A), (C_curr) -> (T, C_prev, C_curr, A)
#                 tr_per_class = tr[:, :, None, :] + log_class_probs[None, None, :, None]
                
#                 # like dot product with C_prev, C_curr
#                 # output is T, C, B
#                 return e + logsumexp(in_carry[:, :, None, :] + tr_per_class, 
#                                       axis=1)
            
#             def end(in_carry):
#                 # output is T, C, B
#                 return tr + in_carry
            
#             ### alpha update, in log space ONLY if curr_state is not pad
#             alpha = jnp.where(curr_state != 0,
#                                   jnp.where( curr_state != 4,
#                                               main_body(alpha),
#                                               end(alpha) ),
#                                   alpha )
            
            
#         # T, B
#         logprob_perSamp_perTime = logsumexp(alpha, axis=1)
        
#         ### marginalize over times
#         # (B,)
#         if t_array.shape[0] > 1:
#             neg_logP = self.marginalize_over_times(logprob_perSamp_perTime = logprob_perSamp_perTime,
#                                         exponential_dist_param = self.exponential_dist_param,
#                                         t_array = t_array)
#         else:
#             neg_logP = logprob_perSamp_perTime[0,:]
        
        
#         ### normalize
#         if self.norm_loss_by == 'desc_len':
#             # don't count <bos>
#             predicate = (aligned_inputs[...,0] !=0) & (aligned_inputs[...,0] !=43)
#             length_for_normalization = predicate.sum(axis=1) - 1
        
#         elif self.norm_loss_by == 'align_len':
#             # don't count <bos>
#             length_for_normalization = (aligned_inputs[...,0] != 0).sum(axis=1) - 1
        
#         logprob_perSamp_length_normed = neg_logP / length_for_normalization
#         loss = -jnp.mean(logprob_perSamp_length_normed)
        
#         out = {'loss': loss,
#                'neg_logP': neg_logP,
#                'neg_logP_length_normed': logprob_perSamp_length_normed}
        
#         return out