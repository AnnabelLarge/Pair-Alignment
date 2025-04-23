#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 04:33:00 2025

@author: annabel

models:
=======
MarkovPairHMM
MarkovPairHMMLoadAll
MarkovHKY85PairHMM
MarkovHKY85PairHMMLoadAll

"""
import pickle
import numpy as np

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

def _expand_dims_like(x, target):
    return x.reshape(list(x.shape) + [1] * (target.ndim - x.ndim))

def flip_sequences( inputs, 
                    seq_lengths, 
                    flip_along_axis,
                    num_features_dims = None
                   ):
    """
    this is taken from flax RNN
    
    https://flax.readthedocs.io/en/v0.6.10/_modules/flax/linen/recurrent.html \
        #RNN:~:text=.ndim))-,def%20flip_sequences(,-inputs%3A%20Array
    """
    # Compute the indices to put the inputs in flipped order as per above example.
    max_steps = inputs.shape[flip_along_axis]
    
    if seq_lengths is None:
        # reverse inputs and return
        inputs = jnp.flip(inputs, axis=flip_along_axis)
        return inputs
    
    seq_lengths = jnp.expand_dims(seq_lengths, axis=flip_along_axis)
    
    # create indexes
    idxs = jnp.arange(max_steps - 1, -1, -1) # [max_steps]
    
    if flip_along_axis == 0:
        num_batch_dims = len( inputs.shape[1:-num_features_dims] )
        idxs = jnp.reshape(idxs, [max_steps] + [1] * num_batch_dims)
    elif flip_along_axis > 0:
        num_batch_dims = len( inputs.shape[:flip_along_axis] )
        idxs = jnp.reshape(idxs, [1] * num_batch_dims + [max_steps])
    
    idxs = (idxs + seq_lengths) % max_steps # [*batch, max_steps]
    idxs = _expand_dims_like(idxs, target=inputs) # [*batch, max_steps, *features]
    # Select the inputs in flipped order.
    outputs = jnp.take_along_axis(inputs, idxs, axis=flip_along_axis)
    
    return outputs


    
class MarkovPairHMM(ModuleBase):
    """
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
        self.gap_tok = self.config['gap_tok']
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        
        ### how to score emissions from indel sites
        if self.num_site_classes == 1:
            self.indel_prob_module = LogEqulVecFromCounts(config = self.config,
                                                       name = f'get equilibrium')
        elif self.num_site_classes > 1:
            self.indel_prob_module = LogEqulVecPerClass(config = self.config,
                                                     name = f'get equilibrium')
        
        
        ### probability of site classes
        self.site_class_probability_module = SiteClassLogprobs(config = self.config,
                                                       name = 'class_logits')
    
    
        ### rate matrix to score emissions from match sites
        # init with values from LG08
        out = self._init_rate_matrix_module(self.config)
        self.rate_matrix_module, self.subst_model_type = out
        del out
        
        
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
        used_tkf_beta_approx = out['used_tkf_beta_approx']
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
        start_any = joint_transit[:, 0, :, -1, :]
        init_trans_logprob = start_any[...,curr_state-1]
        
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
                tr_per_class = joint_transit[..., -1, prev_state-1, -1]
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
            banned_toks = jnp.array([0,1,2,self.gap_tok])
            
        elif self.norm_loss_by == 'align_len':
            # where descendant is not pad (but could be gap)
            banned_toks = jnp.array([0,1,2])
        
        mask = ~jnp.isin( aligned_inputs[...,1], banned_toks)
        length_for_normalization = mask.sum(axis=1)
        del mask
        
        joint_neg_logP_length_normed = joint_neg_logP / length_for_normalization
        loss = jnp.mean(joint_neg_logP_length_normed)
        
        aux_dict = {'joint_neg_logP': joint_neg_logP,
                    'joint_neg_logP_length_normed': joint_neg_logP_length_normed,
                    'used_tkf_beta_approx': used_tkf_beta_approx}
        
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
        B = aligned_inputs.shape[0]
        L_align = aligned_inputs.shape[1]
        
        # get lengths, not including <bos> and <eos>
        align_len = ~jnp.isin( aligned_inputs[...,0], jnp.array([0,1,2]) )
        anc_len = ~jnp.isin( aligned_inputs[...,0], jnp.array([0,1,2,self.gap_tok]) )
        desc_len = ~jnp.isin( aligned_inputs[...,1], jnp.array([0,1,2,self.gap_tok]) )
        align_len = align_len.sum(axis=1)
        anc_len = anc_len.sum(axis=1)
        desc_len = desc_len.sum(axis=1)
        
        # get score matrices
        out = self._get_scoring_matrices( t_array=t_array,
                                          sow_intermediates=sow_intermediates )
        
        logprob_emit_at_indel = out['logprob_emit_at_indel']
        joint_logprob_emit_at_match = out['joint_logprob_emit_at_match']
        joint_transit = out['all_transit_matrices']['joint']
        marginal_transit = out['all_transit_matrices']['marginal'] 
        used_tkf_beta_approx = out['used_tkf_beta_approx']
        del out
        
        C = logprob_emit_at_indel.shape[0]
        
        anc_alpha = jnp.zeros( (C, B) )
        desc_alpha = jnp.zeros( (C, B) )
        md_seen = jnp.zeros( B, ).astype(bool)
        mi_seen = jnp.zeros( B, ).astype(bool)
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
        start_any = joint_transit[:, 0, :, -1, :]
        init_joint_tr = start_any[...,curr_state-1]
        
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
        # init_anc_tr = marginal_transit[0,:,1,0][...,None] * anc_mask
        first_anc_emission_flag = (~md_seen) & anc_mask
        anc_first_tr = marginal_transit[0,:,1,0][...,None]
        
        continued_anc_emission_flag = md_seen & anc_mask
        anc_cont_tr = log_dot_bigger( log_vec = anc_alpha[None,...],
                                      log_mat = marginal_transit[...,0,0][None,...,None])[0,...]
        init_anc_tr = ( anc_cont_tr * continued_anc_emission_flag +
                        anc_first_tr * first_anc_emission_flag )
        
        
        # things to remember are:
        #   alpha: for forward algo
        #   md_seen: used to remember if <s> -> emit has been used yet
        #   (there could be gaps in between <s> and first emission)
        init_anc_alpha = init_anc_e + init_anc_tr
        del init_anc_e, init_anc_tr, anc_mask
        
        
        ### P(desc); (C, B)
        # emissions; only valid if current position is match or ins
        desc_mask = (curr_state == 1) | (curr_state == 2)
        init_desc_e = logprob_emit_at_indel[:, desc_toks - 3] * desc_mask 
        
        # transitions
        # init_desc_tr = marginal_transit[0,:,1,0][...,None] * desc_mask
        first_desc_emission_flag = (~mi_seen) & desc_mask
        desc_first_tr = marginal_transit[0,:,1,0][...,None]
        
        continued_desc_emission_flag = mi_seen & desc_mask
        desc_cont_tr = log_dot_bigger( log_vec = desc_alpha[None,...],
                                       log_mat = marginal_transit[...,0,0][None,...,None])[0,...]
        init_desc_tr = ( desc_cont_tr * continued_desc_emission_flag +
                         desc_first_tr * first_desc_emission_flag )
        
        # things to remember are:
        #   alpha: for forward algo
        #   mi_seen: used to remember if <s> -> emit has been used yet
        #   (there could be gaps in between <s> and first emission)
        init_desc_alpha = init_desc_e + init_desc_tr
        del init_desc_e, init_desc_tr, desc_mask, curr_state
        
        init_dict = {'joint_alpha': init_joint_alpha,
                     'anc_alpha': init_anc_alpha,
                     'desc_alpha': init_desc_alpha,
                     'md_seen': first_anc_emission_flag,
                     'mi_seen': first_desc_emission_flag}
        
        
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
                joint_out = joint_e + logsumexp(joint_carry[:, :, None, :] + joint_tr_per_class, axis=1)
                
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
                joint_tr_per_class = joint_transit[..., -1, prev_state-1, -1]
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
                         'md_seen': (first_anc_emission_flag + prev_md_seen).astype(bool),
                         'mi_seen': (first_desc_emission_flag + prev_mi_seen).astype(bool) }
            
            return (out_dict, None)
    
        ### scan over remaining length
        idx_arr = jnp.array( [i for i in range(2, L_align)] )
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
                'cond_neg_logP_length_normed': cond_neg_logP_length_normed,
                'used_tkf_beta_approx': used_tkf_beta_approx
                }
        
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
        
        rate_mat_times_rho = out['rate_mat_times_rho']
        for c in range(rate_mat_times_rho.shape[0]):
            mat_to_save = rate_mat_times_rho[c,...]
            
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
            
        for key, mat in out['all_transit_matrices'].items():
            mat = np.exp(mat)
            new_key = key.replace('logprob','prob')
            
            with open(f'{out_folder}/{new_key}_transit_matrix.npy', 'wb') as g:
                np.save(g, mat)
            
            mat = np.squeeze(mat)
            if len(mat.shape) <= 2:
                np.savetxt( f'{out_folder}/ASCII_{new_key}_transit_matrix.tsv', 
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
                
                
        ### transitions
        # always write lambda and mu
        if 'tkf_lam_mu_logits' in dir(self.transitions_module):
            lam_min_val = self.transitions_module.lam_min_val
            lam_max_val = self.transitions_module.lam_max_val
            offs_min_val = self.transitions_module.offs_min_val
            offs_max_val = self.transitions_module.offs_max_val
            lam_mu_logits = self.transitions_module.tkf_lam_mu_logits
        
            lam = bounded_sigmoid(x = lam_mu_logits[0],
                                  min_val = lam_min_val,
                                  max_val = lam_max_val)
            
            offset = bounded_sigmoid(x = lam_mu_logits[1],
                                     min_val = offs_min_val,
                                     max_val = offs_max_val)
            mu = lam / ( 1 -  offset) 
            
            r_extend_min_val = self.transitions_module.r_extend_min_val
            r_extend_max_val = self.transitions_module.r_extend_max_val
            r_extend_logits = self.transitions_module.r_extend_logits
            
            r_extend = bounded_sigmoid(x = r_extend_logits,
                                       min_val = r_extend_min_val,
                                       max_val = r_extend_max_val)
            
            mean_indel_lengths = 1 / (1 - r_extend)
            
            with open(f'{out_folder}/PARAMS_tkf92_indel_params.txt','w') as g:
                g.write(f'insert rate, lambda: {lam}\n')
                g.write(f'deletion rate, mu: {mu}\n')
                g.write(f'used tkf beta approximation? {out["used_tkf_beta_approx"]}\n\n')
                g.write(f'extension prob, r: ')
                [g.write(f'{elem}\t') for elem in r_extend]
                g.write('\n')
                g.write(f'mean indel length: ')
                [g.write(f'{elem}\t') for elem in mean_indel_lengths]
                g.write('\n')
                
    
    def forward_with_interms( self,
                              aligned_inputs,
                              logprob_emit_at_indel,
                              joint_logprob_emit_at_match,
                              joint_transit ):
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
        start_any = joint_transit[:, 0, :, -1, :]
        init_trans_logprob = start_any[...,curr_state-1]
        
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
                tr_per_class = joint_transit[..., -1, prev_state-1, -1]
                return tr_per_class + in_carry
            
            
            ### alpha update, in log space ONLY if curr_state is not pad
            new_alpha = jnp.where(curr_state != 0,
                                  jnp.where( curr_state != 4,
                                              main_body(prev_alpha),
                                              end(prev_alpha) ),
                                  prev_alpha )
            
            return (new_alpha, new_alpha)
        
        ### end scan function definition, use scan
        idx_arr = jnp.array( [ i for i in range(2, L_align) ] )
        _, stacked_outputs = jax.lax.scan( f = scan_fn,
                                           init = init_alpha,
                                           xs = idx_arr,
                                           length = idx_arr.shape[0] )
        
        # stacked_outputs is cumulative sum PER POSITION, PER TIME
        # append the first return value (from sentinel -> first alignment column)
        stacked_outputs = jnp.concatenate( [ init_alpha[None,...],
                                             stacked_outputs ],
                                          axis=0)
        
        # transpose to (T, C, B, L-1) for convenience
        stacked_outputs = jnp.transpose( stacked_outputs, (1,2,3,0) )
        
        return stacked_outputs
    
    
    def backward_with_interms( self,
                               aligned_inputs,
                               logprob_emit_at_indel,
                               joint_logprob_emit_at_match,
                               joint_transit ):
        ######################################
        ### flip inputs, transition matrix   #
        ######################################
        align_len = (aligned_inputs[...,-1] != 0).sum(axis=1)
        flipped_seqs = flip_sequences( inputs = aligned_inputs, 
                                       seq_lengths = align_len, 
                                       flip_along_axis = 1
                                       )
        B = flipped_seqs.shape[0]
        
        # transits needs to be flipped from (T, C_from, C_to, S_from, S_to)
        #   to (T, C_to, C_from, S_to, S_from)
        flipped_transit = jnp.transpose(joint_transit, (0, 2, 1, 4, 3) )
        
        
        ### init with <sentinel> -> any, but don't count transition found at any,
        ###   since this is already taken care of by forward alignment
        # these are all (B,)
        future_state = flipped_seqs[:,1,2]
        
        # transits is (T, C_prev, C_curr, S_prev, S_curr)
        # under forward: prev=from, curr=to
        # under backwards: prev=to, curr=from
        sentinel_to_any = flipped_transit[:, 0, :, -1, :]
        init_alpha = sentinel_to_any[:,:,future_state-1]
        del future_state, sentinel_to_any
        
        
        def scan_fn(prev_alpha, index):
            curr_state = flipped_seqs[:,index,2]
            future_state = flipped_seqs[:,index-1,2] 
            
            #################
            ### emissions   #
            #################
            e = self._joint_emit_scores( aligned_inputs = flipped_seqs,
                                         pos = index-1,
                                         joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                         logprob_emit_at_indel = logprob_emit_at_indel )
            
            ###################
            ### transitions   #
            ###################
            def main_body(in_carry):
                # (T, C_future, C_curr, B)
                tr = flipped_transit[:,:,:, future_state-1, curr_state-1]
                
                # add emission at C_future to transition C_curr -> C_future
                for_log_dot = e[:,:, None, :] + tr
                
                # return dot product with previous carry
                return log_dot_bigger(log_vec = in_carry, log_mat = for_log_dot)
            
            def begin(in_carry):
                # e is always associated with prev_state, so still add here
                any_to_sentinel = flipped_transit[:, :, -1, :, -1]
                final_tr = any_to_sentinel[:,:,future_state-1]
                return e + in_carry + final_tr
            
            ### alpha update, in log space ONLY if curr_state is not pad
            new_alpha = jnp.where(curr_state != 0,
                                  jnp.where( curr_state != 4,
                                             main_body(prev_alpha),
                                             begin(prev_alpha) ),
                                  prev_alpha )
            
            return (new_alpha, new_alpha)
        
        idx_arr = jnp.array( [i for i in range(2, L)] )
        out = jax.lax.scan( f = scan_fn,
                            init = init_alpha,
                            xs = idx_arr,
                            length = idx_arr.shape[0] )
        _, stacked_outputs = out
        del out
        
        
        # stacked_outputs is cumulative sum PER POSITION
        # append the first return value (from sentinel -> last alignment column)
        stacked_outputs = jnp.concatenate( [ init_alpha[None,...],
                                             stacked_outputs ],
                                          axis=0)
        
        ### swap the sequence back; final output should always be (T, C, B, L-1)
        # reshape: (L-1, T, C, B) -> (B, L-1, T, C)
        reshaped_stacked_outputs = jnp.transpose( stacked_outputs,
                                                  (3, 0, 1, 2) )
        flipped_stacked_outputs = flip_sequences( inputs = reshaped_stacked_outputs, 
                                                  seq_lengths = align_len-1, 
                                                  flip_along_axis = 1
                                                  )
        # reshape back to (T, C, B, L-1)
        stacked_outputs = jnp.transpose( flipped_stacked_outputs,
                                          (2, 3, 0, 1) )
        
        return stacked_outputs
        
    
    def get_class_posterior_marginals(self,
                                      aligned_inputs,
                                      t_array):
        """
        Label P(C | anc, desc, align, t) post-hoc using the 
          forward-backard algorithm
        
        ASSUMES pad is 0, bos is 1, and eos is 2
        
        returns:
            - posterior_marginals
        
        
        extra notes:
        ------------
        posterior_marginals will be of shape: (T, C, B, L-2)

        posterior_marginals[...,0] corresponds to the marginals at the FIRST valid 
          alignment column (right after <bos>)

        increases from there, and the posterior marginal at <eos> should be all zeros 
          (because it's an invalid value)
        """
        T = time.shape[0]
        B = aligned_inputs.shape[0]
        L = aligned_inputs.shape[1]
        
        out = self._get_scoring_matrices( t_array=t_array,
                                          sow_intermediates=False )
        
        logprob_emit_at_indel = out['logprob_emit_at_indel']
        joint_logprob_emit_at_match = out['joint_logprob_emit_at_match']
        joint_transit = out['all_transit_matrices']['joint']
        used_tkf_beta_approx = out['used_tkf_beta_approx']
        del out
        
        C = joint_logprob_emit_at_match.shape[0]
        
        
        ### each are: (T, C, B, L-1) 
        forward_stacked_outputs = self.forward_with_interms( aligned_inputs,
                                                             logprob_emit_at_indel,
                                                             joint_logprob_emit_at_match,
                                                             joint_transit )
        
        backward_stacked_outputs = self.backward_with_interms( aligned_inputs,
                                                               logprob_emit_at_indel,
                                                               joint_logprob_emit_at_match,
                                                               joint_transit )
        
        # can get the denominator from backwards outputs before masking
        joint_logprob = logsumexp(backward_stacked_outputs[..., 0], axis=1)
       
        
        ### mask and combine 
        invalid_tok_mask = ~jnp.isin(aligned_inputs[...,0], jnp.array([0,1,2]))
        
        # mask out padding tokens and final value at <eos>
        # reduce to (T, C, B, L-2) 
        forward_pad = invalid_tok_mask[:, 1:]
        forward_pad = jnp.broadcast_to( forward_pad[None,None,...], (T, C, B, L-1) )
        forward_stacked_outputs = forward_stacked_outputs * forward_pad
        
        # mask out padding tokens and "final" value at <bos>
        # reduce to (T, C, B, L-2) 
        backwards_pad = invalid_tok_mask[:, :-1]
        backwards_pad = jnp.broadcast_to( backwards_pad[None,None,...], (T, C, B, L-1) )
        backward_stacked_outputs = backward_stacked_outputs * backwards_pad
        
        # combine; wherever there's padding and <eos>, the sum should be zero
        for_times_back = forward_logprobs_per_pos[..., :-1] + backward_logprobs_per_pos[..., 1:]
        
        invalid_pos = (( forward_logprobs_per_pos[..., :-1] == 0 ) &
                       ( backward_logprobs_per_pos[..., :-1] == 0 ) &
                       ( for_times_back == 0 ) )
        
        posterior_log_marginals = jnp.where( ~invalid_pos, 
                                             for_times_back - joint_logprob[:, None, :, None],
                                             0 )
        
        return posterior_log_marginals
        
    
    def _init_rate_matrix_module(self, config):
        mod = RateMatFitBoth( config = self.config,
                               name = f'get rate matrix' )
        return mod, 'GTR'
    
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
        
        
        ### probability of being in any particular class
        log_class_probs = self.site_class_probability_module( sow_intermediates = sow_intermediates )


        ### transition logprobs
        # (T,C,C,4,4)
        all_transit_matrices, used_tkf_beta_approx = self.transitions_module( t_array = t_array,
                                                        class_probs = jnp.exp( log_class_probs ),
                                                        sow_intermediates = sow_intermediates )
        
        out_dict = {'logprob_emit_at_indel': logprob_emit_at_indel,
                    'joint_logprob_emit_at_match': joint_logprob_emit_at_match,
                    'cond_logprob_emit_at_match': cond_logprob_emit_at_match,
                    'rate_mat_times_rho': rate_mat_times_rho,
                    'to_expm': to_expm,
                    'all_transit_matrices': all_transit_matrices,
                    'used_tkf_beta_approx': used_tkf_beta_approx}
        
        return out_dict
    
    
    def _joint_emit_scores( self,
                            aligned_inputs,
                            pos,
                            joint_logprob_emit_at_match,
                            logprob_emit_at_indel ):
        T = joint_logprob_emit_at_match.shape[0]
        B = aligned_inputs.shape[0]
        C = self.num_site_classes
        
        # unpack
        anc_toks = aligned_inputs[:,pos,0]
        desc_toks = aligned_inputs[:,pos,1]
        state_at_pos = aligned_inputs[:,pos,2]
        
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
        state_at_pos_expanded = jnp.broadcast_to( state_at_pos[None, None, None, :]-1, 
                                                  (1, T, C, B) )

        # gather, remove temporary leading axis
        joint_e = jnp.take_along_axis( joint_emissions, 
                                       state_at_pos_expanded,
                                       axis=0 )[0, ...]
        
        return joint_e
    
    
    def _marginalize_over_times(self,
                               logprob_perSamp_perTime,
                               exponential_dist_param,
                               t_array):
        # logP(t_k) = exponential distribution
        logP_time = ( jnp.log(exponential_dist_param) - 
                      (exponential_dist_param * t_array) )
        log_t_grid = jnp.log( t_array[1:] - t_array[:-1] )
        
        # kind of a hack, but repeat the last time array value
        log_t_grid = jnp.concatenate( [log_t_grid, log_t_grid[-1][None] ], axis=0)
        
        logP_perSamp_perTime_withConst = ( logprob_perSamp_perTime +
                                           logP_time[:,None] +
                                           log_t_grid[:,None] )
        
        logP_perSamp_raw = logsumexp(logP_perSamp_perTime_withConst, axis=0)
        
        return logP_perSamp_raw
    
    
    def _return_bound_sigmoid_limits(self):
        ### rate_matrix_module
        # exchangeabilities
        exchange_min_val = self.rate_matrix_module.exchange_min_val
        exchange_max_val = self.rate_matrix_module.exchange_max_val
        
        #rate multiplier
        if self.rate_matrix_module.rate_mult_activation == 'bound_sigmoid':
            rate_mult_min_val = self.rate_matrix_module.rate_mult_min_val
            rate_mult_max_val = self.rate_matrix_module.rate_mult_max_val
        
        
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
        
        params_range = { "exchange_min_val": exchange_min_val,
                         "exchange_max_val": exchange_max_val,
                         "rate_mult_min_val": rate_mult_min_val,
                         "rate_mult_max_val": rate_mult_max_val,
                         "lam_min_val": lam_min_val,
                         "lam_max_val": lam_max_val,
                         "offs_min_val": offs_min_val,
                         "offs_max_val": offs_max_val,
                         "r_extend_min_val": r_extend_min_val,
                         "r_extend_max_val": r_extend_max_val,
                         }
        
        return params_range
    

###############################################################################
### Variants   ################################################################
###############################################################################
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
        self.gap_tok = self.config['gap_tok']
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
                                                               name = 'class_logits')
        
        ### transitions modele
        self.transitions_module = TKF92TransitionLogprobsFromFile(config = self.config,
                                                     name = f'tkf92 indel model')
    
    def write_params(self, **kwargs):
        ##########################
        ### the final matrices   #
        ##########################                
        out = self._get_scoring_matrices(t_array=t_array,
                                        sow_intermediates=False)
        
        rate_mat_times_rho = out['rate_mat_times_rho']
        for c in range(rate_mat_times_rho.shape[0]):
            mat_to_save = rate_mat_times_rho[c,...]
            
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
            
        for key, mat in out['all_transit_matrices'].items():
            mat = np.exp(mat)
            new_key = key.replace('logprob','prob')
            
            with open(f'{out_folder}/{new_key}_transit_matrix.npy', 'wb') as g:
                np.save(g, mat)
            
            mat = np.squeeze(mat)
            if len(mat.shape) <= 2:
                np.savetxt( f'{out_folder}/ASCII_{new_key}_transit_matrix.tsv', 
                            np.array(mat), 
                            fmt = '%.4f',
                            delimiter= '\t' )
            
            del key, mat, g
    
    def _init_rate_matrix_module(self, config):
        mod = RateMatFromFile( config = self.config,
                               name = f'get rate matrix' )
        return mod, 'GTR'



class MarkovHKY85PairHMM(MarkovPairHMM):
    """
    Identical to MarkovPairHMM, but uses the HKY85 substitution model.
    """
    config: dict
    name: str

    def _init_rate_matrix_module(self, config):
        mod = HKY85( config = self.config,
                               name = f'get rate matrix' )
        return mod, 'HKY85'

class MarkovHKY85PairHMMLoadAll(MarkovPairHMMLoadAll):
    """
    Identical to MarkovPairHMMLoadAll, but uses the HKY85 substitution model.
    """
    config: dict
    name: str

    def _init_rate_matrix_module(self, config):
        mod = HKY85FromFile( config = self.config,
                               name = f'get rate matrix' )
        return mod, 'HKY85'
