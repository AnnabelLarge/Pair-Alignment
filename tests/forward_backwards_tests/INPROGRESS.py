#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 13:33:56 2025

@author: annabel
"""
import pickle
import numpy as np
from itertools import product

import numpy.testing as npt
import unittest

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax.scipy.special import logsumexp

from models.BaseClasses import ModuleBase
from models.simple_site_class_predict.model_functions import (bound_sigmoid, 
                                                              safe_log)
from tests.data_processing import (str_aligns_to_tensor,
                                   summarize_alignment)

from models.simple_site_class_predict.transition_models import TKF92TransitionLogprobs
from models.simple_site_class_predict.model_functions import (stable_tkf,
                                                              MargTKF92TransitionLogprobs,
                                                              CondTransitionLogprobs,
                                                              get_joint_loglike_emission,
                                                              joint_only_forward)

THRESHOLD = 1e-6


def log_space_dot_prod_helper(alpha,
                              marginal_logprob_transit):
    alpha_reshaped = alpha[:,None,:] #(C_prev, 1, B)
    marginal_logprob_transit_reshaped = marginal_logprob_transit[...,0,0][...,None] #(C_prev, C_curr, 1)
    to_logsumexp = alpha_reshaped + marginal_logprob_transit_reshaped
    return logsumexp(to_logsumexp, axis=0) # (C_curr, B)
    

def all_loglikes_forward(aligned_inputs,
                         logprob_emit_at_indel,
                         joint_logprob_emit_at_match,
                         all_transit_matrices):
    """
    forward algo to find joint, conditional and both single-sequence marginals
    """
    joint_logprob_transit = all_transit_matrices['joint']
    marginal_logprob_transit = all_transit_matrices['marginal'] 
    
    B = aligned_inputs.shape[0]
    L_align = aligned_inputs.shape[1]
    C = logprob_emit_at_indel.shape[0]
    
    # memory for single-sequence marginals
    anc_alpha = jnp.zeros( (C, B) ) #(C, B)
    desc_alpha = jnp.zeros( (C, B) ) #(C, B)
    md_seen = jnp.zeros( B, ).astype(bool) #(B,)
    mi_seen = jnp.zeros( B, ).astype(bool) #(B,)
    
    ######################################################
    ### initialize with <start> -> any (curr pos is 1)   #
    ######################################################
    pos = 1
    anc_toks =   aligned_inputs[:, pos, 0] #(B,)
    desc_toks =  aligned_inputs[:, pos, 1] #(B,)
    curr_state = aligned_inputs[:, pos, 2] #(B,)

    
    ### joint: P(anc, desc, align)
    # emissions; 
    e = get_joint_loglike_emission( aligned_inputs=aligned_inputs,
                                    pos=pos,
                                    joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                    logprob_emit_at_indel=logprob_emit_at_indel ) # (T, C, B)
    
    # transitions; assume there's never start -> end; 
    # joint_logprob_transit is (T, C_prev, C_curr, S_prev, S_curr)
    # initial state is 4 (<start>); take the last row
    # use C_prev=0 for start class (but it doesn't matter, because the 
    # transition probability is the same for all C_prev)
    start_any = joint_logprob_transit[:, 0, :, -1, :] # (T, C_curr, S_curr)
    tr = start_any[...,curr_state-1] # (T, C_curr, B)
    
    # carry value
    init_joint_alpha = e + tr # (T, C, B)
    del e, tr, start_any
    
    
    ### logP(anc)
    # emissions; only valid if current position is match or delete
    anc_mask = (curr_state == 1) | (curr_state == 3)  # (B,)
    init_anc_e = logprob_emit_at_indel[:, anc_toks - 3] * anc_mask  # (C, B)
    
    # transitions
    # marginal_logprob_transit is (C_prev, C_curr, S_prev, S_curr), where:
    #   (S_prev=0, S_curr=0) is emit->emit
    #   (S_prev=1, S_curr=0) is <s>->emit
    #   (S_prev=0, S_curr=1) is emit-><e>
    # use C_prev=0 for start class (but it doesn't matter, because the 
    # transition probability is the same for all C_prev)
    # transition prob for <s>->emit
    first_anc_emission_flag = (~md_seen) & anc_mask  # (B,)
    anc_first_tr = marginal_logprob_transit[0,:,1,0][...,None] #(C_curr, 1)
    
    # transition prob for emit->emit
    continued_anc_emission_flag = md_seen & anc_mask  # (B,)
    anc_cont_tr = log_space_dot_prod_helper(alpha = anc_alpha,
                                            marginal_logprob_transit = marginal_logprob_transit)  # (C_curr, B)
    
    # possibilities are: <s>->emit transition, emit->emit transition, or  
    #   nothing happened (at an indel site where ancestor was not emitted yet)
    init_anc_tr = ( anc_cont_tr * continued_anc_emission_flag +
                    anc_first_tr * first_anc_emission_flag ) # (C, B)
    
    # things to remember are:
    #   alpha: for forward algo
    #   md_seen: used to remember if <s> -> emit has been used yet
    #   (there could be gaps in between <s> and first emission)
    init_anc_alpha = init_anc_e + init_anc_tr # (C, B)
    del init_anc_e, init_anc_tr, anc_mask
    
    
    ### logP(desc); (C, B)
    # emissions; only valid if current position is match or ins
    desc_mask = (curr_state == 1) | (curr_state == 2) #(B,)
    init_desc_e = logprob_emit_at_indel[:, desc_toks - 3] * desc_mask # (C, B)
    
    # transitions
    first_desc_emission_flag = (~mi_seen) & desc_mask # (B,)
    desc_first_tr = marginal_logprob_transit[0,:,1,0][...,None] #(C_curr, 1)
    
    continued_desc_emission_flag = mi_seen & desc_mask # (B,)
    desc_cont_tr = log_space_dot_prod_helper(alpha = desc_alpha,
                                             marginal_logprob_transit = marginal_logprob_transit)  # (C_curr, B)
    
    init_desc_tr = ( desc_cont_tr * continued_desc_emission_flag +
                     desc_first_tr * first_desc_emission_flag ) # (C, B)
    
    # things to remember are:
    #   alpha: for forward algo
    #   mi_seen: used to remember if <s> -> emit has been used yet
    #   (there could be gaps in between <s> and first emission)
    init_desc_alpha = init_desc_e + init_desc_tr  # (C, B)
    del init_desc_e, init_desc_tr, desc_mask, curr_state
    
    init_dict = {'joint_alpha': init_joint_alpha, # (T, C, B)
                 'anc_alpha': init_anc_alpha,  # (C, B)
                 'desc_alpha': init_desc_alpha,  # (C, B),
                 'md_seen': first_anc_emission_flag, # (B,)
                 'mi_seen': first_desc_emission_flag} # (B,)
    
    
    ######################################################
    ### scan down length dimension to end of alignment   #
    ######################################################
    def scan_fn(carry_dict, pos):
        ### unpack 
        # carry dict
        prev_joint_alpha = carry_dict['joint_alpha'] #(T, C, B)
        prev_anc_alpha = carry_dict['anc_alpha'] #(C, B)
        prev_desc_alpha = carry_dict['desc_alpha'] #(C, B)
        prev_md_seen = carry_dict['md_seen'] #(B,)
        prev_mi_seen = carry_dict['mi_seen'] #(B,)
        
        # batch
        anc_toks =   aligned_inputs[:,   pos, 0] #(B,)
        desc_toks =  aligned_inputs[:,   pos, 1] #(B,)
        prev_state = aligned_inputs[:, pos-1, 2] #(B,)
        curr_state = aligned_inputs[:,   pos, 2] #(B,)
        curr_state = jnp.where( curr_state!=5, curr_state, 4 ) #(B,)
        
        
        ### emissions
        joint_e = get_joint_loglike_emission( aligned_inputs=aligned_inputs,
                                              pos=pos,
                                              joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                              logprob_emit_at_indel=logprob_emit_at_indel ) #(T, C, B)
        
        anc_mask = (curr_state == 1) | (curr_state == 3) #(B,)
        anc_e = logprob_emit_at_indel[:, anc_toks - 3] * anc_mask  #(C,B)

        desc_mask = (curr_state == 1) | (curr_state == 2)  #(B,)
        desc_e = logprob_emit_at_indel[:, desc_toks - 3] * desc_mask #(C,B)
        
        
        ### flags needed for transitions
        # first_emission_flag: is the current position <s> -> emit?
        # continued_emission_flag: is the current postion emit -> emit?
        # need these because gaps happen in between single sequence 
        #   emissions...
        first_anc_emission_flag = (~prev_md_seen) & anc_mask  #(B,)
        continued_anc_emission_flag = prev_md_seen & anc_mask  #(B,)
        first_desc_emission_flag = (~prev_mi_seen) & desc_mask  #(B,)
        continued_desc_emission_flag = (prev_mi_seen) & desc_mask  #(B,)
        
        ### transition probabilities
        def main_body(joint_carry, anc_carry, desc_carry):
            # logP(anc, desc, align)
            joint_tr_per_class = joint_logprob_transit[..., prev_state-1, curr_state-1] #(T, C_prev, C_curr, B)         
            joint_out = joint_e + logsumexp(joint_carry[:, :, None, :] + joint_tr_per_class, axis=1) #(T, C, B)
            
            # logP(anc)
            anc_first_tr = marginal_logprob_transit[0,:,1,0][...,None] #(C_curr, 1)
            anc_cont_tr = log_space_dot_prod_helper(alpha = anc_carry,
                                                    marginal_logprob_transit = marginal_logprob_transit) #(C_curr, B)
            anc_tr = ( anc_cont_tr * continued_anc_emission_flag +
                       anc_first_tr * first_anc_emission_flag ) # (C_curr, B)
            anc_out = anc_e + anc_tr # (C, B)
            
            # logP(desc)
            desc_first_tr = marginal_logprob_transit[0,:,1,0][...,None] #(C_curr, 1)
            desc_cont_tr = log_space_dot_prod_helper(alpha = desc_carry,
                                                    marginal_logprob_transit = marginal_logprob_transit) #(C_curr, B)
            desc_tr = ( desc_cont_tr * continued_desc_emission_flag +
                        desc_first_tr * first_desc_emission_flag ) # (C_curr, B)
            desc_out = desc_e + desc_tr # (C, B)
            
            return (joint_out, anc_out, desc_out)
        
        def end(joint_carry, anc_carry, desc_carry):
            # note for all: if end, then curr_state = -1 (<end>)
            # logP(anc, desc, align)
            joint_tr_per_class = joint_logprob_transit[..., -1, prev_state-1, -1] #(T,C,B)
            joint_out = joint_tr_per_class + joint_carry #(T,C,B)
            
            # logP(anc)
            final_anc_tr = marginal_logprob_transit[:,-1,0,1] #(C,)
            final_anc_tr = jnp.broadcast_to( final_anc_tr[:,None], anc_carry.shape ) #(C, B)
            anc_out = anc_carry + final_anc_tr #(C, B)
            
            # logP(desc)
            final_desc_tr = marginal_logprob_transit[:,-1,0,1] #(C,)
            final_desc_tr = jnp.broadcast_to( final_desc_tr[:,None], desc_carry.shape ) #(C,B)
            desc_out = desc_carry + final_desc_tr #(C,B)
            
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
    
    return out_dict


# fake inputs
fake_aligns = [ ('AC-A','D-ED'),
                ('D-ED','AC-A'),
                ('ECDAD','-C-A-'),
                ('-C-A-','ECDAD'),
                ('-C-A-','ECDAD') ]

fake_aligns =  str_aligns_to_tensor(fake_aligns) #(B, L, 3)

# fake params
rngkey = jax.random.key(42) # note: reusing this rngkey over and over
t_array = jnp.array([0.3, 0.5, 0.7, 0.9])
C = 3
A = 20
lam = jnp.array(0.3)
mu = jnp.array(0.5)
offset = 1 - (lam/mu)
r = nn.sigmoid( jax.random.randint(key=rngkey, 
                                   shape=(C,), 
                                   minval=-3.0, 
                                   maxval=3.0).astype(float) )
class_probs = nn.softmax( jax.random.randint(key=rngkey, 
                                             shape=(C,), 
                                             minval=1.0, 
                                             maxval=10.0).astype(float) )

# other dims (blt; yum)
B = fake_aligns.shape[0]
L_align = fake_aligns.shape[1]
T = t_array.shape[0]
        
# fake scoring matrices (not coherently normalized; just some example values)
def generate_fake_scoring_mat(dim_tuple):
    logits = jax.random.uniform(key=rngkey, 
                                shape=dim_tuple,
                                minval=-10.0,
                                maxval=-1e-4)
    return nn.log_softmax(logits, axis=-1)

joint_logprob_emit_at_match = generate_fake_scoring_mat( (T,C,A,A) )
logprob_emit_at_indel = generate_fake_scoring_mat( (C,A) )

# be more careful about generating a fake transition matrix
my_tkf_params, _ = stable_tkf(mu = mu, 
                              offset = offset,
                              t_array = t_array)
my_tkf_params['log_offset'] = jnp.log(offset)
my_tkf_params['log_one_minus_offset'] = jnp.log1p(-offset)
my_model = TKF92TransitionLogprobs(config={'num_tkf_site_classes': C},
                                   name='tkf92')
fake_params = my_model.init(rngs=jax.random.key(0),
                            t_array = t_array,
                            class_probs = class_probs,
                            sow_intermediates = False)

joint_logprob_transit =  my_model.apply(variables = fake_params,
                                        out_dict = my_tkf_params,
                                        r_extend = r,
                                        class_probs = class_probs,
                                        method = 'fill_joint_tkf92') #(T, C, C, 4, 4)

marg_logprob_transit = MargTKF92TransitionLogprobs( offset = offset,
                                                    class_probs = class_probs,
                                                    r_ext_prob = r )

del my_tkf_params, my_model, fake_params
    
    
### pred
pred_out = all_loglikes_forward( aligned_inputs = fake_aligns,
                                 logprob_emit_at_indel = logprob_emit_at_indel,
                                 joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                 all_transit_matrices = {'joint': joint_logprob_transit,
                                                         'marginal': marg_logprob_transit} )

pred_joint = logsumexp( pred_out['joint_alpha'], axis=1 ) #(T,B)
pred_anc_marg = logsumexp( pred_out['anc_alpha'], axis=0 ) #(B,)
pred_desc_marg = logsumexp( pred_out['desc_alpha'], axis=0 ) #(B,)

### test 1: check against (already verified) forward joint function
tmp = joint_only_forward( aligned_inputs = fake_aligns,
                          joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                          logprob_emit_at_indel = logprob_emit_at_indel,
                          joint_logprob_transit = joint_logprob_transit ) #(L_align, T, C, B)

# (L_align, T, C, B) -> (T, B)
true_joint = logsumexp(tmp[-1,...], axis=1)

print( jnp.allclose(pred_joint, true_joint) )


### test 2: manually enumerate ugh
batch = fake_aligns[...,0] # ancestor for now
pred = pred_anc_marg

for b in range(B):
    invalid_toks = jnp.array([0,1,2,43])
    sample_gapped_seq = batch[b,:]
    sample_seq = sample_gapped_seq[~jnp.isin(sample_gapped_seq, invalid_toks)]
    n = (  ~jnp.isin(sample_gapped_seq, invalid_toks) ).sum()
    paths = [list(p) for p in product(range(C), repeat= int(n) )]
    del sample_gapped_seq

    # manually score each possible path
    score_per_path = []
    
    for path in paths:
        path_logprob = 0

        ### first start -> emit
        l = 0
        curr_site_class = path[0]
        seq_tok = sample_seq[0]
        
        e = logprob_emit_at_indel[curr_site_class, seq_tok-3]
        tr = marg_logprob_transit[0, curr_site_class, 1, 0]
        path_logprob += (tr + e)
        del l, curr_site_class, seq_tok, e, tr
        
        
        ### all emitted sequences
        for l in range(1, sample_seq.shape[0]):
            prev_site_class = path[l-1]
            curr_site_class = path[l]
            seq_tok = sample_seq[l]
            
            e = logprob_emit_at_indel[curr_site_class, seq_tok-3]
            tr = marg_logprob_transit[prev_site_class, curr_site_class, 0, 0]
            path_logprob += (tr + e)
        
        
        ### ending
        last_site_class = path[-1]
        path_logprob += marg_logprob_transit[last_site_class, -1, 0, 1]
        
        score_per_path.append(path_logprob)
        
    true = logsumexp( jnp.array(score_per_path) )
    npt.assert_allclose(pred[b], true, rtol=THRESHOLD)
