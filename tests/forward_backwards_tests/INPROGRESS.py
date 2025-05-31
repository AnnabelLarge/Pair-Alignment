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
                                                              joint_only_forward)

THRESHOLD = 1e-6

def _expand_dims_like(x, target):
    """
    this is taken from flax RNN
    
    https://flax.readthedocs.io/en/v0.6.10/_modules/flax/linen/recurrent.html \
        #RNN:~:text=.ndim))-,def%20flip_sequences(,-inputs%3A%20Array
    """
    return x.reshape(list(x.shape) + [1] * (target.ndim - x.ndim))

def _flip_sequences( inputs, 
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

def backwards_joint_only(aligned_inputs,
                         logprob_emit_at_indel,
                         joint_logprob_emit_at_match,
                         joint_logprob_transit):
    L_align = aligned_inputs.shape[1]
     
    ######################################
    ### flip inputs, transition matrix   #
    ######################################
    align_len = (aligned_inputs[...,-1] != 0).sum(axis=1)
    flipped_seqs = _flip_sequences( inputs = aligned_inputs, 
                                   seq_lengths = align_len, 
                                   flip_along_axis = 1
                                   )
    B = flipped_seqs.shape[0]
    
    # transits needs to be flipped from (T, C_from, C_to, S_from, S_to)
    #   to (T, C_to, C_from, S_to, S_from)
    flipped_transit = jnp.transpose(joint_logprob_transit, (0, 2, 1, 4, 3) )
    
    
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
        e = _joint_emit_scores( aligned_inputs = flipped_seqs,
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
            return _log_dot_bigger(log_vec = in_carry, log_mat = for_log_dot)
        
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
    
    idx_arr = jnp.array( [i for i in range(2, L_align)] )
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
    
    
    # ### swap the sequence back; final output should always be (T, C, B, L-1)
    # # reshape: (L-1, T, C, B) -> (B, L-1, T, C)
    # reshaped_stacked_outputs = jnp.transpose( stacked_outputs,
    #                                           (3, 0, 1, 2) )
    # flipped_stacked_outputs = _flip_sequences( inputs = reshaped_stacked_outputs, 
    #                                           seq_lengths = align_len-1, 
    #                                           flip_along_axis = 1
    #                                           )
    # # reshape back to (T, C, B, L-1)
    # stacked_outputs = jnp.transpose( flipped_stacked_outputs,
    #                                   (2, 3, 0, 1) )
    
    return stacked_outputs




# class TestJointOnlyForward(unittest.TestCase):
#     """
#     FORWARD-BACKWARD TEST 1
    
    
#     About
#     ------
#     compare forward algo implementation to manual enumeration over all 
#       possible paths, given example alignments
    
#     this is done with a fixed time grid
    
#     """
#     def setUp(self):
    
    
# fake inputs
fake_aligns = [ ('AC-A','D-ED'),
                ('D-ED','AC-A'),
                ('ECDAD','-C-A-'),
                ('-C-A-','ECDAD'),
                ('-C-A-','ECDAD') ]

fake_aligns =  str_aligns_to_tensor(fake_aligns) #(B, L, 3)

# fake params
rngkey = jax.random.key(42) # note: reusing this rngkey over and over
t_array = jnp.array([0.3, 0.5])
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
my_model = TKF92TransitionLogprobs(config={'num_tkf_fragment_classes': C},
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
del my_tkf_params, my_model, fake_params
    
    
    
    
#     def test_forward(self):
#         ### pred
#         forward_intermeds = joint_only_forward(aligned_inputs = fake_aligns,
#                                                joint_logprob_emit_at_match = joint_logprob_emit_at_match,
#                                                logprob_emit_at_indel = logprob_emit_at_indel,
#                                                joint_logprob_transit = joint_logprob_transit,
#                                                unique_time_per_branch = False)
        
#         # (L_align, T, C, B) -> (T, B)
#         pred = logsumexp(forward_intermeds[-1,...], axis=1)
        
        
#         ### true
#         for t in range(T):
#             for b in range(B):
#                 sample_seq = fake_aligns[b,:,:]
                
#                 # all possible path combinations
#                 invalid_toks = jnp.array([0,1,2])
#                 n = (  ~jnp.isin(sample_seq[:, 0], invalid_toks) ).sum()
#                 paths = [list(p) for p in product(range(C), repeat= int(n) )]
                
#                 # manually score each possible path
#                 score_per_path = []
#                 for path in paths:
#                     to_pad = L_align - (len(path)+1)
#                     path = [-999] + path + [-999]*to_pad
#                     path_logprob = 0
#                     prev_state = sample_seq[0, -1]
#                     for l in range(1,L_align):
#                         prev_site_class = path[l-1]
#                         curr_site_class = path[l]
#                         anc_tok, desc_tok, curr_state = sample_seq[l,:]
                        
#                         if curr_state == 0:
#                             break
                        
#                         curr_state = jnp.where(curr_state != 5, curr_state, 4)
                        
#                         ### emissions
#                         e = 0
                        
#                         if curr_state == 1:
#                             e = joint_logprob_emit_at_match[t, curr_site_class, anc_tok - 3, desc_tok - 3]
                        
#                         elif curr_state == 2:
#                             e = logprob_emit_at_indel[curr_site_class, desc_tok-3]
                        
#                         elif curr_state == 3:
#                             e = logprob_emit_at_indel[curr_site_class, anc_tok-3]
                        
#                         ### transitions
#                         tr = joint_logprob_transit[t, prev_site_class, curr_site_class, prev_state-1, curr_state-1]
#                         path_logprob += (tr + e)
#                         prev_state = curr_state
                    
#                     score_per_path.append(path_logprob)
                    
#                 true = logsumexp( jnp.array(score_per_path) )
#                 npt.assert_allclose(pred[t,b], true, rtol=THRESHOLD)


# if __name__ == '__main__':
#     unittest.main()