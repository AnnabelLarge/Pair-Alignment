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
from models.simple_site_class_predict.model_functions import (regular_tkf,
                                                              joint_only_forward)

THRESHOLD = 1e-6

class TestJointOnlyForwardUniqBranchLen(unittest.TestCase):
    """
    About
    ------
    compare forward algo implementation to manual enumeration over all 
      possible paths, given example alignments
    
    this is done with a unique time per sample
    
    """
    def setUp(self):
        # fake inputs
        self.fake_aligns = [ ('AC-A','D-ED'),
                            ('D-ED','AC-A'),
                            ('ECDAD','-C-A-'),
                            ('-C-A-','ECDAD'),
                            ('-C-A-AA','ECDADAA') ]
        
        self.fake_aligns =  str_aligns_to_tensor(self.fake_aligns) #(B, L, 3)
        
        # fake params
        rngkey = jax.random.key(42) # note: reusing this rngkey over and over
        t_array = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.C = 3
        self.A = 20
        lam = jnp.array(0.3)
        mu = jnp.array(0.5)
        offset = 1 - (lam/mu)
        r = nn.sigmoid( jax.random.randint(key=rngkey, 
                                           shape=(self.C,), 
                                           minval=-3.0, 
                                           maxval=3.0).astype(float) )
        class_probs = nn.softmax( jax.random.randint(key=rngkey, 
                                                     shape=(self.C,), 
                                                     minval=1.0, 
                                                     maxval=10.0).astype(float) )
        
        # other dims (blt; yum)
        self.B = self.fake_aligns.shape[0]
        self.L_align = self.fake_aligns.shape[1]
        
        # fake scoring matrices (not coherently normalized; just some example values)
        def generate_fake_scoring_mat(dim_tuple):
            logits = jax.random.uniform(key=rngkey, 
                                        shape=dim_tuple,
                                        minval=-10.0,
                                        maxval=-1e-4)
            return nn.log_softmax(logits, axis=-1)
        
        self.joint_logprob_emit_at_match = generate_fake_scoring_mat( (self.B,self.C,self.A,self.A) )
        self.logprob_emit_at_indel = generate_fake_scoring_mat( (self.C,self.A) )
        
        # be more careful about generating a fake transition matrix
        my_tkf_params, _ = regular_tkf(mu = mu, 
                                      offset = offset,
                                      t_array = t_array)
        my_tkf_params['log_offset'] = jnp.log(offset)
        my_tkf_params['log_one_minus_offset'] = jnp.log1p(-offset)
        my_model = TKF92TransitionLogprobs(config={'num_tkf_fragment_classes': self.C},
                                           name='tkf92')
        fake_params = my_model.init(rngs=jax.random.key(0),
                                    t_array = t_array,
                                    log_class_probs = jnp.log(class_probs),
                                    sow_intermediates = False)
        
        self.joint_logprob_transit =  my_model.apply(variables = fake_params,
                                                out_dict = my_tkf_params,
                                                r_extend = r,
                                                class_probs = class_probs,
                                                method = 'fill_joint_tkf92') #(B, C, C, 4, 4)
        del my_tkf_params, my_model, fake_params
    
    
    def test_forward(self):
        ### pred
        forward_intermeds = joint_only_forward(aligned_inputs = self.fake_aligns,
                                               joint_logprob_emit_at_match = self.joint_logprob_emit_at_match,
                                               logprob_emit_at_indel = self.logprob_emit_at_indel,
                                               joint_logprob_transit = self.joint_logprob_transit,
                                               unique_time_per_sample = True,
                                               return_all_intermeds = True)
        
        # (L_align, C, B) -> (B)
        pred = logsumexp(forward_intermeds[-1,...], axis=0)
        
        
        ### true
        for b in range(self.B):
            sample_seq = self.fake_aligns[b,:,:]
            
            # all possible path combinations
            invalid_toks = jnp.array([0,1,2])
            n = (  ~jnp.isin(sample_seq[:, 0], invalid_toks) ).sum()
            paths = [list(p) for p in product(range(self.C), repeat= int(n) )]
            
            # manually score each possible path
            score_per_path = []
            for path in paths:
                to_pad = self.L_align - (len(path)+1)
                path = [-999] + path + [-999]*to_pad
                path_logprob = 0
                prev_state = sample_seq[0, -1]
                for l in range(1,self.L_align):
                    prev_site_class = path[l-1]
                    curr_site_class = path[l]
                    anc_tok, desc_tok, curr_state = sample_seq[l,:]
                    
                    if curr_state == 0:
                        break
                    
                    curr_state = jnp.where(curr_state != 5, curr_state, 4)
                    
                    ### emissions
                    e = 0
                    
                    if curr_state == 1:
                        e = self.joint_logprob_emit_at_match[b, curr_site_class, anc_tok - 3, desc_tok - 3]
                    
                    elif curr_state == 2:
                        e = self.logprob_emit_at_indel[curr_site_class, desc_tok-3]
                    
                    elif curr_state == 3:
                        e = self.logprob_emit_at_indel[curr_site_class, anc_tok-3]
                    
                    ### transitions
                    tr = self.joint_logprob_transit[b, prev_site_class, curr_site_class, prev_state-1, curr_state-1]
                    path_logprob += (tr + e)
                    prev_state = curr_state
                
                score_per_path.append(path_logprob)
                
            true = logsumexp( jnp.array(score_per_path) )
            npt.assert_allclose(pred[b], true, rtol=THRESHOLD)


if __name__ == '__main__':
    unittest.main()