#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 13:33:56 2025

@author: annabel
"""
import pickle
import numpy as np
from itertools import product
from tqdm import tqdm

import numpy.testing as npt
import unittest

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax.scipy.special import logsumexp

from models.BaseClasses import ModuleBase
from models.latent_class_mixtures.model_functions import (bound_sigmoid, 
                                                              safe_log)
from tests.data_processing import (str_aligns_to_tensor,
                                   summarize_alignment)

from models.latent_class_mixtures.transition_models import TKF92TransitionLogprobs
from models.latent_class_mixtures.model_functions import switch_tkf
from models.latent_class_mixtures.one_dim_backward_joint_loglikes import joint_only_one_dim_backward as joint_only_backward

THRESHOLD = 1e-6

class TestJointOnlyBackwardIntermeds(unittest.TestCase):
    """
    this is done with a fixed time grid
    """
    def setUp(self):
        # fake inputs
        self.fake_aligns = [ ('AC','AD') ]
        self.fake_aligns =  str_aligns_to_tensor(self.fake_aligns) #(B, L, 3)
        
        # fake params
        rngkey = jax.random.key(0) # note: reusing this rngkey over and over
        t_array = jnp.array([1.0])
        self.T = t_array.shape[0]
        self.C_frag = 2
        self.A = 20
        lam = jnp.array([0.3])
        mu = jnp.array([0.5])
        offset = 1 - (lam/mu)
        r = nn.sigmoid( jax.random.randint(key=rngkey, 
                                           shape=(1, self.C_frag), 
                                           minval=-3.0, 
                                           maxval=3.0).astype(float) )
        frag_class_probs = nn.softmax( jax.random.randint(key=rngkey, 
                                                          shape=(1,self.C_frag), 
                                                          minval=1.0, 
                                                          maxval=10.0).astype(float) )
        
        # other dims
        self.B = self.fake_aligns.shape[0]
        self.L_align = self.fake_aligns.shape[1]
        
        # use dummy scoring matrices for emissions
        sub_emit_logits = jax.random.normal( key = jax.random.key(0),
                                             shape = (self.T, self.C_frag, self.A, self.A) ) #(T, C_transit, A, A)
        self.joint_logprob_emit_at_match = nn.log_softmax(sub_emit_logits, axis=(-1,-2)) #(T, C_transit, A, A)
        del sub_emit_logits
        
        indel_emit_logits = jax.random.normal( key = jax.random.key(0),
                                              shape = (self.C_frag, self.A) ) #(C_trans, A)
        self.logprob_emit_at_indel = jax.nn.log_softmax(indel_emit_logits, axis=-1) #(C_trans, A)
        del indel_emit_logits

        # be more careful about generating a fake transition matrix
        my_tkf_params, _ = switch_tkf(mu = mu, 
                                      offset = offset,
                                      t_array = t_array)
        my_tkf_params['log_offset'] = jnp.log(offset)
        my_tkf_params['log_one_minus_offset'] = jnp.log1p(-offset)
        my_model = TKF92TransitionLogprobs(config={'num_domain_mixtures':1,
                                                   'num_fragment_mixtures': self.C_frag,
                                                   'tkf_function': 'regular_tkf'},
                                           name='tkf92')
        fake_params = my_model.init(rngs=jax.random.key(0),
                                    t_array = t_array,
                                    return_all_matrices=False,
                                    sow_intermediates = False)
        
        self.joint_logprob_transit = my_model.apply(variables = fake_params,
                                                tkf_param_dict = my_tkf_params,
                                                r_extend = r,
                                                frag_class_probs = frag_class_probs,
                                                method = 'fill_joint_tkf92')[:,0,...] #(T, C, C, 4, 4)
        del my_tkf_params, my_model, fake_params
    
    
    def test_backward(self):
        ############
        ### pred   #
        ############
        backward_intermeds = joint_only_backward(aligned_inputs = self.fake_aligns,
                                               joint_logprob_emit_at_match = self.joint_logprob_emit_at_match,
                                               logprob_emit_at_indel = self.logprob_emit_at_indel,
                                               joint_logprob_transit = self.joint_logprob_transit,
                                               unique_time_per_sample = False) # (L_align-1, T, C, B)
        
        true = np.zeros( backward_intermeds.shape )  # (L_align-1, T, C, B)
        
        
        #####################################
        ### Helpers for future iterations   #
        #####################################
        def index_match_mat(from_class, 
                            anc_tok,
                            desc_tok):
            return self.joint_logprob_emit_at_match[:, from_class, anc_tok, desc_tok] #(T,)
        
        def index_transit_mat(from_class,
                              to_class,
                              from_state,
                              to_state):
            return self.joint_logprob_transit[:, from_class, to_class, from_state, to_state] #(T,)
        
        def message_passing(anc_tok, 
                            desc_tok, 
                            from_state, 
                            to_state, 
                            from_class, 
                            to_class,
                            has_emit: bool=True,
                            has_cache: bool=True):
            # transit
            tr = index_transit_mat(from_class = from_class,
                                    to_class = to_class,
                                    from_state = from_state,
                                    to_state = to_state)
            
            # cache update
            cache = true[l+1, :, to_class, :] if has_cache else 0
            
            # emission score
            if has_emit:
                e = index_match_mat(from_class = from_class, 
                                    anc_tok = anc_tok,
                                    desc_tok = desc_tok)
            else:
                e = 0
                
            # return
            return e + tr + cache
        
        def marginalize_over_future_class(anc_tok, 
                            desc_tok, 
                            from_state, 
                            to_state, 
                            from_class):
            # to class 0
            to_0 = message_passing(anc_tok=anc_tok, 
                                   desc_tok=desc_tok, 
                                   from_state=from_state, 
                                   to_state=to_state, 
                                   from_class=from_class, 
                                   to_class=0)
            
            # to class 1
            to_1  = message_passing(anc_tok=anc_tok, 
                                    desc_tok=desc_tok, 
                                    from_state=from_state, 
                                    to_state=to_state, 
                                    from_class=from_class, 
                                    to_class=1)
            
            # cell should be sum of above
            return jnp.logaddexp( to_0, to_1 ) #(T)
        
        
        ###################################
        ### col 2: <end> -> (C,D) match   #
        ###################################
        l = 2
        anc_tok = 1
        desc_tok = 2
        from_state = 0 # match at 2
        to_state = 3   # <end> at 3
        to_class = -1  # <end>_class at 3
        
        # Match_0 -> End
        from_class = 0
        true[l, :, from_class, :] = message_passing(anc_tok = anc_tok, 
                                                    desc_tok = desc_tok, 
                                                    from_state = from_state, 
                                                    to_state = to_state, 
                                                    from_class = from_class, 
                                                    to_class = to_class,
                                                    has_cache = False)
        
        # Match_1 -> End
        from_class = 1 
        true[l, :, from_class, :] = message_passing(anc_tok = anc_tok, 
                                                    desc_tok = desc_tok, 
                                                    from_state = from_state, 
                                                    to_state = to_state, 
                                                    from_class = from_class, 
                                                    to_class = to_class,
                                                    has_cache = False)
        
        del l, anc_tok, desc_tok, from_state, to_state, from_class, to_class
        
        
        ##########################################
        ### col 1: (C, D) match -> (A,A) match   #
        ##########################################
        l = 1
        anc_tok = 0
        desc_tok = 0
        from_state = 0 # match at 1
        to_state = 0   # match at 2
        
        # M_0 -> M_0
        from_class = 0
        true[l, :, from_class, :] = marginalize_over_future_class(anc_tok = anc_tok, 
                                                                  desc_tok = desc_tok, 
                                                                  from_state = from_state, 
                                                                  to_state = to_state, 
                                                                  from_class = from_class)
        del from_class
        
        # M_0 -> M_1
        from_class = 1 # from class 1
        true[l, :, from_class, :] = marginalize_over_future_class(anc_tok = anc_tok, 
                                                                  desc_tok = desc_tok, 
                                                                  from_state = from_state, 
                                                                  to_state = to_state, 
                                                                  from_class = from_class)
        
        del l, anc_tok, desc_tok, from_state, to_state, from_class
        
        
        ###################################
        ### col 2: (A,A) match -> Start   #
        ###################################
        l = 0
        from_state = 3 # <start> at 1
        from_class = -1 
        to_state = 0   # match at 2
        
        # Start -> M0
        to_class = 0
        true[l, :, to_class, :] = message_passing(anc_tok = -1, 
                                                  desc_tok = -1, 
                                                  from_state = from_state, 
                                                  to_state = to_state, 
                                                  from_class = from_class, 
                                                  to_class = to_class,
                                                  has_emit = False)
        
        # Start -> M1
        to_class = 1
        true[l, :, to_class, :] = message_passing(anc_tok = -1, 
                                                  desc_tok = -1, 
                                                  from_state = from_state, 
                                                  to_state = to_state, 
                                                  from_class = from_class, 
                                                  to_class = to_class,
                                                  has_emit = False)
        
        npt.assert_allclose(backward_intermeds, true)
        
        

if __name__ == '__main__':
    unittest.main()