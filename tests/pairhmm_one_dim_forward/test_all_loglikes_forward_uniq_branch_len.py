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
from models.latent_class_mixtures.model_functions import (switch_tkf,
                                                          get_tkf92_single_seq_marginal_transition_logprobs)
from models.latent_class_mixtures.one_dim_forward_joint_loglikes import joint_only_one_dim_forward as joint_only_forward
from models.latent_class_mixtures.one_dim_forward_all_loglikes import all_loglikes_one_dim_forward as all_loglikes_forward

THRESHOLD = 1e-6



class TestAllLoglikesForwardUniqBranchLen(unittest.TestCase):
    """
    About
    ------
    compare forward algo implementation to manual enumeration over all 
      possible paths, given example alignments, as well as other 
      validated functions
     
    now, each sample has its own branch length
    
    """
    def setUp(self):
        # fake inputs
        self.fake_aligns = [ ('AC-A','D-ED'),
                             ('D-ED','AC-A'),
                             ('ECDAD','-C-A-'),
                             ('-C-A-','ECDAD'),
                             ('-C-A-','ECDAD') ]
        
        self.fake_aligns =  str_aligns_to_tensor(self.fake_aligns) #(B, L, 3)
        
        # fake params
        rngkey = jax.random.key(42) # note: reusing this rngkey over and over
        t_array = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.t_array = t_array
        self.T = t_array.shape[0]
        self.C_frag = 3
        self.A = 20
        lam = jnp.array([0.3])
        mu = jnp.array([0.5])
        offset = 1 - (lam/mu)
        r = nn.sigmoid( jax.random.randint(key=rngkey, 
                                           shape=(1,self.C_frag), 
                                           minval=-3.0, 
                                           maxval=3.0).astype(float) )
        frag_class_probs = nn.softmax( jax.random.randint(key=rngkey, 
                                                     shape=(1,self.C_frag), 
                                                     minval=1.0, 
                                                     maxval=10.0).astype(float) )
        
        # other dims
        self.B = self.fake_aligns.shape[0]
        assert self.B == self.T
        self.L_align = self.fake_aligns.shape[1]
                
        # fake scoring matrices (not coherently normalized; just some example values)
        def generate_fake_scoring_mat(dim_tuple):
            logits = jax.random.uniform(key=rngkey, 
                                        shape=dim_tuple,
                                        minval=-10.0,
                                        maxval=-1e-4)
            return nn.log_softmax(logits, axis=-1)
        
        self.joint_logprob_emit_at_match = generate_fake_scoring_mat( (self.B,self.C_frag,self.A,self.A) )
        self.logprob_emit_at_indel = generate_fake_scoring_mat( (self.C_frag,self.A) )
        
        # be more careful about generating a fake transition matrix
        my_tkf_params, _ = switch_tkf(mu = mu, 
                                      offset = offset,
                                      t_array = self.t_array)
        my_tkf_params['log_offset'] = jnp.log(offset)
        my_tkf_params['log_one_minus_offset'] = jnp.log1p(-offset)
        my_model = TKF92TransitionLogprobs(config={'num_domain_mixtures':1,
                                                   'num_fragment_mixtures': self.C_frag,
                                                   'num_site_mixtures': 10,
                                                   'k_rate_mults':4,
                                                   'tkf_function': 'regular_tkf'},
                                           name='tkf92')
        fake_params = my_model.init(rngs=jax.random.key(0),
                                    t_array = self.t_array,
                                    return_all_matrices = True,
                                    sow_intermediates = False)
        
        self.joint_logprob_transit =  my_model.apply(variables = fake_params,
                                                tkf_param_dict = my_tkf_params,
                                                r_extend = r,
                                                frag_class_probs = frag_class_probs,
                                                method = 'fill_joint_tkf92')[:,0,...] #(B, C, C, 4, 4)
        
        self.marg_logprob_transit = get_tkf92_single_seq_marginal_transition_logprobs( offset = offset,
                                                            frag_class_probs = frag_class_probs,
                                                            r_ext_prob = r ) [0,...] #(C, C, 4, 4)
        
        
        ### run function
        pred_out = all_loglikes_forward( aligned_inputs = self.fake_aligns,
                                         logprob_emit_at_indel = self.logprob_emit_at_indel,
                                         joint_logprob_emit_at_match = self.joint_logprob_emit_at_match,
                                         all_transit_matrices = {'joint': self.joint_logprob_transit,
                                                                 'marginal': self.marg_logprob_transit},
                                         unique_time_per_sample = True )
        
        self.pred_joint = -pred_out['joint_neg_logP'] #(B,)
        self.pred_anc_marg = -pred_out['anc_neg_logP'] #(B,)
        self.pred_desc_marg = -pred_out['desc_neg_logP'] #(B,)
        
    def test_match_with_prev_func(self):
        tmp = joint_only_forward( aligned_inputs = self.fake_aligns,
                                  joint_logprob_emit_at_match = self.joint_logprob_emit_at_match,
                                  logprob_emit_at_indel = self.logprob_emit_at_indel,
                                  joint_logprob_transit = self.joint_logprob_transit,
                                  unique_time_per_sample = True,
                                  return_all_intermeds = True ) #(L_align, C, B)
        
        # (L_align, C, B) -> (B)
        true_joint = logsumexp(tmp[-1,...], axis=0)
        npt.assert_allclose(self.pred_joint, true_joint, rtol=THRESHOLD)

    def _single_seq_marginal_test(self, batch, pred):
        """
        this is the same as last time, since it's unaffected by time
        
        run anyways, just in case
        """
        for b in range(self.B):
            invalid_toks = jnp.array([0,1,2,43])
            sample_gapped_seq = batch[b,:]
            sample_seq = sample_gapped_seq[~jnp.isin(sample_gapped_seq, invalid_toks)]
            n = (  ~jnp.isin(sample_gapped_seq, invalid_toks) ).sum()
            paths = [list(p) for p in product(range(self.C_frag), repeat= int(n) )]
            del sample_gapped_seq
        
            # manually score each possible path
            score_per_path = []
            
            for path in paths:
                path_logprob = 0
        
                ### first start -> emit
                l = 0
                curr_site_class = path[0]
                seq_tok = sample_seq[0]
                
                e = self.logprob_emit_at_indel[curr_site_class, seq_tok-3]
                tr = self.marg_logprob_transit[0, curr_site_class, 1, 0]
                path_logprob += (tr + e)
                del l, curr_site_class, seq_tok, e, tr
                
                
                ### all emitted sequences
                for l in range(1, sample_seq.shape[0]):
                    prev_site_class = path[l-1]
                    curr_site_class = path[l]
                    seq_tok = sample_seq[l]
                    
                    e = self.logprob_emit_at_indel[curr_site_class, seq_tok-3]
                    tr = self.marg_logprob_transit[prev_site_class, curr_site_class, 0, 0]
                    path_logprob += (tr + e)
                
                
                ### ending
                last_site_class = path[-1]
                path_logprob += self.marg_logprob_transit[last_site_class, -1, 0, 1]
                
                score_per_path.append(path_logprob)
                
            true = logsumexp( jnp.array(score_per_path) )
            npt.assert_allclose(pred[b], true, rtol=THRESHOLD)
    
    def test_anc_marginal_against_manual_enumeration(self):
        self._single_seq_marginal_test(batch = self.fake_aligns[...,0], 
                                        pred = self.pred_anc_marg)
        
    def test_desc_marginal_against_manual_enumeration(self):
        self._single_seq_marginal_test(batch = self.fake_aligns[...,1], 
                                        pred = self.pred_desc_marg)

if __name__ == '__main__':
    unittest.main()