#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 13:47:05 2025

@author: annabel

"""
import jax
from jax import numpy as jnp
import flax.linen as nn

import numpy as np
import numpy.testing as npt
import unittest
from itertools import product
from scipy.special import logsumexp
from tqdm import tqdm

from models.latent_class_mixtures.transition_models import TKF92TransitionLogprobs
from models.latent_class_mixtures.one_dim_forward_joint_loglikes import joint_only_one_dim_forward as one_dim_forward
from models.latent_class_mixtures.two_dim_forward_with_time_grid import two_dim_forward_with_time_grid as forward_fn



class TestForwardLoglikeVsBruteForceEnum(unittest.TestCase):
    """
    This works through MARGINALIZING OVER A GRID OF TIMES; T!=B
    """
    def _str_to_align_mat(self, align_tup):
        anc, desc = align_tup
        del align_tup
        
        out = [[1,1,4]]
        
        def str_to_num(letter):
            if letter == 'A':
                return 3
            elif letter == '-':
                return 43
        
        for i in range(len(anc)):
            a = str_to_num(anc[i])
            d = str_to_num(desc[i])
            
            if a == d:
                s = 1
            
            elif a == 43:
                s = 2
            
            elif d == 43:
                s = 3
            
            out.append( [a,d,s] )
        
        out.append( [2,2,5] )
        if len(out) == 5:
            out.append( [0,0,0] )
            out.append( [0,0,0] )
        
        elif len(out) == 6:
            out.append( [0,0,0] )
        
        return jnp.array(out)
    
    
    def test_against_brute_force_enumeration(self):
        # make sure this is turned off, or the test will fail
        jax.config.update("jax_enable_x64", False)
        
        #######################################################################
        ### Fake inputs   #####################################################
        #######################################################################
        # dims
        C_transit = 1
        A = 20
        S = 4
        C_S = C_transit * (S-1) #use this for forward algo carry
        
        # time
        t_array = jnp.array( [1.0] )
        T = t_array.shape[0]
        
        
        ########################
        ### scoring matrices   #
        ########################
        # use real model object for transition matrix
        transit_model = TKF92TransitionLogprobs( config={'tkf_function': 'regular_tkf',
                                                         'num_domain_mixtures': 1,
                                                         'num_fragment_mixtures': C_transit},
                                                 name = 'transit_mat' )
        init_params = transit_model.init( rngs = jax.random.key(0),
                                          t_array = t_array,
                                          return_all_matrices = False,
                                          sow_intermediates = False )
        
        out = transit_model.apply( variables = init_params,
                                   t_array = t_array,
                                   return_all_matrices = False,
                                   sow_intermediates = False )
        joint_logprob_transit_old_dim_order = out[1]['joint'][:,0,...] #(T, C_transit_prev, C_transit_curr, S_prev, S_curr)
        del transit_model, init_params, out
        
        # use dummy scoring matrices for emissions
        sub_emit_logits = jax.random.normal( key = jax.random.key(0),
                                             shape = (T, C_transit, A, A) ) #(T, C_transit, A, A)
        joint_logprob_emit_at_match = nn.log_softmax(sub_emit_logits, axis=(-1,-2)) #(T, C_transit, A, A)
        del sub_emit_logits
        
        indel_emit_logits = jax.random.normal( key = jax.random.key(0),
                                             shape = (C_transit, A) ) #(C_transit, A)
        logprob_emit_at_indel = nn.log_softmax(indel_emit_logits, axis=-1) #(C_transit, A)
        del indel_emit_logits
        
        
        #################
        ### sequences   #
        #################
        ### AAA -> AA: 25 possible alignments
        seqs1 = jnp.array( [[1, 1],
                            [3, 3],
                            [3, 3],
                            [3, 2],
                            [2, 0]] )[None,...] #(B, L_seq, 2)
        
        # alignments in strings; thanks GPT
        str_aligns1 = [("AAA", "AA-"),
                      ("AAA", "A-A"),
                      ("AAA-", "A--A"),
                      ("AA-A", "A-A-"),
                      ("A-AA", "AA--"),
                      ("AAA", "-AA"),
                      ("AAA-", "-A-A"),
                      ("AA-A", "-AA-"),
                      ("AAA-", "--AA"),
                      ("AAA--", "---AA"),
                      ("AA-A", "--AA"),
                      ("AA-A-", "--A-A"),
                      ("AA--A", "--AA-"),
                      ("A-AA", "-AA-"),
                      ("A-AA", "-A-A"),
                      ("A-AA-", "-A--A"),
                      ("A-A-A", "-A-A-"),
                      ("A--AA", "-AA--"),
                      ("-AAA", "AA--"),
                      ("-AAA", "A-A-"),
                      ("-AAA", "A--A"),
                      ("-AAA-", "A---A"),
                      ("-AA-A", "A--A-"),
                      ("-A-AA", "A-A--"),
                      ("--AAA", "AA---")]
        
        alignment_mats1 = [self._str_to_align_mat(al) for al in str_aligns1 ]
        alignment_mats1 = jnp.stack( alignment_mats1 ) #(N_alignments, L_align, 3)
        
        
        ### AA -> AAA: 25 possible alignments
        seqs2 = jnp.array( [[1, 1],
                            [3, 3],
                            [3, 3],
                            [2, 3],
                            [0, 2]] )[None,...] #(B, L_seq, 2)
        
        
        # alignments in strings; thanks GPT
        str_aligns2 = [("AA-", "AAA"),
                       ("A-A", "AAA"),
                       ("A--A", "AAA-"),
                       ("A-A-", "AA-A"),
                       ("AA--", "A-AA"),
                       ("-AA", "AAA"),
                       ("-A-A", "AAA-"),
                       ("-AA-", "AA-A"),
                       ("--AA", "AAA-"),
                       ("---AA", "AAA--"),
                       ("--AA", "AA-A"),
                       ("--A-A", "AA-A-"),
                       ("--AA-", "AA--A"),
                       ("-AA-", "A-AA"),
                       ("-A-A", "A-AA"),
                       ("-A--A", "A-AA-"),
                       ("-A-A-", "A-A-A"),
                       ("-AA--", "A--AA"),
                       ("AA--", "-AAA"),
                       ("A-A-", "-AAA"),
                       ("A--A", "-AAA"),
                       ("A---A", "-AAA-"),
                       ("A--A-", "-AA-A"),
                       ("A-A--", "-A-AA"),
                       ("AA---", "--AAA")]
        
        alignment_mats2 = [self._str_to_align_mat(al) for al in str_aligns2 ]
        alignment_mats2 = jnp.stack( alignment_mats2 ) #(N_alignments, L_align, 3)
        
        
        ### concatenate sequences
        unaligned_seqs = jnp.concatenate( [seqs1, seqs2] ) #(B, L, 2)
        
        
        #######################################################################
        ### Test function   ###################################################
        #######################################################################
        ### True scores from sum over all possible alignments (manually enumerated)
        def sum_over_alignments(all_possible_aligns):
            # one_dim_forward was previously developed to marginalize over classes, given an alignment
            # proven to work in other tests
            score_per_align = one_dim_forward(aligned_inputs = all_possible_aligns,
                                                  joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                                  logprob_emit_at_indel = logprob_emit_at_indel,
                                                  joint_logprob_transit = joint_logprob_transit_old_dim_order,
                                                  unique_time_per_sample = False,
                                                  return_all_intermeds = False) #(T, num_alignments)
            return logsumexp(score_per_align, axis=-1 ) #(T,)
        
        true1 = sum_over_alignments( alignment_mats1 )
        true2 = sum_over_alignments( alignment_mats2 )
        true = np.concatenate( [true1, true2] )
        
        ### Predicted scores from 2D forward algo implementation
        # transpose transition matrix 
        # old: (T, C_transit_prev, C_transit_curr, S_prev, S_curr)
        # new: (T, C_transit_prev, S_prev, C_transit_curr, S_curr)
        joint_logprob_transit = jnp.transpose(joint_logprob_transit_old_dim_order, (0,1,3,2,4)) # (T, C_transit_prev, S_prev, C_transit_curr, S_curr)
        
        pred = forward_fn(unaligned_seqs = unaligned_seqs,
                          joint_logprob_transit = joint_logprob_transit,
                          joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                          logprob_emit_at_indel = logprob_emit_at_indel) #(T, B)
        
        pred = jnp.squeeze(pred)
        
        npt.assert_allclose(pred, true)


if __name__ == '__main__':
    unittest.main() 
    