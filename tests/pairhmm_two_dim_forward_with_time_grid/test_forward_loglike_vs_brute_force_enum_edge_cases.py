#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 13:47:05 2025

@author: annabel

This works through MARGINALIZING OVER A GRID OF TIMES; T!=B


sizes:
------
transition matrx: T, C_transit, C_transit, S_prev, S_curr
equilibrium distribution: C_transit, C_sites, A
  > after marginalizing over site-independent C_sites: C_transit, A
substitution emission matrix: T, C_transit, C_sites, K, A, A
  > after marginalizing over site-independent C_sites and K: T, C_transit, A, A
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
from models.latent_class_mixtures.model_functions import joint_only_forward

from models.latent_class_mixtures.two_dim_forward_with_time_grid import two_dim_forward_with_time_grid as forward_fn



class TestForwardLoglikeVsBruteFroceEnumEdgeCases(unittest.TestCase):
    def test_against_brute_force_enumeration(self):
        # make sure this is turned off, or the test will fail
        jax.config.update("jax_enable_x64", False)
        
        #######################################################################
        ### Fake inputs   #####################################################
        #######################################################################
        # dims
        C_transit = 3
        A = 20
        S = 4
        C_S = C_transit * (S-1) #use this for forward algo carry
        
        # time
        t_array = jnp.array( [1.0, 0.2, 0.3, 0.5] )
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
        ### AG -> A: 5 possible alignment paths
        seqs1 = jnp.array( [[1, 1],
                            [3, 3],
                            [5, 2],
                            [2, 0],
                            [0, 0]] )
        
        # alignment 1:
        # AG-
        # --A
        align1 = jnp.array( [[ 1,  1,  4],
                             [ 3, 43,  3],
                             [ 5, 43,  3],
                             [43,  3,  2],
                             [ 2,  2,  5]] )
        
        # alignment 2:
        # A-G
        # -A-
        align2 = jnp.array( [[ 1,  1,  4],
                             [ 3, 43,  3],
                             [43,  3,  2],
                             [ 5, 43,  3],
                             [ 2,  2,  5]] )
        
        # alignment 3:
        # -AG
        # A--
        align3 = jnp.array( [[ 1,  1,  4],
                             [43,  3,  2],
                             [ 3, 43,  3],
                             [ 5, 43,  3],
                             [ 2,  2,  5]] )
        
        # alignment 4:
        # AG
        # A-
        align4 = jnp.array( [[ 1,  1,  4],
                             [ 3,  3,  1],
                             [ 5, 43,  3],
                             [ 2,  2,  5],
                             [ 0,  0,  0]] )
        
        # alignment 5:
        # AG
        # -A
        align5 = jnp.array( [[ 1,  1,  4],
                             [ 3, 43,  3],
                             [ 5,  3,  1],
                             [ 2,  2,  5],
                             [ 0,  0,  0]] )
        
        aligned_mats1 = jnp.stack( [align1, align2, align3, align4, align5], axis=0 ) #(num_possible_aligns, L_align, 3)
        del align1, align2, align3, align4, align5
        
        
        ### T -> TC: 5 possible alignment paths
        seqs2 = jnp.array( [[1, 1],
                            [6, 6],
                            [2, 4],
                            [0, 2],
                            [0, 0]] )
        
        # alignment 1:
        # --T
        # TC-
        align1 = jnp.array( [[ 1,  1,  4],
                             [43,  6,  2],
                             [43,  4,  2],
                             [ 6, 43,  3],
                             [ 2,  2,  5]] )
        
        # alignment 2:
        # -T-
        # T-C
        align2 = jnp.array( [[ 1,  1,  4],
                             [43,  6,  2],
                             [ 6, 43,  3],
                             [43,  4,  2],
                             [ 2,  2,  5]] )
        
        # alignment 3:
        # T--
        # -TC
        align3 = jnp.array( [[ 1,  1,  4],
                             [ 6, 43,  3],
                             [43,  6,  2],
                             [43,  4,  2],
                             [ 2,  2,  5]] )
        
        # alignment 4:
        # T-
        # TC
        align4 = jnp.array( [[ 1,  1,  4],
                             [ 6,  6,  1],
                             [43,  4,  2],
                             [ 2,  2,  5],
                             [ 0,  0,  0]] )
        
        # alignment 5:
        # -T
        # TC
        align5 = jnp.array( [[ 1,  1,  4],
                             [43,  6,  2],
                             [ 6,  4,  1],
                             [ 2,  2,  5],
                             [ 0,  0,  0]] )
        
        aligned_mats2 = jnp.stack( [align1, align2, align3, align4, align5], axis=0 ) #(num_possible_aligns, L_align, 3)
        del align1, align2, align3, align4, align5
        
        
        ### T -> T: 3 possible alignment paths
        seqs3 = jnp.array( [[1, 1],
                            [6, 6],
                            [2, 2],
                            [0, 0],
                            [0, 0]] )
        
        # alignment 1:
        # -T
        # T-
        align1 = jnp.array( [[ 1,  1,  4],
                             [43,  6,  2],
                             [ 6, 43,  3],
                             [ 2,  2,  5],
                             [ 0,  0,  0]] )
        
        # alignment 2:
        # T
        # T
        align2 = jnp.array( [[ 1,  1,  4],
                             [ 6,  6,  1],
                             [ 2,  2,  5],
                             [ 0,  0,  0],
                             [ 0,  0,  0]] )
        
        # alignment 3:
        # T-
        # -T
        align3 = jnp.array( [[ 1,  1,  4],
                             [ 6, 43,  3],
                             [43,  6,  2],
                             [ 2,  2,  5],
                             [ 0,  0,  0]] )
        
        aligned_mats3 = jnp.stack( [align1, align2, align3], axis=0 ) #(num_possible_aligns, L_align, 3)
        del align1, align2, align3
        
        ### A -> A: 3 possible alignment paths
        seqs4 = jnp.array( [[1, 1],
                            [3, 3],
                            [2, 2],
                            [0, 0],
                            [0, 0]] )
        
        # alignment 1:
        # -A
        # A-
        align1 = jnp.array( [[ 1,  1,  4],
                             [43,  3,  2],
                             [ 3, 43,  3],
                             [ 2,  2,  5],
                             [ 0,  0,  0]] )
        
        # alignment 2:
        # A
        # A
        align2 = jnp.array( [[ 1,  1,  4],
                             [ 3,  3,  1],
                             [ 2,  2,  5],
                             [ 0,  0,  0],
                             [ 0,  0,  0]] )
        
        # alignment 3:
        # A-
        # -A
        align3 = jnp.array( [[ 1,  1,  4],
                             [ 3, 43,  3],
                             [43,  3,  2],
                             [ 2,  2,  5],
                             [ 0,  0,  0]] )
        
        aligned_mats4 = jnp.stack( [align1, align2, align3], axis=0 ) #(num_possible_aligns, L_align, 3)
        del align1, align2, align3
        
        
        ### concate sequences
        unaligned_seqs = jnp.stack([seqs1, seqs2, seqs3, seqs4], axis=0) #(B, L_seq, 2)
        del seqs1, seqs2, seqs3, seqs4
        
        
        #######################################################################
        ### Test function   ###################################################
        #######################################################################
        ### True scores from sum over possible alignments
        def sum_over_alignments(all_possible_aligns):
            score_per_align = joint_only_forward(aligned_inputs = all_possible_aligns,
                                                 joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                                 logprob_emit_at_indel = logprob_emit_at_indel,
                                                 joint_logprob_transit = joint_logprob_transit_old_dim_order,
                                                 unique_time_per_sample = False,
                                                 return_all_intermeds = False) #(T, num_alignments)
            return logsumexp(score_per_align, axis=-1 ) #(T,)
        
        true_score1 = sum_over_alignments(aligned_mats1)
        true_score2 = sum_over_alignments(aligned_mats2)
        true_score3 = sum_over_alignments(aligned_mats3)
        true_score4 = sum_over_alignments(aligned_mats4)
        
        true = np.stack( [true_score1, true_score2, true_score3, true_score4], axis=-1 ) #(T, B)
        del true_score1, true_score2, true_score3, true_score4
        del aligned_mats1, aligned_mats2, aligned_mats3, aligned_mats4
        
        
        ### Predicted scores from 2D forward algo implementation
        # transpose transition matrix 
        # old: (T, C_transit_prev, C_transit_curr, S_prev, S_curr)
        # new: (T, C_transit_prev, S_prev, C_transit_curr, S_curr)
        joint_logprob_transit = jnp.transpose(joint_logprob_transit_old_dim_order, (0,1,3,2,4)) # (T, C_transit_prev, S_prev, C_transit_curr, S_curr)
        
        pred = forward_fn(unaligned_seqs = unaligned_seqs,
                          joint_logprob_transit = joint_logprob_transit,
                          joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                          logprob_emit_at_indel = logprob_emit_at_indel) #(T, B)
        
        npt.assert_allclose(pred, true)


if __name__ == '__main__':
    unittest.main() 
    