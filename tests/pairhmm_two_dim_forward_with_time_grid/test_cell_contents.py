#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 12:01:02 2025

@author: annabel

todo: make this prettier
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



class TestCellContents(unittest.TestCase):
    def test_this(self):
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
        ### T -> TC: 5 possible alignment paths
        unaligned_seqs = jnp.array( [[1, 1],
                            [6, 6],
                            [2, 4],
                            [0, 2],
                            [0, 0]] )[None,...]
        
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
        
        aligned_mats = jnp.stack( [align1, align2, align3, align4, align5], axis=0 ) #(num_possible_aligns, L_align, 3)
        del align1, align2, align3, align4, align5
        
        # widest diagonal for wavefront parallelism is min(anc_len, desc_len) + 1
        seq_lens = (unaligned_seqs != 0).sum(axis=1)-2 #(B, 2)
        seq_lens = seq_lens
        
        min_lens = seq_lens.min(axis=1) #(B,)
        W = min_lens.max() + 1 #float
        del min_lens
        
        # number of diagonals
        K = (seq_lens.sum(axis=1)).max()
        
        
        #######################################################################
        ### True value, calculated by hand   ##################################
        #######################################################################
        # T encoded as: 6
        # C encoded as: 4
        
        # scoring matrices, simplified
        compressed_joint_logprob_transit = jnp.squeeze(joint_logprob_transit_old_dim_order) #(S_prev, S_to)
        compressed_joint_logprob_emit_at_match = jnp.squeeze(joint_logprob_emit_at_match) #(A, A)
        compressed_logprob_emit_at_indel = jnp.squeeze(logprob_emit_at_indel) #(A,)
        
        
        #######################
        ### cell (1,0); K=1   #
        #######################
        # T
        # -
        # S -> del T
        true_cell_1_0 = compressed_joint_logprob_transit[3,2] + compressed_logprob_emit_at_indel[6-3]
        
        
        #######################
        ### cell (0,1); K=1   #
        #######################
        # -
        # T
        # S -> ins T
        true_cell_0_1 = compressed_joint_logprob_transit[3,1] + compressed_logprob_emit_at_indel[6-3]
        
        
        #######################
        ### cell (1,1); K=2   #
        #######################
        # T -
        # - T
        # S -> del T -> ins T
        path1 = ( compressed_joint_logprob_transit[3,2] + compressed_logprob_emit_at_indel[6-3] +
                  compressed_joint_logprob_transit[2,1] + compressed_logprob_emit_at_indel[6-3] )
        
        # - T
        # T -
        # S -> ins T -> del T
        path2 = ( compressed_joint_logprob_transit[3,1] + compressed_logprob_emit_at_indel[6-3] +
                  compressed_joint_logprob_transit[1,2] + compressed_logprob_emit_at_indel[6-3] )
        
        # T
        # T
        # S -> match (T,T)
        path3 = compressed_joint_logprob_transit[3,0] + compressed_joint_logprob_emit_at_match[6-3, 6-3]
        
        true_cell_1_1 = logsumexp( [path1, path2, path3] )
        del path1, path2, path3
        
        
        #######################
        ### cell (0,2); K=2   #
        #######################
        # - -
        # T C
        # S -> ins T -> ins C
        true_cell_0_2 = ( compressed_joint_logprob_transit[3,1] + compressed_logprob_emit_at_indel[6-3] +
                     compressed_joint_logprob_transit[1,1] + compressed_logprob_emit_at_indel[4-3])
        
        
        #######################
        ### cell (1,2); K=3   #
        #######################
        # --T
        # TC-
        # S -> ins T -> ins C -> del T 
        path1 = ( compressed_joint_logprob_transit[3,1] + compressed_logprob_emit_at_indel[6-3] +
                  compressed_joint_logprob_transit[1,1] + compressed_logprob_emit_at_indel[4-3] +
                  compressed_joint_logprob_transit[1,2] + compressed_logprob_emit_at_indel[6-3] )
                 
        
        # -T-
        # T-C
        # S -> ins T -> del T -> ins C 
        path2 = ( compressed_joint_logprob_transit[3,1] + compressed_logprob_emit_at_indel[6-3] +
                  compressed_joint_logprob_transit[1,2] + compressed_logprob_emit_at_indel[6-3] +
                  compressed_joint_logprob_transit[2,1] + compressed_logprob_emit_at_indel[4-3] )
        
        # T--
        # -TC
        # S -> del T -> ins T -> ins C 
        path3 = ( compressed_joint_logprob_transit[3,2] + compressed_logprob_emit_at_indel[6-3] +
                  compressed_joint_logprob_transit[2,1] + compressed_logprob_emit_at_indel[6-3] +
                  compressed_joint_logprob_transit[1,1] + compressed_logprob_emit_at_indel[4-3] )
        
        # T-
        # TC
        # S -> match (T,T) -> ins C 
        path4 = ( compressed_joint_logprob_transit[3,0] + compressed_joint_logprob_emit_at_match[6-3, 6-3] +
                  compressed_joint_logprob_transit[0,1] + compressed_logprob_emit_at_indel[4-3] )
        
        # -T
        # TC
        # S -> ins T -> match (T, C) 
        path5 = ( compressed_joint_logprob_transit[3,1] + compressed_logprob_emit_at_indel[6-3]+
                  compressed_joint_logprob_transit[1,0] + compressed_joint_logprob_emit_at_match[6-3, 4-3] )
        
        true_cell_1_2 = logsumexp( [path1, path2, path3, path4, path5] )
        
        
        #######################################################################
        ### Predicted scores from 2D forward algo implementation   ############
        #######################################################################
        # transpose transition matrix 
        # old: (T, C_transit_prev, C_transit_curr, S_prev, S_curr)
        # new: (T, C_transit_prev, S_prev, C_transit_curr, S_curr)
        joint_logprob_transit = jnp.transpose(joint_logprob_transit_old_dim_order, (0,1,3,2,4)) # (T, C_transit_prev, S_prev, C_transit_curr, S_curr)
        
        _, pred_grid = forward_fn(unaligned_seqs = unaligned_seqs,
                                  joint_logprob_transit = joint_logprob_transit,
                                  joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                                  logprob_emit_at_indel = logprob_emit_at_indel,
                                  return_full_grid=True) #(K, W, 1, C_S, 1)
        
        pred_grid = np.squeeze(pred_grid) #(K, W, C_S)
        pred_grid = nn.logsumexp(pred_grid, axis=-1) #(K, W)
        
        
        ### K=1
        # W=0; (1,0)
        pred_cell_1_0 = pred_grid[0,0]
        npt.assert_allclose( pred_cell_1_0, true_cell_1_0 )
        
        # W=1;(0,1)
        pred_cell_0_1 = pred_grid[0,1]
        npt.assert_allclose( pred_cell_0_1, true_cell_0_1 )
        
        
        ### K=2
        # W=1; (1,1)
        pred_cell_1_1 = pred_grid[1,0]
        npt.assert_allclose( pred_cell_1_1, true_cell_1_1 )
        
        # W=2; (0,2)
        pred_cell_0_2 = pred_grid[1,1]
        npt.assert_allclose( pred_cell_0_2, true_cell_0_2 )
        
        
        ### K=3
        # W=0; (1,2)
        pred_cell_1_2 = pred_grid[2,0]
        npt.assert_allclose( pred_cell_1_2, true_cell_1_2 )
        
        # padding at (2,2)
        npt.assert_allclose( pred_grid[2,1], jnp.finfo(jnp.float32).min )
        

if __name__ == '__main__':
    unittest.main() 