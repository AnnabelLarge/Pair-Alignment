#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 16:03:51 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
import numpy as np

import numpy.testing as npt
import unittest

from models.neural_hmm_predict.scoring_fns import score_transitions
THRESHOLD = 1e-6


class TestScoreTransitions(unittest.TestCase):
    def setUp(self):
        state = np.array([[4, 1, 2, 3, 5],
                          [4, 1, 1, 1, 5],
                          [4, 1, 5, 0, 0]])
        prev = state[:, :-1]
        curr = state[:, 1:]
        
        self.staggered_alignment_state = np.stack( [prev, curr], axis=-1 )
        
    def test_one_trans_marg_over_times(self):
        trans_mat_t1 = np.arange(1, 17).reshape(4, 4) * 0.01
        trans_mat_t2 = np.arange(1, 17).reshape(4, 4) * 0.1
        trans_mat = np.stack([trans_mat_t1, trans_mat_t2], axis=0)
        trans_mat = trans_mat[:, None, None, :, :] #(T, 1, 1, A, A)
        del trans_mat_t1, trans_mat_t2
        
        true_scores_t1 = np.array( [[0.13, 0.02, 0.07, 0.12],
                                    [0.13, 0.01, 0.01, 0.04],
                                    [0.13, 0.04,    0,    0]] ) 
        true_scores_t2 = true_scores_t1 * 10
        true_scores = np.stack([true_scores_t1, true_scores_t2], axis=0)
        del true_scores_t1, true_scores_t2
        
        pred_scores = score_transitions( staggered_alignment_state = self.staggered_alignment_state,
                                         logprob_trans_mat = trans_mat,
                                         unique_time_per_sample = False,
                                         padding_idx=0)
        
        npt.assert_allclose(true_scores, pred_scores)
    
    def test_one_trans_t_per_samp(self):
        trans_mat_b1 = np.arange(1, 17).reshape(4, 4) * 0.01
        trans_mat_b2 = np.arange(1, 17).reshape(4, 4) * 0.1
        trans_mat_b3 = np.arange(1, 17).reshape(4, 4)
        trans_mat = np.stack([trans_mat_b1, trans_mat_b2, trans_mat_b3], axis=0)
        trans_mat = trans_mat[:, None, :, :]
        del trans_mat_b1, trans_mat_b2, trans_mat_b3
        
        true_scores = np.array( [[0.13, 0.02, 0.07, 0.12],
                                 [1.3, 0.1, 0.1, 0.4],
                                 [13., 4.,    0,    0]] ) 
        
        pred_scores = score_transitions( staggered_alignment_state = self.staggered_alignment_state,
                                         logprob_trans_mat = trans_mat,
                                         unique_time_per_sample = True,
                                         padding_idx=0)
        
        npt.assert_allclose(true_scores, pred_scores) 
    
    
    def _setup_trans_mat(self):
        # sample 1
        trans_mat_b1_l1 = np.array([ [-1, -1, -1, -1],
                                     [-1, -1, -1, -1],
                                     [-1, -1, -1, -1],
                                     [.1, -1, -1, -1]])
        
        trans_mat_b1_l2 = np.array([ [-1, .2, -1, -1],
                                     [-1, -1, -1, -1],
                                     [-1, -1, -1, -1],
                                     [-1, -1, -1, -1]])
        
        trans_mat_b1_l3 = np.array([ [-1, -1, -1, -1],
                                     [-1, -1, .3, -1],
                                     [-1, -1, -1, -1],
                                     [-1, -1, -1, -1]])
        
        trans_mat_b1_l4 = np.array([ [-1, -1, -1, -1],
                                     [-1, -1, -1, -1],
                                     [-1, -1, -1, .4],
                                     [-1, -1, -1, -1]])
        
        b1_mat = np.stack( [trans_mat_b1_l1,
                            trans_mat_b1_l2,
                            trans_mat_b1_l3,
                            trans_mat_b1_l4] )[None,None,...]
        del trans_mat_b1_l1, trans_mat_b1_l2, trans_mat_b1_l3, trans_mat_b1_l4
        
        # sample 2
        trans_mat_b2_l1 = np.array([ [-1, -1, -1, -1],
                                     [-1, -1, -1, -1],
                                     [-1, -1, -1, -1],
                                     [.1, -1, -1, -1]])
        
        trans_mat_b2_l2 = np.array([ [.2, -1, -1, -1],
                                     [-1, -1, -1, -1],
                                     [-1, -1, -1, -1],
                                     [-1, -1, -1, -1]])
        
        trans_mat_b2_l3 = np.array([ [.3, -1, -1, -1],
                                     [-1, -1, -1, -1],
                                     [-1, -1, -1, -1],
                                     [-1, -1, -1, -1]])
        
        trans_mat_b2_l4 = np.array([ [-1, -1, -1, .4],
                                     [-1, -1, -1, -1],
                                     [-1, -1, -1, -1],
                                     [-1, -1, -1, -1]])
        
        b2_mat = np.stack( [trans_mat_b2_l1,
                            trans_mat_b2_l2,
                            trans_mat_b2_l3,
                            trans_mat_b2_l4] )[None,None,...]
        del trans_mat_b2_l1, trans_mat_b2_l2, trans_mat_b2_l3, trans_mat_b2_l4
        
        # sample 3
        trans_mat_b3_l1 = np.array([ [-1, -1, -1, -1],
                                     [-1, -1, -1, -1],
                                     [-1, -1, -1, -1],
                                     [.1, -1, -1, -1]])
        
        trans_mat_b3_l2 = np.array([ [-1, -1, -1, .2],
                                     [-1, -1, -1, -1],
                                     [-1, -1, -1, -1],
                                     [-1, -1, -1, -1]])
        
        trans_mat_b3_pad = np.array([ [-1, -1, -1, -1],
                                      [-1, -1, -1, -1],
                                      [-1, -1, -1, -1],
                                      [-1, -1, -1, -1]])
        
        b3_mat = np.stack( [trans_mat_b3_l1,
                            trans_mat_b3_l2,
                            trans_mat_b3_pad,
                            trans_mat_b3_pad] )[None,None,...]
        
        return (b1_mat,
                b2_mat,
                b3_mat)
    
    def test_multi_trans_marg_over_times(self):
        trans_b1_mat, trans_b2_mat, trans_b3_mat = self._setup_trans_mat()

        # concat to final shape: (T, B, L_align-1, S, S)
        # T = 2
        # B = 3
        # L_align-1 = 4
        # A = 4
        trans_mat_t1 = np.concatenate( [trans_b1_mat, trans_b2_mat, trans_b3_mat], axis=1 )
        del trans_b1_mat, trans_b2_mat, trans_b3_mat
        
        trans_mat_t2 = trans_mat_t1 * 10
        trans_mat = np.concatenate( [trans_mat_t1, trans_mat_t2], axis=0 )
        del trans_mat_t1, trans_mat_t2
        
        # compare hand-calculation to function output
        true_t1 = np.array([ [.1, .2, .3, .4],
                             [.1, .2, .3, .4],
                             [.1, .2, 0., 0.]] )
        true_t2 = true_t1 * 10
        true_scores = np.stack( [true_t1, true_t2] )
        del true_t1, true_t2
        
        pred_scores = score_transitions( staggered_alignment_state = self.staggered_alignment_state,
                                         logprob_trans_mat = trans_mat,
                                         unique_time_per_sample = False,
                                         padding_idx=0)
        
        npt.assert_allclose(true_scores, pred_scores)
    
    
    def test_multi_trans_t_per_samp(self):
        trans_b1_mat, trans_b2_mat, trans_b3_mat = self._setup_trans_mat()
        trans_b3_mat = trans_b3_mat * 10 # just to switch things up
        
        # concat to final shape: (B, L_align-1, S, S)
        # B = 3
        # L_align-1 = 4
        # A = 4
        trans_mat = np.concatenate( [trans_b1_mat, trans_b2_mat, trans_b3_mat], axis=0 )[:,0,...]
        del trans_b1_mat, trans_b2_mat, trans_b3_mat
        
        # compare hand-calculation to function output
        true_scores = np.array([ [.1, .2, .3, .4],
                                 [.1, .2, .3, .4],
                                 [1., 2., 0., 0.]] )
        
        pred_scores = score_transitions( staggered_alignment_state = self.staggered_alignment_state,
                                         logprob_trans_mat = trans_mat,
                                         unique_time_per_sample = True,
                                         padding_idx=0)
        
        npt.assert_allclose(true_scores, pred_scores) 
    
        
if __name__ == '__main__':
    unittest.main()
