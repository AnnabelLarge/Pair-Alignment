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

from models.neural_hmm_predict.scoring_fns import (score_f81_substitutions_marg_over_times,
                                                   score_f81_substitutions_t_per_samp)
THRESHOLD = 1e-6


class TestScoreSubsF81(unittest.TestCase):
    def setUp(self):
        a1 = np.array( [[ 3, 4, 5, 6, 3],
                        [ 3, 4, 5, 6, 3]] ).T
        s1 = np.array(   [1, 1, 1, 1, 1] )[:,None]
        a1 = np.concatenate([a1, s1], axis=-1)
        del s1
        
        a2 = np.array( [[ 3, 4, 5, 6, 3],
                        [ 4, 5, 6, 3, 4]] ).T
        s2 = np.array(   [1, 1, 1, 1, 1] )[:,None]
        a2 = np.concatenate([a2, s2], axis=-1)
        del s2
        
        a3 = np.array( [[3, 3, 4, 0, 0],
                        [3, 4, 4, 0, 0]] ).T
        s3 = np.array(  [1, 1, 1, 0, 0] )[:,None]
        a3 = np.concatenate([a3, s3], axis=-1)
        del s3
        
        # B=3
        # L_align-1=5
        self.align = np.stack([a1, a2, a3], axis=0) 
        del a1, a2, a3

    def test_one_f81_marg_over_times(self):
        f81_t1_match = np.array([.01, .02, .03, .04])
        f81_t1_sub = np.array([.05, .06, .07, .08])
        f81_t1 = np.stack([f81_t1_match, f81_t1_sub], axis=-1)
        f81_t2 = f81_t1 * 10
        f81_mat = np.stack([f81_t1, f81_t2])[:,None,None,...]
        del f81_t1_match, f81_t1_sub, f81_t1, f81_t2
        
        true_scores_1 = np.array( [[0.01, 0.02, 0.03, 0.04, 0.01],
                                   [0.06, 0.07, 0.08, 0.05, 0.06],
                                   [0.01, 0.06, 0.02, 0., 0.]]) 
        true_scores_2 = true_scores_1 * 10
        true_scores = np.stack([true_scores_1, true_scores_2], axis=0)
        del true_scores_1, true_scores_2
        
        pred_scores = score_f81_substitutions_marg_over_times( true_alignment_without_start = self.align,
                                                               logprob_scoring_mat = f81_mat )
        
        npt.assert_allclose(true_scores, pred_scores) 
    
    
    def test_one_f81_t_per_samp(self):
        f81_b1_match = np.array([.01, .02, .03, .04])
        f81_b1_sub = np.array([.05, .06, .07, .08])
        f81_b1 = np.stack([f81_b1_match, f81_b1_sub], axis=-1)
        f81_b2 = f81_b1 * 10
        f81_b3 = f81_b1 * 100
        f81_mat = np.stack([f81_b1, f81_b2, f81_b3])[:,None,...]
        del f81_b1, f81_b2, f81_b3, f81_b1_match, f81_b1_sub
        
        true_scores = np.array( [[0.01, 0.02, 0.03, 0.04, 0.01],
                                 [0.6, 0.7, 0.8, 0.5, 0.6],
                                 [1., 6., 2., 0., 0.]]) 
        
        pred_scores = score_f81_substitutions_t_per_samp( true_alignment_without_start = self.align,
                                                          logprob_scoring_mat = f81_mat )
        
        npt.assert_allclose(true_scores, pred_scores) 
    
    
    def _setup_f81_mat(self):
        # sample 1
        f81_mat_b1_l1 = np.array([ [.1, -1, -1, -1],
                                   [-1, -1, -1, -1]]).T
        
        f81_mat_b1_l2 = np.array([ [-1, .2, -1, -1],
                                   [-1, -1, -1, -1]]).T
        
        f81_mat_b1_l3 = np.array([ [-1, -1, .3, -1],
                                   [-1, -1, -1, -1]]).T
        
        f81_mat_b1_l4 = np.array([ [-1, -1, -1, .4],
                                   [-1, -1, -1, -1]]).T
        
        f81_mat_b1_l5 = np.array([ [.5, -1, -1, -1],
                                   [-1, -1, -1, -1]]).T
        
        f81_b1_mat = np.stack( [f81_mat_b1_l1,
                                f81_mat_b1_l2,
                                f81_mat_b1_l3,
                                f81_mat_b1_l4,
                                f81_mat_b1_l5] )[None,None,...]
        del f81_mat_b1_l1, f81_mat_b1_l2, f81_mat_b1_l3, f81_mat_b1_l4, f81_mat_b1_l5
        
        # sample 2
        f81_mat_b2_l1 = np.array([ [-1, -1, -1, -1],
                                   [-1, .1, -1, -1]]).T
        
        f81_mat_b2_l2 = np.array([ [-1, -1, -1, -1],
                                   [-1, -1, .2, -1]]).T
        
        f81_mat_b2_l3 = np.array([ [-1, -1, -1, -1],
                                   [-1, -1, -1, .3]]).T
        
        f81_mat_b2_l4 = np.array([ [-1, -1, -1, -1],
                                   [.4, -1, -1, -1]]).T
        
        f81_mat_b2_l5 = np.array([ [-1, -1, -1, -1],
                                   [-1, .5, -1, -1]]).T
        
        f81_b2_mat = np.stack( [f81_mat_b2_l1,
                                f81_mat_b2_l2,
                                f81_mat_b2_l3,
                                f81_mat_b2_l4,
                                f81_mat_b2_l5] )[None,None,...]
        del f81_mat_b2_l1, f81_mat_b2_l2, f81_mat_b2_l3, f81_mat_b2_l4, f81_mat_b2_l5
        
        # sample 3
        f81_mat_b3_l1 = np.array([ [.1, -1, -1, -1],
                                   [-1, -1, -1, -1]]).T
        
        f81_mat_b3_l2 = np.array([ [-1, -1, -1, -1],
                                   [-1, .2, -1, -1]]).T
        
        f81_mat_b3_l3 = np.array([ [-1, .3, -1, -1],
                                   [-1, -1, -1, -1]]).T
        
        f81_mat_b3_pad = np.array([ [-1, -1, -1, -1],
                                    [-1, -1, -1, -1]]).T
        
        f81_b3_mat = np.stack( [f81_mat_b3_l1,
                                f81_mat_b3_l2,
                                f81_mat_b3_l3,
                                f81_mat_b3_pad,
                                f81_mat_b3_pad] )[None,None,...]
        
        return (f81_b1_mat,
                f81_b2_mat,
                f81_b3_mat)
    
    def test_multi_f81_marg_over_times(self):
        f81_b1_mat, f81_b2_mat, f81_b3_mat = self._setup_f81_mat()

        # concat to final shape: (T, B, L_align-1, A, 2)
        # T = 2
        # B = 3
        # L_align-1 = 5
        # A = 4
        f81_mat_t1 = np.concatenate( [f81_b1_mat, f81_b2_mat, f81_b3_mat], axis=1 )
        del f81_b1_mat, f81_b2_mat, f81_b3_mat
        
        f81_mat_t2 = f81_mat_t1 * 10
        f81_mat = np.concatenate( [f81_mat_t1, f81_mat_t2], axis=0 )
        del f81_mat_t1, f81_mat_t2
        
        # compare hand-calculation to function output
        true_t1 = np.array([ [.1, .2, .3, .4, .5],
                             [.1, .2, .3, .4, .5],
                             [.1, .2, .3, 0., 0.]] )
        true_t2 = true_t1 * 10
        true_scores = np.stack( [true_t1, true_t2] )
        del true_t1, true_t2
        
        pred_scores = score_f81_substitutions_marg_over_times( true_alignment_without_start = self.align,
                                                               logprob_scoring_mat = f81_mat )
        
        npt.assert_allclose(true_scores, pred_scores)
        
    
    def test_multi_f81_t_per_samp(self):
        f81_b1_mat, f81_b2_mat, f81_b3_mat = self._setup_f81_mat()
        f81_b3_mat = f81_b3_mat * 10 # just to switch things up
        
        # concat to final shape: (B, L_align-1, A, A)
        # B = 3
        # L_align-1 = 5
        # A = 4
        f81_mat = np.concatenate( [f81_b1_mat, f81_b2_mat, f81_b3_mat], axis=0 )[:,0,...]
        del f81_b1_mat, f81_b2_mat, f81_b3_mat
        
        # compare hand-calculation to function output
        true_scores = np.array([ [.1, .2, .3, .4, .5],
                                 [.1, .2, .3, .4, .5],
                                 [1., 2., 3., 0., 0.]] )
        
        pred_scores = score_f81_substitutions_t_per_samp( true_alignment_without_start = self.align,
                                                          logprob_scoring_mat = f81_mat )
        
        npt.assert_allclose(true_scores, pred_scores) 
    
if __name__ == '__main__':
    unittest.main()
