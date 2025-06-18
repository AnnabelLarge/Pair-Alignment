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

from models.neural_hmm_predict.scoring_fns import score_gtr_substitutions


class TestScoreSubsGTR(unittest.TestCase):
    def setUp(self):
        a1 = np.array( [[ 3, 3, 3, 3, 3],
                        [ 3, 4, 5, 6, 3]] ).T
        s1 = np.array(   [1, 1, 1, 1, 1] )[:,None]
        a1 = np.concatenate([a1, s1], axis=-1)
        del s1
        
        a2 = np.array( [[ 3, 4, 5, 6, 3],
                        [ 3, 4, 5, 6, 3]] ).T
        s2 = np.array(   [1, 1, 1, 1, 1] )[:,None]
        a2 = np.concatenate([a2, s2], axis=-1)
        del s2
        
        a3 = np.array( [[3, 3, 3, 0, 0],
                        [4, 5, 6, 0, 0]] ).T
        s3 = np.array(  [1, 1, 1, 0, 0] )[:,None]
        a3 = np.concatenate([a3, s3], axis=-1)
        del s3
        
        # B=3
        # L_align-1=5
        self.align = np.stack([a1, a2, a3], axis=0) 
        del a1, a2, a3
    
    def test_one_gtr_marg_over_times(self):
        # T=2
        # B=3
        # L_align-1=5
        # A=4
        gtr_mat_t1 = np.arange(1, 17).reshape(4, 4) * 0.01
        gtr_mat_t2 = np.arange(1, 17).reshape(4, 4) * 0.1
        gtr_mat = np.stack([gtr_mat_t1, gtr_mat_t2], axis=0)
        gtr_mat = gtr_mat[:, None, None, :, :] #(T, 1, 1, A, A)
        del gtr_mat_t1, gtr_mat_t2
        
        true_scores_1 = np.array( [[0.01, 0.02, 0.03, 0.04, 0.01],
                                    [0.01, 0.06, 0.11, 0.16, 0.01],
                                    [0.02, 0.03, 0.04, 0., 0.]]) 
        true_scores_2 = true_scores_1 * 10
        true_scores = np.stack([true_scores_1, true_scores_2], axis=0)
        del true_scores_1, true_scores_2
        
        pred_scores = score_gtr_substitutions( true_alignment_without_start = self.align,
                                                logprob_scoring_mat = gtr_mat,
                                                unique_time_per_sample = False )
        
        npt.assert_allclose(true_scores, pred_scores) 
    
    def test_one_gtr_t_per_samp(self):
        # B=3
        # L_align-1=5
        # A=4
        gtr_mat_b1 = np.arange(1, 17).reshape(4, 4) * 0.01
        gtr_mat_b2 = np.arange(1, 17).reshape(4, 4) * 0.1
        gtr_mat_b3 = np.arange(1, 17).reshape(4, 4)
        gtr_mat = np.stack([gtr_mat_b1, gtr_mat_b2, gtr_mat_b3], axis=0)
        gtr_mat = gtr_mat[:, None, :, :] #(B, 1, 4, 4)
        del gtr_mat_b1, gtr_mat_b2, gtr_mat_b3
        
        true_scores = np.array( [[0.01, 0.02, 0.03, 0.04, 0.01],
                                  [ 0.1,  0.6,  1.1,  1.6,  0.1],
                                  [  2.,   3.,   4.,   0.,   0.]])
        
        pred_scores = score_gtr_substitutions( true_alignment_without_start = self.align,
                                                logprob_scoring_mat = gtr_mat,
                                                unique_time_per_sample = True )
        
        npt.assert_allclose(true_scores, pred_scores) 
    
    def _setup_gtr_mat(self):
        # sample 1
        gtr_mat_b1_l1 = np.array([ [.1, -1, -1, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1]])
        
        gtr_mat_b1_l2 = np.array([ [-1, .2, -1, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1]])
        
        gtr_mat_b1_l3 = np.array([ [-1, -1, .3, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1]])
        
        gtr_mat_b1_l4 = np.array([ [-1, -1, -1, .4],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1]])
        
        gtr_mat_b1_l5 = np.array([ [.5, -1, -1, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1]])
        
        gtr_b1_mat = np.stack( [gtr_mat_b1_l1,
                                gtr_mat_b1_l2,
                                gtr_mat_b1_l3,
                                gtr_mat_b1_l4,
                                gtr_mat_b1_l5] )[None,None,...]
        del gtr_mat_b1_l1, gtr_mat_b1_l2, gtr_mat_b1_l3, gtr_mat_b1_l4, gtr_mat_b1_l5
        
        # sample 2
        gtr_mat_b2_l1 = np.array([ [.1, -1, -1, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1]])
        
        gtr_mat_b2_l2 = np.array([ [-1, -1, -1, -1],
                                   [-1, .2, -1, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1]])
        
        gtr_mat_b2_l3 = np.array([ [-1, -1, -1, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, .3, -1],
                                   [-1, -1, -1, -1]])
        
        gtr_mat_b2_l4 = np.array([ [-1, -1, -1, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, .4]])
        
        gtr_mat_b2_l5 = np.array([ [.5, -1, -1, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1]])
        
        gtr_b2_mat = np.stack( [gtr_mat_b2_l1,
                                gtr_mat_b2_l2,
                                gtr_mat_b2_l3,
                                gtr_mat_b2_l4,
                                gtr_mat_b2_l5] )[None,None,...]
        del gtr_mat_b2_l1, gtr_mat_b2_l2, gtr_mat_b2_l3, gtr_mat_b2_l4, gtr_mat_b2_l5
        
        # sample 3
        gtr_mat_b3_l1 = np.array([ [-1, .1, -1, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1]])
        
        gtr_mat_b3_l2 = np.array([ [-1, -1, .2, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1]])
        
        gtr_mat_b3_l3 = np.array([ [-1, -1, -1, .3],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1],
                                   [-1, -1, -1, -1]])
        
        gtr_mat_b3_pad = np.array([ [-1, -1, -1, -1],
                                    [-1, -1, -1, -1],
                                    [-1, -1, -1, -1],
                                    [-1, -1, -1, -1]])
        
        gtr_b3_mat = np.stack( [gtr_mat_b3_l1,
                                gtr_mat_b3_l2,
                                gtr_mat_b3_l3,
                                gtr_mat_b3_pad,
                                gtr_mat_b3_pad] )[None,None,...]
        
        return (gtr_b1_mat,
                gtr_b2_mat,
                gtr_b3_mat)
    
    def test_multi_gtr_marg_over_times(self):
        gtr_b1_mat, gtr_b2_mat, gtr_b3_mat = self._setup_gtr_mat()

        # concat to final shape: (T, B, L_align-1, A, A)
        # T = 2
        # B = 3
        # L_align-1 = 5
        # A = 4
        gtr_mat_t1 = np.concatenate( [gtr_b1_mat, gtr_b2_mat, gtr_b3_mat], axis=1 )
        del gtr_b1_mat, gtr_b2_mat, gtr_b3_mat
        
        gtr_mat_t2 = gtr_mat_t1 * 10
        gtr_mat = np.concatenate( [gtr_mat_t1, gtr_mat_t2], axis=0 )
        del gtr_mat_t1, gtr_mat_t2
        
        # compare hand-calculation to function output
        true_t1 = np.array([ [.1, .2, .3, .4, .5],
                              [.1, .2, .3, .4, .5],
                              [.1, .2, .3, 0., 0.]] )
        true_t2 = true_t1 * 10
        true_scores = np.stack( [true_t1, true_t2] )
        del true_t1, true_t2
        
        pred_scores = score_gtr_substitutions( true_alignment_without_start = self.align,
                                                logprob_scoring_mat = gtr_mat,
                                                unique_time_per_sample = False )
        
        npt.assert_allclose(true_scores, pred_scores)

    def test_multi_gtr_t_per_samp(self):
        gtr_b1_mat, gtr_b2_mat, gtr_b3_mat = self._setup_gtr_mat()
        gtr_b3_mat = gtr_b3_mat * 10 # just to switch things up
        
        # concat to final shape: (B, L_align-1, A, A)
        # B = 3
        # L_align-1 = 5
        # A = 4
        gtr_mat = np.concatenate( [gtr_b1_mat, gtr_b2_mat, gtr_b3_mat], axis=0 )[:,0,...]
        del gtr_b1_mat, gtr_b2_mat, gtr_b3_mat
        
        # compare hand-calculation to function output
        true_scores = np.array([ [.1, .2, .3, .4, .5],
                                  [.1, .2, .3, .4, .5],
                                  [1., 2., 3., 0., 0.]] )
        
        pred_scores = score_gtr_substitutions( true_alignment_without_start = self.align,
                                                logprob_scoring_mat = gtr_mat,
                                                unique_time_per_sample = True )
        
        npt.assert_allclose(true_scores, pred_scores)  



if __name__ == '__main__':
    unittest.main()