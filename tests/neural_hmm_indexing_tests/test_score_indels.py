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

from models.neural_hmm_predict.scoring_fns import score_indels


THRESHOLD = 1e-6


class TestScoreIndels(unittest.TestCase):
    def setUp(self):
        a1 = np.array( [[ 3,  3,  4,  5,  6],
                        [43, 43, 43, 43, 43]] ).T
        s1 = np.array(   [3,  3,  3,  3,  3] )[:,None]
        a1 = np.concatenate([a1, s1], axis=-1)
        del s1
        
        a2 = np.array( [[43, 43, 43, 43, 43],
                        [ 3,  3,  4,  5,  6]] ).T
        s2 = np.array(   [2,  2,  2,  2,  2] )[:,None]
        a2 = np.concatenate([a2, s2], axis=-1)
        del s2
        
        a3 = np.array( [[ 3,  3,  3, 0, 0],
                        [43, 43, 43, 0, 0]] ).T
        s3 = np.array(   [3,  3,  3, 0, 0] )[:,None]
        a3 = np.concatenate([a3, s3], axis=-1)
        del s3
        
        a4 = np.array( [[43, 43, 43, 0, 0],
                        [ 3,  3,  3, 0, 0]] ).T
        s4 = np.array(  [ 2,  2,  2, 0, 0] )[:,None]
        a4 = np.concatenate([a4, s4], axis=-1)
        del s4
        
        # B=4
        # L=5
        self.align = np.stack([a1, a2, a3, a4], axis=0) 
    
    
    def _make_one_equl(self):
        return np.array( [.1, .2, .3, .4] )[None,None,:]
    
    def test_one_equl_score_anc(self):
        equl = self._make_one_equl()
        
        true_scores = np.array( [[.1, .1, .2, .3, .4],
                                 [ 0,  0,  0,  0,  0],
                                 [.1, .1, .1,  0,  0],
                                 [ 0,  0,  0,  0,  0]] )
        
        pred_scores = score_indels( true_alignment_without_start = self.align, 
                                    logprob_scoring_vec = equl, 
                                    which_seq = 'anc')
        
        npt.assert_allclose(true_scores, pred_scores)
        
    def test_one_equl_score_desc(self):
        equl = self._make_one_equl()
        
        true_scores = np.array( [[ 0,  0,  0,  0,  0],
                                 [.1, .1, .2, .3, .4],
                                 [ 0,  0,  0,  0,  0],
                                 [.1, .1, .1,  0,  0]] )
        
        pred_scores = score_indels( true_alignment_without_start = self.align, 
                                    logprob_scoring_vec = equl, 
                                    which_seq = 'desc')
        
        npt.assert_allclose(true_scores, pred_scores)
    
    
    def _make_multi_equl(self):
        equl1 = np.array( [ [.1, -1, -1, -1],
                            [.2, -1, -1, -1],
                            [-1, .3, -1, -1],
                            [-1, -1, .4, -1],
                            [-1, -1, -1, .5]] )
        
        equl2 = np.array( [ [.1, -1, -1, -1],
                            [.2, -1, -1, -1],
                            [.3, -1, -1, -1],
                            [-1, -1, -1, -1],
                            [-1, -1, -1, -1]] )
        
        # B=4
        # L_align-1=5
        # A=4
        equl = np.stack([equl1, equl1, equl2, equl2], axis=0)
        return equl
    
    def test_multi_equl_score_anc(self):
        equl = self._make_multi_equl()
        
        true_scores = np.array( [[.1, .2, .3, .4, .5],
                                 [ 0,  0,  0,  0,  0],
                                 [.1, .2, .3,  0,  0],
                                 [ 0,  0,  0,  0,  0]] )
        
        pred_scores = score_indels( true_alignment_without_start = self.align, 
                                    logprob_scoring_vec = equl, 
                                    which_seq = 'anc')
        
        npt.assert_allclose(true_scores, pred_scores)
    
    def test_multi_equl_score_desc(self):
        equl = self._make_multi_equl()
        
        true_scores = np.array( [[ 0,  0,  0,  0,  0],
                                 [.1, .2, .3, .4, .5],
                                 [ 0,  0,  0,  0,  0],
                                 [.1, .2, .3,  0,  0]] )
        
        pred_scores = score_indels( true_alignment_without_start = self.align, 
                                    logprob_scoring_vec = equl, 
                                    which_seq = 'desc')
        
        npt.assert_allclose(true_scores, pred_scores)



if __name__ == '__main__':
    unittest.main()
