#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 09:47:56 2025

@author: annabel_large

About:
======
Confirm that scoring function for aggregated counts is working as expected

"""
import jax
import numpy as np

import numpy.testing as npt
import unittest

from models.simple_site_class_predict.model_functions import (lse_over_match_logprobs_per_mixture,
                                                              joint_prob_from_counts)



THRESHOLD = 1e-6


def generate_fake_alignment():
    """
    PURPOSE: some fake alignments to score (not the same format as input 
        data though)
    """
    fake_aligns = np.array([[0,0],
                            [0,1],
                            [0,2],
                            [0,3],
                            [0,3]]) #(L, 2)
    
    match_counts = np.eye(4)
    match_counts[-1,-1] += 1
    match_counts = match_counts[None,...] #(B, 4, 4)
    
    seq_len = match_counts.shape[0]
    return fake_aligns, match_counts, seq_len


class TestScoreAlignment(unittest.TestCase):
    """
    B: batch (samples)
    C: hidden site classes
    K: rate multipliers
    T: branch lengths (time)
    A: alphabet
    
    About
    ------
    test _score_alignment with fake scoring matrices and inputs, compare 
      against hand-calculated scores
    
    """
    def test_score_alignment_one_component(self):
        fake_aligns, match_counts, seq_len = generate_fake_alignment()

        # params to work with
        log_joint = np.array([[ 1, -1, -1, -1],
                              [-1,  2, -1, -1],
                              [-1, -1,  3, -1],
                              [-1, -1, -1,  4]])[None,:,:] #(T, 4, 4)
        P_emit = 0.995
        
        # calculate this by hand
        true_scores = (1 + 2 + 3 + 4*2) + (5 * np.log(0.995) + np.log(0.005) )
        
        B = 1
        T = 1
        A = 4
        
        # score
        fake_batch = (match_counts, 
                      np.zeros((B,A)),
                      np.zeros((B,A)),
                      np.zeros((B, 4, 4)))
        
        scoring_matrices_dict = {'joint_logprob_emit_at_match': log_joint,
                                  'all_transit_matrices': 
                                      {'joint': np.log(np.array([P_emit, 1 - P_emit]))
                                      } 
                                }
        
        out = joint_prob_from_counts( batch = fake_batch,
                                      times_from = 'geometric',
                                      score_indels = False,
                                      scoring_matrices_dict = scoring_matrices_dict,
                                      t_array = np.zeros(1,),
                                      exponential_dist_param = None,
                                      norm_reported_loss_by = None )
        
        pred_scores = -out['joint_neg_logP']
        
        npt.assert_allclose(true_scores, pred_scores, atol=THRESHOLD)
    
    
    def test_score_alignment_mixture(self):
        fake_aligns, match_counts, seq_len = generate_fake_alignment()

        # params to work with
        joint_c1_k1 = np.array([[1, 1, 1, 1],
                                [1, 2, 1, 1],
                                [1, 1, 3, 1],
                                [1, 1, 1, 4]])[None,None,None,...]
        
        joint_c2_k1 = np.array([[5, 1, 1, 1],
                                [1, 6, 1, 1],
                                [1, 1, 7, 1],
                                [1, 1, 1, 8]])[None,None,None,...]
        
        joint_c1_k2 = np.array([[9, 1, 1, 1],
                                [1, 10, 1, 1],
                                [1, 1, 11, 1],
                                [1, 1, 1, 12]])[None,None,None,...]
        
        joint_c2_k2 = np.array([[13, 1, 1, 1],
                                [1, 14, 1, 1],
                                [1, 1, 15, 1],
                                [1, 1, 1, 16]])[None,None,None,...]
        
        row1 = np.concatenate([joint_c1_k1, joint_c2_k1], axis=1) 
        row2 = np.concatenate([joint_c1_k2, joint_c2_k2], axis=1) 
        joint = np.concatenate( [row1, row2], axis=2 ) #(T, C, K, A, A)
        del joint_c1_k1, joint_c2_k1, joint_c1_k2, joint_c2_k2, row1, row2
        
        class_probs = np.array([0.4, 0.6]) #(C,)
        rate_mix_probs = np.array([[0.2, 0.8],
                                   [0.3, 0.7]]) #(C,K)
        P_emit = 0.995
        
        B = 1
        T = 1
        C = 2
        K = 2
        A = 4
        
        # calculate this by hand
        len_score = (0.995)**5 * 0.005
        emit_score = ( (1*0.4*0.2 + 5*0.6*0.3 +  9*0.4*0.8 + 13*0.6*0.7 ) *
                       (2*0.4*0.2 + 6*0.6*0.3 + 10*0.4*0.8 + 14*0.6*0.7) *
                       (3*0.4*0.2 + 7*0.6*0.3 + 11*0.4*0.8 + 15*0.6*0.7) *
                       (4*0.4*0.2 + 8*0.6*0.3 + 12*0.4*0.8 + 16*0.6*0.7)**2 )
        true_scores = emit_score * len_score
        true_log_scores = np.log(true_scores) #(T,B)
        
        # score
        pred_scoring_matrix = lse_over_match_logprobs_per_mixture(log_class_probs = np.log(class_probs),
                                                                  log_rate_mult_probs = np.log(rate_mix_probs),
                                                                  logprob_emit_at_match_per_mixture = np.log(joint)) #(T,A,A)
        
        fake_batch = (match_counts, 
                      np.zeros((B,A)),
                      np.zeros((B,A)),
                      np.zeros((B, 4, 4)))
        
        scoring_matrices_dict = {'joint_logprob_emit_at_match': pred_scoring_matrix,
                                  'all_transit_matrices': 
                                      {'joint': np.log(np.array([P_emit, 1 - P_emit]))
                                      } 
                                }
        
        out = joint_prob_from_counts( batch = fake_batch,
                                      times_from = 'geometric',
                                      score_indels = False,
                                      scoring_matrices_dict = scoring_matrices_dict,
                                      t_array = np.zeros(1,),
                                      exponential_dist_param = None,
                                      norm_reported_loss_by = None )
        
        pred_scores = -out['joint_neg_logP']
        
        npt.assert_allclose(true_log_scores, pred_scores, atol=THRESHOLD)

if __name__ == '__main__':
    unittest.main()
