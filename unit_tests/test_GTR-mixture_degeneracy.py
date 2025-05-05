#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 09:47:56 2025

@author: annabel_large


ABOUT:
======
C mixtures of the same model should achieve the same loglikelihood as just
  one copy of the model

"""
import jax
import numpy as np

import numpy.testing as npt
import unittest

from unit_tests.req_files.data_processing import (str_aligns_to_tensor,
                                                  summarize_alignment)
from models.simple_site_class_predict.emission_models import (_rate_matrix_from_exch_equl,
                                                              get_cond_logprob_emit_at_match_per_class,
                                                              get_joint_logprob_emit_at_match_per_class)
from models.simple_site_class_predict.PairHMM_indp_sites import (_score_alignment,
                                                                 _lse_over_match_logprobs_per_class)




THRESHOLD = 1e-6
NUM_CLASSES = 3



def generate_fake_alignments():
    """
    PURPOSE: some fake alignments to score
    """
    
    
    return fake_aligns, match_counts


def generate_fake_params():
    """
    PURPOSE: parameters for scoring sequences
    """
    
    
    out = {'Q': Q, #(C=1,A,A)
           'Q_copied': Q_copied, #(C=NUM_CLASSES,A,A)
           'class_probs': class_probs, #(C=NUM_CLASSES)
           't_array': t_array} #(T,)





class TestAlignmentLoglikeGTRMixture(unittest.TestCase):
    """
    SUBSTITUTION PROCESS SCORING TEST 7
    
    B: batch (samples)
    L: length (number of alignment columns)
    C: hidden site classes
    T: branch lengths (time)
    A: alphabet
    
    About
    ------
    C mixtures of the same latent site clas mixture model should achieve the  
      same loglikelihood as just one copy of the model
    
    """
    def test_score_alignment(self):
        ### generate fake alignments
        fake_aligns = [ ('AC-A','D-ED'),
                        ('D-ED','AC-A'),
                        ('ECDAD','-C-A-'),
                        ('-C-A-','ECDAD') ]
        
        fake_aligns =  str_aligns_to_tensor(fake_aligns) #(B,L,3)
            
        vmapped_summarize_alignment = jax.vmap(summarize_alignment, 
                                               in_axes=0, 
                                               out_axes=0)
        counts =  vmapped_summarize_alignment( fake_aligns )
        match_counts = counts['match_counts'][:, :4, :4] #(B,A,A)
        
        
        ### generate fake parameters
        exchangeabilities = np.array([[0, 1, 2, 3],
                                      [1, 0, 4, 5],
                                      [2, 4, 0, 6],
                                      [3, 5, 6, 0]]) #(A,A)
        
        # original model; one component
        equilibrium_distributions = np.array([0.1, 0.2, 0.3, 0.4]) #(A,)
        Q = _rate_matrix_from_exch_equl(exchangeabilities,
                                        equilibrium_distributions[None,...],
                                        norm=True) #(C=1,A,A)
        
        # mixture model; NUM_CLASSES copies of the same component
        equilibrium_distributions_copied = np.repeat(equilibrium_distributions[None,...], 
                                                     NUM_CLASSES,
                                                     axis=0) #(C=NUM_CLASSES,A)
        Q_copied = _rate_matrix_from_exch_equl(exchangeabilities,
                                        equilibrium_distributions_copied,
                                        norm=True) #(C=NUM_CLASSES,A,A)
        
        class_probs = np.array([1/NUM_CLASSES] * NUM_CLASSES) #(C=NUM_CLASSES)
        t_array = np.array( [0.3, 1.0, 0.2] ) #(T)
        
        
        ### dims
        B = fake_aligns.shape[0]
        L = fake_aligns.shape[1]
        C = NUM_CLASSES
        T = t_array.shape[0]
        A = Q.shape[1]
        
        
        ### score with original model
        log_cond,_ = get_cond_logprob_emit_at_match_per_class(t_array = t_array,
                                                              scaled_rate_mat_per_class = Q) #(T,C=1,A,A)
        log_joint = get_joint_logprob_emit_at_match_per_class(cond_logprob_emit_at_match_per_class = log_cond,
                                                              log_equl_dist_per_class = np.log(equilibrium_distributions[None,:])) #(T,C=1,A,A)
        del Q, log_cond
        
        original_scores = _score_alignment(subCounts = match_counts,
                                           insCounts = np.zeros((B,A)),
                                           delCounts = np.zeros((B,A)),
                                           transCounts = np.zeros((B, 4, 4)),
                                           logprob_emit_at_match = log_joint[:,0,...],
                                           logprob_emit_at_indel = np.zeros((A)),
                                           transit_mat = np.zeros((T,4,4)))  #(T,B)
        
        
        ### score with mixture of same model
        log_cond_mix,_ = get_cond_logprob_emit_at_match_per_class(t_array = t_array,
                                                              scaled_rate_mat_per_class = Q_copied) #(T,C=NUM_CLASSES,A,A)
        log_joint_mix = get_joint_logprob_emit_at_match_per_class(cond_logprob_emit_at_match_per_class = log_cond_mix,
                                                              log_equl_dist_per_class = np.log(equilibrium_distributions_copied)) #(T,C=NUM_CLASSES,A,A)
        del Q_copied, log_cond_mix
        mix_scoring_matrix = _lse_over_match_logprobs_per_class(log_class_probs = np.log(class_probs),
                                               joint_logprob_emit_at_match_per_class = log_joint_mix) #(T,A,A)
        scores_with_mixture = _score_alignment(subCounts = match_counts,
                                               insCounts = np.zeros((B,A)),
                                               delCounts = np.zeros((B,A)),
                                               transCounts = np.zeros((B, 4, 4)),
                                               logprob_emit_at_match = mix_scoring_matrix,
                                               logprob_emit_at_indel = np.zeros((A)),
                                               transit_mat = np.zeros((T,4,4))) #(T,B)
        
        npt.assert_allclose(original_scores, scores_with_mixture, atol=THRESHOLD)

if __name__ == '__main__':
    unittest.main()
