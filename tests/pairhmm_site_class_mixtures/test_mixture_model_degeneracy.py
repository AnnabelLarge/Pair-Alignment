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
from scipy.special import softmax
from functools import partial

import numpy.testing as npt
import unittest

from tests.data_processing import (str_aligns_to_tensor,
                                   summarize_alignment)
from models.simple_site_class_predict.model_functions import (rate_matrix_from_exch_equl,
                                                              cond_logprob_emit_at_match_per_mixture,
                                                              joint_logprob_emit_at_match_per_mixture,
                                                              lse_over_match_logprobs_per_mixture,
                                                              joint_prob_from_counts)

THRESHOLD = 1e-6
NUM_CLASSES = 3
NUM_RATE_MULTS = 4


class TestMixtureModelDegeneracy(unittest.TestCase):
    """
    B: batch (samples)
    L: length (number of alignment columns)
    C_sites: hidden site classes
    K: rate multipliers
    T: branch lengths (time)
    A: alphabet
    
    About
    ------
    mixtures of the same latent site clas mixture model should achieve the  
      same loglikelihood as just one copy of the model
    
    """
    def test_score_alignment_mix(self):
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
        Q = rate_matrix_from_exch_equl(exchangeabilities,
                                       equilibrium_distributions[None,None,...],
                                       norm=True)[None,...] #(C_fr=1, C_sites=1, K, A, A)
        
        # mixture model; NUM_CLASSES copies of the same component
        equilibrium_distributions_copied = np.repeat( equilibrium_distributions[None,:], NUM_CLASSES, axis=0)[None,...] #(C_fr=1, C_sites, A, A)
        out_shape = (1, NUM_CLASSES, NUM_RATE_MULTS, Q.shape[-2], Q.shape[-1])
        Q_copied = np.broadcast_to( Q, out_shape )
        
        class_probs = np.array([1/NUM_CLASSES] * NUM_CLASSES)[None,:] #(C_fr=1, C_sites)
        rate_mult_probs = np.array( [ (1/NUM_CLASSES) * (1/NUM_RATE_MULTS) ] * (NUM_CLASSES * NUM_RATE_MULTS) )
        rate_mult_probs = np.reshape( rate_mult_probs, (NUM_CLASSES, NUM_RATE_MULTS) )[None,...] #(C_fr=1, C_sites, K)
        
        # other parameters
        P_emit = 0.995
        t_array = np.array( [0.3, 1.0, 0.2, 0.2, 0.1, 0.6] ) #(T)
        
        
        ### dims
        B = fake_aligns.shape[0]
        L = fake_aligns.shape[1]
        C = NUM_CLASSES
        T = t_array.shape[0]
        A = Q.shape[1]
        
        
        ### assemble fake batch for both models, common scoring function
        fake_batch = (match_counts, 
                      np.zeros((B,A)),
                      np.zeros((B,A)),
                      np.zeros((B, 4, 4)))
        
        common_scoring_fn = partial( joint_prob_from_counts,
                                     batch = fake_batch,
                                     times_from = 'geometric',
                                     score_indels = False,
                                     t_array = t_array,
                                     exponential_dist_param = np.array([1.1]),
                                     norm_reported_loss_by = None )
        
        ### score with original model
        log_cond = cond_logprob_emit_at_match_per_mixture(t_array = t_array,
                                                          scaled_rate_mat_per_mixture = Q) #(T,C_fr=1,C_sites=1,K,A,A)
        log_joint = joint_logprob_emit_at_match_per_mixture(cond_logprob_emit_at_match_per_mixture = log_cond,
                                                            log_equl_dist_per_mixture = np.log(equilibrium_distributions[None,None,:])) #(T,C_fr=1,C_sites=1,K,A,A)
        del Q, log_cond
        
        scoring_matrices_dict = {'joint_logprob_emit_at_match': log_joint[:,0,0,0,...],
                                 'all_transit_matrices': 
                                     {'joint': np.log(np.array([P_emit, 1 - P_emit]))
                                      } 
                                }
        out = common_scoring_fn( scoring_matrices_dict = scoring_matrices_dict )
        original_scores = out['joint_neg_logP']
        
        
        ### score with mixture of same model
        log_cond_mix = cond_logprob_emit_at_match_per_mixture(t_array = t_array,
                                                              scaled_rate_mat_per_mixture = Q_copied) #(T,C_fr=1,C_sites,K,A,A)
        log_joint_mix = joint_logprob_emit_at_match_per_mixture(cond_logprob_emit_at_match_per_mixture = log_cond_mix,
                                                              log_equl_dist_per_mixture = np.log(equilibrium_distributions_copied)) #(T,C_fr=1,C_sites,K,A,A)
        del Q_copied, log_cond_mix
        
        mix_scoring_matrix = lse_over_match_logprobs_per_mixture(log_site_class_probs = np.log(class_probs),
                                                                  log_rate_mult_probs = np.log(rate_mult_probs),
                                                                  logprob_emit_at_match_per_mixture = log_joint_mix) #(T,C_fr=1,A,A)
        
        mixture_scoring_matrices_dict = {'joint_logprob_emit_at_match': mix_scoring_matrix[:,0,...],
                                          'all_transit_matrices': 
                                              {'joint': np.log(np.array([P_emit, 1 - P_emit]))
                                              } 
                                        }
        out = common_scoring_fn( scoring_matrices_dict = mixture_scoring_matrices_dict )
        scores_with_mixture = out['joint_neg_logP']
        
        npt.assert_allclose(original_scores, scores_with_mixture, atol=THRESHOLD)

if __name__ == '__main__':
    unittest.main()
