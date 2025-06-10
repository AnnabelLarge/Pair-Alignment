#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 21:00:45 2025

@author: annabel_large


About:
======
Check the F81 matrix two ways:
    1.) by hand
    2.) by making sure that GTR reduces to F81 (GTR already validated)

"""
import jax
from jax import numpy as jnp
import numpy as np

import numpy.testing as npt
import unittest

from models.simple_site_class_predict.emission_models import F81LogProbs

from models.simple_site_class_predict.model_functions import (rate_matrix_from_exch_equl,
                                                              get_cond_logprob_emit_at_match_per_class,
                                                              joint_prob_from_counts)

THRESHOLD = 1e-6


class TestF81(unittest.TestCase):
    def setUp(self):
        self.rate_mult_multiclass = np.array([1, 2, 3]) #(C=3)
        self.equl = np.array([0.1, 0.2, 0.3, 0.4]) #(A=4)
        self.t_array = np.array([0.1, 0.3, 0.5, 0.7, 0.9]) #(T=5)
    
    def test_one_f81_hand_calc(self):
        T = self.t_array.shape[0]
        A = self.equl.shape[0]
        C = 1
        
        ### true
        normed_rate = 1 / ( 1 - np.square(self.equl).sum() )
        cond_prob_true = np.zeros( (T, C, A, A) )
        for t_idx, t in enumerate(self.t_array):
            mat = np.zeros((A,A))
            for i in range(A):
                for j in range(A):
                    pi_j = self.equl[j]
                    
                    if i == j:
                        val = pi_j + (1-pi_j) * np.exp(-normed_rate * t)
                    
                    elif i != j:
                        val = pi_j * (1 - np.exp(-normed_rate*t) )
                    
                    mat[i,j] = val
            
            cond_prob_true[t_idx] = mat
        
        ### by my formula
        my_model = F81LogProbs(config={'num_mixtures': 1},
                               name='mymod')
        

    # one F81: does it match hand calc?
    
    
    # one F81: does reduced GTR match F81?
    
    
    # mixture of F81s: does it match hand calc?
    
    
    # mixture of F81s: does reduced GTR match F81?
    
if __name__ == '__main__':
    unittest.main()