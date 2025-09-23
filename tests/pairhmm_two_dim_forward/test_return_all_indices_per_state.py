#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 14:49:03 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
import numpy as np

import numpy.testing as npt
import unittest

from models.simple_site_class_predict.marg_over_alignments_forward_fns import index_all_classes_one_state
                                                                             

class TestReturnAllIndicesPerState(unittest.TestCase):
    def test_this(self):
        C = 2
        
        ### true
        true_match_coords = []
        true_ins_coords = []
        true_del_coords = []
        
        state = ['M', 'I', 'D'] * C
        for i,s in enumerate(state):
            # match
            if s == 'M':
                true_match_coords.append(i)
            
            # ins
            elif s == 'I':
                true_ins_coords.append(i)
            
            # del
            elif s == 'D':
                true_del_coords.append(i)
        
        true_match_coords = np.array(true_match_coords)
        true_ins_coords = np.array(true_ins_coords)
        true_del_coords = np.array(true_del_coords)
        
        
        ### by my function
        # match
        pred_match_coords = index_all_classes_one_state(state_idx = 0,
                                                        num_transit_classes = C)
        npt.assert_allclose( true_match_coords, pred_match_coords )
        del true_match_coords, pred_match_coords
        
        # ins
        pred_ins_coords = index_all_classes_one_state(state_idx = 1,
                                                      num_transit_classes = C)
        npt.assert_allclose( true_ins_coords, pred_ins_coords )
        del true_ins_coords, pred_ins_coords
        
        # del
        pred_del_coords = index_all_classes_one_state(state_idx = 2,
                                                      num_transit_classes = C)
        npt.assert_allclose( true_del_coords, pred_del_coords )

if __name__ == '__main__':
    unittest.main()