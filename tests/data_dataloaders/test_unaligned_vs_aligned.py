#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 13:32:33 2025

@author: annabel
"""
import numpy as np

import numpy.testing as npt
import unittest


class TestUnalignedVsAligned(unittest.TestCase):
    """
    Make sure that aligned sequences match their unaligned counterparts, after
        removing gap tokens
    
    """
    def setUp(self):
        self.data_dir = 'example_data'
        self.prefix = 'sevenSamp'
    
    def _extract_unaligned(self, 
                           vec, 
                           max_len):
        # filter out padding and gap chars
        vec = vec[ (vec != 0) & (vec != 43) ]
        
        # pad to max length
        vec = np.pad(vec,
                     pad_width=(0, max_len - vec.shape[0]),
                     mode='constant',
                     constant_values=0)
        
        return vec
        
    def test_dset(self,):
        with open(f'{self.data_dir}/{self.prefix}_aligned_mats.npy', 'rb') as f:
            aligned_mats = np.load(f)
            
        B = aligned_mats.shape[0]
        
        with open(f'{self.data_dir}/{self.prefix}_seqs_unaligned.npy', 'rb') as f:
            true_unaligned = np.load(f)

        L_seq = true_unaligned.shape[1]

        pred_unaligned = np.zeros( true_unaligned.shape )
        
        for b in range(B):
            pred_anc = self._extract_unaligned( vec = aligned_mats[b, :, 0],
                                                max_len = L_seq )
            
            pred_desc = self._extract_unaligned( vec = aligned_mats[b, :, 1],
                                                 max_len = L_seq )
            
            pred_unaligned[b,:,0] = pred_anc
            pred_unaligned[b,:,1] = pred_desc
        
        npt.assert_allclose( pred_unaligned, true_unaligned )
        
                
if __name__ == '__main__':
    unittest.main()         
                
                
                
        
