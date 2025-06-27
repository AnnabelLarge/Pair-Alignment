#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 13:19:03 2025

@author: annabel
"""
import numpy as np

import numpy.testing as npt
import unittest


class TestAlignmentIndexEncoding(unittest.TestCase):
    """
    Make sure that index encoding matches given alignment
    
    
    note: iterates over every sample in the batch, every position in
        the alignement; this is kind of slow
    """
    def setUp(self):
        self.data_dir = 'example_data'
        self.prefix = 'sevenSamp'
    
    def test_dset(self,):
        with open(f'{self.data_dir}/{self.prefix}_aligned_mats.npy', 'rb') as f:
            aligned_mats = np.load(f)
        
        B = aligned_mats.shape[0]
        L = aligned_mats.shape[1]
        
        # calculate true indices by loop
        true_m = np.ones( (B, L) ) * -9
        true_n = np.ones( (B, L) ) * -9
        
        # indices always start with (m=1, n=0)
        true_m[:, 0] = 1
        true_n[:, 0] = 0
        
        for b in range(B):
            for l in range(1, L):
                m = true_m[b, l-1]
                n = true_n[b, l-1]
                
                anc_tok = aligned_mats[b,l,0]
                desc_tok = aligned_mats[b,l,1]
                
                # padding
                if anc_tok == 0:
                    break
                
                # end
                if (anc_tok == 2) and (desc_tok == 2):
                    break
                
                # match
                elif (anc_tok != 43) and (desc_tok != 43):
                    m += 1
                    n += 1
                
                # ins
                elif (anc_tok == 43) and (desc_tok != 43):
                    n += 1
                
                # del
                elif (anc_tok != 43) and (desc_tok == 43):
                    m += 1
                
                true_m[b, l] = m
                true_n[b, l] = n
        
        npt.assert_allclose( true_m, aligned_mats[...,2] )
        npt.assert_allclose( true_n, aligned_mats[...,3] )
            
            
if __name__ == '__main__':
    unittest.main()

