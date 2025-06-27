#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 12:58:47 2025

@author: annabel
"""
import numpy as np

import numpy.testing as npt
import unittest


class TestSummaryCounts(unittest.TestCase):
    """
    Make sure that summary counts encode the same information as full alignment
    
    
    note: iterates over every sample in the batch, every position in
        the alignement; this is kind of slow
    """
    def setUp(self):
        self.data_dir = 'example_data'
        self.prefix = 'sevenSamp'
    
    def test_dset(self,):
        with open(f'{self.data_dir}/{self.prefix}_aligned_mats.npy', 'rb') as f:
            aligned_seqs = np.load(f)[..., [0,1]]
        
        B = aligned_seqs.shape[0]
        L = aligned_seqs.shape[1]
        A = 20
        S = 5
        
        # calculate true values by loop
        true_subs = np.zeros( (B, A, A) )
        true_ins = np.zeros( (B, A) )
        true_del = np.zeros( (B, A) )
        true_trans = np.zeros( (B, 5, 5) )
        
        for b in range(B):
            # first position is start
            prev_state = 4
            
            for l in range(1, L):
                anc_tok = aligned_seqs[b,l,0]
                desc_tok = aligned_seqs[b,l,1]
                
                # padding
                if anc_tok == 0:
                    break
                
                # end
                if (anc_tok == 2) and (desc_tok == 2):
                    curr_state = 5
                
                # match
                elif (anc_tok != 43) and (desc_tok != 43):
                    curr_state = 1
                    true_subs[b, anc_tok-3, desc_tok-3] += 1
                
                # ins
                elif (anc_tok == 43) and (desc_tok != 43):
                    curr_state = 2
                    true_ins[b, desc_tok-3] += 1
                
                # del
                elif (anc_tok != 43) and (desc_tok == 43):
                    curr_state = 3
                    true_del[b, anc_tok-3] += 1
                
                # update transitions
                true_trans[b, prev_state-1, curr_state-1] += 1
                prev_state = curr_state
        
        true_dict = {'subCounts': true_subs,
                     'insCounts': true_ins,
                     'delCounts': true_del,
                     'transCounts': true_trans}
        
        # load the matrices and make sure these match
        for suffix in ['subCounts', 
                       'delCounts', 
                       'insCounts', 
                       'transCounts_five_by_five']:
            with open(f'{self.data_dir}/{self.prefix}_{suffix}.npy','rb') as f:
                pred_mat = np.load(f)
            
            true_mat = true_dict[ suffix.split('_')[0] ]
            
            npt.assert_allclose( pred_mat, true_mat )
            
if __name__ == '__main__':
    unittest.main()
