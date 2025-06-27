#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 14:24:54 2025

@author: annabel
"""
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader,default_collate

import numpy.testing as npt
import unittest


from dloaders.CountsDset import CountsDset

TEST_LOC = 'tests/data_dataloaders'


class TestCountsDset(unittest.TestCase):
    """
    Make sure that summary counts encode the same information as full alignment
    
    
    note: iterates over every sample in the batch, every position in
        the alignement; this is kind of slow
    """
    def setUp(self):
        A = 4

        ### hand-code alignments
        # unaligned seqs
        ancs = np.array( [[1, 3, 4, 5, 2, 0], 
                          [1, 3, 2, 0, 0, 0],
                          [1, 3, 4, 5, 6, 2],
                          [1, 3, 4, 5, 6, 2],
                          [1, 3, 4, 2, 0, 0]] )

        descs = np.array( [[ 1,  3,  2,  0,  0,  0],
                           [ 1,  3,  4,  5,  2,  0],
                           [ 1,  3,  4,  5,  6,  2],
                           [ 1,  3,  4,  5,  6,  2],
                           [ 1,  3,  4,  2,  0,  0]] )

        seqs_unaligned = np.stack( [ancs, descs], axis=-1 )
        del ancs, descs

        L_seq = seqs_unaligned.shape[1]

        # aligned matrices
        ancs_aligned = np.array( [[1,  3,  4,  5,  2,  0,  0],
                                  [1, 43,  3, 43,  2,  0,  0],
                                  [1,  3,  4,  5,  6, 43,  2],
                                  [1, 43,  3,  4,  5,  6,  2],
                                  [1,  3,  4,  2,  0,  0,  0]] )

        descs_aligned = np.array( [[ 1, 43,  3, 43,  2,  0,  0],
                                   [ 1,  3,  4,  5,  2,  0,  0],
                                   [ 1, 43,  3,  4,  5,  6,  2],
                                   [ 1,  3,  4,  5,  6, 43,  2],
                                   [ 1,  3,  4,  2,  0,  0,  0]] )

        state = np.array( [[4,3,1,3,5,0,0],
                           [4,2,1,2,5,0,0],
                           [4,3,1,1,1,2,5],
                           [4,2,1,1,1,3,5],
                           [4,1,1,5,0,0,0]] )

        m_idx = np.array( [[1,  2,  3,  4, -9, -9, -9],
                           [1,  1,  2,  2, -9, -9, -9],
                           [1,  2,  3,  4,  5,  5, -9],
                           [1,  1,  2,  3,  4,  5, -9],
                           [1,  2,  3, -9, -9, -9, -9]] )

        n_idx = np.array( [[0,  0,  1,  1, -9, -9, -9],
                           [0,  1,  2,  3, -9, -9, -9],
                           [0,  0,  1,  2,  3,  4, -9],
                           [0,  1,  2,  3,  4,  4, -9],
                           [0,  1,  2, -9, -9, -9, -9]] )

        aligned_mats = np.stack( [ancs_aligned, descs_aligned, state, m_idx, n_idx], axis=-1 )
        B = aligned_mats.shape[0]
        L_align = aligned_mats.shape[1]

        del m_idx, n_idx, ancs_aligned, descs_aligned, state

        # metadata
        meta_df = pd.DataFrame( {'pairID': [f'pair{i}' for i in range(B)], 
                                 'ancestor': [f'anc{i}' for i in range(B)],
                                 'descendant': [f'desc{i}' for i in range(B)],
                                 'pfam': 'PF00000',
                                 'anc_seq_len': [3, 1, 4, 4, 2],
                                 'desc_seq_len': [1, 3, 4, 4, 2], 
                                 'alignment_len': [3, 3, 5, 5, 2], 
                                 'num_matches': [1, 1, 3, 3, 2], 
                                 'num_ins': [0,2, 1, 1, 0], 
                                 'num_del': [2, 0, 1, 1, 0]} )
        
        # time
        times = np.arange( 1, B+1 ) * 0.1


        ### make summary counts
        true_subs = np.zeros( (B, A, A) )
        true_ins = np.zeros( (B, A) )
        true_del = np.zeros( (B, A) )
        true_trans = np.zeros( (B, 4, 4) )
        true_emissions = np.zeros( (A,) )
        true_emit_from_match = np.zeros( (A,) )

        for b in range(B):
            # first position is start
            prev_state = 4
            
            for l in range(1, L_align):
                anc_tok = aligned_mats[b,l,0]
                desc_tok = aligned_mats[b,l,1]
                
                # padding
                if anc_tok == 0:
                    break
                
                # end; use the same encoding as start
                if (anc_tok == 2) and (desc_tok == 2):
                    curr_state = 4
                
                # match
                elif (anc_tok != 43) and (desc_tok != 43):
                    curr_state = 1
                    true_subs[b, anc_tok-3, desc_tok-3] += 1
                    true_emissions[anc_tok-3] += 1
                    true_emissions[desc_tok-3] += 1
                    true_emit_from_match[anc_tok-3] += 1
                    true_emit_from_match[desc_tok-3] += 1
                
                # ins
                elif (anc_tok == 43) and (desc_tok != 43):
                    curr_state = 2
                    true_ins[b, desc_tok-3] += 1
                    true_emissions[desc_tok-3] += 1
                
                # del
                elif (anc_tok != 43) and (desc_tok == 43):
                    curr_state = 3
                    true_del[b, anc_tok-3] += 1
                    true_emissions[anc_tok-3] += 1
                
                # update transitions
                true_trans[b, prev_state-1, curr_state-1] += 1
                prev_state = curr_state
        
        
        ### final attributes
        self.times = times
        
        self.true_subs = true_subs
        self.true_ins = true_ins
        self.true_del = true_del
        self.true_trans = true_trans
        self.true_emissions = true_emissions
        self.true_emit_from_match = true_emit_from_match
        
        self.B = B

    def _run(self,
             subs_only: bool,
             t_per_sample: bool):
        dset_obj = CountsDset( data_dir = f'{TEST_LOC}/fake_inputs',
                               split_prefixes = ['fake_inputs_dna'], 
                               emission_alphabet_size = 4,
                               t_per_sample = t_per_sample,
                               subs_only = subs_only,
                               toss_alignments_longer_than = None,
                               bos_eos_as_match = False )
        # total emission counts
        pred_equl_dist = dset_obj.retrieve_equil_dist()
        
        if not subs_only:
            true_equl_dist = self.true_emissions / ( self.true_emissions.sum() )
        
        elif subs_only:
            true_equl_dist = self.true_emit_from_match / ( self.true_emit_from_match.sum() )
        
        npt.assert_allclose( pred_equl_dist, true_equl_dist )
        
        read_data = list(dset_obj)
        
        for b in range(self.B):
            sample = read_data[b]
            subCounts, insCounts, delCounts, transCounts, time, _ = sample
            del sample
            
            # main counts
            npt.assert_allclose( subCounts, self.true_subs[b] )
            npt.assert_allclose( insCounts, self.true_ins[b] )
            npt.assert_allclose( delCounts, self.true_del[b] )
            npt.assert_allclose( transCounts, self.true_trans[b] )
                    
            # time
            if t_per_sample:
                npt.assert_allclose( time, self.times[b] )
            
            elif not t_per_sample:
                npt.assert_(time is None)
            
            
    def test_all_pos_no_time(self):
        self._run( subs_only = False,
                   t_per_sample = False )
    
    def test_all_pos_t_per_sample(self):
        self._run( subs_only = False,
                   t_per_sample = True )
        
    def test_subs_only_no_time(self):
        self._run( subs_only = True,
                   t_per_sample = False )
        
    def test_subs_only_t_per_sample(self):
        self._run( subs_only = True,
                   t_per_sample = True )
  
        
if __name__ == '__main__':
    unittest.main()      
