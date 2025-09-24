#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 15:13:06 2025

@author: annabel
"""
import pandas as pd
import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader,default_collate

import numpy.testing as npt
import unittest


from dloaders.FullLenDset import FullLenDset

TEST_LOC = 'tests/data_dataloaders'


class TestFullLenDset(unittest.TestCase):
    """
    Make sure that summary counts encode the same information as full alignment
    
    
    note: iterates over every sample in the batch, every position in
        the alignement; this is kind of slow
    """
    def setUp(self):
        A = 4
        max_len = 8

        ### hand-code alignments
        # unaligned seqs
        ancs = np.array( [[1, 3, 4, 5, 2, 0, 0, 0, 0, 0], 
                          [1, 3, 2, 0, 0, 0, 0, 0, 0, 0],
                          [1, 3, 3, 3, 3, 3, 3, 3, 3, 2], # remove sample 3 (idx=2)
                          [1, 3, 4, 5, 6, 2, 0, 0, 0, 0],
                          [1, 3, 4, 5, 6, 2, 0, 0, 0, 0],
                          [1, 3, 4, 2, 0, 0, 0, 0, 0, 0]] )

        descs = np.array( [[1, 3, 2, 0, 0, 0, 0, 0, 0, 0],
                           [1, 3, 4, 5, 2, 0, 0, 0, 0, 0],
                           [1, 3, 3, 3, 3, 3, 3, 3, 3, 2], # remove sample 3 (idx=2)
                           [1, 3, 4, 5, 6, 2, 0, 0, 0, 0],
                           [1, 3, 4, 5, 6, 2, 0, 0, 0, 0],
                           [1, 3, 4, 2, 0, 0, 0, 0, 0, 0]] )

        seqs_unaligned = np.stack( [ancs, descs], axis=-1 ) #(B, L, 2)
        del ancs, descs
        
        # emission counts
        _, true_emissions = np.unique(seqs_unaligned, return_counts=True)
        true_emissions = true_emissions[3:] #(A,)

        L_seq = seqs_unaligned.shape[1]

        # aligned matrices
        ancs_aligned = np.array( [[1,  3,  4,  5,  2,  0,  0, 0, 0, 0],
                                  [1, 43,  3, 43,  2,  0,  0, 0, 0, 0],
                                  [1,  3,  3,  3,  3,  3,  3, 3, 3, 2], # remove sample 3 (idx=2)
                                  [1,  3,  4,  5,  6, 43,  2, 0, 0, 0],
                                  [1, 43,  3,  4,  5,  6,  2, 0, 0, 0],
                                  [1,  3,  4,  2,  0,  0,  0, 0, 0, 0]] )

        descs_aligned = np.array( [[1, 43,  3, 43,  2,  0,  0, 0, 0, 0],
                                   [1,  3,  4,  5,  2,  0,  0, 0, 0, 0],
                                   [1,  3,  3,  3,  3,  3,  3, 3, 3, 2],  # remove sample 3 (idx=2)
                                   [1, 43,  3,  4,  5,  6,  2, 0, 0, 0],
                                   [1,  3,  4,  5,  6, 43,  2, 0, 0, 0],
                                   [1,  3,  4,  2,  0,  0,  0, 0, 0, 0]] )

        state = np.array( [[4,3,1,3,5,0,0,0,0,0],
                           [4,2,1,2,5,0,0,0,0,0],
                           [1,3,3,3,3,3,3,3,3,2], # remove sample 3 (idx=2)
                           [4,3,1,1,1,2,5,0,0,0],
                           [4,2,1,1,1,3,5,0,0,0],
                           [4,1,1,5,0,0,0,0,0,0]] )

        m_idx = np.array( [[1,  2,  3,  4, -9, -9, -9, -9, -9, -9],
                           [1,  1,  2,  2, -9, -9, -9, -9, -9, -9],
                           [-9]*10, # remove sample 3 (idx=2)
                           [1,  2,  3,  4,  5,  5, -9, -9, -9, -9],
                           [1,  1,  2,  3,  4,  5, -9, -9, -9, -9],
                           [1,  2,  3, -9, -9, -9, -9, -9, -9, -9]] )

        n_idx = np.array( [[0,  0,  1,  1, -9, -9, -9, -9, -9, -9],
                           [0,  1,  2,  3, -9, -9, -9, -9, -9, -9],
                           [-9]*10, # remove sample 3 (idx=2),
                           [0,  0,  1,  2,  3,  4, -9, -9, -9, -9],
                           [0,  1,  2,  3,  4,  4, -9, -9, -9, -9],
                           [0,  1,  2, -9, -9, -9, -9, -9, -9, -9]] )

        aligned_mats = np.stack( [ancs_aligned, descs_aligned, state, m_idx, n_idx], axis=-1 ) #(B, L, 5)
        B = aligned_mats.shape[0]
        L_align = aligned_mats.shape[1]

        del m_idx, n_idx, ancs_aligned, descs_aligned, state

        # metadata
        meta_df = pd.DataFrame( {'pairID': [f'pair{i}' for i in range(B)], 
                                 'ancestor': [f'anc{i}' for i in range(B)],
                                 'descendant': [f'desc{i}' for i in range(B)],
                                 'pfam': 'PF00000',
                                 'anc_seq_len': [3, 1, 10, 4, 4, 2],
                                 'desc_seq_len': [1, 3, 10, 4, 4, 2], 
                                 'alignment_len': [3, 3, 10, 5, 5, 2], 
                                 'num_matches': [1, 1, 8, 3, 3, 2], 
                                 'num_ins': [0, 2, 0, 1, 1, 0], 
                                 'num_del': [2, 0, 0, 1, 1, 0]} )
        
        # time
        time_df = pd.DataFrame( meta_df.copy()['pairID'] )
        time_df['times'] = np.arange( 1, B+1 ) * 0.1
        times = time_df['times']
        
        ### save all to load later
        arrs = [seqs_unaligned,
                aligned_mats,
                true_emissions]

        suffixes = ['seqs_unaligned',
                    'aligned_mats',
                    'NuclCounts']

        prefix = 'fake_inputs_dna_withextra'
        for i in range( len(arrs) ):
            a = arrs[i]
            s = suffixes[i]
            
            with open(f'{prefix}_{s}.npy','wb') as g:
                np.save(g, a)

        meta_df.to_csv(f'{prefix}_metadata.tsv', sep='\t')
        time_df.to_csv(f'{prefix}_pair-times.tsv', header=False, index=False, sep='\t')
        
        
        ### final attributes
        self.prefix = prefix
        self.seqs_unaligned = seqs_unaligned
        self.aligned_mats = aligned_mats
        self.times = times
        
        self.B = B
        self.L_align = L_align
        self.A = A
        self.max_len = max_len
    
    def _run(self, 
             pred_model_type,
             use_scan_fns,
             t_per_sample):
        
        ### if using scan functions, need to pad out the aligned matrices to length 9
        if use_scan_fns:
            to_add = np.array( [[0, 0],
                                [0, 0],
                                [0, 0],
                                [-9, -9],
                                [-9, -9]] ).T[None,...]
            to_add = np.broadcast_to( to_add, (self.B, to_add.shape[1], to_add.shape[2]) )
            
            self.aligned_mats = np.concatenate( [self.aligned_mats, to_add], axis =1 )
            self.L_align = self.aligned_mats.shape[1]
        
        
        ### different formats, depending on pred model type
        # don't need indices for pairhmm inputs
        if pred_model_type == 'pairhmm_frag_and_site_classes':
            self.aligned_mats = self.aligned_mats[...,[0,1,2]]
        
        # completely different alphabet for feedforward
        elif pred_model_type == 'feedforward':
            new_mat = np.zeros( (self.B, self.L_align, 4) )
            for b in range(self.B):
                for l in range(self.L_align):
                    row = self.aligned_mats[b,l,:]
                    
                    # at insert position, add A to desc tok
                    if row[2] == 2:
                        desc_tok = ( row[1] + self.A ) 
                    
                    else:
                        desc_tok = row[1]
                        
                    new_mat[b, l, :] = [desc_tok, row[2], row[3], row[4]]
            
            self.aligned_mats = new_mat
            self.L_align = self.aligned_mats.shape[1]
            
        
        ### make dataset object
        dset_obj = FullLenDset( data_dir = '.',
                                split_prefixes = [self.prefix], 
                                pred_model_type = pred_model_type,
                                toss_alignments_longer_than = self.max_len,
                                use_scan_fns = use_scan_fns,
                                emission_alphabet_size = 4,
                                t_per_sample = t_per_sample,
                                chunk_length = 4 )
        
        read_data = list(dset_obj)
        
        pred_b_idx = -1
        for true_b_idx in range(self.B):
            true_unaligned = self.seqs_unaligned[true_b_idx]
            true_aligned = self.aligned_mats[true_b_idx]
            
            if ( true_aligned[:,0] != 0 ).sum() <= self.max_len:
                pred_b_idx += 1
                sample = read_data[pred_b_idx]
                pred_unaligned, pred_aligned, pred_time, _ = sample
                
                true_unaligned = true_unaligned[ :pred_unaligned.shape[0], ... ]
                true_aligned = true_aligned[ :pred_aligned.shape[0], ... ]
                
                # main arrays
                npt.assert_allclose( pred_unaligned, true_unaligned )
                npt.assert_allclose( pred_aligned, true_aligned )
                        
                # time
                if t_per_sample:
                    npt.assert_allclose( pred_time, self.times[true_b_idx] )
                
                elif not t_per_sample:
                    npt.assert_(pred_time is None)
                
        
    
    ###############
    ### pairHMM   #
    ###############
    def test_pairhmm(self):
        self._run( pred_model_type = 'pairhmm_frag_and_site_classes',
                    use_scan_fns = False,
                    t_per_sample = False )
    
    def test_pairhmm_with_scan(self):
        self._run( pred_model_type = 'pairhmm_frag_and_site_classes',
                    use_scan_fns = True,
                    t_per_sample = False )
    
    def test_pairhmm_t_per_sample(self):
        self._run( pred_model_type = 'pairhmm_frag_and_site_classes',
                    use_scan_fns = False,
                    t_per_sample = True )
    
    def test_pairhmm_with_scan_t_per_sample(self):
        self._run( pred_model_type = 'pairhmm_frag_and_site_classes',
                    use_scan_fns = True,
                    t_per_sample = True )
    
    ##################
    ### neural TKF   #
    ##################
    def test_neural_hmm(self):
        self._run( pred_model_type = 'neural_hmm',
                    use_scan_fns = False,
                    t_per_sample = False )
    
    def test_neural_hmm_with_scan(self):
        self._run( pred_model_type = 'neural_hmm',
                    use_scan_fns = True,
                    t_per_sample = False )
    
    def test_neural_hmm_t_per_sample(self):
        self._run( pred_model_type = 'neural_hmm',
                    use_scan_fns = False,
                    t_per_sample = True )
    
    def test_neural_hmm_with_scan_t_per_sample(self):
        self._run( pred_model_type = 'neural_hmm',
                    use_scan_fns = True,
                    t_per_sample = True )
    
    
    ###################
    ### feedforward   #
    ###################
    def test_ff_hmm(self):
        self._run( pred_model_type = 'feedforward',
                    use_scan_fns = False,
                    t_per_sample = False )
    
    def test_ff_hmm_with_scan(self):
        self._run( pred_model_type = 'feedforward',
                    use_scan_fns = True,
                    t_per_sample = False )
    
    def test_ff_hmm_t_per_sample(self):
        self._run( pred_model_type = 'feedforward',
                    use_scan_fns = False,
                    t_per_sample = True )
    
    def test_ff_hmm_with_scan_t_per_sample(self):
        self._run( pred_model_type = 'feedforward',
                    use_scan_fns = True,
                    t_per_sample = True )
        
    def tearDown(self):
        suffixes = ['seqs_unaligned.npy',
                    'aligned_mats.npy',
                    'NuclCounts.npy',
                    'metadata.tsv',
                    'pair-times.tsv']

        for suff in suffixes:
            os.remove(f'./{self.prefix}_{suff}')
    
    
     
if __name__ == '__main__':
    unittest.main()      
        
        
