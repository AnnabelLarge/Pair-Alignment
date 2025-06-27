#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 15:13:06 2025

@author: annabel
"""
import numpy as np

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
        
        
        ### final attributes
        self.seqs_unaligned = seqs_unaligned
        self.aligned_mats = aligned_mats
        self.times = times
        
        self.B =B
        self.L_align = L_align
        self.A = A
    
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
        if pred_model_type in ['pairhmm_indp_sites',
                                'pairhmm_frag_and_site_classes']:
            self.aligned_mats = self.aligned_mats[...,[0,1,2]]
        
        # completely different alphabet for feedforward
        elif pred_model_type == 'feedforward':
            new_mat = np.zeros( (self.B, self.L_align, 4) )
            for b in range(self.B):
                for l in range(self.L_align):
                    row = self.aligned_mats[b,l,:]
                    
                    # at insert position, add A to desc tok
                    if row[2] == 2:
                        desc_tok = ( row[1] + self.A ) - 1
                        
                    else:
                        # move all tokens except pad, start, and gap down by one
                        if not np.isin( row[1], [0, 1, 43] ):
                            desc_tok = row[1] - 1
                        else:
                            desc_tok = row[1]
                    
                    new_mat[b, l, :] = [desc_tok, row[2], row[3], row[4]]
            
            self.aligned_mats = new_mat
            self.L_align = self.aligned_mats.shape[1]
            
        
        ### make dataset object
        dset_obj = FullLenDset( data_dir = f'{TEST_LOC}/fake_inputs',
                                split_prefixes = ['fake_inputs_dna'], 
                                pred_model_type = pred_model_type,
                                use_scan_fns = use_scan_fns,
                                emission_alphabet_size = 4,
                                t_per_sample = t_per_sample,
                                chunk_length = 4 )
        
        read_data = list(dset_obj)
        
        for b in range(self.B):
            sample = read_data[b]
            unaligned, aligned, time, _ = sample
            
            # main arrays
            npt.assert_allclose( unaligned, self.seqs_unaligned[b] )
            npt.assert_allclose( aligned, self.aligned_mats[b] )
                    
            # time
            if t_per_sample:
                npt.assert_allclose( time, self.times[b] )
            
            elif not t_per_sample:
                npt.assert_(time is None)
        
    
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
    
    
     
if __name__ == '__main__':
    unittest.main()      
        
        
