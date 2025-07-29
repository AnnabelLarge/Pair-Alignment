#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 15:59:59 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
import numpy as np

import numpy.testing as npt
import unittest

from train_eval_fns.neural_hmm_predict_train_eval_one_batch import _preproc
from models.sequence_embedders.initial_embedding_blocks import PlaceholderEmbedding
from models.sequence_embedders.concatenation_fns import extract_embs


H = 3

class TestExtractEmbs(unittest.TestCase):
    """
    Test of models.sequence_embedders.concatenation_fns.extract_embs
    
    I worked out how I expect the embeddings to be concatenated by hand. Make
      sure function returns the same result
    """
    def test_extract_embs(self):
        ###################
        ### True inputs   #
        ###################
        # unaligned seqs
        ancs = np.array( [[1, 3, 4, 5, 2, 0], 
                          [1, 3, 2, 0, 0, 0],
                          [1, 3, 4, 5, 6, 2],
                          [1, 3, 4, 5, 6, 2],
                          [1, 3, 4, 2, 0, 0]] )
        
        descs = np.array( [[-1, -3, -2,  0,  0,  0],
                           [-1, -3, -4, -5, -2,  0],
                           [-1, -3, -4, -5, -6, -2],
                           [-1, -3, -4, -5, -6, -2],
                           [-1, -3, -4, -2,  0,  0]] )
        
        unaligned_seqs = np.stack( [ancs, descs], axis=-1 )
        del ancs, descs
        
        L_seq = unaligned_seqs.shape[1]
        
        # aligned seqs
        ancs_aligned = np.array( [[1,  3,  4,  5,  2,  0,  0],
                                  [1, 43,  3, 43,  2,  0,  0],
                                  [1,  3,  4,  5,  6, 43,  2],
                                  [1, 43,  3,  4,  5,  6,  2],
                                  [1,  3,  4,  2,  0,  0,  0]] )
        
        descs_aligned = np.array( [[-1, 43,  -3, 43, -2,  0,  0],
                                   [-1, -3,  -4, -5, -2,  0,  0],
                                   [-1, 43,  -3, -4, -5, -6, -2],
                                   [-1, -3,  -4, -5, -6, 43, -2],
                                   [-1, -3,  -4, -2,  0,  0,  0]] )
        
        # indices
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
        
        aligned_mats = np.stack( [ancs_aligned, descs_aligned, m_idx, n_idx], axis=-1 )
        B = aligned_mats.shape[0]
        L_align = aligned_mats.shape[1]
        
        del m_idx, n_idx, ancs_aligned, descs_aligned
        
        
        ##################################
        ### true concatenation results   #
        ##################################
        true_anc_block = np.zeros( (B, L_align-1, H) )
        true_desc_block = np.zeros( (B, L_align-1, H) )
        for b in range(B):
            for l in range(L_align):
                m = aligned_mats[b,l,2]
                n = aligned_mats[b,l,3]
                
                if (m != -9):
                    anc_tok = unaligned_seqs[b,m,0]
                    desc_tok = unaligned_seqs[b,n,1]
                    
                    for h in range(H):
                        true_anc_block[b,l,h] = anc_tok * 10**h
                        true_desc_block[b,l,h] = desc_tok * 10**h
        
        del b, l, m, n, anc_tok, desc_tok
        
        
        ####################################
        ### mimic training/eval pipeline   #
        ####################################
        out_dict = _preproc( unaligned_seqs = unaligned_seqs,
                             aligned_mats = aligned_mats )
        anc_seqs = out_dict['anc_seqs']
        desc_seqs = out_dict['desc_seqs']
        align_idxes = out_dict['align_idxes']
        from_states = out_dict['from_states']
        true_out = out_dict['true_out']
        del out_dict
        
        embed_model = PlaceholderEmbedding( config = {'hidden_dim': H,
                                                      'seq_padding_idx': 0},
                                            name = 'placehold' )
        
        anc_embeddings,_ = embed_model.apply( variables = {},
                                            datamat = anc_seqs )
        desc_embeddings,_ = embed_model.apply( variables = {},
                                             datamat = desc_seqs )
        
        # make these more distinguishable
        anc_embeddings = np.array(anc_embeddings)
        desc_embeddings = np.array(desc_embeddings)
        for b in range(B):
            for l in range(L_seq):
                for h in range(H):
                    anc_embeddings[b,l,h] = anc_embeddings[b,l,h] * 10**h
                    desc_embeddings[b,l,h] = desc_embeddings[b,l,h] * 10**h
        
        
        out = extract_embs(anc_encoded = anc_embeddings, 
                           desc_encoded = desc_embeddings,
                           idx_lst = align_idxes,
                           seq_padding_idx = 0,
                           align_idx_padding = -9)
        datamat_lst, _ = out
        
        pred_anc_block, pred_desc_block = datamat_lst
        
        del anc_seqs, desc_seqs, align_idxes, from_states, true_out, embed_model
        del anc_embeddings, desc_embeddings, out, datamat_lst
        
        npt.assert_allclose(pred_anc_block, true_anc_block ) 
        npt.assert_allclose(pred_desc_block, pred_desc_block )
        

if __name__ == '__main__':
    unittest.main()