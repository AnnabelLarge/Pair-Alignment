#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 13:00:52 2025

@author: annabel

About:
======
8th test for substitution models

Check the log-probability of some real pfam alignments using xrate-fitted 
    parameters; see if you get same likelihood as xrate

only check up to the precision given by XRATE (which is generally low)

Unlike previous tests, there's a unique time for every sample

ONE MODEL, NO MIXTURES YET
"""
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
import numpy as np
import pandas as pd

import numpy.testing as npt
import unittest

from tests.data_processing import (str_aligns_to_tensor,
                                   summarize_alignment)

from models.simple_site_class_predict.model_functions import (rate_matrix_from_exch_equl,
                                                              get_cond_logprob_emit_at_match_per_class,
                                                              get_joint_logprob_emit_at_match_per_class,
                                                              joint_prob_from_counts)


def load_mat(file):
    with open(file,'rb') as f:
        return np.load(f)

def round_val_arr(arr, precision_arr):
    out = np.zeros( arr.shape[0] )
    for i in range(arr.shape[0]):
        out[i] = np.round( arr[i], decimals=precision_arr[i] )
    return out
        

class TestXRATELikelihoodMatch(unittest.TestCase):
    """
    SUBSTITUTION PROCESS SCORING TEST 8
    
    B: batch (samples)
    L: length (number of alignment columns)
    C: hidden site classes
    A: alphabet
    
    About
    ------
    Check the log-probability of some real pfam alignments using xrate-fitted 
        parameters; see if you get same likelihood as xrate (up to xrate's 
        precision for the sample, which is usually pretty low); Unlike 
        previous tests, there's a unique time for every sample
    
    """
    def test_score_alignment(self):
        ### read inputs
        # counts of substitutions
        match_counts = load_mat(f'tests/substitution_model_tests/req_files/xrate_one_GTR_subs_model/PF07734_subCounts.npy') #(B,A,A)
        
        # params from xrate: numpy files (add a fake class dim)
        Q = load_mat('tests/substitution_model_tests/req_files/xrate_one_GTR_subs_model/PF07734_xrate_fitted_unnormed_Q.npy')[None,:,:] #(C,A,A)
        equilibrium_distributions = load_mat('tests/substitution_model_tests/req_files/xrate_one_GTR_subs_model/PF07734_xrate_fitted_equlibriums.npy')[None,:] #(C,A)
        
        # params from xrate: flat text files
        with open('tests/substitution_model_tests/req_files/xrate_one_GTR_subs_model/PF07734_xrate_other_params.txt','r') as f:
            line = f.readline()
            one_minus_P_emit = float( line.strip().split(': ')[-1] ) #one float value
            line_again = f.readline()
            P_emit = float( line_again.strip().split(': ')[-1] ) #one float value
        
        # params from xrate: tsv text files
        meta_df = pd.read_csv(f'tests/substitution_model_tests/req_files/xrate_one_GTR_subs_model/PF07734_metadata.tsv',
                              sep='\t',
                              usecols=['pairID',
                                       'ancestor',
                                       'descendant',
                                       'TREEDIST_anc-to-desc'])
        t_array = np.squeeze(meta_df['TREEDIST_anc-to-desc']).to_numpy() #(B,)
        
        # true loglike from XRATE; make sure to record precision
        true_bits_str = pd.read_csv(f'tests/substitution_model_tests/req_files/xrate_one_GTR_subs_model/xrate_score_per_sample_parsed.tsv',
                                    sep='\t',
                                    usecols=['xrate_bitscore'], 
                                    dtype=str)  # read everything as string
        precision = true_bits_str['xrate_bitscore'].str.split('.', n=1).str[1].str.rstrip('0').str.len().fillna(0).astype(int)
        precision = precision.to_numpy()
        true_bits = true_bits_str.to_numpy().astype(float)
        true_bits = np.squeeze(true_bits)
        del true_bits_str
        
        # final dims
        assert match_counts.shape[0] == t_array.shape[0]
        B = match_counts.shape[0]
        L = match_counts.sum(axis=(-1, -2))
        C = Q.shape[0]
        A = Q.shape[1]
        
        
        ### get matrices
        log_cond,_ = get_cond_logprob_emit_at_match_per_class(t_array = t_array,
                                                              scaled_rate_mat_per_class = Q) #(B,C,A,A)
        log_joint = get_joint_logprob_emit_at_match_per_class(cond_logprob_emit_at_match_per_class = log_cond,
                                                              log_equl_dist_per_class = np.log(equilibrium_distributions))  #(B,C,A,A)
        log_joint = log_joint[:,0,...]
        
        del Q, log_cond
        
        # check if this log joint matrix matches the previous loopy version
        log_joint_prev_calc = load_mat( ('tests/substitution_model_tests/'+
                                         'req_files/'+
                                         'xrate_one_GTR_subs_model/'+
                                         'PF07734_joint_logprob_emit_per_sample.npy')
                                       )
        assert np.allclose(log_joint, log_joint_prev_calc)
        
        # check if this transition matrix matches the previous loopy version
        transit_vec = np.log(np.array([P_emit, one_minus_P_emit]))
        transit_vec_prev_calc = load_mat( ('tests/substitution_model_tests/'+
                                         'req_files/'+
                                         'xrate_one_GTR_subs_model/'+
                                         'PF07734_transitions.npy')
                                       )
        assert np.allclose(transit_vec, transit_vec_prev_calc)
        
        
        ### calculate with my function
        fake_batch = (match_counts, 
                      np.zeros((B,A)),
                      np.zeros((B,A)),
                      np.zeros((B, 4, 4)))
        
        
        scoring_matrices_dict = {'joint_logprob_emit_at_match': log_joint,
                                 'all_transit_matrices': 
                                     {'joint': transit_vec
                                      } 
                                }
        
        out = joint_prob_from_counts( batch = fake_batch,
                                      times_from = 't_per_sample',
                                      score_indels = False,
                                      scoring_matrices_dict = scoring_matrices_dict,
                                      t_array = t_array,
                                      exponential_dist_param = None,
                                      norm_loss_by = None )
        
        pred_scores_nats = -out['joint_neg_logP']
        pred_scores_bits = pred_scores_nats / np.log(2)
        del out
        
        ### check closeness of both scores BASED ON PRECISION OF XRATE COLUMN
        pred_scores_rounded = round_val_arr(pred_scores_bits, precision)
        true_bits_rounded = round_val_arr(true_bits, precision)
        final_comparison = np.isclose(pred_scores_rounded, true_bits_rounded, equal_nan=True)
        
        diff = np.abs( pred_scores_rounded - true_bits_rounded )

        df = pd.DataFrame( {'pred_rounded': pred_scores_rounded,
                            'true': true_bits_rounded,
                            'precision': precision,
                            'abs_diff': diff} )
        df.to_csv(f'DIFF_single-class.tsv', sep='\t')
        
        npt.assert_array_equal(final_comparison, True)

                      
if __name__ == '__main__':
    unittest.main()

