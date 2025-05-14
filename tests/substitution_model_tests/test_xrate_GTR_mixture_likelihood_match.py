#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 13:00:52 2025

@author: annabel

About:
======
9th test for substitution models

Check the log-probability of some real pfam alignments using xrate-fitted 
    parameters; see if you get same likelihood as xrate

only check up to the precision given by XRATE (which is generally low)

using TWO mixtures now

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
                                                              lse_over_match_logprobs_per_class,
                                                              joint_prob_from_counts)


THRESHHOLD = 1e-2


def load_mat(file):
    with open(file,'rb') as f:
        return np.load(f)

def round_val_arr(arr, precision_arr):
    out = np.zeros( arr.shape[0] )
    for i in range(arr.shape[0]):
        out[i] = np.round( arr[i], decimals=precision_arr[i] )
    return out
        

class TestXRATEMixtureLikelihoodMatch(unittest.TestCase):
    """
    SUBSTITUTION PROCESS SCORING TEST 9
    
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
    def test_xrate_vs_my_loglike(self):
    
        ### read inputs
        # counts of substitutions
        match_counts = load_mat( (f'tests/'+
                                  f'substitution_model_tests/'+
                                  f'req_files/'+
                                  f'xrate_GTR_subs_model_two_mixtures/'+
                                  f'PF07734_subCounts.npy') ) #(B,A,A)
        match_only_align_lens = match_counts.sum( axis=(-1,-2) ) #(B,)
        
        # params from xrate: numpy files (add a fake class dim)
        Q_times_rho = load_mat( (f'tests/'+
                                 f'substitution_model_tests/'+
                                 f'req_files/'+
                                 f'xrate_GTR_subs_model_two_mixtures/'+
                                 f'PARAM-MATS-PF07734_Q_unnormed_times_rho.npy') ) #(C,A,A)
        
        
        equilibrium_distributions = load_mat( (f'tests/'+
                                               f'substitution_model_tests/'+
                                               f'req_files/'+
                                               f'xrate_GTR_subs_model_two_mixtures/'+
                                               f'PARAM-MATS-PF07734_equilibrium_dists.npy') ) #(C,A)
        
        # params from xrate: class probs
        class_probs = load_mat( (f'tests/'+
                                 f'substitution_model_tests/'+
                                 f'req_files/'+
                                 f'xrate_GTR_subs_model_two_mixtures/'+
                                 f'PARAM-MATS-PF07734_class_probs.npy') ) #(C,)
        
        
        # params from xrate: length params
        length_raw = load_mat( (f'tests/'+
                                f'substitution_model_tests/'+
                                f'req_files/'+
                                f'xrate_GTR_subs_model_two_mixtures/'+
                                f'PARAM-MATS-PF07734_length_params.npy') ) #(2,)
        
        P_emit = length_raw[0]
        one_minus_P_emit = length_raw[1]
        transit_vec = np.log( np.array([P_emit, one_minus_P_emit]) )
        del P_emit, one_minus_P_emit
        
        
        # file of branch lengths from PFam
        meta_df = pd.read_csv( (f'tests/'+
                                f'substitution_model_tests/'+
                                f'req_files/'+
                                f'xrate_GTR_subs_model_two_mixtures/'+
                                f'PF07734_metadata.tsv'),
                              sep='\t',
                              usecols=['pairID',
                                       'ancestor',
                                       'descendant',
                                       'TREEDIST_anc-to-desc'])
        t_array = np.squeeze(meta_df['TREEDIST_anc-to-desc']).to_numpy() #(B,)
        
        # true loglike from XRATE; make sure to record precision
        true_bits_str = pd.read_csv( (f'tests/'+
                                      f'substitution_model_tests/'+
                                      f'req_files/'+
                                      f'xrate_GTR_subs_model_two_mixtures/'+
                                      f'OUT-PF07734_score_per_sample_parsed.tsv' ),
                                    sep='\t',
                                    usecols=['inside_loglike_bits'], 
                                    dtype=str)  # read everything as string
        precision = true_bits_str['inside_loglike_bits'].str.split('.', n=1).str[1].str.rstrip('0').str.len().fillna(0).astype(int)
        precision = precision.to_numpy()
        true_bits = true_bits_str.to_numpy().astype(float)
        true_bits = np.squeeze(true_bits)
        del true_bits_str
        
        # final dims
        assert match_counts.shape[0] == t_array.shape[0]
        B = match_counts.shape[0]
        L = match_counts.sum(axis=(-1, -2))
        C = Q_times_rho.shape[0]
        A = Q_times_rho.shape[1]
        
        
        ### get matrices
        log_cond,_ = get_cond_logprob_emit_at_match_per_class(t_array = t_array,
                                                              scaled_rate_mat_per_class = Q_times_rho) #(B,C,A,A)
        log_joint_emit_at_match_per_class = get_joint_logprob_emit_at_match_per_class(cond_logprob_emit_at_match_per_class = log_cond,
                                                              log_equl_dist_per_class = np.log(equilibrium_distributions)) #(B,C,A,A)
        log_joint = lse_over_match_logprobs_per_class(log_class_probs = np.log(class_probs),
                                              joint_logprob_emit_at_match_per_class = log_joint_emit_at_match_per_class) #(B,A,A)
        del Q_times_rho, log_cond, log_joint_emit_at_match_per_class
        
        
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
        
        # a is tested values, b is reference
        diff = np.abs( pred_scores_rounded - true_bits_rounded )
        
        df = pd.DataFrame( {'pred_rounded': pred_scores_rounded,
                            'true': true_bits_rounded,
                            'precision': precision,
                            'abs_diff': diff,
                            'align_len': match_only_align_lens} )
        
        largest_diff = -99
        
        status_file = f'tests/substitution_model_tests/output_files/OUTPUT_subs-unit-test-9_printout.log'
        with open(status_file, 'w') as g:
            g.write('')
        
        precisions = np.unique(df['precision'])
        for prec in precisions:
            sub_df = df[df['precision']==prec]
            true = sub_df['true']
            abs_diff = sub_df['abs_diff']
            relative_diff = abs_diff / np.abs( true )
            
            with open(status_file, 'a') as g:
                g.write(f'for precision {prec}:\n')
                g.write(f'---------------------\n')
                g.write(f'maximum relative difference (rtol): {relative_diff.max()}\n')
                g.write(f'average relative difference (rtol): {relative_diff.mean()}\n')
                g.write(f'maximum absolute difference (atol): {abs_diff.max()}\n')
                g.write(f'average absolute difference (atol): {abs_diff.mean()}\n\n')
            
            
            if relative_diff.max() > largest_diff:
                largest_diff = relative_diff.max()
        
        npt.assert_array_less(largest_diff, THRESHHOLD)
        
        out_file = f'tests/substitution_model_tests/output_files/OUTPUT_subs-unit-test-9_abs-diff-scores.tsv'
        df.to_csv(out_file, sep='\t')



if __name__ == '__main__':
    unittest.main()

