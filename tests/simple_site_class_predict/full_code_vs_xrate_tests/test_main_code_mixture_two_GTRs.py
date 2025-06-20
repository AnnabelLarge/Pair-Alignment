#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 16:13:55 2023

@author: annabel_large

"""
import json
import os
import argparse
import jax
import pickle
import shutil
import pandas as pd
import numpy as np

import numpy.testing as npt
import unittest

from cli.train_pairhmm_indp_sites import train_pairhmm_indp_sites as train_fn
from dloaders.init_counts_dset import init_counts_dset as init_dataloaders

jax.config.update("jax_enable_x64", True)
THRESHHOLD=0.001


def round_val_arr(arr, precision_arr):
    out = np.zeros( arr.shape[0] )
    for i in range(arr.shape[0]):
        out[i] = np.round( arr[i], decimals=precision_arr[i] )
    return out


class TestMainCodeGTRMix(unittest.TestCase):
    def test_main_code_two_gtr_mixtures(self):    
        ### make sure this folder doesn't already exist in the directory
        if 'RESULT_compare_to_xrate' in os.listdir():
            shutil.rmtree('RESULT_compare_to_xrate')
        
        ##############################
        ### Run main training code   #
        ##############################
        # parse the arguments
        parser = argparse.ArgumentParser(prog='Pair_Alignment')
        args = parser.parse_args()
        args.task = 'train'
        args.configs = ('tests/'+
                        'full_code_vs_xrate_tests/'+
                        'two_gtr_models/'+
                        'CONFIG.json')
        
        # helper function to open a single config file and extract additional arguments
        def read_config_file(config_file):
            with open(config_file, 'r') as f:
                contents = json.load(f)
                t_args = argparse.Namespace()
                t_args.__dict__.update(contents)
                args = parser.parse_args(namespace=t_args)
            return args
        
        args = read_config_file(args.configs)
        
        # "train" model i.e. run full evaluation script without updating 
        #   gradients
        dload_lst = init_dataloaders( args,
                                      'train',
                                      training_argparse = None )
        
        train_fn( args, dload_lst )
        
        
        ################################
        ### Compare to xrate results   #
        ################################
        # true loglike from XRATE; make sure to record precision
        true_bits_str = pd.read_csv( (f'tests/'+
                                      'full_code_vs_xrate_tests/'+
                                      'two_gtr_models/'+
                                      'xrate_out.tsv'),
                                    sep='\t', 
                                    dtype=str)  # read everything as string
        precision = true_bits_str['inside_loglike_bits'].str.split('.', n=1).str[1].str.rstrip('0').str.len().fillna(0).astype(int)
        precision = precision.to_numpy()
        true_bits = true_bits_str['inside_loglike_bits'].to_numpy().astype(float)
        true_bits = np.squeeze(true_bits)
        del true_bits_str
        
        # loglikes from running this code
        pred_results = pd.read_csv(f'RESULT_compare_to_xrate/logfiles/test-set_pt0_FINAL-LOGLIKES.tsv',
                                   sep='\t',
                                   usecols=['joint_logP']).to_numpy()
        pred_results = -pred_results
        pred_scores_bits = pred_results / np.log(2)
        
        
        ### check closeness of both scores BASED ON PRECISION OF XRATE COLUMN
        pred_scores_rounded = round_val_arr(pred_scores_bits, precision)
        true_bits_rounded = round_val_arr(true_bits, precision)
        
        # a is tested values, b is reference
        diff = np.abs( pred_scores_rounded - true_bits_rounded )
        
        df = pd.DataFrame( {'pred_rounded': pred_scores_rounded,
                            'true': true_bits_rounded,
                            'precision': precision,
                            'abs_diff': diff} )
        
        largest_diff = -99
        
        status_file = f'tests/full_code_vs_xrate_tests/two_gtr_models/OUTPUT_diff-stats.log'
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
        
        out_file = f'tests/full_code_vs_xrate_tests/two_gtr_models/OUTPUT_diff-per-sample.tsv'
        df.to_csv(out_file, sep='\t')


if __name__ == '__main__':
    unittest.main()