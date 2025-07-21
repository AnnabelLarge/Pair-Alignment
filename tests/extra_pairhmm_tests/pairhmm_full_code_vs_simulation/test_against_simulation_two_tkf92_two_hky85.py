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

from cli.train_pairhmm_frag_and_site_classes import train_pairhmm_frag_and_site_classes as train_fn
from dloaders.init_full_len_dset import init_full_len_dset as init_dataloaders

jax.config.update("jax_enable_x64", True)



# helper function to open a single config file and extract additional arguments
def read_config_file(config_file, parser):
    with open(config_file, 'r') as f:
        contents = json.load(f)
        t_args = argparse.Namespace()
        t_args.__dict__.update(contents)
        args = parser.parse_args(namespace=t_args)
    return args

def remove_folder(name):
    if name in os.listdir():
        shutil.rmtree(name)

def load_mat(file):
    with open(file,'rb') as f:
        return np.load(f)

def retrain():
    response = input("Retrain? It helps to have GPU for this. [y/n] ")
    if response.strip().lower() == 'y':
        ### score with existing parameters 
        # make sure this doesn't exist
        remove_folder('RESULTS_load_params')
        
        # parse the arguments
        parser = argparse.ArgumentParser(prog='Pair_Alignment')
        args = parser.parse_args()
        args.task = 'train'
        args.configs = ('tests/'+
                        'full_code_vs_simulation_tests/'+
                        'two-tkf92_two-hky/'+
                        'two-tkf92_two-hky85_score_original.json')
        
        args = read_config_file(args.configs, parser)
        
        # "train" model i.e. run full evaluation script without updating 
        #   gradients
        dload_lst = init_dataloaders( args,
                                      'train',
                                      training_argparse = None )
        
        train_fn( args, dload_lst )
        del parser, args, dload_lst
        
        ### refit
        # make sure this doesn't exist
        remove_folder('RESULTS_recover_params')
        
        # parse the arguments
        parser = argparse.ArgumentParser(prog='Pair_Alignment')
        args = parser.parse_args()
        args.task = 'train'
        args.configs = ('tests/'+
                        'full_code_vs_simulation_tests/'+
                        'two-tkf92_two-hky/'+
                        'two-tkf92_two-hky85_recover-params.json')
        
        args = read_config_file(args.configs, parser)
        
        # train model 
        dload_lst = init_dataloaders( args,
                                      'train',
                                      training_argparse = None )
        
        train_fn( args, dload_lst )
        

class TestAgainstSimulationTwoTKF92TwoHKY85(unittest.TestCase):
    def test_losses_are_close(self):
        """
        compare joint loglikelihoods per sample
        """
        def load_all_losses(path):
            file_lst = [file for file in os.listdir(path) if file.endswith('FINAL-LOGLIKES.tsv')
                        and file.startswith('test-set')]
            df = []
            for file in file_lst:
                in_file = f'{path}/{file}'
                df.append( pd.read_csv(in_file, sep='\t', index_col=0) )
            return pd.concat(df)
        loaded_losses_df = load_all_losses('RESULTS_load_params/logfiles')
        recovered_losses_df = load_all_losses('RESULTS_recover_params/logfiles')
        
        # make sure results are synced
        assert (loaded_losses_df['ancestor'] == recovered_losses_df['ancestor']).all()
        assert (loaded_losses_df['descendant'] == recovered_losses_df['descendant']).all()
        
        # compile differences, take note of largest one
        df = pd.DataFrame({ 'alignment_len': loaded_losses_df['alignment_len'],
                            'loaded': loaded_losses_df['joint_logP'].to_numpy(),
                            'recovered': recovered_losses_df['joint_logP'].to_numpy()})
        df['absolute_diff']= np.abs( loaded_losses_df['joint_logP'] - recovered_losses_df['joint_logP'] )
        df['relative_diff']= np.abs( loaded_losses_df['joint_logP'] - recovered_losses_df['joint_logP'] ) / np.abs(loaded_losses_df['joint_logP'])
        df = df.sort_values(by='relative_diff', ascending=False)
        
        # save this for later
        sample_with_greatest_skew = df[df["relative_diff"] == df["relative_diff"].max()].iloc[0]
        with open(f'differences_in_loglikes.tsv','w') as g:
            g.write(f'# Largest difference\n')
            for idx in sample_with_greatest_skew.index:
                val = sample_with_greatest_skew.loc[idx].item()
                g.write(f'# {idx}: {val}\n')
            g.write('#\n')
            df.to_csv(g, sep='\t')
        
        # this will be skewed by more unreliable short alignments, so remove those 
        #  before this assert statement
        temp_df = df[df['alignment_len'] >= 10]
        npt.assert_array_less( temp_df["relative_diff"].max(), 0.01 )
        print("CHECK PARAMETER FILES MANUALLY")
        
if __name__ == '__main__':
    retrain()
    unittest.main()