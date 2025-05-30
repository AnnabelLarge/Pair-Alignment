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
                        'one_tkf92_two_hky-mix/'+
                        'two-hky85-tkf92_score_original.json')
        
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
                        'one_tkf92_two_hky-mix/'+
                        'two-hky85-tkf92_recover-params.json')
        
        args = read_config_file(args.configs, parser)
        
        # train model 
        dload_lst = init_dataloaders( args,
                                      'train',
                                      training_argparse = None )
        
        train_fn( args, dload_lst )
        

class TestAgainstSimulationTKF92TwoHKY85Mixes(unittest.TestCase):
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
        assert temp_df["relative_diff"].max() <= 0.01
    
    
    def test_sub_emit_close(self):
        """
        compare joint prob of substitution matrices
        """
        with open(f'differences_in_sub_emit_prob.tsv', 'w') as g:
            ### emission at match sites
            true = load_mat(f'RESULTS_load_params/'+
                            f'out_arrs/'+
                            f'_joint_prob_emit_at_match.npy')
            
            recov = load_mat(f'RESULTS_recover_params/'+
                             f'out_arrs/'+
                             f'_joint_prob_emit_at_match.npy')
            
            abs_diff = np.abs(true - recov)
            rel_diff = abs_diff / np.abs(true)
            g.write(f'Joint Substitution Matrix differences:\n')
            g.write(f'max absolute: {abs_diff.max()}\n')
            g.write(f'max relative: {rel_diff.max()}\n')
            g.write('\n')
        
        npt.assert_allclose( true, recov, rtol=0, atol=0.01)
        
    def test_ins_emit_close(self):
        """
        compare equilibrium distributions
        """
        with open(f'differences_in_ins_emit_prob.tsv', 'w') as g:
            ### emission at ins sites
            true = load_mat(f'RESULTS_load_params/'+
                            f'out_arrs/'+
                            f'_prob_emit_at_indel.npy')
            
            recov = load_mat(f'RESULTS_recover_params/'+
                             f'out_arrs/'+
                             f'_prob_emit_at_indel.npy')
            
            abs_diff = np.abs(true - recov)
            rel_diff = abs_diff / np.abs(true)
            g.write(f'Equilibrium Distribution differences:\n')
            g.write(f'max absolute: {abs_diff.max()}\n')
            g.write(f'max relative: {rel_diff.max()}\n')
            g.write('\n')
            
        npt.assert_allclose( true, recov, rtol=0, atol=0.01)
            
        
    def test_transit_close(self):
        """
        compare transition probabilities
        """
        with open(f'differences_in_trans_prob.tsv', 'w') as g:
            ### transition matrix
            true = load_mat(f'RESULTS_load_params/'+
                            f'out_arrs/'+
                            f'_joint_transit_matrix.npy')
            
            recov = load_mat(f'RESULTS_recover_params/'+
                             f'out_arrs/'+
                             f'_joint_transit_matrix.npy')
        
            abs_diff = np.abs(true - recov)
            rel_diff = abs_diff / np.abs(true)
            g.write(f'Joint Transition Matrix differences:\n')
            g.write(f'max absolute: {abs_diff.max()}\n')
            g.write(f'max relative: {rel_diff.max()}\n')
            g.write('\n')
            
        npt.assert_allclose( true, recov, rtol=0, atol=0.01)

if __name__ == '__main__':
    retrain()
    unittest.main()