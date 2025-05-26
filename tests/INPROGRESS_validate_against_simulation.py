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
def read_config_file(config_file):
    with open(config_file, 'r') as f:
        contents = json.load(f)
        t_args = argparse.Namespace()
        t_args.__dict__.update(contents)
        args = parser.parse_args(namespace=t_args)
    return args

def remove_folder(name):
    if name in os.listdir():
        shutil.rmtree(name)


# class TestMainCodeGTR(unittest.TestCase):
#     def test_main_code_one_gtr(self): 
    
    
# ##########################################
# ### Score original using training code   #
# ##########################################
# # make sure this doesn't exist
# remove_folder('RESULTS_load_params')

# # parse the arguments
# parser = argparse.ArgumentParser(prog='Pair_Alignment')
# args = parser.parse_args()
# args.task = 'train'
# args.configs = ('tests/'+
#                 'full_code_vs_simulation_tests/'+
#                 'one_tkf92_hky/'+
#                 'one-class_hky85-tkf92_score_original.json')

# args = read_config_file(args.configs)

# # "train" model i.e. run full evaluation script without updating 
# #   gradients
# dload_lst = init_dataloaders( args,
#                               'train',
#                               training_argparse = None )

# train_fn( args, dload_lst )

# del parser, args, dload_lst



# ##########################################
# ### Fit parameters using training code   #
# ##########################################
# # make sure this doesn't exist
# remove_folder('RESULTS_recover_params')

# # parse the arguments
# parser = argparse.ArgumentParser(prog='Pair_Alignment')
# args = parser.parse_args()
# args.task = 'train'
# args.configs = ('tests/'+
#                 'full_code_vs_simulation_tests/'+
#                 'one_tkf92_hky/'+
#                 'one-class_hky85-tkf92_recover-params.json')

# args = read_config_file(args.configs)

# # "train" model i.e. run full evaluation script without updating 
# #   gradients
# dload_lst = init_dataloaders( args,
#                               'train',
#                               training_argparse = None )

# train_fn( args, dload_lst )

# del parser, args, dload_lst


###########################################
### check that logprob arrays are close   #
###########################################
losses_from_loaded_dir = 'RESULTS_load_params/logfiles/test-set_pt0_FINAL-LOGLIKES.tsv'
losses_from_recovery_dir = 'RESULTS_recover_params/logfiles/test-set_pt0_FINAL-LOGLIKES.tsv'

loaded_losses_df = pd.read_csv(losses_from_loaded_dir, sep='\t', index_col=0)
recovered_losses_df = pd.read_csv(losses_from_recovery_dir, sep='\t', index_col=0)

# make sure results are synced
assert (loaded_losses_df['ancestor'] == recovered_losses_df['ancestor']).all()
assert (loaded_losses_df['descendant'] == recovered_losses_df['descendant']).all()


diff = np.abs( loaded_losses_df['joint_logP'] - recovered_losses_df['joint_logP'] ) / np.abs(loaded_losses_df['joint_logP'])


# if __name__ == '__main__':
#     unittest.main()