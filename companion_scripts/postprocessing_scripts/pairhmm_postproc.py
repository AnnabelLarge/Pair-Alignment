#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 20:18:57 2025

@author: annabel
"""    
import argparse
import sys
import os

from pairhmm_postprocessing.extract_best_runs_by_total_joint_loglike import main as extract_best
from pairhmm_postprocessing.extract_ave_losses import main as compile_losses



def main(folder, indp_or_frag, t_per_samp_bool_flag):
    assert 'pairhmm_postprocessing' in os.listdir()
    assert indp_or_frag in ['indp','frag']
    
    
    # correct function imports
    if indp_or_frag == 'indp':
        from pairhmm_postprocessing.parse_variable_length_vars_indp_sites import main as reparse_variable_len_params
        from pairhmm_postprocessing.aggregate_params_in_folder_indp_sites  import main as aggregate_variable_len_params
    
    elif indp_or_frag == 'frag':
        from pairhmm_postprocessing.parse_variable_length_vars_fragment_classes import main as reparse_variable_len_params
        from pairhmm_postprocessing.aggregate_params_in_folder_fragment_classes import main as aggregate_variable_len_params
        
    
    
    # extract all and best joint loglikelihoods
    extract_best(fold = folder, 
                 dset_prefix = 'train-set')
    extract_best(fold = folder, 
                 dset_prefix = 'test-set')
    
    # extract all and best likelihood metrics
    compile_losses(fold = folder, 
                   dset_prefix = 'train-set', 
                   all_or_best = 'ALL')
    compile_losses(fold = folder, 
                   dset_prefix = 'train-set', 
                   all_or_best = 'BEST')
    compile_losses(fold = folder, 
                   dset_prefix = 'test-set', 
                   all_or_best = 'ALL')
    compile_losses(fold = folder, 
                   dset_prefix = 'test-set', 
                   all_or_best = 'BEST')
    
    # for every run, reparse the variable-length parameters in a file that's
    #   easier to read
    reparse_variable_len_params(fold = folder)
    
    # aggregate the varaible-length params from the best runs
    aggregate_variable_len_params(fold = folder,
                                  t_per_samp = t_per_samp_bool_flag)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='pairhmm_postproc')
    
    parser.add_argument('-folder',
                        type=str,
                        required=True,
                        help='folder containing result directories, starting with "RESULTS"')
    
    parser.add_argument('-indp_or_frag',
                        type=str,
                        required=True,
                        choices = ['indp','frag'],
                        help='Independent sites, or Fragment-and-site classes?')
    
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument('-frag', 
                        dest='indp_or_frag', 
                        action='store_const', 
                        const='frag',
                        help='Using a fragment AND site class pairhmm')
    group1.add_argument('-indp', 
                        dest='indp_or_frag', 
                        action='store_const', 
                        const='indp',
                        help='Using an indepented site class pairhmm')
    
    group2 = parser.add_mutually_exclusive_group(required=True)
    group2.add_argument('-t_per_samp', 
                        dest='t_per_samp_bool_flag', 
                        action='store_true', 
                        help='Have unique branch length per sample')
    group2.add_argument('-marg_over_grid', 
                        dest='t_per_samp_bool_flag', 
                        action='store_false', 
                        help='Marginalize over time grid')

    args = parser.parse_args()
    
    main(folder = args.folder, 
         indp_or_frag = args.indp_or_frag, 
         t_per_samp_bool_flag = args.t_per_samp_bool_flag)
    
    
    