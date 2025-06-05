#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 20:18:57 2025

@author: annabel
"""
import sys
import os

from pairhmm_postprocessing.extract_best_runs_by_total_joint_loglike import main as extract_best
from pairhmm_postprocessing.extract_ave_losses import main as compile_losses


folder = sys.argv[1]
indp_or_frag = sys.argv[2]
t_per_samp = sys.argv[3]

if t_per_samp in ['true', 'True', 't', 'T']:
    t_per_samp = True

elif t_per_samp in ['false', 'False', 'f', 'F']:
    t_per_samp = False

else:
    raise ValueError

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
                              t_per_samp = t_per_samp)


