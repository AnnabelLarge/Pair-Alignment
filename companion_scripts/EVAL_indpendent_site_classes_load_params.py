#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:33:01 2025

@author: annabel

Load parameters and evaluate likelihoods for an independent
  site class model

"""
# general python
import os
import shutil
from tqdm import tqdm
from time import process_time
from time import time as wall_clock_time
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None # this is annoying
import pickle
from functools import partial
import platform
import argparse
import json

# jax/flax stuff
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import optax

# pytorch imports
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# custom function/classes imports (in order of appearance)
from train_eval_fns.build_optimizer import build_optimizer
from utils.write_config import write_config
from utils.edit_argparse import (enforce_valid_defaults,
                                 fill_with_default_values,
                                 share_top_level_args)
from utils.setup_training_dir import setup_training_dir
from utils.sequence_length_helpers import (determine_seqlen_bin, 
                                           determine_alignlen_bin)
from utils.tensorboard_recording_utils import (write_times,
                                               write_optional_outputs_during_training)

# specific to training this model
from dloaders.init_counts_dset import init_counts_dset
from models.simple_site_class_predict.initializers import init_pairhmm_indp_sites as init_pairhmm
from train_eval_fns.indp_site_classes_training_fns import ( train_one_batch,
                                                            eval_one_batch,
                                                            final_eval_wrapper )


###############################################################################
### INITIALIZE PARSER, DATALOADER   ###########################################
###############################################################################
in_file = 'tkf92_load_params.json'

parser = argparse.ArgumentParser(prog='Pair_Alignment')

def read_config_file(config_file):
    with open(config_file, 'r') as f:
        contents = json.load(f)
        
        t_args = argparse.Namespace()
        t_args.__dict__.update(contents)
        args = parser.parse_args(namespace=t_args)
    return args

args = read_config_file(in_file)

dataloader_dict = init_counts_dset(args, 'train')
del dataloader_dict['training_dset']
del dataloader_dict['training_dl']



###########################################################################
### 0: CHECK CONFIG; IMPORT APPROPRIATE MODULES   #########################
###########################################################################
### edit the argparse object in-place
enforce_valid_defaults(args)
fill_with_default_values(args)
share_top_level_args(args)


### needed these for larger combined training function; see if you can
###   get rid of them
have_acc = False
have_time_array = True
have_full_length_alignments = False



###########################################################################
### 1: SETUP   ############################################################
###########################################################################
### initial setup of misc things
# setup the working directory (if not done yet) and this run's sub-directory
setup_training_dir(args, assert_no_overwrite=False)

# initial random key, to carry through execution
rngkey = jax.random.key(args.rng_seednum)

# setup tensorboard writer
writer = SummaryWriter(args.tboard_dir)

# create a new logfile
with open(args.logfile_name,'w') as g:
    g.write(f'PairHMM TKF92 with independent site classes over emissions\n')
    g.write(f'  - Loading all parameters (unit testing evaluation functions) \n')
    g.write( (f'  - Number of site classes for substitution model: '+
              f'{args.pred_config["num_emit_site_classes"]}\n' )
            )
    g.write(f'  - Loss function: {args.loss_type}\n')
    g.write(f'  - Normalizing losses by: {args.norm_loss_by}\n')


### save updated config, provide filename for saving model parameters
finalpred_save_model_filename = args.model_ckpts_dir + '/'+ f'FINAL_PRED.pkl'
write_config(args = args, out_dir = args.model_ckpts_dir)


### extract data from dataloader_dict
test_dset = dataloader_dict['test_dset']
test_dl = dataloader_dict['test_dl']



###########################################################################
### 2: INITIALIZE MODEL PARTS, OPTIMIZER  #################################
###########################################################################
print('2: model init')
with open(args.logfile_name,'a') as g:
    g.write('\n')
    g.write(f'2: model init\n')


# init the optimizer, split a new rng key
tx = build_optimizer(args)
rngkey, model_init_rngkey = jax.random.split(rngkey, num=2)


### determine shapes for init
# time
num_timepoints = test_dset.retrieve_num_timepoints(times_from = args.pred_config['times_from'])
dummy_t_array = jnp.empty( (num_timepoints, ) )

# counts array
B = args.batch_size
alph = args.emission_alphabet_size
num_states = 4 if args.pred_config['indel_model_type'].startswith('tkf') else 3
dummy_subCounts = jnp.empty( (B, alph, alph) )
dummy_insCounts = jnp.empty( (B, alph) )
dummy_delCounts = jnp.empty( (B, alph) )
dummy_transCounts = jnp.empty( (B, num_states, num_states) )

seq_shapes = [dummy_subCounts,
              dummy_insCounts,
              dummy_delCounts,
              dummy_transCounts]


### initialize functions
all_trainstates = init_pairhmm( seq_shapes = seq_shapes, 
                                dummy_t_array = dummy_t_array,
                                tx = tx, 
                                model_init_rngkey = model_init_rngkey,
                                pred_config = args.pred_config,
                                )


### part+jit eval function
# note: if you want to use a different time per sample, will
#  have to change this jit compilation
t_array = test_dset.return_time_array()
parted_eval_fn = partial( eval_one_batch,
                           interms_for_tboard = {'finalpred_sow_outputs': False},
                           t_array = t_array )
eval_fn_jitted = jax.jit(parted_eval_fn)
del parted_eval_fn



###########################################################################
### 3: EVAL   #############################################################
###########################################################################
print(f'3: main eval loop')
with open(args.logfile_name,'a') as g:
    g.write('\n')
    g.write(f'3: main eval loop\n')


with jax.disable_jit():
    final_eval_wrapper(dataloader = test_dl, 
                       dataset = test_dset, 
                       best_trainstates = all_trainstates, 
                       eval_fn_jitted = eval_fn_jitted,
                       logfile_dir = args.logfile_dir,
                       out_arrs_dir = args.out_arrs_dir,
                       outfile_prefix = f'test-set')

