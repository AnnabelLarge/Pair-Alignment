#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:33:01 2025

@author: annabel

Load parameters and evaluate likelihoods for markovian
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
from dloaders.init_full_len_dset import init_full_len_dset
from models.simple_site_class_predict.initializers import init_pairhmm_markov_sites as init_pairhmm
from train_eval_fns.markovian_site_classes_training_fns import ( train_one_batch,
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

dataloader_dict = init_full_len_dset(args, 'train')
del dataloader_dict['training_dset']
del dataloader_dict['training_dl']



###########################################################################
### 0: CHECK CONFIG; IMPORT APPROPRIATE MODULES   #########################
###########################################################################
### edit the argparse object in-place
enforce_valid_defaults(args)
fill_with_default_values(args)
share_top_level_args(args)


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
    g.write(f'PairHMM TKF92 with markovian site classes over emissions\n')
    g.write(f'  - Loading all parameters (unit testing evaluation functions) \n')
    g.write( (f'  - Number of site classes: '+
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


### init sizes
# (B, L, 3)
max_dim1 = test_dset.global_align_max_length 
largest_aligns = jnp.zeros( (args.batch_size, max_dim1, 3), dtype=int )
del max_dim1

### fn to handle jit-compiling according to alignment length
parted_determine_alignlen_bin = partial(determine_alignlen_bin,  
                                        chunk_length = args.chunk_length,
                                        seq_padding_idx = args.seq_padding_idx)
jitted_determine_alignlen_bin = jax.jit(parted_determine_alignlen_bin)
del parted_determine_alignlen_bin


####################################
### pairHMM with markovian sites   #
####################################
### initialize functions
all_trainstates = init_pairhmm( seq_shapes = largest_aligns, 
                                dummy_t_array = dummy_t_array,
                                tx = tx, 
                                model_init_rngkey = model_init_rngkey,
                                pred_config = args.pred_config)

### part+jit eval function
t_array = test_dset.return_time_array()
parted_eval_fn = partial( eval_one_batch,
                           interms_for_tboard = {'finalpred_sow_outputs': False},
                           t_array = t_array )

eval_fn_jitted = jax.jit(parted_eval_fn, 
                          static_argnames = ['max_align_len'])
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
                       jitted_determine_alignlen_bin = jitted_determine_alignlen_bin,
                       logfile_dir = args.logfile_dir,
                       out_arrs_dir = args.out_arrs_dir,
                       outfile_prefix = f'test-set')

