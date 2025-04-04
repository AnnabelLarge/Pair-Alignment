#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:45:05 2025

@author: annabel
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
from models.neural_hmm_predict.initializers import create_all_tstates 
from train_eval_fns.neural_hmm_training_fns import ( train_one_batch,
                                                      eval_one_batch )
from train_eval_fns.full_length_final_eval_wrapper import final_eval_wrapper


###############################################################################
### INITIALIZE PARSER, DATALOADER   ###########################################
###############################################################################
in_file = 'neural_hmm_load_all.json'

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
    g.write(f'Neural TKF92 with markovian site classes over emissions\n')
    g.write(f'  - preset name: {args.pred_config["preset_name"]} \n')
    g.write(f'  - Loss function: {args.loss_type}\n')
    g.write(f'  - Normalizing losses by: {args.norm_loss_by}\n')


### save updated config, provide filename for saving model parameters
encoder_save_model_filename = args.model_ckpts_dir + '/'+ f'ANC_ENC.pkl'
decoder_save_model_filename = args.model_ckpts_dir + '/'+ f'DESC_DEC.pkl'
finalpred_save_model_filename = args.model_ckpts_dir + '/'+ f'FINAL_PRED.pkl'
all_save_model_filenames = [encoder_save_model_filename, 
                            decoder_save_model_filename,
                            finalpred_save_model_filename]


### extract data from dataloader_dict
test_dset = dataloader_dict['test_dset']
test_dl = dataloader_dict['test_dl']
args.pred_config['equilibr_config']['training_dset_aa_counts'] = test_dset.aa_counts
print('filling in amino acid counts with TEST dataset')


###########################################################################
### 2: INITIALIZE MODEL PARTS, OPTIMIZER  #################################
###########################################################################
print('2: model init')
with open(args.logfile_name,'a') as g:
    g.write('\n')
    g.write(f'2: model init\n')

######################
### for all models   #
######################
# init the optimizer
tx = build_optimizer(args)

# initialize dummy array of times
num_timepoints = test_dset.retrieve_num_timepoints(times_from = args.pred_config['times_from'])
dummy_t_array = jnp.empty( (num_timepoints, 1) ) #(T, B=1)

# split a new rng key
rngkey, model_init_rngkey = jax.random.split(rngkey, num=2)


### init sizes
global_seq_max_length = test_dset.global_seq_max_length
largest_seqs = (args.batch_size, global_seq_max_length)

if args.use_scan_fns:
    max_dim1 = args.chunk_length

elif not args.use_scan_fns:
    max_dim1 = test_dset.global_align_max_length - 1
  
largest_aligns = (args.batch_size, max_dim1)
del max_dim1

seq_shapes = [largest_seqs, largest_aligns]


### fn to handle jit-compiling according to alignment length
parted_determine_alignlen_bin = partial(determine_alignlen_bin,  
                                        chunk_length = args.chunk_length,
                                        seq_padding_idx = args.seq_padding_idx)
jitted_determine_alignlen_bin = jax.jit(parted_determine_alignlen_bin)
del parted_determine_alignlen_bin

parted_determine_seqlen_bin = partial(determine_seqlen_bin,
                                      chunk_length = args.chunk_length, 
                                      seq_padding_idx = args.seq_padding_idx)
jitted_determine_seqlen_bin = jax.jit(parted_determine_seqlen_bin)
del parted_determine_seqlen_bin


### initialize functions, determine concat_fn
out = create_all_tstates( seq_shapes = seq_shapes, 
                          dummy_t_array = dummy_t_array,
                          tx = tx, 
                          model_init_rngkey = model_init_rngkey,
                          tabulate_file_loc = args.model_ckpts_dir,
                          anc_model_type = args.anc_model_type, 
                          desc_model_type = args.desc_model_type, 
                          pred_model_type = args.pred_model_type, 
                          anc_enc_config = args.anc_enc_config, 
                          desc_dec_config = args.desc_dec_config, 
                          pred_config = args.pred_config,
                          )  
all_trainstates, all_model_instances, concat_fn = out
del out


### parted and jit-compiled training_fn
t_array = test_dset.return_time_array()
# parted_train_fn = partial( train_one_batch,
#                            all_model_instances = all_model_instances,
#                            norm_loss_by = args.norm_loss_by,
#                            interms_for_tboard = args.interms_for_tboard,
#                            t_array = t_array,  
#                            loss_type = args.loss_type,
#                            exponential_dist_param = args.pred_config['exponential_dist_param'],
#                            concat_fn = concat_fn
#                           )

# train_fn_jitted = jax.jit(parted_train_fn, 
#                           static_argnames = ['max_seq_len',
#                                              'max_align_len'])
# del parted_train_fn


### eval_fn used in training loop (to monitor progress)
# pass arguments into eval_one_batch; make a parted_eval_fn that doesn't
#   return any intermediates
# no_returns = {'encoder_sow_outputs': False,
#               'decoder_sow_outputs': False,
#               'finalpred_sow_outputs': False,
#               'gradients': False,
#               'weights': False,
#               'ancestor_embeddings': False,
#               'descendant_embeddings': False,
#               'forward_pass_outputs': False,
#               'final_logprobs': False}
extra_args_for_eval = dict()

# if this is a transformer model, will have extra arguments for eval funciton
if (args.anc_model_type == 'Transformer' or args.desc_model_type == 'Transformer'):
    extra_args_for_eval['output_attn_weights'] = False

parted_eval_fn = partial( eval_one_batch,
                          all_model_instances = all_model_instances,
                          norm_loss_by = args.norm_loss_by,
                          interms_for_tboard = args.interms_for_tboard,
                          t_array = t_array,  
                          loss_type = args.loss_type,
                          exponential_dist_param = args.pred_config['exponential_dist_param'],
                          concat_fn = concat_fn
                          )

# jit compile this eval function
eval_fn_jitted = jax.jit(parted_eval_fn, 
                          static_argnames = ['max_seq_len',
                                             'max_align_len'])
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
                       jitted_determine_seqlen_bin = jitted_determine_seqlen_bin,
                       jitted_determine_alignlen_bin = jitted_determine_alignlen_bin,
                       eval_fn_jitted = eval_fn_jitted,
                       out_alph_size = args.full_alphabet_size, 
                       save_arrs = args.save_arrs,
                       interms_for_tboard = args.interms_for_tboard, 
                       logfile_dir = args.logfile_dir,
                       out_arrs_dir = args.out_arrs_dir,
                       outfile_prefix = f'test-set',
                       tboard_writer = writer)
    
