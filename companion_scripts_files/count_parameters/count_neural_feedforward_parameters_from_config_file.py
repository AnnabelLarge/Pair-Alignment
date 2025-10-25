#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 15:50:37 2025

@author: annabel
"""
# general python
import os
import shutil
import glob
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None # this is annoying
import pickle
from functools import partial
import argparse
import json

# jax/flax stuff
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import optax

# custom function/classes imports
from train_eval_fns.build_optimizer import build_optimizer

# specific to training this model
from models.neural_shared.neural_initializer import create_all_tstates 

from utils.edit_argparse import enforce_valid_defaults
from utils.edit_argparse import feedforward_fill_with_default_values as fill_with_default_values
from utils.edit_argparse import feedforward_share_top_level_args as share_top_level_args

def main(args): 
    fill_with_default_values(args)
    enforce_valid_defaults(args)
    share_top_level_args(args)
    
    args.pred_config['training_dset_emit_counts'] = jnp.array( [1/20]*20 )
    # args.pred_config['emissions_postproc_config']['training_dset_emit_counts'] = jnp.array( [1/20]*20 )
    
    # init the optimizer, split a new rng key
    tx = build_optimizer(args)
    
    # batch provided to train/eval functions consist of:
    # 1.) unaligned sequences (B, L_seq, 2)
    # 2.) aligned data matrices (B, L_align, 5)
    # 3.) time per sample (if applicable) (B,)
    # 4, not used.) sample index (B,)
    largest_seqs = (args.batch_size, 513) 
    largest_aligns = (args.batch_size, 513) 
    dummy_t_for_each_sample = jnp.empty( (args.batch_size,) )
    seq_shapes = [largest_seqs, largest_aligns, dummy_t_for_each_sample]
    
    
    ### initialize trainstate objects, concat_fn
    out = create_all_tstates( seq_shapes = seq_shapes, 
                              tx = tx, 
                              model_init_rngkey = jax.random.key(0),
                              tabulate_file_loc = None,
                              anc_model_type = args.anc_model_type, 
                              desc_model_type = args.desc_model_type, 
                              pred_model_type = args.pred_model_type, 
                              anc_enc_config = args.anc_enc_config, 
                              desc_dec_config = args.desc_dec_config, 
                              pred_config = args.pred_config,
                              t_array_for_all_samples = None,
                              )  
    all_trainstates, _, _ = out
    
    def flatten_dict(d, parent_key="", sep="/"):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    anc_params = flatten_dict( all_trainstates[0].params['params'] )
    desc_params = flatten_dict( all_trainstates[1].params['params'] )
    outproj_params = flatten_dict( all_trainstates[2].params['params'] )

    param_count = 0
    anc_count = 0
    desc_count = 0
    final_count = 0

    for val in anc_params.values():
        param_count += val.size
        anc_count += val.size

    for val in desc_params.values():
        param_count += val.size
        desc_count += val.size

    for val in outproj_params.values():
        param_count += val.size
        final_count += val.size

    print(f'anc seq embedder: {anc_count}')
    print(f'desc seq embedder: {desc_count}')
    print(f'prediction head: {final_count}')
    print(f'TOTAL: {param_count}')
    


if __name__ == '__main__':
    pass