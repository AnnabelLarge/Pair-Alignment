#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 18:16:22 2025

@author: annabel
"""
def share_top_level_args(args):    
    # these models have sequence embedding trunks
    if args.pred_model_type == 'feedforward':
        args.anc_enc_config['base_alphabet_size'] = args.base_alphabet_size
        args.desc_dec_config['base_alphabet_size'] = args.base_alphabet_size
        args.anc_enc_config['seq_padding_idx'] = args.seq_padding_idx
        args.desc_dec_config['seq_padding_idx'] = args.seq_padding_idx
    
    elif args.pred_model_type.startswith('neural_hmm'):
        args.anc_enc_config['base_alphabet_size'] = args.base_alphabet_size
        args.desc_dec_config['base_alphabet_size'] = args.base_alphabet_size
        args.anc_enc_config['seq_padding_idx'] = args.seq_padding_idx
        args.desc_dec_config['seq_padding_idx'] = args.seq_padding_idx
        args.pred_config['loss_type'] = args.loss_type
        args.pred_config['emission_alphabet_size'] = args.emission_alphabet_size
        
    args.pred_config['full_alphabet_size'] = args.full_alphabet_size
    args.pred_config['seq_padding_idx'] = args.seq_padding_idx
    
    
    
def fill_with_default_values(args):
    ### top-level defaults
    provided_args = list(vars(args).keys())
    
    if 'seq_padding_idx' not in provided_args:
        args.seq_padding_idx = 0
    
    if 'align_padding_idx' not in provided_args:
        args.align_padding_idx = -9
    
    if 'emission_alphabet_size' not in provided_args:
        args.emission_alphabet_size = 20

    if 'base_alphabet_size' not in provided_args:
        args.base_alphabet_size = args.emission_alphabet_size + 3
        
    if 'full_alphabet_size' not in provided_args:
        args.full_alphabet_size = 44
    
    
    ### extra defaults
    if args.pred_model_type == 'feedforward':
        if 'chunk_length' not in provided_args:
            args.chunk_length = 512
        
        # missing some things from tkf models
        args.times_from = None
        args.num_site_classes = 1
        args.loss_type = 'cond'
    
    elif args.pred_model_type.startswith('neural_hmm'):
        if 'chunk_length' not in provided_args:
            args.chunk_length = 512
        
        if 'num_site_classes' not in provided_args:
            args.num_site_classes = 1
            
        indel_config_keys = list( args.pred_config['indels_config'].keys() )
        if 'safe_grads' not in indel_config_keys:
            args.pred_config['indels_config']['safe_grads '] = False
            
        if 'tkf_err' not in indel_config_keys:
            args.pred_config['indels_config']['tkf_err '] = 1e-4
    
    elif args.pred_model_type == 'pair_hmm':
        if 'num_site_classes' not in provided_args:
            args.num_site_classes = 1
            
        indel_config_keys = list( args.pred_config['indels_config'].keys() )
        if 'safe_grads' not in indel_config_keys:
            args.pred_config['indels_config']['safe_grads '] = False
            
        if 'tkf_err' not in indel_config_keys:
            args.pred_config['indels_config']['tkf_err '] = 1e-4
    
    
    
def enforce_valid_defaults(args):
    if args.anc_model_type is None:
        args.interms_for_tboard['ancestor_embeddings'] = False
        args.interms_for_tboard['encoder_sow_outputs'] = False

    if args.desc_model_type is None:
        args.interms_for_tboard['descendant_embeddings'] = False
        args.interms_for_tboard['decoder_sow_outputs'] = False
    