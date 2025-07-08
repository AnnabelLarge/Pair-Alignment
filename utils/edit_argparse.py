#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 18:16:22 2025

@author: annabel
"""
def fill_with_default_values(args):
    ### top-level defaults
    provided_args = list(vars(args).keys())
    
    if 'seq_padding_idx' not in provided_args:
        args.seq_padding_idx = 0
    
    if 'align_padding_idx' not in provided_args:
        args.align_padding_idx = -9
    
    if 'base_alphabet_size' not in provided_args:
        args.base_alphabet_size = args.emission_alphabet_size + 3
    
    if 'update_grads' not in provided_args:
        args.update_grads = True
    
    if 'gap_tok' not in provided_args:
        args.gap_tok = 43
    
    ### extra defaults
    if args.pred_model_type == 'feedforward':
        if 'chunk_length' not in provided_args:
            args.chunk_length = 512
        
        # don't include bos
        if 'full_alphabet_size' not in provided_args:
            args.full_alphabet_size = 43
        
        # missing some things from tkf models
        args.times_from = None
        
    
    elif args.pred_model_type.startswith('neural_hmm'):
        # keep bos for now
        if 'full_alphabet_size' not in provided_args:
            args.full_alphabet_size = 44
            
        if 'chunk_length' not in provided_args:
            args.chunk_length = 512
        
    elif args.pred_model_type in ['pairhmm_indp_sites',
                                  'pairhmm_frag_and_site_classes']:
    
        if args.pred_model_type == 'pairhmm_indp_sites':
            args.pred_config['num_tkf_fragment_classes'] = 1
        
        elif args.pred_model_type == 'pairhmm_frag_and_site_classes':
            args.pred_config['num_tkf_fragment_classes'] = args.pred_config['num_mixtures']

        
    
def enforce_valid_defaults(args):
    provided_args = list(vars(args).keys())
    
    if ('anc_model_type' in provided_args) and (args.anc_model_type is None):
        args.interms_for_tboard['ancestor_embeddings'] = False
        args.interms_for_tboard['encoder_sow_outputs'] = False

    if ('desc_model_type' in provided_args) and (args.desc_model_type is None):
        args.interms_for_tboard['descendant_embeddings'] = False
        args.interms_for_tboard['decoder_sow_outputs'] = False
    
    ### if you're not updating gradients, don't run any training updates
    if not args.update_grads:
        args.num_epochs = 1
    

def share_top_level_args(args):    
    args.pred_config['seq_padding_idx'] = args.seq_padding_idx
    args.pred_config['align_padding_idx'] = args.align_padding_idx
    args.pred_config['base_alphabet_size'] = args.base_alphabet_size
    
    if args.pred_model_type == 'feedforward':
        args.anc_enc_config['base_alphabet_size'] = args.base_alphabet_size
        args.anc_enc_config['seq_padding_idx'] = args.seq_padding_idx

        args.desc_dec_config['base_alphabet_size'] = args.base_alphabet_size
        args.desc_dec_config['seq_padding_idx'] = args.seq_padding_idx
        
        args.pred_config['full_alphabet_size'] = args.full_alphabet_size
    
    elif args.pred_model_type.startswith('neural_hmm'):
        args.anc_enc_config['base_alphabet_size'] = args.base_alphabet_size
        args.anc_enc_config['seq_padding_idx'] = args.seq_padding_idx
        
        args.desc_dec_config['base_alphabet_size'] = args.base_alphabet_size
        args.desc_dec_config['seq_padding_idx'] = args.seq_padding_idx
        
        args.pred_config['emission_alphabet_size'] = args.emission_alphabet_size
        args.pred_config['full_alphabet_size'] = args.full_alphabet_size

        args.pred_config['emissions_postproc_config']['emission_alphabet_size'] = args.emission_alphabet_size
        args.pred_config['transitions_postproc_config']['emission_alphabet_size'] = args.emission_alphabet_size
        args.pred_config['emissions_postproc_config']['full_alphabet_size'] = args.full_alphabet_size
        args.pred_config['transitions_postproc_config']['full_alphabet_size'] = args.full_alphabet_size

    elif args.pred_model_type in ['pairhmm_indp_sites',
                                  'pairhmm_frag_and_site_classes']:
        args.pred_config['gap_tok'] = args.gap_tok
        args.pred_config['norm_loss_by'] = args.norm_loss_by
        args.pred_config['emission_alphabet_size'] = args.emission_alphabet_size
    

    