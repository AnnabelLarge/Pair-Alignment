#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 15:47:38 2025

@author: annabel
"""
import numpy as np
import argparse
import json
from tqdm import tqdm
from typing import Optional, Any

import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import optax

from dloaders.init_full_len_dset import init_full_len_dset
from models.neural_utils.neural_initializer import create_all_tstates
from train_eval_fns.build_optimizer import build_optimizer
from utils.sequence_length_helpers import (determine_seqlen_bin, 
                                           determine_alignlen_bin)
from models.sequence_embedders.concatenation_fns import extract_embs
from train_eval_fns.neural_tkf_train_eval import ( _preproc,
                                                   _one_hot_pad_with_zeros )



def test_neural_tkf_model_is_causal(config_file):
    print(f'Testing: {config_file}')
    
    ##############
    ### inputs   #
    ##############
    # read json file
    parser = argparse.ArgumentParser(prog='Pair_Alignment')
    with open(config_file, 'r') as f:
        contents = json.load(f)
        t_args = argparse.Namespace()
        t_args.__dict__.update(contents)
        args = parser.parse_args(namespace=t_args)
    del t_args, contents, f, parser
    
    # add to pred_config
    A = args.emission_alphabet_size
    args.pred_config['emission_alphabet_size'] = A
    args.pred_config['base_alphabet_size'] = A+3
    args.pred_config['training_dset_emit_counts'] = np.array( [1/A]*A ) 
    
    args.anc_enc_config['base_alphabet_size'] = A+3
    args.desc_dec_config['base_alphabet_size'] = A+3
    
    
    if args.pred_model_type == 'neural_hmm':
        args.pred_config['emissions_postproc_config']['emission_alphabet_size'] = A
        args.pred_config['emissions_postproc_config']['base_alphabet_size'] = A+3
        
        args.pred_config['transitions_postproc_config']['emission_alphabet_size'] = A
        args.pred_config['transitions_postproc_config']['base_alphabet_size'] = A+3
        
    
    
    #########################
    ### init a fake batch   #
    #########################
    ### unaligned seqs
    ancs = np.array( [[1, 3, 4, 5, 2, 0], 
                      [1, 3, 2, 0, 0, 0],
                      [1, 3, 4, 5, 6, 2],
                      [1, 3, 4, 5, 6, 2],
                      [1, 3, 4, 2, 0, 0]] )
    
    descs = np.array( [[1, 3, 2, 0, 0, 0],
                       [1, 3, 4, 5, 2, 0],
                       [1, 3, 4, 5, 6, 2],
                       [1, 3, 4, 5, 6, 2],
                       [1, 3, 4, 2, 0, 0]] )
    
    unaligned_seqs = np.stack( [ancs, descs], axis=-1 )
    del ancs, descs
    
    L_seq = unaligned_seqs.shape[1]
    
    
    ### aligned seqs
    ancs_aligned = np.array( [[1,  3,  4,  5,  2,  0,  0],
                              [1, 43,  3, 43,  2,  0,  0],
                              [1,  3,  4,  5,  6, 43,  2],
                              [1, 43,  3,  4,  5,  6,  2],
                              [1,  3,  4,  2,  0,  0,  0]] )
    
    descs_aligned = np.array( [[1, 43,  3, 43,  2,  0,  0],
                               [1,  3,  4,  5,  2,  0,  0],
                               [1, 43,  3,  4,  5,  6,  2],
                               [1,  3,  4,  5,  6, 43,  2],
                               [1,  3,  4,  2,  0,  0,  0]] )
    
    alignment_state = np.array( [[4, 3, 1, 3, 5, 0, 0],
                                 [4, 2, 1, 2, 5, 0, 0],
                                 [4, 3, 1, 1, 1, 2, 5],
                                 [4, 2, 1, 1, 1, 3, 5],
                                 [4, 1, 1, 5, 0, 0, 0]] )
    
    m_idx = np.array( [[1,  2,  3,  4, -9, -9, -9],
                       [1,  1,  2,  2, -9, -9, -9],
                       [1,  2,  3,  4,  5,  5, -9],
                       [1,  1,  2,  3,  4,  5, -9],
                       [1,  2,  3, -9, -9, -9, -9]] )
    
    n_idx = np.array( [[0,  0,  1,  1, -9, -9, -9],
                       [0,  1,  2,  3, -9, -9, -9],
                       [0,  0,  1,  2,  3,  4, -9],
                       [0,  1,  2,  3,  4,  4, -9],
                       [0,  1,  2, -9, -9, -9, -9]] )
    
    aligned_mats = np.stack( [ancs_aligned, descs_aligned, alignment_state, m_idx, n_idx], axis=-1 )
    del m_idx, n_idx, ancs_aligned, descs_aligned, alignment_state
    
    # dims
    B = aligned_mats.shape[0]
    L_align = aligned_mats.shape[1]
    L_seq = unaligned_seqs.shape[1]
    
    
    ### time
    if args.pred_config['times_from'] == 't_per_sample':
        t_per_sample = np.arange( 1, B+1 ) * 0.1 #(B,)
        t_array_for_all_samples = None
        unique_t_per_sample = True
    
    else:
        t_per_sample = None
        t_array_for_all_samples = np.arange( 1, 9 ) * 0.1 #(T)
        T = t_array_for_all_samples.shape[0]
        unique_t_per_sample = False
    
    batch = (unaligned_seqs, aligned_mats, t_per_sample, None)
        
        
    ######################
    ### init the model   #
    ######################
    tx = optax.adamw(learning_rate=0.005)
    
    ### determine shapes for init
    # sequence inputs
    largest_seqs = (args.batch_size, unaligned_seqs.shape[1])
    largest_aligns = (args.batch_size, aligned_mats.shape[1])
    
    # time
    if t_array_for_all_samples is not None:
        dummy_t_array_for_all_samples = jnp.empty( (t_array_for_all_samples.shape[0], ) )
        dummy_t_for_each_sample = None
    
    else:
        dummy_t_array_for_all_samples = None
        dummy_t_for_each_sample = jnp.empty( (args.batch_size,) )
    
    # batch provided to train/eval functions consist of:
    # 1.) unaligned sequences (B, L_seq, 2)
    # 2.) aligned data matrices (B, L_align, 5)
    # 3.) time per sample (if applicable) (B,)
    # 4, not used.) sample index (B,)
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
                              t_array_for_all_samples = dummy_t_array_for_all_samples
                              )  
    all_trainstates, all_model_instances, concat_fn = out
    del out, tx, largest_seqs, largest_aligns, dummy_t_array_for_all_samples
    del dummy_t_for_each_sample, seq_shapes
    
    
    ##################
    ### fake apply   #
    ##################
    def unit_test_eval(batch, 
                       all_trainstates,
                       all_model_instances,
                       t_array_for_all_samples,  
                       true_out: Optional[Any] = None,
                       extra_args_for_eval: dict = dict(),
                       gap_idx: int = 43,
                       seq_padding_idx: int = 0,
                       align_idx_padding: int = -9,
                       *args,
                       **kwargs):
        """
        modified version of train_eval_fns.neural_tkf_train_eval.eval_one_batch
        """
        # which times to use 
        if t_array_for_all_samples is None:
            times_for_matrices = batch[2] #(B,)
        
        elif t_array_for_all_samples is not None:
            times_for_matrices = t_array_for_all_samples #(T,)
            
        
        ##############
        ### UNPACK   #
        ##############
        # unpack
        encoder_trainstate, decoder_trainstate, finalpred_trainstate = all_trainstates
        encoder_instance, decoder_instance, _ = all_model_instances
        del all_trainstates, all_model_instances
        
        # clip to max lengths, split into prefixes and suffixes
        unaligned_seqs = batch[0] #(B, L, 2)
        aligned_mats = batch[1] #(B, L, 5)
        del batch
        
        
        ##########################
        ### PREPARE THE INPUTS   #
        ##########################
        # preprocess with helper
        out_dict = _preproc( unaligned_seqs = unaligned_seqs, 
                             aligned_mats = aligned_mats )
        anc_seqs = out_dict['anc_seqs']
        desc_seqs = out_dict['desc_seqs']
        align_idxes = out_dict['align_idxes']
        from_states = out_dict['from_states']
        
        # if this is the first time, generate true_out and return it
        if true_out is None:
            del true_out
            true_out = out_dict['true_out']
            return_true_out = True
        
        elif true_out is not None:
            return_true_out = False
        
        del out_dict
        
        
        #######################
        ### Apply the model   #
        #######################
        ### embed with ancestor encoder
        # anc_embeddings is (B, L_seq-1, H)
        out = encoder_instance.apply_seq_embedder_in_eval( seqs = anc_seqs,
                                                           tstate = encoder_trainstate,
                                                           sow_intermediates = False )
        anc_embeddings, _ = out
        del out
        
        
        ### embed with descendant decoder
        # desc_embeddings is (B, L_seq-1, H)
        out = decoder_instance.apply_seq_embedder_in_eval( seqs = desc_seqs,
                                                           tstate = decoder_trainstate,
                                                           sow_intermediates = False )
        desc_embeddings, _ = out
        del out
        
        
        ### extract embeddings
        out = extract_embs(anc_encoded = anc_embeddings, 
                           desc_encoded = desc_embeddings,
                           idx_lst = align_idxes,
                           seq_padding_idx = seq_padding_idx,
                           align_idx_padding = align_idx_padding)
        datamat_lst, padding_mask = out
        del out
        
        # add previous alignment state, one-hot encoded
        from_states_one_hot = _one_hot_pad_with_zeros( mat=from_states,
                                                       num_classes=6,
                                                       axis=-1 ) #(B, L_align-1, 5)
        datamat_lst.append( from_states_one_hot ) 
        del from_states_one_hot
        
        # save this
        input_to_network = np.concatenate( datamat_lst, axis=-1 ) #(B, L_align-1, H)
        
            
        ### forward pass through prediction head, to get scoring matrices
        # forward_pass_scoring_matrices has the keys:
        # logprob_emit_match: (T,B,L_align-1,A,A) or (B,L_align-1,A,A)
        # logprob_emit_indel: (B,L_align-1,A)
        # logprob_transits: (T,B,L_align-1,S,S) or (B,L_align-1,S,S)
        # corr: a tuple of two arrays; each either (T,B) or (B,)
        # approx_flags_dict: a dictionary of things (see model code)
        # subs_model_params: a dictionary of things (see model code)
        # indel_model_params: a dictionary of things (see model code)
        forward_pass_scoring_matrices = finalpred_trainstate.apply_fn( variables = finalpred_trainstate.params,
                                             datamat_lst = datamat_lst,
                                             t_array = times_for_matrices,
                                             padding_mask = padding_mask,
                                             training = False,
                                             sow_intermediates = False )
        
        
        ### evaluate negative loglike (but don't sum over positions yet)
        logprob_perSamp_perPos_perTime = finalpred_trainstate.apply_fn(  variables = finalpred_trainstate.params,
                                             logprob_emit_match = forward_pass_scoring_matrices['logprob_emit_match'],
                                             logprob_emit_indel = forward_pass_scoring_matrices['logprob_emit_indel'],
                                             logprob_transits = forward_pass_scoring_matrices['logprob_transits'],
                                             corr = forward_pass_scoring_matrices['corr'],
                                             true_out = true_out,
                                             return_result_before_sum = True,
                                             method = 'neg_loglike_in_scan_fn') #(T, B, L_align-1) or (B, L_align-1)
        
        if not return_true_out:
            out_dict =  {'forward_pass_scoring_matrices': forward_pass_scoring_matrices,
                         'logprob_perSamp_perPos_perTime': logprob_perSamp_perPos_perTime,
                         'input_to_network': input_to_network}
            
        elif return_true_out:
            out_dict =  {'forward_pass_scoring_matrices': forward_pass_scoring_matrices,
                         'logprob_perSamp_perPos_perTime': logprob_perSamp_perPos_perTime,
                         'input_to_network': input_to_network,
                         'true_out': true_out}
        
        return out_dict
            
    
    #####################################################
    ### True output: run eval function on whole input   #
    #####################################################
    out_dict = unit_test_eval(batch = batch, 
                              all_trainstates = all_trainstates,
                              all_model_instances = all_model_instances,
                              t_array_for_all_samples = t_array_for_all_samples,
                              true_out = None)
    
    true_score_dict = out_dict['forward_pass_scoring_matrices'] # dict to unpack
    true_logprob_per_pos = out_dict['logprob_perSamp_perPos_perTime'] #(T, B, L_align-1) or (B, L_align-1)
    true_input_to_network = out_dict['input_to_network']  #(B, L_align-1, H)
    true_out_from_full_seqs = out_dict['true_out'] #(T, B, L_align-1)
    del out_dict
    
    true_logprob_emit_match = true_score_dict['logprob_emit_match'] #(T, B, L_align-1, A, A) or (B, L_align-1, A, A)
    true_logprob_emit_indel = true_score_dict['logprob_emit_indel'] #(B, L_align-1, A)
    true_logprob_transits = true_score_dict['logprob_transits'] #(T, B, L_align-1, S, S) or (B, L_align-1, S, S)
    del true_score_dict
    
    
    #####################################################################
    ### Pred output: redo eval, but go one alignment column at a time   #
    #####################################################################
    ### initialize buckets
    # substitution probability matrix
    if (args.pred_config['subst_model_type'] == 'gtr') and (not unique_t_per_sample):
        final_shape = (T, B, L_align-1, A, A)
        
    elif (args.pred_config['subst_model_type'] == 'gtr') and (unique_t_per_sample):
        final_shape = (B, L_align-1, A, A)
        
    elif (args.pred_config['subst_model_type'] == 'f81') and (not unique_t_per_sample):
        final_shape = (T, B, L_align-1, A, 2)
        
    elif (args.pred_config['subst_model_type'] == 'f81') and (unique_t_per_sample):
        final_shape = (B, L_align-1, A, 2)
    
    pred_logprob_emit_match = np.zeros( final_shape )
    del final_shape
    
    # equilibrium distribution
    pred_logprob_emit_indel = np.zeros( (B, L_align-1, A) ) #(B, L_align-1, A)
    
    # transition probability matrix
    if not unique_t_per_sample:
        final_shape =  (T, B, L_align-1, 4, 4)

    elif unique_t_per_sample:
        final_shape =  (B, L_align-1, 4, 4)
                
    pred_logprob_transits = np.zeros( final_shape )
    del final_shape
    
    # logprob per position
    if not unique_t_per_sample:
        final_shape =  (T, B, L_align-1)

    elif unique_t_per_sample:
        final_shape =  (B, L_align-1)
                
    pred_logprob_per_pos = np.zeros( final_shape )
    del final_shape
    
    # inputs to network
    pred_input_to_network = np.zeros( true_input_to_network.shape ) #(B, L_align-1, H)
    
    
    # l is the position to be aligned
    # know the ancestor, but do NOT know the descendant at l
    for l in range(0, L_align-1): 
        ### adjust inputs
        # clip the aligned inputs; mask everything AFTER l
        clipped_aligned_mats = np.array( aligned_mats )
        clipped_aligned_mats[:, l+1:, [0,1,2]] = 0
        clipped_aligned_mats[:, l+1:, [3,4]] = -9
        
        # mask the descendant sequence
        clipped_unaligned_seqs = np.array( unaligned_seqs )
        for b in range( B ):
            max_desc_idx = clipped_aligned_mats[b,:,-1].max()
            clipped_unaligned_seqs[b, max_desc_idx+1:, 1] = 0
        del b, max_desc_idx
        
        clipped_batch = (clipped_unaligned_seqs, clipped_aligned_mats, t_per_sample, None)
        
        
        ### run eval function
        out_dict = unit_test_eval(batch = clipped_batch, 
                                           all_trainstates = all_trainstates,
                                           all_model_instances = all_model_instances,
                                           t_array_for_all_samples = t_array_for_all_samples,
                                           true_out = true_out_from_full_seqs)
        
        scor_dict_up_to_l = out_dict['forward_pass_scoring_matrices'] # dict to unpack
        logprob_up_to_l = out_dict['logprob_perSamp_perPos_perTime'] #(T, B, L_align-1) or (B, L_align-1)
        input_to_network_up_to_l = out_dict['input_to_network']  #(B, L_align-1, H)
        del out_dict
        
        
        ### take corresponding indices from output_at_l
        logprob_emit_match_to_return = scor_dict_up_to_l['logprob_emit_match'][...,l,:,:] # (T, B, A, A) or (B, A, A)
        logprob_emit_indel_to_return = scor_dict_up_to_l['logprob_emit_indel'][:, l, :] # (B, A)
        logprob_transits_to_return = scor_dict_up_to_l['logprob_transits'][...,l,:,:] # (T, B, S, S) or (B, S, S)
        loglike_to_return = logprob_up_to_l[...,l] #(T, B) or (B,)
        i_to_return = input_to_network_up_to_l[...,l,:] #(B, H)
        
        
        ### validate that future masking worked
        # for inputs to network, everything AFTER l should be zero, since
        #   it's padded
        assert np.allclose( input_to_network_up_to_l[...,l+1:,:],
                            np.zeros( input_to_network_up_to_l[...,l+1:,:].shape ) )
        
        # for all matrices, make sure all positions AFTER l are the same (they should be, since
        #   they're information for padding positions)
        if (L_align-1) - l >= 2:
            emit_match_ref = scor_dict_up_to_l['logprob_emit_match'][...,l+1,:,:]  # (T, B, A, A) or (B, A, A)
            emit_indel_ref = scor_dict_up_to_l['logprob_emit_indel'][:, l+1, :] # (B, A)
            transits_ref = scor_dict_up_to_l['logprob_transits'][...,l+1,:,:] # (T, B, S, S) or (B, S, S)
            
            for l_after in range( l+2, L_align-1 ):
                new_match = scor_dict_up_to_l['logprob_emit_match'][...,l_after,:,:] # (T, B, A, A) or (B, A, A)
                new_indel = scor_dict_up_to_l['logprob_emit_indel'][:, l_after, :] # (B, A)
                new_transits = scor_dict_up_to_l['logprob_transits'][...,l_after,:,:] # (T, B, S, S) or (B, S, S)
                
                assert np.allclose(new_match, emit_match_ref)
                assert np.allclose(new_indel, emit_indel_ref)
                assert np.allclose(new_transits, transits_ref)
                
                emit_match_ref = new_match  # (T, B, A, A) or (B, A, A)
                emit_indel_ref = new_indel # (B, A)
                transits_ref = new_transits # (T, B, S, S) or (B, S, S)
                
                del l_after, new_match, new_indel, new_transits
            
            del emit_match_ref, emit_indel_ref, transits_ref
                
        # update buckets
        pred_logprob_emit_match[...,l,:,:] = logprob_emit_match_to_return
        pred_logprob_emit_indel[...,l,:] = logprob_emit_indel_to_return
        pred_logprob_transits[...,l,:,:] = logprob_transits_to_return
        pred_logprob_per_pos[...,l] = loglike_to_return
        pred_input_to_network[...,l,:] = i_to_return
    
    
    ### compare to results using full sequence
    assert np.allclose(true_logprob_emit_match, pred_logprob_emit_match)
    assert np.allclose(true_logprob_emit_indel, pred_logprob_emit_indel)
    assert np.allclose(true_logprob_transits, pred_logprob_transits)
    assert np.allclose(true_logprob_per_pos, pred_logprob_per_pos)
    assert np.allclose(true_input_to_network, pred_input_to_network)
    
    print('Model in config passes causal test!')
