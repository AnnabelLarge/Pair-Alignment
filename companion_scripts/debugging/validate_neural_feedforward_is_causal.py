#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 17:17:00 2025

@author: annabel_large
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
from models.neural_shared.neural_initializer import create_all_tstates
from train_eval_fns.build_optimizer import build_optimizer
from utils.train_eval_utils import (determine_seqlen_bin, 
                                    determine_alignlen_bin)
from models.sequence_embedders.concatenation_fns import extract_embs

from train_eval_fns.feedforward_predict_train_eval_one_batch import ( _preproc,
                                                                     _one_hot_pad_with_zeros )


def test_feedforward_model_is_causal(config_file):
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
    args.pred_config['in_alph_size'] = A+3
    args.pred_config['out_alph_size'] = 2*A + 4
    args.pred_config['training_dset_emit_counts'] = np.array( [1/A]*A ) 
    
    args.anc_enc_config['in_alph_size'] = A+3
    args.desc_dec_config['in_alph_size'] = A+3
    
    # get things from config
    use_prev_align_info = args.pred_config['use_prev_align_info']
    use_time = args.pred_config['t_per_sample']
    
    
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
    
    for b in range(alignment_state.shape[0]):
        for l in range(alignment_state.shape[1]):
            align = alignment_state[b,l]
            if align == 2:
                assert ancs_aligned[b,l] == 43
                descs_aligned[b,l] += A
    
    del ancs_aligned
    
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
    
    aligned_mats = np.stack( [descs_aligned, alignment_state, m_idx, n_idx], axis=-1 )
    del m_idx, n_idx, descs_aligned, alignment_state
    
    # dims
    B = aligned_mats.shape[0]
    L_align = aligned_mats.shape[1]
    L_seq = unaligned_seqs.shape[1]
    
    
    ### time
    if use_time:
        t_per_sample = np.arange( 1, B+1 ) * 0.1 #(B,)
    
    else:
        t_per_sample = None
        
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
    if t_per_sample is not None:
        dummy_t_per_sample = jnp.empty( (t_per_sample.shape[0], ) )
    
    else:
        dummy_t_per_sample = None
    
    # batch provided to train/eval functions consist of:
    # 1.) unaligned sequences (B, L_seq, 2)
    # 2.) aligned data matrices (B, L_align, 4)
    # 3.) time per sample (if applicable) (B,) or None
    # 4, not used.) sample index (B,)
    seq_shapes = [largest_seqs, largest_aligns, dummy_t_per_sample]
    
    
    ### initialize trainstate objects
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
                              t_array_for_all_samples = None
                              )  
    all_trainstates, all_model_instances, _ = out
    del out, tx, largest_seqs, largest_aligns, dummy_t_per_sample, seq_shapes
    
    
    ##################
    ### fake apply   #
    ##################
    def unit_test_eval(batch, 
                       all_trainstates,  
                       all_model_instances,
                       use_prev_align_info,
                       true_out,
                       gap_idx: int = 43,
                       seq_padding_idx: int = 0,
                       align_idx_padding: int = -9,
                       *args,
                       **kwargs):
        """
        modified version of train_eval_fns.neural_tkf_train_eval.eval_one_batch
        """
        ####################
        ### UNPACK, CLIP   #
        ####################
        # unpack
        encoder_trainstate, decoder_trainstate, finalpred_trainstate = all_trainstates
        encoder_instance, decoder_instance, finalpred_instance = all_model_instances
        del all_model_instances, all_trainstates
        
        # clip to max lengths, split into prefixes and suffixes
        unaligned_seqs = batch[0] #(B, L, 2)
        aligned_mats = batch[1] #(B, L, 4)
        del batch
        
        
        
        ##################
        ### PREPROCESS   #
        ##################
        # preprocess with helper
        out_dict = _preproc( unaligned_seqs = unaligned_seqs, 
                             aligned_mats = aligned_mats,
                             use_prev_align_info = use_prev_align_info )
        anc_seqs = out_dict['anc_seqs'] # (B, L_seq)
        desc_seqs = out_dict['desc_seqs'] # (B, L_seq) 
        align_idxes = out_dict['align_idxes'] #(B, L_align-1, 2)
        prev_state_path = out_dict.get('prev_state_path', None) #(B, L_align-1) or None
        
        # if this is the first time, generate true_out and return it
        if true_out is None:
            del true_out
            true_out = out_dict['true_out']
            return_true_out = True
        
        elif true_out is not None:
            return_true_out = False
        
        del out_dict
        
            
        ##################################
        ### APPLY MODEL, EVALUATE LOSS   #
        ##################################
        ### embed with ancestor encoder
        # anc_embeddings is (B, L_seq-1, H)
        out = encoder_instance.apply_seq_embedder_in_eval(seqs = anc_seqs,
                                                          tstate = encoder_trainstate,
                                                          sow_intermediates = False,
                                                          extra_args_for_eval = dict())
        
        anc_embeddings, embeddings_aux_dict = out
        del out
        
        
        ### embed with descendant decoder
        # desc_embeddings is (B, L_seq-1, H)
        out = decoder_instance.apply_seq_embedder_in_eval(seqs = desc_seqs,
                                                          tstate = decoder_trainstate,
                                                          sow_intermediates = False,
                                                          extra_args_for_eval = dict())
        desc_embeddings, to_add = out
        del out
        
        # at minimum, embeddings_aux_dict contains:
        #     - anc_aux
        #     - anc_layer_metrics
        #     - desc_layer_metrics
        
        # could also contain
        #     - anc_attn_weights (for transformers)
        #     - desc_attn_weights (for transformers)
        #     - anc_carry (for LSTMs)
        #     - desc_carry (for LSTMs)
        embeddings_aux_dict = {**embeddings_aux_dict, **to_add}
        del to_add
        
        
        ### extract embeddings
        out = extract_embs(anc_encoded = anc_embeddings, 
                        desc_encoded = desc_embeddings,
                        idx_lst = align_idxes,
                        seq_padding_idx = seq_padding_idx,
                        align_idx_padding = align_idx_padding)
        datamat_lst, padding_mask = out
        del out
        
        # optionally, add previous alignment state, one-hot encoded
        if use_prev_align_info:
            from_states_one_hot = _one_hot_pad_with_zeros( mat=prev_state_path,
                                                           num_classes=6,
                                                           axis=-1 ) #(B, L_align-1, 5)
            datamat_lst.append( from_states_one_hot ) 
            del from_states_one_hot
            
        elif not use_prev_align_info:
            datamat_lst.append( None )
        
        # save this
        input_to_network = np.concatenate( datamat_lst, axis=-1 ) #(B, L_align-1, H)
        
        
        ### forward pass through prediction head, to get logits
        final_logits = finalpred_trainstate.apply_fn(variables = finalpred_trainstate.params,
                                            datamat_lst = datamat_lst,
                                            t_array = t_per_sample,
                                            padding_mask = padding_mask,
                                            training = False,
                                            sow_intermediates = False )
        
        
        ### evaluate loglike per position
        loss_intermeds = finalpred_trainstate.apply_fn( variables = finalpred_trainstate.params,
                                                       final_logits = final_logits,
                                                       true_out = true_out,
                                                       padding_mask = padding_mask,
                                                       return_result_before_sum = True,
                                                       method = 'neg_loglike_in_scan_fn' ) 
        logprob_perSamp_perPos = loss_intermeds['logprob_perSamp_perPos'] #(B, L_align-1, A_aug)
        del loss_intermeds 
        
        out_dict =  {'final_logits': final_logits,
                      'logprob_perSamp_perPos': logprob_perSamp_perPos,
                      'input_to_network': input_to_network}
            
        if return_true_out:
            out_dict['true_out'] = true_out
        
        return out_dict
        
    
    #####################################################
    ### True output: run eval function on whole input   #
    #####################################################
    out_dict = unit_test_eval(batch = batch, 
                              all_trainstates = all_trainstates,
                              all_model_instances = all_model_instances,
                              use_prev_align_info = use_prev_align_info,
                              true_out = None)
    
    true_final_logits = out_dict['final_logits'] # (B, L_align-1, A_aug)
    true_logprob_per_pos = out_dict['logprob_perSamp_perPos'] #(B, L_align-1)
    true_input_to_network = out_dict['input_to_network']  #(B, L_align-1, H)
    true_out_from_full_seqs = out_dict['true_out'] #(B, L_align-1)
    del out_dict
    
    
    #####################################################################
    ### Pred output: redo eval, but go one alignment column at a time   #
    #####################################################################
    ### initialize buckets
    pred_final_logits = np.zeros( true_final_logits.shape )
    pred_logprob_per_pos = np.zeros( true_logprob_per_pos.shape )
    
    # inputs to network
    pred_input_to_network = np.zeros( true_input_to_network.shape ) #(B, L_align-1, H)
    
    
    # l is the position to be aligned
    # know the ancestor, but do NOT know the descendant at l
    for l in range(0, L_align-1): 
        ### adjust inputs
        # clip the aligned inputs; mask everything AFTER l
        clipped_aligned_mats = np.array( aligned_mats )
        clipped_aligned_mats[:, l+1:, [0,1]] = 0
        clipped_aligned_mats[:, l+1:, [2,3]] = -9
        
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
                                  use_prev_align_info = use_prev_align_info,
                                  true_out = true_out_from_full_seqs)
        
        
        logits_up_to_l = out_dict['final_logits'] # (B, L_align-1, A_aug)
        logprob_up_to_l = out_dict['logprob_perSamp_perPos'] #(B, L_align-1)
        input_to_network_up_to_l = out_dict['input_to_network']  #(B, L_align-1, H)
        del out_dict
        
        
        ### take corresponding indices from output_at_l
        logits_to_return = logits_up_to_l[:, l, :] # (B, A_aug)
        loglike_to_return = logprob_up_to_l[:, l] # (B,)
        i_to_return = input_to_network_up_to_l[..., l, :] #(B, H)
        
        
        ### validate that future masking worked
        # for inputs to network, everything AFTER l should be zero, since
        #   it's padded
        assert np.allclose( input_to_network_up_to_l[...,l+1:,:],
                            np.zeros( input_to_network_up_to_l[...,l+1:,:].shape ) )
        
        # for all matrices, make sure all positions AFTER l are the same (they should be, since
        #   they're information for padding positions)
        if (L_align-1) - l >= 2:
            logits_ref = logits_up_to_l[:, l+1, :] #(B, A_aug)
            loglike_ref = logprob_up_to_l[:, l+1] # (B,)
            
            for l_after in range( l+2, L_align-1 ):
                new_logits = logits_up_to_l[:, l_after, :]
                new_loglike = logprob_up_to_l[:, l_after]
                
                assert np.allclose(new_logits, logits_ref)
                assert np.allclose(new_loglike, loglike_ref)
                
                del l_after, new_logits, new_loglike
            
            del logits_ref, loglike_ref
                
        # update buckets
        pred_final_logits[:,l,:] = logits_to_return
        pred_logprob_per_pos[...,l] = loglike_to_return
        pred_input_to_network[...,l,:] = i_to_return
    
    
    ### compare to results using full sequence
    assert np.allclose(true_final_logits, pred_final_logits)
    assert np.allclose(true_logprob_per_pos, pred_logprob_per_pos)
    assert np.allclose(true_input_to_network, pred_input_to_network)
    
    print('Model in config passes causal test!')


if __name__ == '__main__':
    import os
    
    test_feedforward_model_is_causal('CONFIG_ffwd_cnn_two-blocks.json')
