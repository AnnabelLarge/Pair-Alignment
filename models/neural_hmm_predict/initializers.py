#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABOUT:
======
Helpers to create train state objects; assumes all layers could have dropout

Also save the text outputs of nn.tabulate

Have option to initialize the final bias, but generally found this to be 
  unhelpfulnot using it; leaving default init of a zero vector


TODO:
=====
- Incorporate batch stats (whenever you use BatchNorm)
"""
import importlib

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
from typing import Literal, Union

from models.feedforward_predict.initializers import create_seq_model_tstate
from models.neural_hmm_predict.model_parts import (postprocess_concat_feats,
                                                   concat_feats_to_params,
                                                   params_to_match_emission_logprobs,
                                                   params_to_transition_logprobs)

# # additional key entry for trainstate object (do I need this?)
# class TrainState(train_state.TrainState):
#     key: jax.Array

###############################################################################
### CONFIGURATIONS FOR UNIT TESTS   ###########################################
###############################################################################
def base_hmm_load_all(indel_model_type: Literal["tkf91", "tkf92"],
                      loss_type: Literal["joint", "cond", "conditional"]):
    ### don't preprocess anything
    process_embeds_for_exchang_module = postprocess_concat_feats.Placeholder
    process_embeds_for_equilibr_module = postprocess_concat_feats.Placeholder
    process_embeds_for_lam_mu_module = postprocess_concat_feats.Placeholder
    process_embeds_for_r_module = postprocess_concat_feats.Placeholder
    
    
    ### blocks for datamat_lst -> evolparams (except for TKF92 r, defined later)
    exchang_module = concat_feats_to_params.EvoparamsFromFile
    equilibr_module = concat_feats_to_params.GlobalEqulVecFromCounts
    lam_mu_module = concat_feats_to_params.EvoparamsFromFile
    
    
    ### blocks for other logprobs
    # P(align, anc, desc)
    if loss_type == 'joint':
        emit_match_logprobs_module = params_to_match_emission_logprobs.JointMatchEmissionsLogprobs
        
        if indel_model_type == 'tkf91':
            r_extend_module = concat_feats_to_params.Placeholder
            transits_logprobs_module = params_to_transition_logprobs.JointTKF91TransitionLogprobs
        
        elif indel_model_type == 'tkf92':
            r_extend_module = concat_feats_to_params.EvoparamsFromFile
            transits_logprobs_module = params_to_transition_logprobs.JointTKF92TransitionLogprobs
        
    # P(align, desc | anc)
    elif loss_type in ['cond', 'conditional']:
        emit_match_logprobs_module = params_to_match_emission_logprobs.CondMatchEmissionsLogprobs
        
        if indel_model_type == 'tkf91':
            r_extend_module = concat_feats_to_params.Placeholder
            transits_logprobs_module = params_to_transition_logprobs.CondTKF91TransitionLogprobs
        
        elif indel_model_type == 'tkf92':
            r_extend_module = concat_feats_to_params.EvoparamsFromFile
            transits_logprobs_module = params_to_transition_logprobs.CondTKF92TransitionLogprobs
    
    
    ### return dictionary to use for instantiating model
    prediction_head_dict = {'process_embeds_for_exchang_module': process_embeds_for_exchang_module,
                            'exchang_module': exchang_module,
                            'process_embeds_for_equilibr_module': process_embeds_for_equilibr_module,
                            'equilibr_module': equilibr_module,
                            'process_embeds_for_lam_mu_module': process_embeds_for_lam_mu_module,
                            'lam_mu_module': lam_mu_module,
                            'process_embeds_for_r_module': process_embeds_for_r_module,
                            'r_extend_module': r_extend_module,
                            'emit_match_logprobs_module': emit_match_logprobs_module,
                            'transits_logprobs_module': transits_logprobs_module,
                            'name': f'{indel_model_type} pairhmm'}
    
    return prediction_head_dict


def base_hmm_fit_indel_params(indel_model_type: Literal["tkf91", "tkf92"],
                              loss_type: Literal["joint", "cond", "conditional"]):
    # mostly the same as above, so just do above and change a few entries
    dict_to_modify = base_hmm_load_all(indel_model_type = indel_model_type,
                                       loss_type = loss_type)
    del dict_to_modify['lam_mu_module']
    del dict_to_modify['r_extend_module']
    del dict_to_modify['name']
    
    dict_to_modify['lam_mu_module'] = concat_feats_to_params.GlobalTKFLamMuRates
    
    if indel_model_type == 'tkf92':
        dict_to_modify['r_extend_module'] = concat_feats_to_params.GlobalTKF92ExtProb
    
    return dict_to_modify


###############################################################################
### HIDDEN: CONFIGURATIONS FOR SIMPLE LOCAL/GLOBAL   ##########################
###############################################################################
def _all_global_blocks(indel_model_type: Literal["tkf91", "tkf92"],
                       loss_type: Literal["joint", "cond", "conditional"]):
    ### blocks for features -> params
    # exchangeabilities: from one-hot embeddings of alignment
    process_embeds_for_exchang_module = postprocess_concat_feats.Placeholder
    exchang_module = concat_feats_to_params.GlobalExchMat
    
    # equilibrium distribution: global set
    process_embeds_for_equilibr_module = postprocess_concat_feats.Placeholder
    equilibr_module = concat_feats_to_params.GlobalEqulVec
        
    # indel params: global set
    process_embeds_for_lam_mu_module = postprocess_concat_feats.Placeholder
    lam_mu_module = concat_feats_to_params.GlobalTKFLamMuRates
    process_embeds_for_r_module = postprocess_concat_feats.Placeholder
    
    
    ### blocks for params -> logprobs
    # P(align, anc, desc)
    if loss_type == 'joint':
        emit_match_logprobs_module = params_to_match_emission_logprobs.JointMatchEmissionsLogprobs
        
        if indel_model_type == 'tkf91':
            r_extend_module = concat_feats_to_params.Placeholder
            transits_logprobs_module = params_to_transition_logprobs.JointTKF91TransitionLogprobs
        
        elif indel_model_type == 'tkf92':
            r_extend_module = concat_feats_to_params.EvoparamsFromFile
            transits_logprobs_module = params_to_transition_logprobs.JointTKF92TransitionLogprobs
        
    # P(align, desc | anc)
    elif loss_type in ['cond', 'conditional']:
        emit_match_logprobs_module = params_to_match_emission_logprobs.CondMatchEmissionsLogprobs
        
        if indel_model_type == 'tkf91':
            r_extend_module = concat_feats_to_params.Placeholder
            transits_logprobs_module = params_to_transition_logprobs.CondTKF91TransitionLogprobs
        
        elif indel_model_type == 'tkf92':
            r_extend_module = concat_feats_to_params.GlobalTKF92ExtProb
            transits_logprobs_module = params_to_transition_logprobs.CondTKF92TransitionLogprobs
    
    
    ### return dictionary to use for instantiating model
    prediction_head_dict = {'process_embeds_for_exchang_module': process_embeds_for_exchang_module,
                            'exchang_module': exchang_module,
                            'process_embeds_for_equilibr_module': process_embeds_for_equilibr_module,
                            'equilibr_module': equilibr_module,
                            'process_embeds_for_lam_mu_module': process_embeds_for_lam_mu_module,
                            'lam_mu_module': lam_mu_module,
                            'process_embeds_for_r_module': process_embeds_for_r_module,
                            'r_extend_module': r_extend_module,
                            'emit_match_logprobs_module': emit_match_logprobs_module,
                            'transits_logprobs_module': transits_logprobs_module,
                            'name': f'{indel_model_type} pairhmm'}
    
    return prediction_head_dict


def _local_exch(indel_model_type: Literal["tkf91", "tkf92"],
               loss_type: Literal["joint", "cond", "conditional"],
               in_dict: Union[dict, None]):
    if in_dict is None:
        dict_to_modify = _all_global_blocks(indel_model_type = indel_model_type,
                                            loss_type = loss_type)
    else:
        dict_to_modify = in_dict
    
    del dict_to_modify['process_embeds_for_exchang_module']
    dict_to_modify['process_embeds_for_exchang_module'] = postprocess_concat_feats.FeedforwardToEvoparams
    
    del dict_to_modify['exchang_module']
    dict_to_modify['exchang_module'] = concat_feats_to_params.LocalExchMat
    
    return dict_to_modify


def _local_equilbr(indel_model_type: Literal["tkf91", "tkf92"],
                  loss_type: Literal["joint", "cond", "conditional"],
                  in_dict: Union[dict, None]):
    if in_dict is None:
        dict_to_modify = _all_global_blocks(indel_model_type = indel_model_type,
                                            loss_type = loss_type)
    else:
        dict_to_modify = in_dict
    
    del dict_to_modify['process_embeds_for_equilibr_module']
    dict_to_modify['process_embeds_for_equilibr_module'] = postprocess_concat_feats.FeedforwardToEvoparams
    
    del dict_to_modify['equilibr_module']
    dict_to_modify['equilibr_module'] = concat_feats_to_params.LocalEqulVec
    
    return dict_to_modify


def _local_lam_mu(indel_model_type: Literal["tkf91", "tkf92"],
                 loss_type: Literal["joint", "cond", "conditional"],
                 in_dict: Union[dict, None]):
    if in_dict is None:
        dict_to_modify = _all_global_blocks(indel_model_type = indel_model_type,
                                            loss_type = loss_type)
    else:
        dict_to_modify = in_dict
    
    del dict_to_modify['process_embeds_for_lam_mu_module']
    dict_to_modify['process_embeds_for_lam_mu_module'] = postprocess_concat_feats.FeedforwardToEvoparams
    
    del dict_to_modify['lam_mu_module']
    dict_to_modify['lam_mu_module'] = concat_feats_to_params.LocalTKFLamMuRates
    
    return dict_to_modify


def _local_r_extend(indel_model_type: Literal["tkf91", "tkf92"],
                   loss_type: Literal["joint", "cond", "conditional"],
                   in_dict: Union[dict, None]):
    if in_dict is None:
        dict_to_modify = _all_global_blocks(indel_model_type = indel_model_type,
                                          loss_type = loss_type)
    else:
        dict_to_modify = in_dict  
    
    del dict_to_modify['process_embeds_for_r_module']
    dict_to_modify['process_embeds_for_r_module'] = postprocess_concat_feats.FeedforwardToEvoparams
    
    if indel_model_type == 'tkf92':
        del dict_to_modify['r_extend_module']
        dict_to_modify['r_extend_module'] = concat_feats_to_params.LocalTKF92ExtProb
    
    return dict_to_modify




###############################################################################
### CONFIGURATIONS TO USE   ###################################################
###############################################################################
def local_exch_equilibr(indel_model_type: Literal["tkf91", "tkf92"],
                        loss_type: Literal["joint", "cond", "conditional"],
                        in_dict: Union[dict, None]):
    if in_dict is None:
        dict_to_modify = _all_global_blocks(indel_model_type = indel_model_type,
                                          loss_type = loss_type)
    else:
        dict_to_modify = in_dict
    
    dict_to_modify = _local_exch( indel_model_type = indel_model_type,
                                 loss_type = loss_type,
                                 in_dict = dict_to_modify )
    dict_to_modify = _local_equilbr( indel_model_type = indel_model_type,
                                    loss_type = loss_type,
                                    in_dict = dict_to_modify )
    return dict_to_modify


def local_exch_equilibr_r(indel_model_type: Literal["tkf91", "tkf92"],
                          loss_type: Literal["joint", "cond", "conditional"],
                          in_dict: Union[dict, None]):
    if in_dict is None:
        dict_to_modify = _all_global_blocks(indel_model_type = indel_model_type,
                                          loss_type = loss_type)
    else:
        dict_to_modify = in_dict
    
    dict_to_modify = local_exch_equilibr( indel_model_type = indel_model_type,
                                          loss_type = loss_type,
                                          in_dict = dict_to_modify )
    dict_to_modify = _local_r_extend( indel_model_type = indel_model_type,
                                      loss_type = loss_type,
                                      in_dict = dict_to_modify )
    return dict_to_modify


def all_local(indel_model_type: Literal["tkf91", "tkf92"],
              loss_type: Literal["joint", "cond", "conditional"],
              in_dict: Union[dict, None]):
    dict_to_modify = _all_global_blocks(indel_model_type = indel_model_type,
                                        loss_type = loss_type)
    dict_to_modify = local_exch_equilibr_r( indel_model_type = indel_model_type,
                                            loss_type = loss_type,
                                            in_dict = dict_to_modify )
    dict_to_modify = _local_lam_mu( indel_model_type = indel_model_type,
                                     loss_type = loss_type,
                                     in_dict = dict_to_modify )
    
    return dict_to_modify
    



def neural_hmm_params_instance( input_shapes,
                                dummy_t_array,
                                tx, 
                                model_init_rngkey, 
                                tabulate_file_loc,
                                preset_name, 
                                model_config ):

    # instantiate a specific model by name
    initializers = { 'base_hmm_fit_indel_params': base_hmm_fit_indel_params,
                     'base_hmm_load_all': base_hmm_load_all,
                     'local_exch_equilibr': local_exch_equilibr,
                     'local_exch_equilibr_r': local_exch_equilibr_r,
                     'all_local': all_local }
    
    indel_model_type = model_config['indel_model_type']
    loss_type = model_config['loss_type']
    argsdict = initializers[preset_name]( indel_model_type = indel_model_type,
                                          loss_type = loss_type )
    
    from models.neural_hmm_predict.NeuralHmmBase import NeuralHmmBase
    finalpred_instance = NeuralHmmBase( config = model_config,
                                        **argsdict )
    
    
    ##################
    ### initialize   #
    ##################
    dummy_mat_lst = [jnp.empty(s) for s in input_shapes]
    dim0 = dummy_mat_lst[0].shape[0]
    dim1 = dummy_mat_lst[0].shape[1]
    dummy_masking_mat = jnp.empty( (dim0, dim1) )
    
    ### tabulate and save the model
    if (tabulate_file_loc is not None):
        tab_fn = nn.tabulate(finalpred_instance, 
                              rngs=model_init_rngkey,
                              console_kwargs = {'soft_wrap':True,
                                                'width':250})
        str_out = tab_fn(datamat_lst=dummy_mat_lst, 
                          padding_mask = dummy_masking_mat,
                          t_array = dummy_t_array,
                          training=False,
                          sow_intermediates = False,
                          mutable = ['params'])
        with open(f'{tabulate_file_loc}/OUT-PROJ_tabulate.txt','w') as g:
            g.write(str_out)
        
    
    ### turn into a train state
    init_params = finalpred_instance.init(rngs = model_init_rngkey,
                                          datamat_lst = dummy_mat_lst,
                                          padding_mask = dummy_masking_mat,
                                          t_array = dummy_t_array,
                                          training = False,
                                          sow_intermediates = False,
                                          mutable=['params'])
    
    return (finalpred_instance, init_params)


def create_all_tstates(seq_shapes, 
                        tx, 
                        model_init_rngkey, 
                        tabulate_file_loc: str,
                        anc_model_type: str, 
                        desc_model_type: str, 
                        pred_model_type: str, 
                        anc_enc_config: dict, 
                        desc_dec_config: dict, 
                        pred_config: dict,
                        dummy_t_array: jnp.array
                        ):
    largest_seqs, largest_aligns = seq_shapes
    
    # keep track of dim3 size
    expected_dim3_size = 0
    
    # split input key
    keys = jax.random.split(model_init_rngkey, num=3)
    anc_rngkey, desc_rngkey, outproj_rngkey = keys
    del keys
    
    
    ### ancestor encoder
    out = create_seq_model_tstate( embedding_which = 'anc',
                                    seq_shape = largest_seqs, 
                                    tx = tx, 
                                    model_init_rngkey = anc_rngkey, 
                                    tabulate_file_loc = tabulate_file_loc,
                                    model_type = anc_model_type,
                                    model_config = anc_enc_config )
    ancestor_trainstate = out[0]
    ancestor_instance = out[1]
    ancestor_emb_size = (largest_seqs[0], largest_aligns[1], out[2])
    
    
    ### descendant decoder
    out = create_seq_model_tstate( embedding_which = 'desc',
                                    seq_shape = largest_seqs, 
                                    tx = tx, 
                                    model_init_rngkey = desc_rngkey, 
                                    tabulate_file_loc = tabulate_file_loc,
                                    model_type = desc_model_type,
                                    model_config = desc_dec_config )
    descendant_trainstate = out[0]
    descendant_instance = out[1]
    descendant_emb_size = (largest_seqs[0], largest_aligns[1], out[2])
    
    list_of_shapes = [ancestor_emb_size, descendant_emb_size]
    
    
    ### final prediction network
    # init
    preset_name = pred_model_type.split('/')[-1]
    out = neural_hmm_params_instance(input_shapes = list_of_shapes, 
                                    dummy_t_array = dummy_t_array, 
                                    tx = tx, 
                                    model_init_rngkey = outproj_rngkey, 
                                    tabulate_file_loc = tabulate_file_loc,
                                    preset_name = preset_name,
                                    model_config = pred_config)
    
    finalpred_instance, init_params = out
    
    finalpred_trainstate = TrainState.create( apply_fn=finalpred_instance.apply, 
                                              params=init_params,
                                              # key=model_init_rngkey,
                                              tx=tx)
    
    all_trainstates = (ancestor_trainstate, 
                        descendant_trainstate, 
                        finalpred_trainstate)
    
    all_instances = (ancestor_instance, 
                      descendant_instance, 
                      finalpred_instance)
    
    
    ### determine concatenation function
    if pred_config['use_precomputed_indices']:
        from models.sequence_embedders.concatenation_fns import extract_embs as concat_fn
    
    elif not pred_config['use_precomputed_indices']:
        from models.sequence_embedders.concatenation_fns import combine_one_hot_embeddings as concat_fn
        
    return all_trainstates, all_instances, concat_fn
