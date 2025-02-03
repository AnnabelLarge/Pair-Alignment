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
from flax.training import train_state
from train_state import TrainState

from models.feedforward_predict import create_seq_model_tstate

# # additional key entry for trainstate object (do I need this?)
# class TrainState(train_state.TrainState):
#     key: jax.Array


def evolpair_params_instance(input_shapes,
                              dummy_t_array,
                              tx, 
                              model_init_rngkey, 
                              tabulate_file_loc,
                              model_config: dict = dict()):
    #############
    ### imports #
    #############
    from models.EvolPairPredict import (evoparam_blocks,
                                        pairHMM_emissions_blocks,
                                        pairHMM_transitions_blocks)
    ### blocks for datamat_lst -> evolparams
    # exchangeabilities
    process_embeds_for_exchang_module = getattr(evoparam_blocks, model_config['process_embeds_for_exchang_module'])
    exchang_module = getattr(evoparam_blocks, model_config['exchang_module'])

    # equilibrium distribution
    process_embeds_for_equilibr_module = getattr(evoparam_blocks, model_config['process_embeds_for_equilibr_module'])
    equilibr_module = getattr(evoparam_blocks, model_config['equilibr_module'])
    
    # indel model params
    process_embeds_for_indels_module = getattr(evoparam_blocks, model_config['process_embeds_for_indels_module'])
    indels_module = getattr(evoparam_blocks, model_config['indels_module'])
    

    ### blocks for logprobs
    emit_match_logprobs_module = getattr(pairHMM_emissions_blocks, 
                                          model_config['emit_match_logprobs_module'])
    emit_ins_logprobs_module = getattr(pairHMM_emissions_blocks, 
                                        model_config['emit_ins_logprobs_module'])
    transits_logprobs_module = getattr(pairHMM_transitions_blocks, 
                                        model_config['transits_logprobs_module'])
    
    # get the name of the indel model
    if model_config['transits_logprobs_module'].startswith('TKF91'):
        indel_model_type = 'TKF91'
    elif model_config['transits_logprobs_module'].startswith('TKF92'):
        indel_model_type = 'TKF92'
    
    # instantiate
    from models.EvolPairPredict.EvolPairPredict import EvolPairPredict
    finalpred_instance = EvolPairPredict(process_embeds_for_exchang_module = process_embeds_for_exchang_module,
                                          exchang_module = exchang_module,
                                          process_embeds_for_equilibr_module = process_embeds_for_equilibr_module,
                                          equilibr_module = equilibr_module,
                                          process_embeds_for_indels_module = process_embeds_for_indels_module,
                                          indels_module = indels_module,
                                          emit_match_logprobs_module = emit_match_logprobs_module,
                                          emit_ins_logprobs_module = emit_ins_logprobs_module,
                                          transits_logprobs_module = transits_logprobs_module,
                                          config = model_config,
                                          name = f'NEURAL {indel_model_type}')
        
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
    out = evolpair_params_instance(input_shapes = list_of_shapes, 
                                   dummy_t_array = dummy_t_array, 
                                   tx = tx, 
                                   model_init_rngkey = outproj_rngkey, 
                                   tabulate_file_loc = tabulate_file_loc,
                                   model_config = pred_config)
    
    finalpred_trainstate, finalpred_instance = out
    
    finalpred_trainstate = TrainState.create(apply_fn=finalpred_instance.apply, 
                                             params=init_params,
                                             # key=model_init_rngkey,
                                             tx=tx)
    
    all_trainstates = (ancestor_trainstate, 
                       descendant_trainstate, 
                       finalpred_trainstate)
    
    all_instances = (ancestor_instance, 
                     descendant_instance, 
                     finalpred_instance)
    
    return all_trainstates, all_instances
