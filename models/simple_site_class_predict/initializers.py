#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 05:47:08 2025

@author: annabel
"""
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState

def init_pairhmm_indp_sites( seq_shapes, 
                             dummy_t_array,
                             tx, 
                             model_init_rngkey,
                             pred_config,
                             tabulate_file_loc
                             ):
    """
    for independent site classses over substitution models, either TKF91 or TKF92
    """
    preset_names = ['load_all',
                    'fit_rate_mult_only',
                    'fit_rate_mult_and_matrix']
    assert pred_config['preset_name'] in preset_names, f'valid options: {preset_names}'
    
    # enforce this default
    pred_config['num_tkf_site_classes'] = 1
    
    if pred_config['preset_name'] == 'load_all':
        from models.simple_site_class_predict.PairHMM_indp_sites import IndpPairHMMLoadAll
        pairhmm_instance = IndpPairHMMLoadAll(config = pred_config,
                                       name = 'IndpPairHMMLoadAll')
    
    elif pred_config['preset_name'] == 'fit_rate_mult_only':
        from models.simple_site_class_predict.PairHMM_indp_sites import IndpPairHMMFitRateMult
        pairhmm_instance = IndpPairHMMFitRateMult(config = pred_config,
                                                   name = 'IndpPairHMMFitRateMult')
    
    elif pred_config['preset_name'] == 'fit_rate_mult_and_matrix':
        from models.simple_site_class_predict.PairHMM_indp_sites import IndpPairHMMFitBoth
        pairhmm_instance = IndpPairHMMFitBoth(config = pred_config,
                                               name = 'IndpPairHMMFitBoth')
    
    ### tabulate and save the model
    if (tabulate_file_loc is not None):
        tab_fn = nn.tabulate(pairhmm_instance, 
                              rngs=model_init_rngkey,
                              console_kwargs = {'soft_wrap':True,
                                                'width':250})
        str_out = tab_fn(batch = seq_shapes,
                         t_array = dummy_t_array,
                         sow_intermediates = False,
                         mutable = ['params'])
        with open(f'{tabulate_file_loc}/PAIRHMM_tabulate.txt','w') as g:
            g.write(str_out)
    
    init_params = pairhmm_instance.init(rngs = model_init_rngkey,
                                        batch = seq_shapes,
                                        t_array = dummy_t_array,
                                        sow_intermediates = False,
                                        mutable=['params'])
        
    pairhmm_trainstate = TrainState.create( apply_fn=pairhmm_instance.apply, 
                                              params=init_params,
                                              tx=tx)
        
    return pairhmm_trainstate, pairhmm_instance


    
def init_pairhmm_markov_sites( seq_shapes, 
                               dummy_t_array,
                               tx, 
                               model_init_rngkey,
                               pred_config,
                               tabulate_file_loc
                               ):
    """
    for markovian site classses
    
    TODO: update with whichever indel model works best from independent
     site modeling experiments: fitting both rate matrix and rate multiplier,
     or just fitting rate multiplier
    """
    if not pred_config['load_all_params']:
        ### uncomment for jax.lax.scan version
        from models.simple_site_class_predict.PairHMM_markovian_sites import MarkovPairHMM
        pairhmm_instance = MarkovPairHMM(config = pred_config,
                                         name = 'MarkovPairHMM')
    
    elif pred_config['load_all_params']:
        from models.simple_site_class_predict.PairHMM_markovian_sites import MarkovPairHMMLoadAll
        pairhmm_instance = MarkovPairHMMLoadAll(config = pred_config,
                                                name = 'MarkovPairHMMLoadAll')
    
    ### tabulate and save the model
    if (tabulate_file_loc is not None):
        tab_fn = nn.tabulate(pairhmm_instance, 
                              rngs=model_init_rngkey,
                              console_kwargs = {'soft_wrap':True,
                                                'width':250})
        str_out = tab_fn(aligned_inputs = seq_shapes,
                         t_array = dummy_t_array,
                         sow_intermediates = False,
                         mutable = ['params'])
        with open(f'{tabulate_file_loc}/PAIRHMM_tabulate.txt','w') as g:
            g.write(str_out)
    
    init_params = pairhmm_instance.init(rngs = model_init_rngkey,
                                        aligned_inputs = seq_shapes,
                                        t_array = dummy_t_array,
                                        sow_intermediates = False,
                                        mutable=['params'])
        
    pairhmm_trainstate = TrainState.create( apply_fn=pairhmm_instance.apply, 
                                              params=init_params,
                                              tx=tx)
        
    return pairhmm_trainstate, pairhmm_instance
