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
                             ):
    """
    for independent site classses over substitution models
    """
    preset_names = ['load_all',
                    'fit_rate_mult_only',
                    'fit_rate_mult_and_matrix']
    assert pred_config['preset_name'] in preset_names, f'valid options: {preset_names}'
    
    
    # enforce this default
    pred_config['num_tkf_site_classes'] = 1
    
    if pred_config['preset_name'] == 'load_all':
        from models.simple_site_class_predict.PairHMM_indp_sites import JointPairHMMLoadAll
        pairhmm_instance = JointPairHMMLoadAll(config = pred_config,
                                       name = 'JointPairHMMLoadAll')
    
    elif pred_config['preset_name'] == 'fit_rate_mult_only':
        from models.simple_site_class_predict.PairHMM_indp_sites import JointPairHMMFitRateMult
        pairhmm_instance = JointPairHMMFitRateMult(config = pred_config,
                                                   name = 'JointPairHMMFitRateMult')
    
    elif pred_config['preset_name'] == 'fit_rate_mult_and_matrix':
        if pred_config['loss_type'] in ['cond', 'conditional']:
            from models.simple_site_class_predict.PairHMM_indp_sites import CondPairHMM
            pairhmm_instance = CondPairHMM(config = pred_config,
                                           name = 'CondPairHMM')
        
        elif pred_config['loss_type'] == 'joint':
            from models.simple_site_class_predict.PairHMM_indp_sites import JointPairHMMFitBoth
            pairhmm_instance = JointPairHMMFitBoth(config = pred_config,
                                                   name = 'JointPairHMMFitBoth')
        
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
                               ):
    """
    for markovian site classses
    
    TODO: update with whichever indel model works best from independent
     site modeling experiments: fitting both rate matrix and rate multiplier,
     or just fitting rate multiplier
    """
    if not pred_config['load_all_params']:
        from models.simple_site_class_predict.PairHMM_markovian_sites import MarkovSitesJointPairHMM
        pairhmm_instance = MarkovSitesJointPairHMM(config = pred_config,
                                                   name = 'JointPairHMM')
    
    elif pred_config['load_all_params']:
        from models.simple_site_class_predict.PairHMM_markovian_sites import JointPairHMMLoadAll
        pairhmm_instance = JointPairHMMLoadAll(config = pred_config,
                                       name = 'JointPairHMMLoadAll')
        
    init_params = pairhmm_instance.init(rngs = model_init_rngkey,
                                        aligned_inputs = seq_shapes,
                                        t_array = dummy_t_array,
                                        sow_intermediates = False,
                                        mutable=['params'])
        
    pairhmm_trainstate = TrainState.create( apply_fn=pairhmm_instance.apply, 
                                              params=init_params,
                                              tx=tx)
        
    return pairhmm_trainstate, pairhmm_instance

    

