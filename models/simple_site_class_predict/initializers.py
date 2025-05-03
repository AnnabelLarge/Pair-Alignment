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
                    'fit_all',
                    'hky85_load_all',
                    'hky85_fit_all',
                    'gtr_load_all',
                    'gtr_fit_all']
    assert pred_config['preset_name'] in preset_names, f'valid options: {preset_names}'
    
    # enforce this default
    pred_config['num_tkf_site_classes'] = 1
    
    
    ######################
    ### Protein models   #
    ######################
    if pred_config['preset_name'] == 'load_all':
        from models.simple_site_class_predict.PairHMM_indp_sites import IndpSitesLoadAll
        pairhmm_instance = IndpSitesLoadAll(config = pred_config,
                                       name = 'IndpSitesLoadAll')
    
    elif pred_config['preset_name'] == 'fit_all':
        from models.simple_site_class_predict.PairHMM_indp_sites import IndpSites
        pairhmm_instance = IndpSites(config = pred_config,
                                     name = 'IndpSites')
    
    elif pred_config['preset_name'] == 'gtr_load_all':
        from models.simple_site_class_predict.PairHMM_GTR import GTRPairHMMLoadAll
        pairhmm_instance = GTRPairHMMLoadAll(config = pred_config,
                                               name = 'GTRPairHMMLoadAll')
        
    elif pred_config['preset_name'] == 'gtr_fit_all':
        from models.simple_site_class_predict.PairHMM_GTR import GTRPairHMM
        pairhmm_instance = GTRPairHMM(config = pred_config,
                                       name = 'GTRPairHMM')
    
    
    ##########################
    ### DNA (HKY85) models   #
    ##########################
    elif pred_config['preset_name'] == 'hky85_load_all':
        from models.simple_site_class_predict.PairHMM_indp_sites import IndpSitesHKY85LoadAll
        pairhmm_instance = IndpSitesHKY85LoadAll(config = pred_config,
                                            name = 'IndpSitesHKY85LoadAll')
    
    elif pred_config['preset_name'] == 'hky85_fit_all':
        from models.simple_site_class_predict.PairHMM_indp_sites import IndpSitesHKY85
        pairhmm_instance = IndpSitesHKY85(config = pred_config,
                                            name = 'IndpSitesHKY85')
    
        
    ###################################
    ### tabulate and save the model   #
    ###################################
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
    """
    preset_names = ['load_all',
                    'fit_all',
                    'hky85_load_all',
                    'hky85_fit_all']
    assert pred_config['preset_name'] in preset_names, f'valid options: {preset_names}'
    
    
    ######################
    ### Protein models   #
    ######################
    if pred_config['preset_name'] == 'fit_all':
        from models.simple_site_class_predict.PairHMM_markovian_sites import MarkovFrags
        pairhmm_instance = MarkovFrags(config = pred_config,
                                         name = 'MarkovFrags')
    
    elif pred_config['preset_name'] == 'load_all':
        from models.simple_site_class_predict.PairHMM_markovian_sites import MarkovFragsLoadAll
        pairhmm_instance = MarkovFragsLoadAll(config = pred_config,
                                              name = 'MarkovFragsLoadAll')
    
    
    ##########################
    ### DNA (HKY85) models   #
    ##########################
    elif pred_config['preset_name'] == 'hky85_fit_all':
        from models.simple_site_class_predict.PairHMM_markovian_sites import MarkovFragsHKY85
        pairhmm_instance = MarkovFragsHKY85(config = pred_config,
                                            name = 'MarkovFragsHKY85')
    
    elif pred_config['preset_name'] == 'hky85_load_all':
        from models.simple_site_class_predict.PairHMM_markovian_sites import MarkovFragsHKY85LoadAll
        pairhmm_instance = MarkovFragsHKY85LoadAll(config = pred_config,
                                            name = 'MarkovFragsHKY85LoadAll')
    
    
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
