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
    for independent site classses over substitution models
    """
    if not pred_config['load_all']:
        from models.simple_site_class_predict.IndpSites import IndpSites as model
        
    elif pred_config['load_all']:
        from models.simple_site_class_predict.IndpSites import IndpSitesLoadAll as model
    
    pairhmm_instance = model(config = pred_config,
                                 name = 'IndpSites')
    
        
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
                                              tx=tx )
        
    return pairhmm_trainstate, pairhmm_instance


    
def init_pairhmm_frag_and_site_classes( seq_shapes, 
                                        dummy_t_array,
                                        tx, 
                                        model_init_rngkey,
                                        pred_config,
                                        tabulate_file_loc
                                        ):
    """
    for pairHMM using latent fragment and site classses
    """
    if not pred_config['load_all']:
        from models.simple_site_class_predict.FragAndSiteClasses import FragAndSiteClasses as model
        
    elif pred_config['load_all']:
        from models.simple_site_class_predict.FragAndSiteClasses import FragAndSiteClassesLoadAll as model
    
    pairhmm_instance = model(config = pred_config,
                             name = 'FragAndSiteClasses')
    
    
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
                                            tx=tx )
        
    return pairhmm_trainstate, pairhmm_instance
