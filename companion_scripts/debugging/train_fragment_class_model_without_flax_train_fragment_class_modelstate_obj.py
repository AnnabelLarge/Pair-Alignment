#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 18:40:55 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
import optax

import numpy as np
from tqdm import tqdm

from tests.data_processing import (str_aligns_to_tensor,
                                   summarize_alignment)
from models.simple_site_class_predict.transition_models import (TKF91TransitionLogprobs,
                                                                TKF92TransitionLogprobs)
from models.simple_site_class_predict.model_functions import (MargTKF91TransitionLogprobs,
                                                              MargTKF92TransitionLogprobs,
                                                              CondTransitionLogprobs)
from models.simple_site_class_predict.FragAndSiteClasses import FragAndSiteClasses


jax.config.update("jax_enable_x64", False)



###################################
### fake alignments, time array   #
###################################
# fake_aligns = [ ('AC-A','D-ED'),
#                 ('D-ED','AC-A'),
#                 ('ECDAD','-C-A-'),
#                 ('-C-A-','ECDAD') ]

# t_array = jnp.array([0.3, 0.5, 0.9, 1.1])

fake_aligns = [ ('AC-A','D-ED'),
                # ('AC-A','D-ED')]
                ('AC-AA','D-EDA')]

t_array = jnp.array([0.3, 0.5])


fake_aligns =  str_aligns_to_tensor(fake_aligns) #(B, L, 3)
batch = (fake_aligns, t_array)
        
        
##################################
### template for the config file #
##################################
config = {'load_all': False,
          'norm_loss_by_length': False,
          'num_mixtures': 2,
          
          'subst_model_type': 'gtr',
          "norm_rate_matrix": True,
          "norm_rate_mults": True,
          "random_init_exchanges": True,

          'indel_model_type': 'tkf92',
          "tkf_function_name": "switch_tkf",
                   
          # "times_from": "t_array_from_file",
           "times_from": "t_per_sample",
          "exponential_dist_param": 1.0,
                   
          'emission_alphabet_size': 20
          }

config['num_tkf_fragment_classes'] = config['num_mixtures']


############
### init   #
############
my_model_instance = FragAndSiteClasses(config=config,
                                       name='my_pairHMM')

init_params = my_model_instance.init(rngs=jax.random.key(0),
                                     batch = batch,
                                     t_array = t_array,
                                     sow_intermediates = False)

optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(init_params)


#############
### train   #
#############
def train_step(batch, t_array, params, optimizer, opt_state):
    def apply(p):
        joint_loglike,_ = my_model_instance.apply(variables=p,
                                                batch=batch,
                                                t_array=t_array,
                                                sow_intermediates=False)
        return joint_loglike
    loss, grads = jax.value_and_grad(apply)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss

params = init_params
loss_traj = []
print('BEGIN TRAIN')
for i in range(3):
    with jax.disable_jit():
        params, opt_state, loss = train_step(batch, 
                                             t_array, 
                                             params, 
                                             optimizer, 
                                             opt_state)
    loss_traj.append( loss.item() )
    print(f'loss at epoch {i}: {loss.item()}')
        
    