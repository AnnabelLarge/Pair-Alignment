#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 10:50:31 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
import numpy as np
from scipy.special import log_softmax

from models.simple_site_class_predict.transition_models import TKF92TransitionLogprobs


###########
### init  #
###########
# dims
C_dom = 2
C_frag = 3
T = 5

# time
t_array = ( np.random.rand( T ) )**2 #(T,)

# P(c_domain)
log_dom_class_probs = log_softmax( np.random.rand( C_dom ) ) #(C_dom)

# model
transit_config = {'tkf_function': 'regular_tkf',
                  'num_domain_mixtures': 2,
                  'num_fragment_mixtures': 3}

fragment_model = TKF92TransitionLogprobs(config=transit_config,
                                         name='mymod')

init_params = fragment_model.init( rngs=jax.random.key(0),
                                    t_array=t_array,
                                    return_all_matrices=False,
                                    sow_intermediates=False )


###################################################################
### generate TKF92 transition params for every domain, fragment   #
###################################################################
out = fragment_model.apply( variables=init_params, 
                            t_array=t_array,
                            return_all_matrices=False,
                            sow_intermediates=False )

# log_frag_class_probs is (C_dom, C_frag)
log_frag_class_probs, matrix_dict, _ = out
del out, init_params

log_joint_tkf92 = matrix_dict['joint'] #(T, C_dom, C_frag_to, C_frag_from, S_from, S_to)
lam_frag = matrix_dict['lam'] #float
mu_frag = matrix_dict['mu'] #float
offset_frag = matrix_dict['offset'] #float
r_frag = matrix_dict['r_extend'] #(C_dom, C_frag)
del matrix_dict



