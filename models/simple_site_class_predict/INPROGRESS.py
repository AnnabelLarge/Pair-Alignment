#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 10:50:31 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
import numpy as np
from jax.nn import log_softmax
from jax.scipy.special import logsumexp

from models.simple_site_class_predict.transition_models import (TKF91TransitionLogprobs,
                                                                TKF92TransitionLogprobs,
                                                                TKF91DomainTransitionLogprobs)
from models.simple_site_class_predict.model_functions import (logsumexp_with_arr_lst,
                                                              log_one_minus_x,
                                                              two_by_two_log_inverse,
                                                              regular_tkf)

###########
### init  #
###########
# dims
C_dom = 2
C_frag = 3
T = 5

# time
t_array = jax.random.uniform(key = jax.random.key(0), 
                             shape=(T,), 
                             minval=0.01, 
                             maxval=1.0) #(T,)



################################################################
### NESTED MODEL: generate TKF92 transition scoring matrices   #
################################################################
### init model
transit_config = {'tkf_function': 'regular_tkf',
                  'num_domain_mixtures': 2,
                  'num_fragment_mixtures': 3}

fragment_model = TKF92TransitionLogprobs(config=transit_config,
                                         name='mymod')

init_params = fragment_model.init( rngs=jax.random.key(0),
                                    t_array=t_array,
                                    return_all_matrices=False,
                                    sow_intermediates=False )

### get indel params, scoring matrix
out = fragment_model.apply( variables=init_params, 
                            t_array=t_array,
                            return_all_matrices=False,
                            sow_intermediates=False )

# log_frag_class_probs is (C_dom, C_frag)
log_frag_class_probs, matrix_dict, _, frag_tkf_params_dict = out
del out, init_params

frag_joint_transit_mat = matrix_dict['joint'] #(T, C_dom, C_frag_to, C_frag_from, S_from, S_to)
lam_frag = matrix_dict['lam'] #(C_dom,)
mu_frag = matrix_dict['mu'] #(C_dom,)
offset_frag = matrix_dict['offset'] #(C_dom,)
r_frag = matrix_dict['r_extend'] #(C_dom, C_frag)
del matrix_dict, fragment_model



###################################################################
### TOP-LEVEL MODEL: generate TKF91 transition scoring matrices   #
###################################################################
### init model
domain_model = TKF91DomainTransitionLogprobs(config=transit_config,
                                             name='mydomainmod')

init_params = domain_model.init( rngs=jax.random.key(0),
                                 t_array=t_array,
                                 return_all_matrices=False,
                                 sow_intermediates=False )

### get indel params, scoring matrix
out = domain_model.apply( variables=init_params, 
                          t_array=t_array,
                          return_all_matrices=False,
                          sow_intermediates=False )

# log_domain_class_probs is (C_dom,)
log_domain_class_probs, matrix_dict, _, _ = out
del out, init_params

dom_joint_transit_mat = matrix_dict['joint'] #(T, S_from, S_to)
lam_dom = matrix_dict['lam'] #float
mu_dom = matrix_dict['mu'] #float
offset_dom = matrix_dict['offset'] #float
del matrix_dict, domain_model, transit_config


##############################################################
### Construct raw transition matrix; eliminate null cycles   #
##############################################################
# offset = 1 - lam/mu
# log_domain_class_probs is (C_dom,)
# entries in frag_tkf_params_dict are (T, C_dom)

### helper values
log_z_t = logsumexp( (log_domain_class_probs[None,:] +
                      frag_tkf_params_dict['log_offset'][None,:] +
                      frag_tkf_params_dict['log_one_minus_beta']), axis=-1) #(T,)

log_z_0 = logsumexp( log_domain_class_probs+frag_tkf_params_dict['log_offset'], axis=-1) #float

log_one_minus_z_t = log_one_minus_x( log_z_t ) #(T,)
log_one_minus_z_0 = log_one_minus_x( log_z_0 ) #(T,)


### create (MIDS, MIDE) to modify
# multiply any -> M by (1 - z_t)
S = dom_joint_transit_mat.shape[-1]
mask = jnp.concatenate( [jnp.ones(  (T, S, 1), dtype = bool),
                         jnp.zeros( (T, S, 3), dtype=bool )], axis=2 )
v = jnp.where(mask, 
              dom_joint_transit_mat + log_one_minus_z_t[:,None,None], 
              dom_joint_transit_mat) #(T, 6, 4)
assert v.shape == (T, S, S)
del mask

# multiply any ->ID by (1 - z_0)
mask = jnp.concatenate( [jnp.zeros( (T, S, 1), dtype = bool),
                         jnp.ones(  (T, S, 2), dtype = bool),
                         jnp.zeros( (T, S, 1), dtype = bool)], axis=2 )
v = jnp.where(mask, v + log_one_minus_z_0, v) #(T, 6, 4)
assert v.shape == (T, S, S)
del mask


### get (MIDSAB, AB)


### get (AB, MIDSAB)






# log_U_aa = logsumexp_with_arr_lst( [log_z_t + dom_joint_transit_mat[:, 0, 0],
#                                     log_z_0 + dom_joint_transit_mat[:, 0, 1]] ) #(T,)

# log_U_ba = logsumexp_with_arr_lst( [log_z_t + dom_joint_transit_mat[:, 2, 0],
#                                     log_z_0 + dom_joint_transit_mat[:, 2, 1]] ) #(T,)

# log_U_ab = log_z_0 + dom_joint_transit_mat[:, 0, 2] #(T,)

# log_U_bb = log_z_0 + dom_joint_transit_mat[:, 2, 2] #(T,)

# log_U = jnp.stack( [jnp.stack([log_U_aa, log_U_ab], axis=-1),
#                     jnp.stack([log_U_ba, log_U_bb], axis=-1)],
#                   axis=-2) #(T, 2, 2)
# del log_z_t, log_z_0, log_U_aa, log_U_bb, log_U_ab, log_U_ba


# ### inverse of U
# log_U_inv = two_by_two_log_inverse(log_U) #(T, 2, 2)


# ### build matrix


