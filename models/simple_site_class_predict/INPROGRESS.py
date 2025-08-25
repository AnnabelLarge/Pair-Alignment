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
from models.simple_site_class_predict.model_functions import (get_tkf92_single_seq_marginal_transition_logprobs,
                                                              logsumexp_with_arr_lst,
                                                              log_one_minus_x,
                                                              logspace_marginalize_inf_transits,
                                                              regular_tkf,
                                                              log_matmul)


###########
### init  #
###########
# dims
C_dom = 2
C_frag = 5
T = 6

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
                  'num_domain_mixtures': C_dom,
                  'num_fragment_mixtures': C_frag}

fragment_model = TKF92TransitionLogprobs(config=transit_config,
                                         name='mymod')

init_params = fragment_model.init( rngs=jax.random.key(0),
                                    t_array=t_array,
                                    return_all_matrices=False,
                                    sow_intermediates=False )

### get indel params, scoring matrix
out = fragment_model.apply( variables=init_params, 
                            t_array=t_array,
                            return_all_matrices=True,
                            sow_intermediates=False)

# log_frag_class_probs is (C_dom, C_frag)
log_frag_class_probs, matrix_dict, _, frag_tkf_params_dict = out
del out, init_params

frag_joint_transit_mat = matrix_dict['joint'] #(T, C_dom, C_frag_to, C_frag_from, S_from, S_to)
frag_marginal_transit_mat = matrix_dict['marginal'] #(C_dom, C_frag_to, C_frag_from, 2, 2)
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


### create T_mat_{MIDS, MIDE} to modify later
# multiply any -> M by (1 - z_t)
S = dom_joint_transit_mat.shape[-1]
mask = jnp.concatenate( [jnp.ones(  (T, S, 1), dtype = bool),
                         jnp.zeros( (T, S, 3), dtype=bool )], axis=2 )
log_T_mat = jnp.where(mask, 
                  dom_joint_transit_mat + log_one_minus_z_t[:,None,None], 
                  dom_joint_transit_mat) #(T, S_from, S_to)
assert log_T_mat.shape == (T, S, S)
del mask

# multiply any ->ID by (1 - z_0)
mask = jnp.concatenate( [jnp.zeros( (T, S, 1), dtype = bool),
                         jnp.ones(  (T, S, 2), dtype = bool),
                         jnp.zeros( (T, S, 1), dtype = bool)], axis=2 )
log_T_mat = jnp.where(mask, log_T_mat + log_one_minus_z_0, log_T_mat) #(T, S_from, S_to)
assert log_T_mat.shape == (T, S, S)
del mask


### get U_{MIDS, AB}
#   M: 0
#   I: 1
#   D: 2
# S/E: 3

# U_{M,A} = z_t \tau_{M,M} + z_0 \tau_{M,I}
log_u_m_a = jnp.logaddexp( log_z_t + dom_joint_transit_mat[:, 0, 0],
                           log_z_0 + dom_joint_transit_mat[:, 0, 1] ) #(T,)

# U_{I,A} = z_t \tau_{I,I} + z_0 \tau_{I,I}
log_u_i_a = jnp.logaddexp( log_z_t + dom_joint_transit_mat[:, 1, 1],
                           log_z_0 + dom_joint_transit_mat[:, 1, 1] ) #(T,)

# U_{D,A} = z_t \tau_{D,D} + z_0 \tau_{D,I}
log_u_d_a = jnp.logaddexp( log_z_t + dom_joint_transit_mat[:, 2, 2],
                           log_z_0 + dom_joint_transit_mat[:, 2, 1] ) #(T,)

# U_{S,A} = z_t \tau_{S,M} + z_0 \tau_{S,I}
log_u_s_a = jnp.logaddexp( log_z_t + dom_joint_transit_mat[:, 3, 0],
                           log_z_0 + dom_joint_transit_mat[:, 3, 1] ) #(T,)

# concatenate all into a column
log_u_mids_a = jnp.stack([log_u_m_a, log_u_i_a, log_u_d_a, log_u_s_a], axis=1) #(T, 4)
del log_u_m_a, log_u_i_a, log_u_d_a, log_u_s_a

# U_{MIDS, D} = z_0 \tau_{MIDS,D}
log_u_mids_b = log_z_0 + dom_joint_transit_mat[..., 2] #(T, 4)

# final mat
log_u_mids_ab = jnp.stack([log_u_mids_a, log_u_mids_b], axis=2) #(T, 4, 2)
del log_u_mids_a, log_u_mids_b


### get U_{AB, MIDS} from already-created log_T_mat
# U_{A, MIDE} = T_mat_{M, any}
# U_{B, MIDE} = T_mat_{D, any}
log_u_ab_mide = log_T_mat[:, [0,2], :] #(T, 2, 4)


### get U_{AB, AB}
# U_{A, AB} = U_{M,AB}
log_u_a_ab = log_u_mids_ab[:,0,:] #(T, 2)

# U_{B, A} = z_t \tau_{D,M} + z_0 \tau_{DI}
log_u_b_a = jnp.logaddexp( log_z_t + dom_joint_transit_mat[:, 2, 0],
                           log_z_0 + dom_joint_transit_mat[:, 2, 1] ) #(T,)

# U_{B, B} = U_{D,B}
log_u_b_b = log_u_mids_ab[:,2,1] #(T,)

# create the full matrix from parts
log_u_b_ab = jnp.stack([log_u_b_a, log_u_b_b], axis=1) #(T, 2)

log_u_ab_ab = jnp.stack([log_u_a_ab, log_u_b_ab], axis=1) #(T, 2, 2)

# make sure this matrix was put together correctly
assert np.allclose(log_u_ab_ab[:,0,0], log_u_a_ab[:,0])
assert np.allclose(log_u_ab_ab[:,0,1], log_u_a_ab[:,1])
assert np.allclose(log_u_ab_ab[:,1,0], log_u_b_a)
assert np.allclose(log_u_ab_ab[:,1,1], log_u_b_b)
del log_u_b_a, log_u_b_b, log_u_a_ab, log_u_b_ab


### T_{MIDS, MIDE} = U_{MIDS, MIDE} + U_{MIDS,AB} * (I-U_{AB,AB})^-1 * U_{AB,MIDE}
# modifying matrix: U_{MIDS,AB} * (I-U_{AB,AB})^-1 * U_{AB,MIDE}
log_inv_arg = logspace_marginalize_inf_transits( log_u_ab_ab ) #(T, 2, 2)
mod_first_half = log_matmul( log_A = log_u_mids_ab,
                             log_B = log_inv_arg ) #(T, 4, 2)
modifier = log_matmul( log_A = mod_first_half, 
                       log_B = log_u_ab_mide ) #(T, 4, 4)
log_T_mat = jnp.logaddexp( log_T_mat, modifier ) #(T, S_from, S_to)
del log_inv_arg, mod_first_half, modifier, log_u_mids_ab, log_u_ab_mide, log_u_ab_ab
del log_one_minus_z_0, log_one_minus_z_t, log_z_0, log_z_t


##############################################
### Precompute some values that get reused   #
##############################################
# frag_joint_transit_mat: (T, C_dom, C_frag_from, C_frag_to, S_from, S_to)
# log_domain_class_probs: (C_dom,)
# log_frag_class_probs: (C_dom, C_frag)
# log_T_mat = (T, S_from, S_to)

### v_m * (lam_m / mu_m) * w_{mg}; used to open single-sequence fragments
start_single_seq_frag_g = ( log_domain_class_probs[:,None] +
                            frag_tkf_params_dict['log_one_minus_offset'][:,None] + 
                            log_frag_class_probs ) #(C_dom_to, C_frag_to)
assert start_single_seq_frag_g.shape == (C_dom, C_frag)


### v_m * \tau_{SY}^(m) * w_{mg}; used to open pair-aligned fragments
log_domain_class_probs_reshaped = log_domain_class_probs[None,:,None,None] #(1, C_dom_to, 1, 1)

# for every C_frag_from -> C_frag_to, the S -> any transition row is the same
# (since "start" has no class label); so just index the first instance here
# w_{mg} is already included in frag_joint_transit_mat
#
# frag_joint_transit_mat indexing:
# dim0: T; take all
# dim1: C_dom_to; take all
# dim2: C_frag_from; take the 0th element
# dim3: C_frag_to; take all
# dim4: S_from; take the last element (corresponding to START)
# dim5: S_to; take element 0, 1, 2 (corresponding to MATCH, INS, DEL)
frag_joint_transit_mat_reshaped = frag_joint_transit_mat[:, :, 0, :, 3, 0:3] #(T, C_dom_to, C_frag_to, (S_to \in MID) )

# final mat
start_pair_frag_g = log_domain_class_probs_reshaped + frag_joint_transit_mat_reshaped #(T, C_dom_to, C_frag_to, (S_to \in MID) )
assert start_pair_frag_g.shape == (T, C_dom, C_frag, S-1)
del log_domain_class_probs_reshaped, frag_joint_transit_mat_reshaped


### (1 - r_f) (1 - (lam_l / mu_l)); used to close single-sequence fragments
end_single_seq_frag_f = jnp.log1p( -r_frag ) + frag_tkf_params_dict['log_offset'][:,None] #(C_dom_from, C_frag_from)
assert end_single_seq_frag_f.shape == (C_dom, C_frag)


### (1 - r_f) \tau_{XE}^{l}; used to close pair-aligned fragments
# for every C_frag_from -> C_frag_to, the any -> E transition column is the same
# (since "end" has no class label); so just index the last instance here
#
# frag_joint_transit_mat indexing:
# dim0: T; take all
# dim1: C_dom_from; take all
# dim2: C_frag_from; take all
# dim3: C_frag_to; take the last element
# dim4: S_from; take element 0, 1, 2 (corresponding to MATCH, INS, DEL) 
# dim5: S_to; take the last element (corresponding to END)
end_pair_frag_f = frag_joint_transit_mat[:, :, :, -1, 0:3, 3] #(T, C_dom_from, C_frag_from, (S_from \in MID) )
assert end_pair_frag_f.shape == (T, C_dom, C_frag, S-1)



############################################
### Calculate all the transitions needed   #
############################################
### MX -> MY
#   end_pair_frag_f: (T, C_dom_from,        1, C_frag_from,         1, (S_from \in MID),              1 )
#  log_T_mat[:,0,0]: (T,          1,        1,           1,         1,                1,              1 )
# start_pair_frag_g: (T,          1, C_dom_to,           1, C_frag_to,                1, (S_to \in MID) )
#          mx_to_my: (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to, (S_from \in MID), (S_to \in MID) )
mx_to_my = ( end_pair_frag_f[:, :, None, :, None, :, None] +
             log_T_mat[:,0,0][:, None, None, None, None, None, None] +
             start_pair_frag_g[:, None, :, None, :, None, :] ) #(T, C_dom_from, C_dom_to, C_frag_from, C_frag_to, (S_from \in MID), (S_to \in MID) )
assert mx_to_my.shape == (T, C_dom, C_dom, C_frag, C_frag, S-1, S-1)

# if extending the domain and/or fragment, add probabilities from JOINT transitions of fragment-level mixture model
prev_values = jnp.transpose(jnp.diagonal(mx_to_my, axis1=1, axis2=2), (0, 5, 1, 2, 3, 4) )  #(T, C_dom, C_frag_from, C_frag_to, (S_from \in MID), (S_to \in MID) )
new_values = jnp.logaddexp(prev_values, frag_joint_transit_mat[..., 0:3, 0:3]) #(T, C_dom, C_frag_from, C_frag_to, (S_from \in MID), (S_to \in MID) )

assert prev_values.shape == (T, C_dom, C_frag, C_frag, S-1, S-1)
assert new_values.shape == (T, C_dom, C_frag, C_frag, S-1, S-1)

idx = jnp.arange(C_dom)
mx_to_my = mx_to_my.at[:, idx, idx, ...].set(new_values) #(T, C_dom_from, C_dom_to, C_frag_from, C_frag_to, (S_from \in MID), (S_to \in MID) )
assert mx_to_my.shape == (T, C_dom, C_dom, C_frag, C_frag, S-1, S-1)

del prev_values, new_values, idx


### MX -> II, DD, EE
#         end_pair_frag_f: (T, C_dom_from,        1, C_frag_from,         1, (S_from \in MID) )
#        log_T_mat[:,0,1]: (T,          1,        1,           1,         1,                1 )
# start_single_seq_frag_g: (1,          1, C_dom_to,           1, C_frag_to,                1 )
#                mx_to_ii: (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to, (S_from \in MID) )
mx_to_ii = ( end_pair_frag_f[:, :, None, :, None, :] +
             log_T_mat[:,0,1][:, None, None, None, None, None] +
             start_single_seq_frag_g[None, None, :, None, :, None] ) #(T, C_dom_from, C_dom_to, C_frag_from, C_frag_to, (S_from \in MID) )

#         end_pair_frag_f: (T, C_dom_from,        1, C_frag_from,         1, (S_from \in MID) )
#        log_T_mat[:,0,2]: (T,          1,        1,           1,         1,                1 )
# start_single_seq_frag_g: (1,          1, C_dom_to,           1, C_frag_to,                1 )
#                mx_to_dd: (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to, (S_from \in MID) )
mx_to_dd = ( end_pair_frag_f[:, :, None, :, None, :] +
             log_T_mat[:,0,2][:, None, None, None, None, None] +
             start_single_seq_frag_g[None, None, :, None, :, None] ) #(T, C_dom_from, C_dom_to, C_frag_from, C_frag_to, (S_from \in MID) )

#  end_pair_frag_f: (T, C_dom_from, C_frag_from, (S_from \in MID) )
# log_T_mat[:,0,3]: (T,          1,           1,                1 )
#         mx_to_ee: (T, C_dom_from, C_frag_from, (S_from \in MID) )
mx_to_ee = ( end_pair_frag_f + 
             log_T_mat[:,0,3][:, None, None, None] ) #(T, C_dom_from, C_frag_from, (S_from \in MID) )

assert mx_to_ii.shape == (T, C_dom, C_dom, C_frag, C_frag, S-1)
assert mx_to_dd.shape == (T, C_dom, C_dom, C_frag, C_frag, S-1)
assert mx_to_ee.shape == (T, C_dom, C_frag, S-1)


### II -> II
#   end_single_seq_frag_f: (1, C_dom_from,        1, C_frag_from,         1)
#        log_T_mat[:,1,1]: (T,          1,        1,           1,         1)
# start_single_seq_frag_g: (1,          1, C_dom_to,           1, C_frag_to)
#                ii_to_ii: (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to)
ii_to_ii = ( end_single_seq_frag_f[None, :, None, :, None] +
             log_T_mat[:,1,1][:, None, None, None, None] +
             start_single_seq_frag_g[None, None, :, None, :] ) # (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to)
assert ii_to_ii.shape == (T, C_dom, C_dom, C_frag, C_frag)

# if extending the domain and/or fragment, add probabilities from MARGINAL transitions of fragment-level mixture model
prev_values = jnp.transpose( jnp.diagonal(ii_to_ii, axis1=1, axis2=2), (0, 3, 1, 2) ) #(T, C_dom, C_frag_from, C_frag_to)
new_values = jnp.logaddexp( prev_values, frag_marginal_transit_mat[..., 0,0] ) #(T, C_dom, C_frag_from, C_frag_to)
assert prev_values.shape == (T, C_dom, C_frag, C_frag)
assert new_values.shape == (T, C_dom, C_frag, C_frag)

idx = jnp.arange(C_dom)
ii_to_ii = ii_to_ii.at[:, idx, idx, ...].set(new_values) # (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to)
assert ii_to_ii.shape == (T, C_dom, C_dom, C_frag, C_frag)
del prev_values, new_values, idx


### II -> MY, DD, EE
# end_single_seq_frag_f: (1, C_dom_from,        1, C_frag_from,         1               1 )
#      log_T_mat[:,1,0]: (T,          1,        1,           1,         1,              1 )
#     start_pair_frag_g: (T,          1, C_dom_to,           1, C_frag_to, (S_to \in MID) )
#              ii_to_mx: (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to, (S_to \in MID) )
ii_to_mx = ( end_single_seq_frag_f[None, :, None, :, None, None] +
             log_T_mat[:,1,0][:, None, None, None, None, None] +
             start_pair_frag_g[:, None, :, None, :, :] )  # (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to, (S_to \in MID) )

#   end_single_seq_frag_f: (1, C_dom_from,        1, C_frag_from,         1)
#        log_T_mat[:,1,2]: (T,          1,        1,           1,         1)
# start_single_seq_frag_g: (1,          1, C_dom_to,           1, C_frag_to)
#                ii_to_dd: (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to)
ii_to_dd = ( end_single_seq_frag_f[None, :, None, :, None] +
             log_T_mat[:,1,2][:, None, None, None, None] +
             start_single_seq_frag_g[None, None, :, None, :] ) #(T, C_dom_from, C_dom_to, C_frag_from, C_frag_to )

# end_single_seq_frag_f: (1, C_dom_from, C_frag_from)
#      log_T_mat[:,1,3]: (T,          1,           1)
#              ii_to_ee: (T, C_dom_from, C_frag_from)
ii_to_ee = ( end_single_seq_frag_f[None, :, :] +
             log_T_mat[:,1,3][:,None,None] ) # (T, C_dom_from, C_frag_from)

assert ii_to_mx.shape == (T, C_dom, C_dom, C_frag, C_frag, S-1)
assert ii_to_dd.shape == (T, C_dom, C_dom, C_frag, C_frag)
assert ii_to_ee.shape == (T, C_dom, C_frag)


### DD -> DD
dd_to_dd = ( end_single_seq_frag_f[None, :, None, :, None] +
             log_T_mat[:,2,2][:, None, None, None, None] +
             start_single_seq_frag_g[None, None, :, None, :] ) # (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to)
assert dd_to_dd.shape == (T, C_dom, C_dom, C_frag, C_frag)

# if extending the domain and/or fragment, add probabilities from MARGINAL transitions of fragment-level mixture model
prev_values = jnp.transpose( jnp.diagonal(dd_to_dd, axis1=1, axis2=2), (0, 3, 1, 2) ) #(T, C_dom, C_frag_from, C_frag_to)
new_values = jnp.logaddexp( prev_values, frag_marginal_transit_mat[..., 0,0] ) #(T, C_dom, C_frag_from, C_frag_to)
assert prev_values.shape == (T, C_dom, C_frag, C_frag)
assert new_values.shape == (T, C_dom, C_frag, C_frag)

idx = jnp.arange(C_dom)
dd_to_dd = dd_to_dd.at[:, idx, idx, ...].set(new_values) # (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to)
assert dd_to_dd.shape == (T, C_dom, C_dom, C_frag, C_frag)
del prev_values, new_values, idx


### DD -> MY, II, EE
# end_single_seq_frag_f: (1, C_dom_from,        1, C_frag_from,         1               1 )
#      log_T_mat[:,2,0]: (T,          1,        1,           1,         1,              1 )
#     start_pair_frag_g: (T,          1, C_dom_to,           1, C_frag_to, (S_to \in MID) )
#              dd_to_mx: (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to, (S_to \in MID) )
dd_to_mx = ( end_single_seq_frag_f[None, :, None, :, None, None] +
             log_T_mat[:,2,0][:, None, None, None, None, None] +
             start_pair_frag_g[:, None, :, None, :, :] )  # (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to, (S_to \in MID) )

#   end_single_seq_frag_f: (1, C_dom_from,        1, C_frag_from,         1)
#        log_T_mat[:,2,1]: (T,          1,        1,           1,         1)
# start_single_seq_frag_g: (1,          1, C_dom_to,           1, C_frag_to)
#                dd_to_ii: (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to)
dd_to_ii = ( end_single_seq_frag_f[None, :, None, :, None] +
             log_T_mat[:,2,1][:, None, None, None, None] +
             start_single_seq_frag_g[None, None, :, None, :] ) #(T, C_dom_from, C_dom_to, C_frag_from, C_frag_to )

# end_single_seq_frag_f: (1, C_dom_from, C_frag_from)
#      log_T_mat[:,2,3]: (T,          1,           1)
#              dd_to_ee: (T, C_dom_from, C_frag_from)
dd_to_ee = ( end_single_seq_frag_f[None, ...] +
             log_T_mat[:,2,3][:,None,None] ) # (T, C_dom_from, C_frag_from)

assert dd_to_mx.shape == (T, C_dom, C_dom, C_frag, C_frag, S-1)
assert dd_to_dd.shape == (T, C_dom, C_dom, C_frag, C_frag)
assert dd_to_ee.shape == (T, C_dom, C_frag)


### SS -> MY,II,DD
# ss -> ee is just log_T_mat[:,3,3]; no other modifications needed

#  log_T_mat[:,3,0]: (T,        1,         1,              1 )
# start_pair_frag_g: (T, C_dom_to, C_frag_to, (S_to \in MID) )
#          ss_to_my: (T, C_dom_to, C_frag_to, (S_to \in MID) )
ss_to_my = log_T_mat[:,3,0][:,None, None, None] + start_pair_frag_g #(T, C_dom_to, C_frag_to, (S_to \in MID) )

#  log_T_mat[:,3,1]: (T,        1,         1)
# start_pair_frag_g: (T, C_dom_to, C_frag_to)
#          ss_to_ii: (T, C_dom_to, C_frag_to)
ss_to_ii = log_T_mat[:,3,1][:,None, None] + start_single_seq_frag_g[None, :, :] #(T, C_dom_to, C_frag_to)

#  log_T_mat[:,3,2]: (T,        1,         1)
# start_pair_frag_g: (T, C_dom_to, C_frag_to)
#          ss_to_dd: (T, C_dom_to, C_frag_to)
ss_to_dd = log_T_mat[:,3,2][:,None, None] + start_single_seq_frag_g[None, :, :] #(T, C_dom_to, C_frag_to)

assert ss_to_my.shape == (T, C_dom, C_frag, S-1)
assert ss_to_ii.shape == (T, C_dom, C_frag)
assert ss_to_dd.shape == (T, C_dom, C_frag)



#####################################
### Build final transition matrix   #
#####################################
# Row 1: Match -> Any
m_to_m = mx_to_my[..., 0, 0] # (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to)
m_to_i = jnp.logaddexp( mx_to_my[..., 0, 1], mx_to_ii[..., 0]) # (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to)
m_to_d = jnp.logaddexp( mx_to_my[..., 0, 2], mx_to_dd[..., 0]) # (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to)
m_to_e = mx_to_ee[..., 0][:,:,None,:,None] # (T, C_dom_from, 1, C_frag_from, 1)
m_to_e = jnp.broadcast_to( m_to_e, m_to_m.shape ) # (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to)
match_to_any = jnp.stack( [m_to_m, m_to_i, m_to_d, m_to_e], axis=-1 ) # (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to, 4)
del m_to_m, m_to_i, m_to_d, m_to_e

# Row 2: Ins -> Any
i_to_m = jnp.logaddexp( mx_to_my[..., 1, 0], ii_to_mx[..., 0] ) # (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to)
i_to_i = logsumexp_with_arr_lst( [mx_to_my[..., 1, 1],
                                  mx_to_ii[..., 1],
                                  ii_to_mx[..., 1], 
                                  ii_to_ii] ) # (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to)
i_to_d = logsumexp_with_arr_lst( [mx_to_my[..., 1, 2],
                                  mx_to_dd[..., 1],
                                  ii_to_mx[..., 2], 
                                  ii_to_dd] ) # (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to)
i_to_e = ii_to_ee[:, :, None, :, None] # (T, C_dom_from, 1, C_frag_from, 1)
i_to_e = jnp.broadcast_to( i_to_e, i_to_m.shape ) # (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to)
ins_to_any = jnp.stack( [i_to_m, i_to_i, i_to_d, i_to_e], axis=-1 ) # (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to, 4)
del i_to_m, i_to_i, i_to_d, i_to_e

# Row 3: Del -> Any
d_to_m = jnp.logaddexp( mx_to_my[..., 2, 0], dd_to_mx[..., 0] ) # (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to)
d_to_i = logsumexp_with_arr_lst( [mx_to_my[..., 2, 1],
                                  mx_to_ii[..., 2],
                                  dd_to_mx[..., 1], 
                                  dd_to_ii] ) # (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to)
d_to_d = logsumexp_with_arr_lst( [mx_to_my[..., 2, 2],
                                  mx_to_dd[..., 2],
                                  dd_to_mx[..., 2],
                                  dd_to_dd] ) # (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to)
d_to_e = dd_to_ee[:, :, None, :, None] # (T, C_dom_from, 1, C_frag_from, 1)
d_to_e = jnp.broadcast_to( d_to_e, d_to_m.shape ) # (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to)
del_to_any = jnp.stack( [d_to_m, d_to_i, d_to_d, d_to_e], axis=-1 ) # (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to, 4)
del d_to_m, d_to_i, d_to_d, d_to_e

# Row 4: Start -> any
s_to_e = log_T_mat[:,3,3][:, None, None] #(T, 1, 1)
s_to_e = jnp.broadcast_to( s_to_e, ss_to_my[..., 0].shape ) # (T, C_dom_to, C_frag_to)
start_to_any = jnp.stack( [ss_to_my[..., 0],
                           jnp.logaddexp( ss_to_my[..., 1], ss_to_ii),
                           jnp.logaddexp( ss_to_my[..., 2], ss_to_dd),
                           s_to_e ], axis=-1 ) # (T, C_dom_to, C_frag_to, 4)
start_to_any = start_to_any[:, None, :, None, :, :] # (T, 1, C_dom_to, 1, C_frag_to, 4)
start_to_any = jnp.broadcast_to(start_to_any, match_to_any.shape)  # (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to, 4)
del s_to_e

transit_mat = jnp.stack( [match_to_any, ins_to_any, del_to_any, start_to_any], axis=-2 ) # (T, C_dom_from, C_dom_to, C_frag_from, C_frag_to, 4, 4)


