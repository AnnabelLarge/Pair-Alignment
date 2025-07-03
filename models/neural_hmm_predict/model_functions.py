#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 18:54:38 2025

@author: annabel

functions here:
===============
'approx_beta',
'approx_one_minus_gamma',
'approx_tkf',
'bound_sigmoid',
'bound_sigmoid_inverse',
'concat_transition_matrix',
'log_one_minus_x',
'log_x_minus_one',
'logprob_f81',
'logprob_gtr',
'logprob_tkf91',
'logprob_tkf92',
'logsumexp_with_arr_lst',
'rate_matrix_from_exch_equl',
'regular_tkf',
'safe_log',
'stable_log_one_minus_x',
'switch_tkf',
'true_beta',
'upper_tri_vector_to_sym_matrix'
"""
import jax
from jax import numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.linalg import expm
from jax._src.typing import Array, ArrayLike

from functools import partial

# make this slightly more than true jnp.finfo(jnp.float32).eps, 
#  for numerical safety at REALLY small parameter values
SMALL_POSITIVE_NUM = 5e-7


###############################################################################
### general helpers for all pairHMM models   ##################################
###############################################################################
def safe_log(x):
    return jnp.log( jnp.where( x>0, 
                               x, 
                               jnp.finfo('float32').smallest_normal ) )

def bound_sigmoid(x, min_val, max_val, *args, **kwargs):
    return min_val + (max_val - min_val) / (1 + jnp.exp(-x))

def bound_sigmoid_inverse(y, min_val, max_val, eps=1e-4):
    """
    note: this is only for logit initialization; jnp.clip has bad 
          gradients at extremes
    """
    y = jnp.clip(y, min_val + eps, max_val - eps)
    return safe_log( (y - min_val) / (max_val - y) )

def logsumexp_with_arr_lst(array_of_log_vals, coeffs = None):
    """
    concatenate a list of arrays, then use logsumexp
    """
    a_for_logsumexp = jnp.stack(array_of_log_vals, axis=-1)
    out = logsumexp(a = a_for_logsumexp,
                    b = coeffs,
                    axis=-1)
    return out

def log_one_minus_x(log_x):
    """
    calculate log( exp(log(1)) - exp(log(x)) ), which is log( 1 - x )
    """
    return jnp.log1p( -jnp.exp(log_x) )

def log_x_minus_one(log_x):
    """
    calculate log( exp(log(x)) - exp(log(1)) ), which is log( x - 1 )
    """
    return jnp.log( jnp.expm1(log_x) )

def stable_log_one_minus_x(log_x):
    """
    use log_one_minus_x if value is not too small, but return -log_x otherwise 
    """
    return jax.lax.cond( log_x < -SMALL_POSITIVE_NUM,
                         log_one_minus_x,
                         lambda x: jnp.log(-x),
                         log_x)

###############################################################################
### substitution models, emissions from match positions   #####################
###############################################################################
def logprob_f81(equl,
                rate_multiplier,
                t_array,
                unique_time_per_sample):
    """
    this is the CONDITIONAL LOG-PROBABILITY P(desc|anc,t,align=Match)
    
    this also always normalizes rate matrix to t = one substitution, THEN 
      multiplies the entire rate matrix by rate multiplier
    
    if comparing back to pairHMM implementation, need to make sure 
      norm_rate_matrix is True and norm_rate_mult is False
    
    B: batch size
    L_align: length of alignment
    T: number of times in the grid
    A: alphabet size
    
    
    Arguments
    ----------
    equl : ArrayLike
        > if per-site: (B, L_align, A)
        > if global: (1, 1, A)
    
    rate_multiplier : ArrayLike
        > if per-site: (B, L_align)
        > if global: (1, 1)
    
    t_array : ArrayLike, 
    
    unique_time_per_sample : Bool
        whether there's one time per sample, or a grid of times you'll 
        marginalize over
     
    Returns
    --------
    ArrayLike
        > if either per-site equl or rate_multiplier: 
            > if given time grid: (T, B, L_align, A, 2)
            > if unique time per sample: (B, L_align, A, 2)
        > if equl and rate_multiplier are global: 
            > if given time grid: (T, 1, 1, A, 2)
            > if unique time per sample: (1, 1, A, 2)
    """
    normalizing_factor = 1 / ( 1 - jnp.square(equl).sum(axis=-1) ) #(B, L_align)
    
    # expand to compatible dims
    if not unique_time_per_sample:
        normalizing_factor = normalizing_factor[None, ...] #(1, B, L_align) 
        rate_multiplier = rate_multiplier[None, ...] #(1, B, L_align) 
        t_array = t_array[..., None, None] #(T, 1, 1)
        equl = equl[None,...] #(1, B, L_align, A) 
    
    elif unique_time_per_sample:
        t_array = t_array[..., None] #(B, 1)
    
    ### calculate probs
    # shapes of exp_term:
    #   if per-site equlibrium distribution or rates, and unique time per 
    #     sample: (T, B, L_align, 1)
    #   if per-site equlibrium distribution or rates, and not unique time per 
    #     sample: (B, L_align, 1)
    exp_term = jnp.exp(-rate_multiplier * normalizing_factor * t_array)[..., None]
    
    # shapes of match_prob, subs_prob:
    #   if per-site equlibrium distribution or rates, and unique time per 
    #     sample: (T, B, L_align, A)
    #   if per-site equlibrium distribution or rates, and not unique time per 
    #     sample: (B, L_align, A)
    match_prob = equl + (1-equl) * exp_term
    subs_prob = equl * (1 - exp_term )
    
    # shape of final output:
    #   if per-site equlibrium distribution or rates, and unique time per 
    #     sample: (T, B, L_align, A, 2)
    #   if per-site equlibrium distribution or rates, and not unique time per 
    #     sample: (B, L_align, A, 2)
    return safe_log( jnp.stack( [match_prob, subs_prob], axis = -1 ) )

def upper_tri_vector_to_sym_matrix(vec: ArrayLike):
    """
    Given upper triangular values, fill in a symmetric matrix

    B: batch size
    L_align: length of alignment
    A: alphabet size

    Arguments
    ----------
    vec : ArrayLike, (B, L, n,) 
        upper triangular values
    
    Returns
    -------
    mat : ArrayLike, (B, L, A, A) 
        final matrix; A = ( n * (n-1) ) / 2
    
    Example at one sample, one column
    -----------------------------------
    vec = [a, b, c, d, e, f]
    
    upper_tri_vector_to_sym_matrix(vec) = [[0, a, b, c],
                                            [a, 0, d, e],
                                            [b, d, 0, f],
                                            [c, e, f, 0]]

    """
    B, L, n = vec.shape
    
    
    ### automatically detect emission alphabet size
    # 6 = DNA (alphabet size = 4)
    # 190 = proteins (alphabet size = 20)
    # 2016 = codons (alphabet size = 64)
    if vec.shape[-1] == 6:
        A = 4
    
    elif vec.shape[-1] == 190:
        A = 20
    
    elif vec.shape[-1] == 2016:
        A = 64
    
    else:
        raise ValueError(f'input dimensions are: {vec.shape}')
        

    # Get upper triangle indices (excluding diagonal)
    i_idx, j_idx = jnp.triu_indices(A, k=1) #(A,) and (A,)

    # Initialize zero matrix (B, L, A, A)
    mat = jnp.zeros((B, L, A, A))

    # Fill upper triangle
    mat = mat.at[:, :, i_idx, j_idx].set(vec)

    # Reflect to lower triangle
    mat = mat.at[:, :, j_idx, i_idx].add(vec)

    return mat #(B, L, A, A)

def rate_matrix_from_exch_equl(exchangeabilities: ArrayLike,
                               equilibrium_distributions: ArrayLike,
                               norm: bool=True):
    """
    computes rate matrix Q = \chi * \pi; normalizes to substution 
      rate of one if desired
    
    only one exchangeability; rho and pi are properties of the class
    
    B: batch size
    L_align: length of alignment
    A: alphabet size
    
    
    Arguments
    ----------
    exchangeabilities : ArrayLike, (B, L_align, A, A) 
        symmetric exchangeability parameter matrix
        
    equilibrium_distributions : ArrayLike, (B, L_align, A) 
        amino acid equilibriums per site
    
    norm : bool, optional; default is True

    Returns
    -------
    subst_rate_mat : ArrayLike, (B, L_align, A, A) 

    """
    # reshape for einsum
    B = max( [exchangeabilities.shape[0],
              equilibrium_distributions.shape[0]] )
    L_align = max( [exchangeabilities.shape[1],
                    equilibrium_distributions.shape[1] ] )
    A = exchangeabilities.shape[-1]

    # Q = chi * diag(pi); q_ij = chi_ij * pi_j
    rate_mat_without_diags = exchangeabilities * equilibrium_distributions[:, :, None, :] #(B, L_align, A, A)
    
    # put the row sums in the diagonals
    neg_row_sums = -rate_mat_without_diags.sum(axis=-1)  # (B, L_align, A) 
    diags = jnp.eye( A, dtype=bool )[None,None,...]   # (1, 1, A, A)
    diags = jnp.broadcast_to( diags, (B, L_align, A, A) )  # (B, L_align, A, A) 
    neg_row_sums_to_add = neg_row_sums[..., None] * diags # (B, L_align, A, A) 
    subst_rate_mat = rate_mat_without_diags + neg_row_sums_to_add  # (B, L_align, A, A) 
    del neg_row_sums, diags
    
    # normalize (true by default)
    if norm:
        diags = jnp.diagonal(subst_rate_mat, axis1=-2, axis2=-1) # (B, L_align, A) 
        norm_factor = -jnp.sum(diags * equilibrium_distributions, axis=-1)[...,None,None] #(B, L_align, 1, 1)
        subst_rate_mat = subst_rate_mat / norm_factor # (B, L_align, A, A) 
    
    return subst_rate_mat

def logprob_gtr( exch_upper_triag_values,
                 equilibrium_distributions,
                 rate_multiplier,
                 t_array,
                 unique_time_per_sample ):
    """
    this is the CONDITIONAL LOG-PROBABILITY P(desc|anc,t,align=Match)
    
    from exchangeabilities and equililbrium distributions, use matrix
      exponential to get log-probability of emissions at match sites
    
    B: batch size
    L_align: length of alignment
    T: number of times in the grid
    A: alphabet size
    
    
    Arguments
    ----------
    exch_upper_triag_values : ArrayLike
        > if per-site: (B, L_align, n)
        > if global: (1, 1, n)
    
    equilibrium_distributions : ArrayLike
        > if per-site: (B, L_align, A)
        > if global: (1, 1, A)
    
    rate_multiplier : ArrayLike
        > if per-site: (B, L_align)
        > if global: (1, 1)
    
    t_array : ArrayLike, (T,) or (B,)
    
    unique_time_per_sample : Bool
        whether there's one time per sample, or a grid of times you'll 
        marginalize over
     
    Returns
    --------
    ArrayLike
        > if any parameter set is per-site: 
            > if given time grid: (T, B, L_align, A, A)
            > if unique time per sample: (B, L_align, A, A)
        > if all are global: 
            > if given time grid: (T, 1, 1, A, A)
            > if unique time per sample: (1, 1, A, A)
    """
    L_align = max( [exch_upper_triag_values.shape[1],
                    equilibrium_distributions.shape[1],
                    rate_multiplier.shape[1]] )
    A = equilibrium_distributions.shape[-1]
    
    # place these in a square matrix
    exchangeabilities = upper_tri_vector_to_sym_matrix(vec = exch_upper_triag_values) # (B, L_align, A, A) 
    
    # generate rate matrix, and normalize it 
    normed_rate_mat = rate_matrix_from_exch_equl(exchangeabilities = exchangeabilities,
                                                 equilibrium_distributions = equilibrium_distributions,
                                                 norm = True) #(B, L_align, A, A) 
    
    # scale by rate multiplier AFTER normalization
    rate_mat = rate_multiplier[..., None, None] * normed_rate_mat #(B, L_align, A, A) 
    
    # adjust dims
    if not unique_time_per_sample:
        T = t_array.shape[0]
        B = max( [exch_upper_triag_values.shape[0],
                  equilibrium_distributions.shape[0],
                  rate_multiplier.shape[0]] )
        
        before_reshape = (T*B*L_align, A, A)
        after_reshape = (T, B, L_align, A, A)
        t_array = jnp.expand_dims(t_array, (1,2,3,4)) #(T, 1, 1, 1, 1)
        rate_mat = rate_mat[None, ...] #(1, B, L_align, A, A) 
    
    elif unique_time_per_sample:
        B = max( [exch_upper_triag_values.shape[0],
                  equilibrium_distributions.shape[0],
                  rate_multiplier.shape[0],
                  t_array.shape[0]] )
    
        before_reshape = (B*L_align, A, A)
        after_reshape = (B, L_align, A, A)
        t_array = jnp.expand_dims(t_array, (1,2,3)) #(B, 1, 1, 1)
    
    oper = rate_mat * t_array # (T, B, L_align, A, A) or (B, L_align, A, A)
    
    # apply matrix exponential with vmap
    reshaped_oper = jnp.reshape( oper, before_reshape ) #(T*B*L, A, A) or (B*L, A, A)
    vmapped_expm = jax.vmap(expm, in_axes=0)
    cond_prob_raw = vmapped_expm( reshaped_oper ) #(T*B*L, A, A) or (B*L, A, A)
    cond_prob = jnp.reshape( cond_prob_raw, after_reshape )
    
    return safe_log(cond_prob) # (T, B, L_align, A, A) or (B, L_align, A, A)


###############################################################################
### for tkf91, tkf92: tkf parameters and their approximations   ###############
###############################################################################
def true_beta(oper):
    """
    the true formula for beta, assuming mu = lambda * (1 - offset)
    """
    mu, offset, t = oper
    
    # log( (1 - offset) * (exp(mu*offset*t) - 1) )
    log_num = jnp.log1p(-offset) + log_x_minus_one( mu*offset*t )
    
    # work out common shape, if batched
    if len(mu.shape) > 0:
        dim0 = max([mu.shape[0], t.shape[0]])
        dim1 = max([mu.shape[1], t.shape[1]])
        if len(mu.shape) == 3:
            dim2 = max([mu.shape[2], t.shape[2]])
            final_shape = (dim0, dim1, dim2)
        elif len(mu.shape) == 2:
            final_shape = (dim0, dim1)
            
        # a = mu*offset*t
        # b = jnp.log( 1 - offset )
        # logsumexp with coeffs does: 
        #   log( exp(a) - exp(b) ) = log( exp(mu*offset*t) - (1-offset) )
        a = jnp.broadcast_to( mu*offset*t, final_shape )
        b = jnp.broadcast_to( jnp.log1p(-offset), final_shape )
    
    # do computation as intended, otherwise
    else:
        a = mu*offset*t
        b = jnp.log1p(-offset)
    
    log_denom = logsumexp_with_arr_lst( [a, b], coeffs = jnp.array([1.0, -1.0]) )
    
    return log_num - log_denom

def approx_beta(oper):
    """
    as lambda approaches mu (or as time shrinks to small values), use 
      first-order taylor approximation
    """
    mu, offset, t = oper
    
    # log(  (1 - offset) * mu * t  )
    log_num = jnp.log1p(-offset) + jnp.log(mu) + jnp.log(t)
    
    # log( mu*t + 1 )
    log_denom = jnp.log1p( mu * t )
    
    return log_num - log_denom

def approx_one_minus_gamma(oper):
    """
    where 1 - gamma is unstable, use this second-order taylor approximation
        instead
    """
    mu, offset, t = oper
    
    # log( 1 + 0.5*mu*offset*t )
    log_num = jnp.log1p( 0.5 * mu * offset * t )
    
    # log( (1 - 0.5*mu*t) (mu*t + 1) )
    # there's another squared term here:
    #   0.5 * offset * (mu*t)**2
    # but it's so small that it's negligible
    log_denom = jnp.log1p( -0.5*mu*t ) + jnp.log1p( mu*t )
    
    return log_num - log_denom

def switch_tkf( mu, 
                offset, 
                t_array, 
                unique_time_per_sample ):
    """
    return alpha, beta, gamma for TKF models

    use real formulas where you can, and taylor-approximations where you can't
    
    T: number of branch lengths in t_array
    
    returns:
    --------
    out_dict: the tkf values
        out_dict['log_alpha']: ArrayLike[float32], 
        out_dict['log_one_minus_alpha']: ArrayLike[float32], 
        out_dict['log_beta']: ArrayLike[float32], 
        out_dict['log_one_minus_beta']: ArrayLike[float32], 
        out_dict['log_gamma']: ArrayLike[float32], 
        out_dict['log_one_minus_gamma']: ArrayLike[float32], 
    
    approx_flags_dict: where you used approx formulas
        out_dict['log_one_minus_alpha']: ArrayLike[bool], 
        out_dict['log_beta']: ArrayLike[bool], 
        out_dict['log_one_minus_gamma']: ArrayLike[bool], 
        out_dict['log_gamma']: ArrayLike[bool], 
    
    """
    L_align = mu.shape[1]
    
    # mu: (B, L_align)
    # offset: (B, L_align)
    # t_array: either (B,) or (T,)
    if not unique_time_per_sample:
        B = mu.shape[0]
        T = t_array.shape[0]
        
        mu = mu[None,...] #(1, B, L_align)
        offset = offset[None,...] #(1, B, L_align)
        t_array = t_array[:,None,None] #(T, 1, 1)
        final_shape = (T, B, L_align)
    
    elif unique_time_per_sample:
        B = t_array.shape[0]
        t_array = t_array[:,None] #(B, 1)
        final_shape = (B, L_align)
    
    
    ######################################################
    ### Some operations can be done with entire arrays   #
    ######################################################
    ### alpha = exp(-mu*t)
    ### log(alpha) = -mu*t
    log_alpha = -mu*t_array #(B, L_align) or (T, B, L_align)
    
    
    ### start of calculation for 1 - gamma
    # numerator:
    # log( exp(mu*offset*t) - 1 )
    gamm_full_log_num = log_x_minus_one( log_x = mu*offset*t_array ) #(B, L_align) or (T, B, L_align)
    
    # denominator, term 1
    # x = mu*offset*t
    # y = jnp.log( 1 - offset )
    # logsumexp with coeffs does: 
    #   log( exp(x) - exp(y) ) = log( exp(mu*offset*t) - (1-offset) )
    constant = jnp.broadcast_to(jnp.log1p(-offset), final_shape)
    gamma_full_log_denom_term1 = logsumexp_with_arr_lst( [mu*offset*t_array, constant],
                                              coeffs = jnp.array([1.0, -1.0]) )
    
    
    ###############################################################
    ### Most have to be done one-at-a-time, due to jax.lax.cond   #
    ###############################################################
    def tkf_params_indv(log_alpha_indv, 
                        gamma_log_numerator_indv,
                        gamma_log_denom_term1_indv,
                        mu_indv,
                        offset_indv,
                        t):
        ### 1 - alpha
        log_one_minus_alpha = stable_log_one_minus_x(log_x = log_alpha_indv)
        
        
        ### beta, 1 - beta
        # beta
        log_beta = jax.lax.cond( mu_indv*offset_indv*t > SMALL_POSITIVE_NUM ,
                                  true_beta,
                                  approx_beta,
                                  (mu_indv, offset_indv, t) )  
        
        # regardless of approx or not, 1-beta calculated from beta
        log_one_minus_beta = log_one_minus_x(log_x = log_beta)
        
        
        ### 1 - gamma, gamma
        # need log(1 - alpha) to finish calculating denominator for log(1 - gamma)
        gamma_log_denom = gamma_log_denom_term1_indv + log_one_minus_alpha
        
        # ad hoc series of conditionals to determine if you approx
        #   1-gamma or not (meh)
        valid_frac = gamma_log_numerator_indv < gamma_log_denom
        large_product = mu_indv * offset_indv * t > 1e-3
        log_diff_large = jnp.abs(gamma_log_numerator_indv - gamma_log_denom) > 0.1
        approx_formula_will_fail = (0.5*mu_indv*t) > 1.0
        
        cond1 = large_product
        cond2 = ~large_product & log_diff_large
        cond3 = ~large_product & ~log_diff_large & approx_formula_will_fail
        use_real_function = valid_frac & ( cond1 | cond2 | cond3 )
        
        # the final value
        log_one_minus_gamma = jax.lax.cond( use_real_function,
                                            lambda _: gamma_log_numerator_indv - gamma_log_denom,
                                            approx_one_minus_gamma,
                                            (mu_indv, offset_indv, t) )
        
        # gamma
        log_gamma = stable_log_one_minus_x(log_x = log_one_minus_gamma)
        
        
        ### output everything
        out_dict = {'log_one_minus_alpha': log_one_minus_alpha,
                    'log_beta': log_beta,
                    'log_one_minus_beta': log_one_minus_beta,
                    'log_gamma': log_gamma,
                    'log_one_minus_gamma': log_one_minus_gamma}
        
        used_one_minus_alpha_approx = ~(log_alpha_indv < -SMALL_POSITIVE_NUM)
        used_beta_approx = ~(mu_indv*offset_indv*t > SMALL_POSITIVE_NUM)
        used_log_one_minus_gamma_approx = ~use_real_function
        used_log_gamma_approx = ~(log_one_minus_gamma < -SMALL_POSITIVE_NUM)
        
        # if you used an approx anywhere, save the time
        flag = used_one_minus_alpha_approx | used_beta_approx | used_log_one_minus_gamma_approx | used_log_gamma_approx
        t_to_add = jnp.where( flag,
                              t,
        
                              -1. )
        # all outputs will be: (T*B*L_align) or (B*L_align)
        approx_flags_dict = {'log_one_minus_alpha': used_one_minus_alpha_approx,
                              'log_beta': used_beta_approx,
                              'log_one_minus_gamma': used_log_one_minus_gamma_approx,
                              'log_gamma': used_log_gamma_approx} 
        
        return out_dict, approx_flags_dict
    
    # vmap over B*L*T dim, instead of just T dim
    log_alpha_reshaped = log_alpha.flatten() #(T*B*L_align) or (B*L_align)
    gamma_full_log_num_reshaped = gamm_full_log_num.flatten() #(T*B*L_align) or (B*L_align)
    gamma_full_log_denom_term1_reshaped = gamma_full_log_denom_term1.flatten() #(T*B*L_align) or (B*L_align)
    t_array_reshaped = jnp.broadcast_to( t_array, log_alpha.shape ) #(T, B, L_align) or (B, L_align)
    t_array_reshaped = t_array_reshaped.flatten() #(T*B*L_align) or (B*L_align)
    mu_reshaped = jnp.broadcast_to( mu, log_alpha.shape )
    mu_reshaped = mu_reshaped.flatten()
    offset_reshaped = jnp.broadcast_to( offset, log_alpha.shape )
    offset_reshaped = offset_reshaped.flatten()
    
    # vmap the function
    vmapped_tkf_params_indv = jax.vmap(tkf_params_indv)
    out = vmapped_tkf_params_indv(log_alpha_reshaped,
                                  gamma_full_log_num_reshaped,
                                  gamma_full_log_denom_term1_reshaped,
                                  mu_reshaped,
                                  offset_reshaped,
                                  t_array_reshaped)
    tkf_params_dict, approx_flags_dict = out
    del out
    
    # reshape all
    def my_reshape(m):
        return jnp.reshape(m, final_shape)
    
    tkf_params_dict['log_one_minus_alpha'] = my_reshape( tkf_params_dict['log_one_minus_alpha'] )
    tkf_params_dict['log_beta'] = my_reshape( tkf_params_dict['log_beta'] )
    tkf_params_dict['log_one_minus_beta'] = my_reshape( tkf_params_dict['log_one_minus_beta'] )
    tkf_params_dict['log_gamma'] = my_reshape( tkf_params_dict['log_gamma'] )
    tkf_params_dict['log_one_minus_gamma'] = my_reshape( tkf_params_dict['log_one_minus_gamma'] )
    tkf_params_dict['log_alpha'] = log_alpha
    
    # just aggregate counts from this one
    approx_flags_dict['log_one_minus_alpha'] = approx_flags_dict['log_one_minus_alpha'].sum()
    approx_flags_dict['log_beta'] = approx_flags_dict['log_beta'].sum()
    approx_flags_dict['log_one_minus_gamma'] = approx_flags_dict['log_one_minus_gamma'].sum()
    approx_flags_dict['log_gamma'] = approx_flags_dict['log_gamma'].sum()
    
    return tkf_params_dict, approx_flags_dict


def regular_tkf( mu, 
                 offset, 
                 t_array,
                 unique_time_per_sample ):
    """
    return alpha, beta, gamma for TKF models; no approximations made, 
        except still allow use of switch between approx and real for 
        log(1-x) function

    T: number of branch lengths in t_array
    
    returns:
    --------
    out_dict: the tkf values
        out_dict['log_alpha']: ArrayLike[float32], 
        out_dict['log_one_minus_alpha']: ArrayLike[float32], 
        out_dict['log_beta']: ArrayLike[float32], 
        out_dict['log_one_minus_beta']: ArrayLike[float32], 
        out_dict['log_gamma']: ArrayLike[float32], 
        out_dict['log_one_minus_gamma']: ArrayLike[float32], 
    
    approx_flags_dict: None (placeholder)
    
    """
    L_align = mu.shape[1]
    
    # mu: (B, L_align)
    # offset: (B, L_align)
    # t_array: either (B,) or (T,)
    if not unique_time_per_sample:
        T = t_array.shape[0]
        B = mu.shape[0]
        
        mu = mu[None,...] #(1, B, L_align)
        offset = offset[None,...] #(1, B, L_align)
        t_array = t_array[:,None,None] #(T, 1, 1)
        final_shape = (T, B, L_align)
    
    elif unique_time_per_sample:
        B = t_array.shape[0]
        t_array = t_array[:,None] #(B, 1)
        final_shape = (B, L_align)
        
    ### alpha
    # alpha = exp(-mu*t)
    # log(alpha) = -mu*t
    log_alpha = -mu*t_array
    
    
    ### beta
    # log( (1 - offset) * (exp(mu*offset*t) - 1) )
    # x = mu*offset*t
    # y = jnp.log( 1 - offset )
    # logsumexp with coeffs does: 
    #   log( exp(x) - exp(y) ) = log( exp(mu*offset*t) - (1-offset) )
    log_beta = true_beta( (mu, offset, t_array) )
    
    # 1 - beta; never use stable log one minus x for this
    log_one_minus_beta = log_one_minus_x(log_x = log_beta)
    
    
    ### vmap + jax.lax.cond solely for stable_log_one_minus_x function
    def tkf_params_indv(log_alpha_indv,
                        log_beta_indv,
                        offset_indv):
        # 1 - alpha
        log_one_minus_alpha = stable_log_one_minus_x(log_x = log_alpha_indv)
        
        # 1 - gamma
        log_one_minus_gamma = (log_beta_indv - 
                               ( jnp.log( 1-offset_indv) + log_one_minus_alpha )
                               )
        
        # gamma
        log_gamma = stable_log_one_minus_x(log_x = log_one_minus_gamma)
        
        return {'log_one_minus_alpha': log_one_minus_alpha,
                'log_gamma': log_gamma,
                'log_one_minus_gamma': log_one_minus_gamma}
    
    log_alpha_reshaped = log_alpha.flatten() #(T*B*L_align) or (B*L_align)
    log_beta_reshaped = log_beta.flatten() #(T*B*L_align) or (B*L_align)
    offset_reshaped = jnp.broadcast_to( offset, log_alpha.shape ) #(T, B, L_align) or (B, L_align)
    offset_reshaped = offset_reshaped.flatten() #(T*B*L_align) or (B*L_align)
    
    vmapped_tkf_params_indv = jax.vmap(tkf_params_indv)
    tkf_params_dict = vmapped_tkf_params_indv( log_alpha_reshaped,
                                               log_beta_reshaped, 
                                               offset_reshaped )
    
    # reshape all
    def my_reshape(m):
        return jnp.reshape(m, final_shape)
    
    tkf_params_dict['log_one_minus_alpha'] = my_reshape( tkf_params_dict['log_one_minus_alpha'] )
    tkf_params_dict['log_gamma'] = my_reshape( tkf_params_dict['log_gamma'] )
    tkf_params_dict['log_one_minus_gamma'] = my_reshape( tkf_params_dict['log_one_minus_gamma'] )
    tkf_params_dict['log_alpha'] = log_alpha
    tkf_params_dict['log_beta'] = log_beta
    tkf_params_dict['log_one_minus_beta'] = log_one_minus_beta
    
    return tkf_params_dict, None


def approx_tkf( mu, 
                offset, 
                t_array,
                unique_time_per_sample ):
    """
    return alpha, beta, gamma for TKF models; only use approx formulas, 
        except still allow use of switch between approx and real for 
        log(1-x) function

    T: number of branch lengths in t_array
    
    returns:
    --------
    out_dict: the tkf values
        out_dict['log_alpha']: ArrayLike[float32], 
        out_dict['log_one_minus_alpha']: ArrayLike[float32], 
        out_dict['log_beta']: ArrayLike[float32], 
        out_dict['log_one_minus_beta']: ArrayLike[float32], 
        out_dict['log_gamma']: ArrayLike[float32], 
        out_dict['log_one_minus_gamma']: ArrayLike[float32], 
    
    approx_flags_dict: None (placeholder)
    
    """
    L_align = mu.shape[1]
    
    # mu: (B, L_align)
    # offset: (B, L_align)
    # t_array: either (B,) or (T,)
    if not unique_time_per_sample:
        T = t_array.shape[0]
        B = mu.shape[0]
        
        mu = mu[None,...] #(1, B, L_align)
        offset = offset[None,...] #(1, B, L_align)
        t_array = t_array[:,None,None] #(T, 1, 1)
        final_shape = (T, B, L_align)
    
    elif unique_time_per_sample:
        B = t_array.shape[0]
        t_array = t_array[:,None] #(B, 1)
        final_shape = (B, L_align)
        
    ### alpha
    # alpha = exp(-mu*t)
    # log(alpha) = -mu*t
    log_alpha = -mu*t_array
    
    
    ### beta
    # log( (1 - offset) * (exp(mu*offset*t) - 1) )
    # x = mu*offset*t
    # y = jnp.log( 1 - offset )
    # logsumexp with coeffs does: 
    #   log( exp(x) - exp(y) ) = log( exp(mu*offset*t) - (1-offset) )
    log_beta = approx_beta( (mu, offset, t_array) )
    
    # 1 - beta; never use stable log one minus x for this
    log_one_minus_beta = log_one_minus_x(log_x = log_beta)
    
    
    ### 1-gamma
    log_one_minus_gamma = approx_one_minus_gamma( (mu, offset, t_array) )
    
    
    ### vmap + jax.lax.cond for stable_log_one_minus_x
    def tkf_params_indv(log_alpha_indv,
                        log_one_minus_gamma_indv):
        # 1 - alpha
        log_one_minus_alpha = stable_log_one_minus_x(log_x = log_alpha_indv)
        
        # gamma
        log_gamma = stable_log_one_minus_x(log_x = log_one_minus_gamma_indv)
        
        return {'log_one_minus_alpha': log_one_minus_alpha,
                'log_gamma': log_gamma}
    
    # vmap over B*L*T dim, instead of just T dim
    log_alpha_reshaped = log_alpha.flatten() #(T*B*L_align) or (B*L_align)
    log_one_minus_gamma_reshaped = log_one_minus_gamma.flatten() #(T*B*L_align) or (B*L_align)
    
    vmapped_tkf_params_indv = jax.vmap(tkf_params_indv)
    tkf_params_dict = vmapped_tkf_params_indv( log_alpha_reshaped,
                                      log_one_minus_gamma_reshaped )
    
    # reshape all
    def my_reshape(m):
        return jnp.reshape(m, final_shape)
    
    tkf_params_dict['log_one_minus_alpha'] = my_reshape( tkf_params_dict['log_one_minus_alpha'] )
    tkf_params_dict['log_gamma'] = my_reshape( tkf_params_dict['log_gamma'] )
    tkf_params_dict['log_alpha'] = log_alpha
    tkf_params_dict['log_beta'] = log_beta
    tkf_params_dict['log_one_minus_beta'] = log_one_minus_beta
    tkf_params_dict['log_one_minus_gamma'] = log_one_minus_gamma
    
    return tkf_params_dict, None


###############################################################################
### transition models   #######################################################
###############################################################################
def concat_transition_matrix(m_m, m_i, m_d, m_e,
                             i_m, i_i, i_d, i_e,
                             d_m, d_i, d_d, d_e,
                             s_m, s_i, s_d, s_e):
    """
    stacks along axis to (....,4,4)
    """
    return jnp.stack([ jnp.stack([m_m, m_i, m_d, m_e], axis=-1),
                       jnp.stack([i_m, i_i, i_d, i_e], axis=-1),
                       jnp.stack([d_m, d_i, d_d, d_e], axis=-1),
                       jnp.stack([s_m, s_i, s_d, s_e], axis=-1)
                      ], axis=-2)

def logprob_tkf91(tkf_params_dict,
                  *args,
                  **kwargs ):
    """
    T = times
    B = batch size
    L_align = length of alignment
    S = number of regular transitions, 4 here: M, I, D, START/END
    
    Arguments
    ----------
    tkf_params_dict : dict
        > (B, L_align) if unique_time_per_sample
        > (T, B, L_align) if not unique_time_per_sample
        contains values for calculating matrix terms: 
        alpha, beta, gamma, 1 - alpha, 1 - beta, 1 - gamma
        (all in log space)
      
    Returns
    -------
    out : ArrayLike
        > (B, L, S_from=4, S_to=4) if unique time per sample
        > (T, B, L, S_from=4, S_to=4) if not unique time per sample
        conditional loglike of transitions
    """
    # a_f = (1-beta)*alpha;     log(a_f) = log(1-beta) + log(alpha)
    # b_g = beta;               log(b_g) = log(beta)
    # c_h = (1-beta)*(1-alpha); log(c_h) = log(1-beta) + log(1-alpha)
    log_a_f = tkf_params_dict['log_one_minus_beta'] + tkf_params_dict['log_alpha']
    log_b_g = tkf_params_dict['log_beta']
    log_c_h = tkf_params_dict['log_one_minus_beta'] + tkf_params_dict['log_one_minus_alpha']
    log_mis_e = tkf_params_dict['log_one_minus_beta']

    # p = (1-gamma)*alpha;     log(p) = log(1-gamma) + log(alpha)
    # q = gamma;               log(q) = log(gamma)
    # r = (1-gamma)*(1-alpha); log(r) = log(1-gamma) + log(1-alpha)
    log_p = tkf_params_dict['log_one_minus_gamma'] + tkf_params_dict['log_alpha']
    log_q = tkf_params_dict['log_gamma']
    log_r = tkf_params_dict['log_one_minus_gamma'] + tkf_params_dict['log_one_minus_alpha']
    log_d_e = tkf_params_dict['log_one_minus_gamma']
    
    # logprob_trans is (T, B, L, 4, 4) or (B, L, 4, 4)
    logprob_trans = concat_transition_matrix(m_m = log_a_f, 
                                             m_i = log_b_g,
                                             m_d = log_c_h, 
                                             m_e = log_mis_e,
                                            
                                             i_m = log_a_f, 
                                             i_i = log_b_g, 
                                             i_d = log_c_h, 
                                             i_e = log_mis_e,
                                            
                                             d_m = log_p, 
                                             d_i = log_q, 
                                             d_d = log_r, 
                                             d_e = log_d_e,
                                            
                                             s_m = log_a_f, 
                                             s_i = log_b_g, 
                                             s_d = log_c_h, 
                                             s_e = log_mis_e)
    return logprob_trans


def logprob_tkf92(tkf_params_dict,
                  r_extend,
                  offset,
                  unique_time_per_sample ):
    """
    T = times
    B = batch size
    L_align = length of alignment
    S = number of regular transitions, 4 here: M, I, D, START/END
    
    Arguments
    ----------
    tkf_params_dict : dict
        > (B, L_align) if unique_time_per_sample
        > (T, B, L_align) if not unique_time_per_sample
        contains values for calculating matrix terms: 
        alpha, beta, gamma, 1 - alpha, 1 - beta, 1 - gamma
        (all in log space)
    
    r_extend : ArrayLike, (B, L_align)
        fragment extension probabilities
    
    unique_time_per_sample : Bool
        whether there's one time per sample, or a grid of times you'll 
        marginalize over
      
    Returns
    -------
    out : ArrayLike
        > (B, L, S_from=4, S_to=4) if unique time per sample
        > (T, B, L, S_from=4, S_to=4) if not unique time per sample
        conditional loglike of transitions
    """
    ### get dims  
    ref_shape = tkf_params_dict['log_alpha'] #(T,B,L) or (B,L)
    
    if not unique_time_per_sample:
        T = ref_shape.shape[0]
        B = max( [ ref_shape.shape[1], r_extend.shape[0] ] )
        L_align = max( [ ref_shape.shape[2], r_extend.shape[1] ] )  
        final_shape = (T, B, L_align)
        
        r_extend = r_extend[None,...] #(1, B, L_align)
    
    elif unique_time_per_sample:        
        B = max( [ ref_shape.shape[0], r_extend.shape[0] ] )
        L_align = max( [ ref_shape.shape[1], r_extend.shape[1] ] )  
        final_shape = (B, L_align)
    
    # reshape tensors with broadcasting, where needed
    def my_reshape(m):
        return jnp.broadcast_to(m, final_shape)
    
    tkf_params_dict['log_alpha'] = my_reshape( tkf_params_dict['log_alpha'] ) #(T, B, L_align) or  #(B, L_align)
    tkf_params_dict['log_beta'] = my_reshape( tkf_params_dict['log_beta'] ) #(T, B, L_align) or  #(B, L_align)
    tkf_params_dict['log_gamma'] = my_reshape( tkf_params_dict['log_gamma'] ) #(T, B, L_align) or  #(B, L_align)
    tkf_params_dict['log_one_minus_alpha'] = my_reshape( tkf_params_dict['log_one_minus_alpha'] ) #(T, B, L_align) or  #(B, L_align)
    tkf_params_dict['log_one_minus_beta'] = my_reshape( tkf_params_dict['log_one_minus_beta'] ) #(T, B, L_align) or  #(B, L_align)
    tkf_params_dict['log_one_minus_gamma'] = my_reshape( tkf_params_dict['log_one_minus_gamma'] ) #(T, B, L_align) or  #(B, L_align)
    r_extend = my_reshape( r_extend )
    
    
    ##############################
    ### start filling in matrix  #
    ##############################
    # log-transform variables
    log_r_extend = safe_log(r_extend)
    log_one_minus_r_extend = jnp.log1p(-r_extend)
    log_lam_div_mu = jnp.log1p(-offset)
    log_one_minus_lam_div_mu = safe_log(offset)
    log_one_div_nu = -safe_log( r_extend + (1-r_extend)*(1-offset) )
    
    
    ### match -> (match, ins, del, end)
    # a = (1/nu) (r_extend + (lam/mu)*(1-r_extend)*(1-beta)*alpha)
    # log(a) = log(1/nu) + logsumexp([r_extend, 
    #                                 log(lam/mu) + log(1-r_extend) + log(1-beta) + log(alpha)
    #                                 ])
    log_a_second_half = ( log_lam_div_mu +
                          log_one_minus_r_extend + 
                          tkf_params_dict['log_one_minus_beta'] +
                          tkf_params_dict['log_alpha'] )
    log_a = log_one_div_nu + logsumexp_with_arr_lst([log_r_extend, log_a_second_half])
    
    # b = (1-r_extend)*beta
    # log(b) = log(1-r_extend) + log(beta)
    log_b = log_one_minus_r_extend + tkf_params_dict['log_beta']
    
    # c_h = (1/nu) ( (lam/mu)*(1-r_extend)*(1-beta)*(1-alpha) )
    # log(c_h) = log(1/nu) + log(lam/mu) + log(1-r_extend) + log(1-beta) + log(1-alpha)
    log_c_h = ( log_one_div_nu +
                log_lam_div_mu +
                log_one_minus_r_extend +
                tkf_params_dict['log_one_minus_beta'] +
                tkf_params_dict['log_one_minus_alpha'] )
    
    # m_e = (1-beta)
    # log(mi_e) = log(1-beta)
    log_mis_e = tkf_params_dict['log_one_minus_beta']
    
    
    ### ins -> (match, ins, del, end)
    # f = (1/nu)*(lam/mu)*(1-r_extend)*(1-beta)*alpha
    # log(f) = log(1/nu) + log(lam/mu) + log(1-r_extend) +log(1-beta) +log(alpha)
    log_f = ( log_one_div_nu +
              log_lam_div_mu +
              log_one_minus_r_extend +
              tkf_params_dict['log_one_minus_beta'] +
              tkf_params_dict['log_alpha'] )
    
    # g = r_extend + (1-r_extend)*beta
    # log(g) = logsumexp([r_extend, 
    #                     log(1-r_extend) + log(beta)
    #                     ]
    #                    )
    log_g_second_half = log_one_minus_r_extend + tkf_params_dict['log_beta']
    log_g = logsumexp_with_arr_lst([log_r_extend, log_g_second_half])
    
    # h and log(h) are the same as c and log(c) 
    # ins->end is same as match->end


    ### del -> (match, ins, del, end)
    # p = (1/nu)*(lam/mu)*(1-r_extend)*(1-gamma)*alpha
    # log(p) = log(1/nu) + log(lam/mu) + log(1-r_extend) + log(1-gamma) +log(alpha)
    log_p = ( log_one_div_nu +
              log_lam_div_mu +
              log_one_minus_r_extend +
              tkf_params_dict['log_one_minus_gamma'] +
              tkf_params_dict['log_alpha'] )

    # q = (1-r_extend)*gamma
    # log(q) = log(1-r_extend) + log(gamma)
    log_q = log_one_minus_r_extend + tkf_params_dict['log_gamma']

    # r = (1/nu) * ( r_extend + (lam/mu)*(1-r_extend)*(1-gamma)*(1-alpha) )
    # log(r) = log(1/nu) + logsumexp([r_extend, 
    #                                 log(lam/mu) + log(1-r_extend) + log(1-gamma) + log(1-alpha)
    #                                 ])
    log_r_second_half = ( log_lam_div_mu +
                          log_one_minus_r_extend +
                          tkf_params_dict['log_one_minus_gamma'] +
                          tkf_params_dict['log_one_minus_alpha'] )
    log_r = log_one_div_nu + logsumexp_with_arr_lst([log_r_extend, log_r_second_half])
    
    # d_e = (1-gamma)
    # log(d_e) = log(1-gamma)
    log_d_e = tkf_params_dict['log_one_minus_gamma']
    
    
    ### final row: start -> any
    log_s_m = tkf_params_dict['log_one_minus_beta'] + tkf_params_dict['log_alpha']
    log_s_i = tkf_params_dict['log_beta']
    log_s_d = tkf_params_dict['log_one_minus_beta'] + tkf_params_dict['log_one_minus_alpha']
    # start->end is same as match->end
    
    # final mat is (T, B, L, 4, 4) or (B, L, 4, 4)
    logprob_trans = concat_transition_matrix(m_m = log_a, 
                                             m_i = log_b,
                                             m_d = log_c_h, 
                                             m_e = log_mis_e,
                                                  
                                             i_m = log_f, 
                                             i_i = log_g, 
                                             i_d = log_c_h, 
                                             i_e = log_mis_e,
                                                  
                                             d_m = log_p, 
                                             d_i = log_q, 
                                             d_d = log_r, 
                                             d_e = log_d_e,
                                                  
                                             s_m = log_s_m, 
                                             s_i = log_s_i, 
                                             s_d = log_s_d, 
                                             s_e = log_mis_e)
    return logprob_trans



###############################################################################
### neural helpers   ##########################################################
###############################################################################
def process_datamat_lst(datamat_lst: list,
                        padding_mask: jnp.array,
                        use_anc_emb: bool,
                        use_desc_emb: bool,
                        use_prev_align_info: bool):
    """
    select which embedding, then mask out padding tokens
    
    B: batch size
    L_align: length of alignment
    
    
    Arguments
    ----------
    datamat_lst : list[ArrayLike, ArrayLike, ArrayLike]
        > first array: ancestor embedding (B, L_align, H)
        > second array: descendant embedding (B, L_align, H)
        > third array: previous position alignment info (B, L_align, 1)
    
    padding_mask : ArrayLike, (B, L_align)
    
    use_anc_emb : bool
        > use ancestor embedding information to generate evolutionary 
          model parameters?
        
    use_desc_emb : bool
        > use descendant embedding information to generate evolutionary 
          model parameters?
    
    use_prev_align_info : bool
        > use previous position alignment label?
    
    Returns
    --------
    datamat : ArrayLike, (B, L_align, n*H + d*6)
        concatenated and padding-masked features
        > n=1, if only using ancestor embedding OR descendant embedding
        > n=2, if using both embeddings
        > d=1 if use_prev_align_info, otherwise 0
        
    masking_mat: ArrayLike, (B, L_align, n*H + d*6)
        location of padding in alignment
        > n=1, if only using ancestor embedding OR descendant embedding
        > n=2, if using both embeddings
        > d=1 if use_prev_align_info, otherwise 0
    """
    to_concat = []
    
    if use_anc_emb:
        to_concat.append( datamat_lst[0] )
    
    if use_desc_emb:
        to_concat.append( datamat_lst[1] )
    
    if use_prev_align_info:
        to_concat.append( datamat_lst[2] )
    
    # datamat could be:
    #   (B, L_align, H): (use_anc_emb | use_anc_emb) & ~use_prev_align_info
    #   (B, L_align, H+6): (use_anc_emb | use_anc_emb) & use_prev_align_info 
    #   (B, L_align, 2*H): use_anc_emb & use_anc_emb & ~use_prev_align_info
    #   (B, L_align, 2*H+6): use_anc_emb & use_anc_emb & use_prev_align_info
    datamat = jnp.concatenate( to_concat, axis = -1 ) 
    
    # masking_mat could be:
    #   (B, L_align, H): (use_anc_emb | use_anc_emb) & ~use_prev_align_info
    #   (B, L_align, H+6): (use_anc_emb | use_anc_emb) & use_prev_align_info 
    #   (B, L_align, 2*H): use_anc_emb & use_anc_emb & ~use_prev_align_info
    #   (B, L_align, 2*H+6): use_anc_emb & use_anc_emb & use_prev_align_info
    new_shape = (padding_mask.shape[0],
                 padding_mask.shape[1],
                 datamat.shape[2]) 
    
    masking_mat = jnp.broadcast_to(padding_mask[...,None], new_shape)
    del new_shape
    
    # datamat could be:
    #   (B, L_align, H): (use_anc_emb | use_anc_emb) & ~use_prev_align_info
    #   (B, L_align, H+6): (use_anc_emb | use_anc_emb) & use_prev_align_info 
    #   (B, L_align, 2*H): use_anc_emb & use_anc_emb & ~use_prev_align_info
    #   (B, L_align, 2*H+6): use_anc_emb & use_anc_emb & use_prev_align_info
    datamat = jnp.multiply(datamat, masking_mat)
    return datamat
