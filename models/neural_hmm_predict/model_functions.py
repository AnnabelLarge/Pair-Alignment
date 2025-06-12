#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 18:54:38 2025

@author: annabel
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
    
    t_array : ArrayLike, (T,) or (B,)
    
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
    if unique_time_per_sample:
        t_array = t_array[..., None] #(B, 1)
    
    elif not unique_time_per_sample:
        normalizing_factor = normalizing_factor[None, ...] #(1, B, L_align) 
        rate_multiplier = rate_multiplier[None, ...] #(1, B, L_align) 
        t_array = t_array[..., None, None] #(T, 1, 1)
        equl = equl[None,...] #(1, B, L_align, A) 
    
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
    return jnp.log( jnp.stack( [match_prob, subs_prob], axis = -1 ) )

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
    mat = jnp.zeros((B, L, A, A), dtype=vec.dtype)

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
    B = max( [exch_upper_triag_values.shape[0],
              equilibrium_distributions.shape[0] )
    L_align = max( [exch_upper_triag_values.shape[1],
                    equilibrium_distributions.shape[1] )
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
    B = max( [exch_upper_triag_values.shape[0],
              equilibrium_distributions.shape[0],
              rate_multiplier.shape[0]] )
    L_align = max( [exch_upper_triag_values.shape[1],
                    equilibrium_distributions.shape[1],
                    rate_multiplier.shape[1]] )
    A = equilibrium_distributions.shape[-1]
    
    # place these in a square matrix
    exchangeabilities = upper_tri_vector_to_sym_matrix(vec = exch_upper_triag_values) # (B, L_align, A, A) 
    
    # generate rate matrix, and normalize it 
    equilibrium_distributions = jnp.exp(log_equl) #(B, L_align, A) 
    normed_rate_mat = rate_matrix_from_exch_equl(exchangeabilities = exchangeabilities,
                                                 equilibrium_distributions = equilibrium_distributions,
                                                 norm = True) #(B, L_align, A, A) 
    
    # scale by rate multiplier
    rate_mat = rate_multiplier[..., None, None] * normed_rate_mat #(B, L_align, A, A) 
    
    # adjust dims
    if unique_time_per_sample:
        T = t_array.shape[0]
        before_reshape = (T*B*L_align, A, A)
        after_reshape = (T, B, L_align, A, A)
        t_array = jnp.expand_dims(t_array, (1,2,3,4)) #(T, 1, 1, 1, 1)
        rate_mat = rate_mat[None, ...] #(1, B, L_align, A, A) 
    
    elif not unique_time_per_sample:
        before_reshape = (B*L_align, A, A)
        after_reshape = (B, L_align, A, A)
        t_array = jnp.expand_dims(t_array, (1,2,3)) #(B, 1, 1, 1)
    
    oper = rate_mat * t_array # (T, B, L_align, A, A) or (B, L_align, A, A)
    
    # apply matrix exponential with vmap
    reshaped_oper = oper.reshape( oper, before_reshape ) #(T*B*L, A, A) or (B*L, A, A)
    vmapped_expm = jax.vmap(expm, axis=0)
    cond_prob_raw = vmapped_expm( reshaped_oper ) #(T*B*L, A, A) or (B*L, A, A)
    cond_prob = jnp.reshape( cond_prob_raw, after_reshape )
    
    return jnp.log(cond_prob) # (T, B, L_align, A, A) or (B, L_align, A, A)
