#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 17:10:38 2025

@author: annabel

About:
======
standalone functions for pairHMM models; are NOT flax modules and do NOT 
  have parameters; also unable to record to tensorboard with these


functions:
---------
'all_loglikes_forward',
'anc_marginal_probs_from_counts',
'approx_beta',
'approx_one_minus_gamma',
'approx_tkf',
'bound_sigmoid',
'bound_sigmoid_inverse',
'cond_logprob_emit_at_match_per_mixture',
'cond_prob_from_counts',
'desc_marginal_probs_from_counts',
'fill_f81_logprob_matrix',
'get_cond_transition_logprobs',
'get_joint_loglike_emission_branch_len_per_samp',
'get_joint_loglike_emission_time_grid',
'get_tkf91_single_seq_marginal_transition_logprobs',
'get_tkf92_single_seq_marginal_transition_logprobs',
'joint_logprob_emit_at_match_per_mixture',
'joint_only_forward',
'joint_prob_from_counts',
'log_one_minus_x',
'log_x_minus_one',
'logsumexp_with_arr_lst',
'lse_over_equl_logprobs_per_mixture',
'lse_over_match_logprobs_per_mixture',
'marginalize_over_times',
'maybe_write_matrix_to_ascii',
'rate_matrix_from_exch_equl',
'regular_tkf',
'safe_log',
'scale_rate_matrix',
'scale_rate_multipliers',
'stable_log_one_minus_x',
'switch_tkf',
'true_beta',
'upper_tri_vector_to_sym_matrix',
'write_matrix_to_npy'


internal:
---------
'_selectively_add_time_dim',
"""
import jax
from jax import numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.linalg import expm
from jax._src.typing import Array, ArrayLike

from functools import partial
import numpy as np

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
### functions used in calculating substitution rate matrix   ##################
###############################################################################
def upper_tri_vector_to_sym_matrix( vec: ArrayLike ):
    """
    Given upper triangular values, fill in a symmetric matrix


    Arguments
    ----------
    vec : ArrayLike, (n,)
        upper triangular values
    
    Returns
    -------
    mat : ArrayLike, (A, A)
        final matrix; A = ( n * (n-1) ) / 2
    
    Example
    -------
    vec = [a, b, c, d, e, f]
    
    upper_tri_vector_to_sym_matrix(vec) = [[0, a, b, c],
                                            [a, 0, d, e],
                                            [b, d, 0, f],
                                            [c, e, f, 0]]

    """
    ### automatically detect emission alphabet size
    # 6 = DNA (alphabet size = 4)
    # 190 = proteins (alphabet size = 20)
    # 2016 = codons (alphabet size = 64)
    if vec.shape[-1] == 6:
        emission_alphabet_size = 4
    
    elif vec.shape[-1] == 190:
        emission_alphabet_size = 20
    
    elif vec.shape[-1] == 2016:
        emission_alphabet_size = 64
    
    else:
        raise ValueError(f'input dimensions are: {vec.shape}')
    
    
    ### fill upper triangular part of matrix
    out_size = (emission_alphabet_size, emission_alphabet_size)
    upper_tri_exchang = jnp.zeros( out_size )
    idxes = jnp.triu_indices(emission_alphabet_size, k=1)  
    upper_tri_exchang = upper_tri_exchang.at[idxes].set(vec) # (A, A)
    
    
    ### reflect across diagonal
    mat = (upper_tri_exchang + upper_tri_exchang.T) # (A, A)
    
    return mat


def rate_matrix_from_exch_equl(exchangeabilities: ArrayLike,
                                equilibrium_distributions: ArrayLike,
                                norm: bool):
    """
    computes rate matrix Q = \chi * \pi_c; normalizes to substution 
      rate of one if desired
    
    only one exchangeability; rho and pi are properties of the class
    
    C_trans: number of mixtures associated with transitions (variable) 
    C_sites: number of latent site classes 
    A: alphabet size 
    
    
    Arguments
    ----------
    exchangeabilities : ArrayLike, (A, A)
        symmetric exchangeability parameter matrix
        
    equilibrium_distributions : ArrayLike, (C_trans, C_sites, A)
        amino acid equilibriums per site
    
    norm : bool

    Returns
    -------
    subst_rate_mat : ArrayLike, (C_trans, C_sites, A, A)
        rate matrix Q, for every latent site class

    """
    C_tr = equilibrium_distributions.shape[0] # f
    C_s = equilibrium_distributions.shape[1] # g
    A = equilibrium_distributions.shape[2] # i,j 

    # just in case, zero out the diagonals of exchangeabilities
    exchangeabilities_without_diags = exchangeabilities * ~jnp.eye(A, dtype=bool)

    # Q = chi * diag(pi); q_ij = chi_ij * pi_j
    rate_mat_without_diags = jnp.multiply( exchangeabilities_without_diags[None, None, :, :],
                                           equilibrium_distributions[:, :, None, :] ) # (C_tr, C_s, A, A)
    
    # put the row sums in the diagonals
    row_sums = rate_mat_without_diags.sum(axis=-1)  # (C_tr, C_s, A)
    ones_diag = jnp.eye( A, dtype=bool )[None,None,...]   # (1, 1, A, A)
    ones_diag = jnp.broadcast_to( ones_diag, (C_tr,
                                              C_s,
                                              ones_diag.shape[-2],
                                              ones_diag.shape[-1]) ) # (C_tr, C_s, A, A)
    diags_to_add = -jnp.multiply( row_sums[...,None], ones_diag ) # (C_tr, C_s, A, A)  
    subst_rate_mat = rate_mat_without_diags + diags_to_add  # (C_tr, C_s, A, A)
    
    # normalize (true by default)
    if norm:
        diag = jnp.diagonal(subst_rate_mat, axis1=-2, axis2=-1) # (C_tr, C_s, A )
        norm_factor = -jnp.multiply( diag, equilibrium_distributions ).sum(-1) #(C_tr, C_s)
        subst_rate_mat = subst_rate_mat / ( norm_factor[...,None,None] )  # (C_tr, C_s, A, A)
    
    return subst_rate_mat # (C_tr, C_s, A, A)

def scale_rate_matrix(subst_rate_mat: ArrayLike,
                      rate_multipliers: ArrayLike):
    """
    Scale Q by rate multipliers, rho
    
    C_trans: number of mixtures associated with transitions (variable) 
    C_sites: number of latent site classes 
    K = number of rate multipliers 
    A = alphabet size 
    
    
    Arguments
    ----------
    subst_rate_mat : ArrayLike, (C_trans, C_sites, A, A)
    
    rate_multipliers : ArrayLike, (C_trans, C_sites, K)

    Returns
    -------
    scaled rate matrix : ArrayLike, (C_trans, C_sites, K, A, A)

    """
    subst_rate_mat = subst_rate_mat[:,:,None,...] #(C_tr, C_s, 1, A, A)
    rate_multipliers = rate_multipliers[...,None,None] #(C_tr, C_s, K, 1, 1)
    return jnp.multiply( subst_rate_mat, rate_multipliers )#(C_tr, C_s, K, A, A)


###############################################################################
### functions used to calculate scoring matrix for substitution sites   #######
###############################################################################
def cond_logprob_emit_at_match_per_mixture( t_array: ArrayLike,
                                            scaled_rate_mat_per_mixture: ArrayLike ):
    """
    P(y|x,t,c_tr,c_s,k) = expm( \rho_{c_tr, c_s, k} * Q_{c_tr, c_s} * t )

    C_trans: number of mixtures associated with transitions (variable)
    C_sites: number of latent site classes 
    K = number of rate multipliers 
    A = alphabet size 
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
    

    Arguments
    ----------
    t_array : ArrayLike, (T,) or (B,)
        branch lengths
        
    scaled_rate_mat_per_mixture : ArrayLike, (C_trans, C_sites, K, A, A)
        \rho_{c,k} * Q_c

    Returns
    -------
    cond_logprob_emit_at_match_per_mixture :  ArrayLike, (T, C_trans, C_sites, K, A, A)
        final conditional log-probability; NOT YET SCALED BY 
        CLASS/RATE PROBABILITIES!!!

    """
    operand = jnp.multiply( scaled_rate_mat_per_mixture[None,...],
                            t_array[:, None, None, None, None, None] ) #(T, C_tr, C_s, K, A, A)
    
    cond_prob_emit_at_match_per_mixture = expm(operand) #(T, C_tr, C_s, K, A, A)
    cond_logprob_emit_at_match_per_mixture = safe_log( cond_prob_emit_at_match_per_mixture ) #(T, C_tr, C_s, K, A, A)
    return cond_logprob_emit_at_match_per_mixture


def joint_logprob_emit_at_match_per_mixture( cond_logprob_emit_at_match_per_mixture: ArrayLike,
                                             log_equl_dist_per_mixture: ArrayLike ):
    """
    P(x,y|t,c,k) = P(x|c) * P(y|x,t,c,k)

    C_trans: number of mixtures associated with transitions (variable)
    C_sites: number of latent site classes 
    K = number of rate multipliers 
    A = alphabet size 
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
    

    Arguments
    ----------
    cond_logprob_emit_at_match_per_mixture : ArrayLike, (T, C_tr, C_s, K, A, A)
        P(y|x,c,t), calculated before
    
    log_equl_dist_per_mixture : ArrayLike, (C_tr, C_s, A)
        equlibrium distribution

    Returns
    -------
    ArrayLike, (T, C_tr, C_s, K, A, A)
        joint logprob

    """
    log_equl_dist_per_mixture = log_equl_dist_per_mixture[None, :, :, None, :, None] #(1, C_tr, C_s, 1, A, 1)
    return ( cond_logprob_emit_at_match_per_mixture + log_equl_dist_per_mixture ) #(T, C_tr, C_s, K, A, A)


###############################################################################
### F81 solution   ############################################################
###############################################################################
def fill_f81_logprob_matrix(equl: jnp.array,
                            rate_multipliers: jnp.array, 
                            t_array: jnp.array, 
                            norm_rate_matrix: bool = True,
                            return_cond: bool = False):
    """
    C_trans: number of mixtures associated with transitions (variable)
    C_sites: number of latent site classes
    K = number of site rates
    A = alphabet size
    
    Use solution for F81 (no need for matrix exponential)
    
    
    Arguments
    ----------
    equl : ArrayLike, (C_trans, C_sites, A)
        log-transformed equilibrium distribution
    
    rate_multipliers : ArrayLike, (C_trans, C_sites, K)
        rate multiplier k for site classes c_tr and c_s; \rho_{c_tr, c_s, k}
    
    t_array : ArrayLike, (T,) or (B,)
        either one time grid for all samples (T,) or unique branch
        length for each sample (B,)
    
    norm_rate_matrix : bool; default is true
        whether or not to normalize the rate matrix to 1 substitution per t
    
    return_cond : bool; default is false
        whether or not to return conditional logprob; only really used
        when unit testing parts, not in full training script
    
    """
    T = t_array.shape[0]
    C_tr = equl.shape[0] 
    C_s = equl.shape[1] 
    A = equl.shape[2]
    K = rate_multipliers.shape[2]
    
    # possibly normalize to one substitution per time t
    if norm_rate_matrix:
        # \sum_i pi_i chi_{ii} = \sum_i pi_i (1-\pi_i) = 1 - \sum_i pi_i^2
        norm_factor = 1 / ( 1 - jnp.square(equl).sum(axis=(-1)) ) # (C_tr, C_s)
        
    elif not norm_rate_matrix:
        norm_factor = jnp.ones( (C_tr, C_s) ) #(C_tr, C_s)
    
    # the exponential operand
    oper = -( rate_multipliers[None,...] * 
              norm_factor[None,..., None] * 
              t_array[:, None, None, None] ) #(T, C_tr, C_s, K)
    exp_oper = jnp.exp(oper) #(T, C_tr, C_s, K)

    # expand permanently, for further use
    exp_oper = exp_oper[...,None] #(T, C_tr, C_s, K, 1)
    equl = equl[None, :, :, None, :] #(1, C_tr, C_s, 1, A)
    
    # all off-diagonal entries, i != j
    # pi_j * ( 1 - exp(-rate*t) )
    row = jnp.multiply( equl, ( 1 - exp_oper ) ) #(T, C_tr, C_s, K, A)
    cond_probs_raw = jnp.broadcast_to( row[..., None, :], (T, C_tr, C_s, K, A, A) )  #(T, C_tr, C_s, K, A, A)
    
    # diagonal entries, i = j
    #   pi_j + (1-pi_j) * exp(-rate*t)
    diags = equl + jnp.multiply( (1-equl), exp_oper ) #(T, C_tr, C_s, K, A)
    diag_indices = jnp.arange(A)  # (A,)
    cond_probs = cond_probs_raw.at[..., diag_indices, diag_indices].set(diags) #(T, C_tr, C_s, K, A, A)
    
    if return_cond:
        return safe_log( cond_probs ) #(T, C_tr, C_s, K, A, A)
    
    elif not return_cond:
        # return to original dimension, and get logprob_equl (as before)
        equl = equl[0, :, :, 0, :] #(C_tr, C_s, A)
        logprob_equl = safe_log(logprob_equl) #(C_tr, C_s, A)
        
        # P(x) P(y|x,t) for all T, C_tr, C_s, K
        joint_logprobs = joint_logprob_emit_at_match_per_mixture( cond_logprob_emit_at_match_per_mixture = cond_logprobs,
                                                                  log_equl_dist_per_mixture = logprob_equl )
        
        return joint_logprobs #(T, C_tr, C_s, K, A, A)
    
    
#######################################################################
### for emissions, functions to marginalize over possible classes   ###
#######################################################################
def lse_over_match_logprobs_per_mixture(log_site_class_probs: jnp.array,
                                        log_rate_mult_probs: jnp.array, 
                                        logprob_emit_at_match_per_mixture: jnp.array):
    """
    Sum over mixtures of rate multipliers and emission site classes 
        (leave transition site classes untouched)
    
    for joint probability:
        P(x,y|t,c_trans) = 
        \sum_{c_st} \sum_k 
        P(c_sites|c_trans) * P(k|c_trans,c_sites) * P(x,y|t, c_trans, c_sites, k)
    
    for cond probability:
        P(y|x,t,c_trans) = 
        \sum_{c_st} \sum_k 
        P(c_sites|c_trans) * P(k|c_trans,c_sites) * P(y|x, t, c_trans, c_sites, k)
    
    C_trans: number of mixtures associated with transitions (variable)
    C_sites: number of latent site classes
    K = number of rate multipliers
    A = alphabet size
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
    

    Arguments
    ----------
    log_site_class_probs : ArrayLike, (C_trans, C_sites)
        the log-probability of latent class assignment for the emission and 
        transition latent site classes
    
    log_rate_mult_probs : ArrayLike (C_trans, C_sites, K)
        the log-probability of having rate class k, given that the column 
        is assigned to latent site class c_st, in transit class c_trans
    
    logprob_emit_at_match_per_mixture : ArrayLike, (T, C_trans, C_sites, K, A, A)
        log-probability of emissions at match sites (either cond or joint)
        
    Returns
    -------
    ArrayLike, (T, C_trans, A, A)

    """
    # P(C_sites, K | C_trans) = P(C_sites | C_trans) * P(K | C_trans, C_sites)
    log_mixture_weight = log_site_class_probs[..., None] + log_rate_mult_probs #(C_tr, C_s, K)
    
    # apply per-class and per-mixture weighting
    weighted_logprobs = ( log_mixture_weight[None, :, :, :, None, None] + 
                          logprob_emit_at_match_per_mixture )  #(T, C_tr, C_s, K, A, A)
    
    # logsumexp over C_site and K dimensions
    lse_over_site_classes = logsumexp( weighted_logprobs, axis=2 ) #(T, C_tr, K, A, A)
    lse_over_rate_mults = logsumexp( lse_over_site_classes, axis=2 ) #(T, C_tr, A, A)
    
    return lse_over_rate_mults #(T, C_tr, A, A)


def lse_over_equl_logprobs_per_mixture(log_site_class_probs: ArrayLike,
                                        log_equl_dist_per_mixture: ArrayLike):
    """
    P(x | c_trans) = \sum_{c_sites} P(c_sites | c_trans) * P(x|c_trans,c_sites)
    
    C_trans: number of mixtures associated with transitions (variable)
    C_sites: number of latent site classes
    A = alphabet size
    
    
    Arguments
    ----------
    log_site_class_probs : ArrayLike, (C_trans, C_sites)
        log-transformed class probabilities (i.e. mixture weights for emissions)
    
    log_equl_dist_per_mixture : ArrayLike, (C_trans, C_sites, A)
        log-transformed equilibrium distributions for every latent class label
        
    Returns
    -------
    ArrayLike, (C_trans, A)
    
    """
    weighted_logprobs = log_equl_dist_per_mixture + log_site_class_probs[..., None] #(C_tr, C_s, A)
    return logsumexp( weighted_logprobs, axis=1 ) #(C_trans, A)



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
    
    # x = mu*offset*t
    # y = jnp.log( 1 - offset )
    # logsumexp with coeffs does: 
    #   log( exp(x) - exp(y) ) = log( exp(mu*offset*t) - (1-offset) )
    log_one_minus_offset = jnp.broadcast_to( jnp.log1p(-offset), t.shape )
    log_denom = logsumexp_with_arr_lst( [mu*offset*t, log_one_minus_offset],
                                    coeffs = jnp.array([1.0, -1.0]) )
    
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

def switch_tkf( mu, offset, t_array ):
    """
    return alpha, beta, gamma for TKF models

    use real formulas where you can, and taylor-approximations where you can't
    
    T: number of branch lengths in t_array
    
    returns:
    --------
    out_dict: the tkf values
        out_dict['log_alpha']: ArrayLike[float32], (T,)
        out_dict['log_one_minus_alpha']: ArrayLike[float32], (T,)
        out_dict['log_beta']: ArrayLike[float32], (T,)
        out_dict['log_one_minus_beta']: ArrayLike[float32], (T,)
        out_dict['log_gamma']: ArrayLike[float32], (T,)
        out_dict['log_one_minus_gamma']: ArrayLike[float32], (T,)
    
    approx_flags_dict: where you used approx formulas
        out_dict['log_one_minus_alpha']: ArrayLike[bool], (T,)
        out_dict['log_beta']: ArrayLike[bool], (T,)
        out_dict['log_one_minus_gamma']: ArrayLike[bool], (T,)
        out_dict['log_gamma']: ArrayLike[bool], (T,)
    
    """
    ######################################################
    ### Some operations can be done with entire arrays   #
    ######################################################
    ### alpha = exp(-mu*t)
    ### log(alpha) = -mu*t
    log_alpha = -mu*t_array
    
    
    ### start of calculation for 1 - gamma
    # numerator:
    # log( exp(mu*offset*t) - 1 )
    gamm_full_log_num = log_x_minus_one( log_x = mu*offset*t_array )
    
    # denominator, term 1
    # x = mu*offset*t
    # y = jnp.log( 1 - offset )
    # logsumexp with coeffs does: 
    #   log( exp(x) - exp(y) ) = log( exp(mu*offset*t) - (1-offset) )
    constant = jnp.broadcast_to(jnp.log1p(-offset), t_array.shape)
    gamma_full_log_denom_term1 = logsumexp_with_arr_lst( [mu*offset*t_array, constant],
                                              coeffs = jnp.array([1.0, -1.0]) )
    
    
    ###############################################################
    ### Most have to be done one-at-a-time, due to jax.lax.cond   #
    ###############################################################
    def tkf_params_per_timepoint(log_alpha_at_t, 
                                 gamma_log_numerator_at_t,
                                 gamma_log_denom_term1_at_t,
                                 t):
        ### 1 - alpha
        log_one_minus_alpha = stable_log_one_minus_x(log_x = log_alpha_at_t)
        
        
        ### beta, 1 - beta
        # beta
        log_beta = jax.lax.cond( mu*offset*t > SMALL_POSITIVE_NUM ,
                                  true_beta,
                                  approx_beta,
                                  (mu, offset, t) )  
        
        # regardless of approx or not, 1-beta calculated from beta
        log_one_minus_beta = log_one_minus_x(log_x = log_beta)
        
        
        ### 1 - gamma, gamma
        # need log(1 - alpha) to finish calculating denominator for log(1 - gamma)
        gamma_log_denom = gamma_log_denom_term1_at_t + log_one_minus_alpha
        
        # ad hoc series of conditionals to determine if you approx
        #   1-gamma or not (meh)
        valid_frac = gamma_log_numerator_at_t < gamma_log_denom
        large_product = mu * offset * t > 1e-3
        log_diff_large = jnp.abs(gamma_log_numerator_at_t - gamma_log_denom) > 0.1
        approx_formula_will_fail = (0.5*mu*t) > 1.0
        
        cond1 = large_product
        cond2 = ~large_product & log_diff_large
        cond3 = ~large_product & ~log_diff_large & approx_formula_will_fail
        use_real_function = valid_frac & ( cond1 | cond2 | cond3 )
        
        # the final value
        log_one_minus_gamma = jax.lax.cond( use_real_function,
                                            lambda _: gamma_log_numerator_at_t - gamma_log_denom,
                                            approx_one_minus_gamma,
                                            (mu, offset, t) )
        
        # gamma
        log_gamma = stable_log_one_minus_x(log_x = log_one_minus_gamma)
        
        
        ### output everything
        out_dict = {'log_one_minus_alpha': log_one_minus_alpha,
                    'log_beta': log_beta,
                    'log_one_minus_beta': log_one_minus_beta,
                    'log_gamma': log_gamma,
                    'log_one_minus_gamma': log_one_minus_gamma}
        
        used_one_minus_alpha_approx = ~(log_alpha_at_t < -SMALL_POSITIVE_NUM)
        used_beta_approx = ~(mu*offset*t > SMALL_POSITIVE_NUM)
        used_log_one_minus_gamma_approx = ~use_real_function
        used_log_gamma_approx = ~(log_one_minus_gamma < -SMALL_POSITIVE_NUM)
        
        # if you used an approx anywhere, save the time
        flag = used_one_minus_alpha_approx | used_beta_approx | used_log_one_minus_gamma_approx | used_log_gamma_approx
        t_to_add = jnp.where( flag,
                              t,
                              -1. )
        
        approx_flags_dict = {'log_one_minus_alpha': used_one_minus_alpha_approx,
                              'log_beta': used_beta_approx,
                              'log_one_minus_gamma': used_log_one_minus_gamma_approx,
                              'log_gamma': used_log_gamma_approx,
                              'valid_frac_log_one_minus_gamma': ~valid_frac,
                              'cond1_log_one_minus_gamma': ~cond1,
                              'cond2_log_one_minus_gamma': ~cond2,
                              'cond3_log_one_minus_gamma': ~cond3,
                              't_array': t_to_add}
        
        return out_dict, approx_flags_dict
    
    vmapped_tkf_params_per_timepoint = jax.vmap(tkf_params_per_timepoint,
                                                in_axes=(0,0,0,0))
    out = vmapped_tkf_params_per_timepoint(log_alpha,
                                           gamm_full_log_num,
                                           gamma_full_log_denom_term1,
                                           t_array)
    out_dict, approx_flags_dict = out
    del out
    
    out_dict['log_alpha'] = log_alpha
    return out_dict, approx_flags_dict


def regular_tkf( mu, offset, t_array ):
    """
    return alpha, beta, gamma for TKF models; no approximations made, 
        except still allow use of switch between approx and real for 
        log(1-x) function

    T: number of branch lengths in t_array
    
    returns:
    --------
    out_dict: the tkf values
        out_dict['log_alpha']: ArrayLike[float32], (T,)
        out_dict['log_one_minus_alpha']: ArrayLike[float32], (T,)
        out_dict['log_beta']: ArrayLike[float32], (T,)
        out_dict['log_one_minus_beta']: ArrayLike[float32], (T,)
        out_dict['log_gamma']: ArrayLike[float32], (T,)
        out_dict['log_one_minus_gamma']: ArrayLike[float32], (T,)
    
    approx_flags_dict: None (placeholder)
    
    """
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
    def tkf_params_per_timepoint(idx):
        log_alpha_at_this_t = log_alpha[idx]
        log_beta_at_this_t = log_beta[idx]
        
        # 1 - alpha
        log_one_minus_alpha = stable_log_one_minus_x(log_x = log_alpha_at_this_t)
        
        # 1 - gamma
        log_one_minus_gamma = (log_beta_at_this_t - 
                               ( jnp.log( 1-offset) + log_one_minus_alpha )
                               )
        
        # gamma
        log_gamma = stable_log_one_minus_x(log_x = log_one_minus_gamma)
        
        return {'log_one_minus_alpha': log_one_minus_alpha,
                'log_gamma': log_gamma,
                'log_one_minus_gamma': log_one_minus_gamma}
    
    vmapped_tkf_params_per_timepoint = jax.vmap(tkf_params_per_timepoint)
    to_add = vmapped_tkf_params_per_timepoint( jnp.arange(t_array.shape[0]) )
        
    
    ### output
    out_dict = {'log_alpha': log_alpha,
                'log_beta': log_beta,
                'log_one_minus_beta': log_one_minus_beta}
    out_dict = {**out_dict, **to_add}
    
    return out_dict, None


def approx_tkf( mu, offset, t_array ):
    """
    return alpha, beta, gamma for TKF models; only use approx formulas, 
        except still allow use of switch between approx and real for 
        log(1-x) function

    T: number of branch lengths in t_array
    
    returns:
    --------
    out_dict: the tkf values
        out_dict['log_alpha']: ArrayLike[float32], (T,)
        out_dict['log_one_minus_alpha']: ArrayLike[float32], (T,)
        out_dict['log_beta']: ArrayLike[float32], (T,)
        out_dict['log_one_minus_beta']: ArrayLike[float32], (T,)
        out_dict['log_gamma']: ArrayLike[float32], (T,)
        out_dict['log_one_minus_gamma']: ArrayLike[float32], (T,)
    
    approx_flags_dict: None (placeholder)
    
    """
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
    
    
    ### vmap + jax.lax.cond solely for stable_log_one_minus_x function
    def tkf_params_per_timepoint(idx):
        t = t_array[idx]
        log_alpha_at_this_t = log_alpha[idx]
        log_beta_at_this_t = log_beta[idx]
        
        # 1 - alpha
        log_one_minus_alpha = stable_log_one_minus_x(log_x = log_alpha_at_this_t)
        
        # 1 - gamma
        log_one_minus_gamma = approx_one_minus_gamma( (mu, offset, t) )
        
        # gamma
        log_gamma = stable_log_one_minus_x(log_x = log_one_minus_gamma)
        
        return {'log_one_minus_alpha': log_one_minus_alpha,
                'log_gamma': log_gamma,
                'log_one_minus_gamma': log_one_minus_gamma}
    
    vmapped_tkf_params_per_timepoint = jax.vmap(tkf_params_per_timepoint)
    to_add = vmapped_tkf_params_per_timepoint( jnp.arange(t_array.shape[0]) )
        
    
    ### output
    out_dict = {'log_alpha': log_alpha,
                'log_beta': log_beta,
                'log_one_minus_beta': log_one_minus_beta}
    out_dict = {**out_dict, **to_add}
    
    return out_dict, None



###############################################################################
### for tkf91, tkf92: functions to get marginal and    ########################
### conditional transition matrices                    ########################
###############################################################################
def get_tkf91_single_seq_marginal_transition_logprobs(offset,
                                                      **kwargs):
    """
    For scoring single-sequence marginals under TKF91 model
    
    Arguments
    ----------
    offset : ArrayLike, ()
        1 - (lam/mu)
    
        
    Returns
    -------
    log_arr : ArrayLike, (2,2)
        
        emit -> emit   |  emit -> end
        -------------------------------
        start -> emit  |  start -> end
        
    """
    # lam / mu = 1 - offset
    log_lam_div_mu = jnp.log1p(-offset)
    log_one_minus_lam_div_mu = jnp.log(offset)

    log_arr = jnp.array( [[log_lam_div_mu, log_one_minus_lam_div_mu],
                          [log_lam_div_mu, log_one_minus_lam_div_mu]] ) #(2,2)
    return log_arr


def get_tkf92_single_seq_marginal_transition_logprobs(offset,
                                                      frag_class_probs,
                                                      r_ext_prob,
                                                      **kwargs):
    """
    For scoring single-sequence marginals under TKF92 model
    
    C_dom: number of mixtures associated with nested TKF92 models (domain-level)
    C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
        > for tkf91, there can NOT be mixtures over transitions (i.e. C_frag=1)
       
    
    Arguments
    ----------
    offset : ArrayLike, ()
        1 - (lam/mu)
    
    r_ext_prob : ArrayLike, (C_dom, C_frag)
        fragment extension probabilities
    
    frag_class_probs : ArrayLike, (C_dom, C_frag)
        support for the classes i.e. P(end at class c_frag)
     
        
    Returns
    -------
    log_arr : ArrayLike, (C_dom, C_{frag_from}, C_{frag_to}, 2, 2)
        
        emit -> emit   |  emit -> end
        -------------------------------
        start -> emit  |  start -> end
        
    """
    C_dom = frag_class_probs.shape[0] #domain-level classes
    C_frag = frag_class_probs.shape[1] #fragment-level classes
    
    ### move values to log space
    log_frag_class_prob = safe_log(frag_class_probs) #(C_dom, C_{frag_to})
    log_r_ext_prob = safe_log(r_ext_prob) #(C_dom, C_{frag_from})
    log_one_minus_r = log_one_minus_x(log_r_ext_prob) #(C_dom, C_{frag_from})
    
    # lam / mu = 1 - offset
    # offset = 1 - (lam/mu)
    log_lam_div_mu = jnp.log1p(-offset) #float
    log_one_minus_lam_div_mu = jnp.log(offset) #float
    
    
    ### build cells
    # cell 1: emit -> emit 
    # (1-r_c) * (lam/mu) * P(d)
    # (log_one_minus_r + log_lam_div_mu)[..., None]: (C_dom, C_{frag_from},           1)
    #               log_frag_class_prob[:, None, :]: (C_dom,             1, C_{frag_to})
    #                                     log_cell1: (C_dom, C_{frag_from}, C_{frag_to})
    log_cell1 = (log_one_minus_r + log_lam_div_mu)[..., None] + log_frag_class_prob[:, None, :] 
    
    # cell 2: emit -> end 
    # (1-r) * (1 - lam/mu)
    log_cell2 = ( log_one_minus_r + log_one_minus_lam_div_mu )[...,None] #(C_dom, C_{frag_from},1)
    log_cell2 = jnp.broadcast_to( log_cell2, (C_dom, C_frag, C_frag) ) # (C_dom, C_{frag_from}, C_{frag_to})
    
    # cell 3: start -> emit
    # (lam/mu) * P(d)
    log_cell3 = ( log_lam_div_mu + log_frag_class_prob )[:,None,:] # (C_dom, 1, C_{frag_to})
    log_cell3 = jnp.broadcast_to( log_cell3, (C_dom, C_frag, C_frag) ) # (C_dom, C_{frag_from}, C_{frag_to})
    
    # cell 4: start -> end
    # (1-lam/mu)
    log_cell4 = jnp.broadcast_to( log_one_minus_lam_div_mu, (C_dom, C_frag, C_frag) ) # (C_dom, C_{frag_from}, C_{frag_to})
    

    ### build matrix
    log_single_seq_tkf92 = jnp.stack( [jnp.stack( [log_cell1, log_cell2], axis=-1 ),
                                       jnp.stack( [log_cell3, log_cell4], axis=-1 )],
                                     axis = -2 ) #(C_dom, C_{frag_from}, C_{frag_to}, 2, 2)
    
    # add fragment extension probability to transitions between same class
    # at cell 1: emit -> emit 
    # r + (1-r) * (lam/mu) * P(c)
    i_idx = jnp.arange(C_frag)
    prev_vals = log_single_seq_tkf92[:, i_idx, i_idx, 0, 0] #(C_dom, C_frag)
    new_vals = logsumexp_with_arr_lst([log_r_ext_prob, prev_vals]) #(C_dom, C_frag)
    log_single_seq_tkf92 = log_single_seq_tkf92.at[:, i_idx, i_idx, 0, 0].set(new_vals) #(C_dom, C_{frag_from}, C_{frag_to}, 2, 2)
    return log_single_seq_tkf92 #(C_dom, C_{frag_from}, C_{frag_to}, 2, 2)


def get_cond_transition_logprobs(marg_matrix, 
                                 joint_matrix):
    """
    obtain the conditional log probability by composing the joint with the marginal
    
    S = full number of states; 4 total: {Match, Ins, Del, Start/End}
    
    Arguments
    ----------
    marg_matrix : ArrayLike, (...,2,2)
        scoring matrix for marginal transition probabilities
        P(seq)
    
    joint_matrix : ArrayLike, (...,S,S) 
        scoring matrix for joint transition probabilities
        P(desc, anc, align | t)
        
    Returns
    -------
    cond_matrix : ArrayLike, joint_matrix.shape
        scoring matrix for conditional transition probabilities
        P(desc, align | anc, t)
        
    """
    cond_matrix = joint_matrix.at[...,[0,1,2], 0].add(-marg_matrix[..., 0,0][None,...,None])
    cond_matrix = cond_matrix.at[...,[0,1,2], 2].add(-marg_matrix[..., 0,0][None,...,None])
    cond_matrix = cond_matrix.at[...,3,0].add(-marg_matrix[..., 1,0][None,...])
    cond_matrix = cond_matrix.at[...,3,2].add(-marg_matrix[..., 1,0][None,...])
    cond_matrix = cond_matrix.at[...,[0,1,2],3].add(-marg_matrix[..., 0,1][None,...,None])
    cond_matrix = cond_matrix.at[...,3,3].add(-marg_matrix[..., 1,1][None,...])
    return cond_matrix



###############################################################################
### logprob of alignments from summary counts    ##############################
###############################################################################
def _selectively_add_time_dim(x, ref):
    """
    add extra dimension for time, compared to ref
    confirmed to be jit-comptable, since ref is passed statically
    
    Arguments
    ----------
    x : ArrayLike
        array to modify
    
    ref : int
        number of dims the array is supposed to have
    
    Returns
    -------
    ArrayLike
        x with possible extra dimension (or None, to intentionally 
        trigger an error)
    
    """
    if len(x.shape) == ref-1:
        return x[None,:]
    
    elif len(x.shape) == ref:
        return x
    
    # silently trigger an error otherwise
    else:
        return None
    

def marginalize_over_times(logprob_perSamp_perTime,
                            exponential_dist_param,
                            t_array):
    """
    marginalize according to an exponential time prior
    
    B = batch; number of alignments
    T = branch length; time
    
    
    Arguments
    ----------
    logprob_perSamp_perTime : ArrayLike, (T,B)
        log probabilities per possible branch length
    
    exponential_dist_param : float
        the parameter for the exponential distribution
    
    t_array : ArrayLike, (T,)
        array of possible branch lengths
    
    Returns
    -------
    ArrayLike, (B,)
        scores after marginalizing out time
    
    """
    ### constants to add (multiply by)
    # logP(t_k) = exponential distribution
    logP_time = ( jnp.log(exponential_dist_param) - 
                  (exponential_dist_param * t_array) ) #(T,)
    log_t_grid = jnp.log( t_array[1:] - t_array[:-1] ) #(T-1,)
    
    # kind of a hack, but repeat the last time array value
    log_t_grid = jnp.concatenate( [log_t_grid, log_t_grid[-1][None] ], axis=0) #(T,)
    
    
    ### add in log space, multiply in probability space; logsumexp
    logP_perSamp_perTime_withConst = ( logprob_perSamp_perTime +
                                        logP_time[:,None] +
                                        log_t_grid[:,None] ) #(T,B)
    
    return logsumexp(logP_perSamp_perTime_withConst, axis=0) #(B,)
    

def joint_prob_from_counts( batch: tuple[ArrayLike],
                            times_from: str,
                            score_indels: bool,
                            scoring_matrices_dict: dict,
                            t_array: ArrayLike or None,
                            exponential_dist_param: float,
                            norm_reported_loss_by: str or None,
                            return_intermeds: bool=False ):
    """
    score an alignment from summary counts
    
    B = batch; number of alignments
    A = alphabet size
    S = number of transition states; here, it's 4: M, I, D, [S or E]
    T = branch length; time
    
    
    Arguments
    ----------
    batch : Tuple (from pytorch dataloader)
        batch[0] : ArrayLike, (B,A,A)
            subCounts

        batch[1] : ArrayLike, (B,A)
            insCounts

        batch[2] : ArrayLike, (B,A)
            delCounts

        batch[3] : ArrayLike, (B,S,S)
            transCounts
        
        batch[4] : ArrayLike, (B,)
            branch length for each sample

    times_from :  {geometric, t_array_from_file, t_per_sample}
        STATIC ARGUEMENT FOR JIT COMPILATION
        how to handle time
    
    score_indels : bool
        STATIC ARGUEMENT FOR JIT COMPILATION
        whether or not to score indel positions

    scoring_matrices_dict : dict
        scoring_matrices_dict['logprob_emit_at_match'] : ArrayLike
            logprob of emissions at match sites
            if time_from in [geometric, t_array_from_file]: (T,A,A)
            elif time_from == 't_per_sample': (B,A,A)
        
        scoring_matrices_dict['logprob_emit_at_indel'] : ArrayLike, (A,)
            logprob of emissions at ins and del sites
        
        scoring_matrices_dict['all_transit_matrices']['joint'] : ArrayLike
            logprob transitions
            if time_from in [geometric, t_array_from_file] and score_indels: (T,S,S)
            elif time_from == 't_per_sample' and score_indels: (B,S,S)
            elif not score_indels: (2,)
            
    t_array : ArrayLike, (T,)
        branch lengths to apply to all samples
        if time_from in [geometric, t_array_from_file]: (T,)
        else t_array = None (never used) 
    
    exponential_dist_param : float or None
        when marginalizing over time, use an exponential prior; this is the 
        parameter for that exponential distribution
    
    norm_reported_loss_by : {desc_len, align_len}
        how to normalize the loss, if including indels (if not scoring indels, 
        normalization length is already decided)
        
    Returns
    -------
    out['joint_neg_logP'] : ArrayLike, (B,)
        raw loglikelihood of alignment
    
    out['joint_neg_logP_length_normed'] : ArrayLike, (B,)
        loglikelihood of alignment, after normalizing by length
    
    """
    ####################################################################
    ### static arguments that determine shape during jit compilation   #
    ####################################################################
    if times_from in ['geometric', 't_array_from_file']:
        time_dep_score_fn = partial( jnp.einsum, 'tij,bij->tb' )
        expected_num_output_dims = 2
    
    elif times_from == 't_per_sample':
        time_dep_score_fn = partial( jnp.einsum, 'bij,bij->b' )
        expected_num_output_dims = 1
    
    
    ####################
    ### unpack batch   #
    ####################
    subCounts = batch[0] #(B, A, A)
    insCounts = batch[1] #(B, A)
    delCounts = batch[2] #(B, A)
    transCounts = batch[3] #(B, S)
        
    
    #######################
    ### score emissions   #
    #######################
    ### emissions at match sites
    # subCounts is (B,A,A)
    #
    # scoring_matrices_dict['logprob_emit_at_match'] has following sizes-
    #   if time_from in [geometric, t_array_from_file]: (T,A,A)
    #   elif time_from == 't_per_sample': (B,A,A)
    #
    # emission_score has following sizes- 
    #   if time_from in [geometric, t_array_from_file]: (T,B)
    #   elif time_from == t_per_sample: (B,)
    joint_match_score = time_dep_score_fn(scoring_matrices_dict['joint_logprob_emit_at_match'], 
                                          subCounts)
    
    if score_indels:
        ### emissions at insert sites
        # insCounts is (B,A)
        # scoring_matrices_dict['logprob_emit_at_indel'] is (A,)
        # ins_emit_score is (B)
        ins_emit_score = jnp.einsum('i,bi->b',
                                    scoring_matrices_dict['logprob_emit_at_indel'], 
                                    insCounts) #(B,)
        
        ins_emit_score = _selectively_add_time_dim( ins_emit_score,
                                                   expected_num_output_dims )
        
        ### emissions at delete sites
        # delCounts is (B,A)
        # del_emit_score is (B)
        del_emit_score = jnp.einsum('i,bi->b',
                                    scoring_matrices_dict['logprob_emit_at_indel'], 
                                    delCounts) #(B,)
        del_emit_score = _selectively_add_time_dim( del_emit_score,
                                                   expected_num_output_dims )
        
        # add to emission score
        emission_score = ( joint_match_score +
                           ins_emit_score +
                           del_emit_score )
    
    elif not score_indels:
        emission_score = joint_match_score
        
    #########################
    ### score transitions   #
    #########################
    # transCounts is (B,S,S)
    #
    # scoring_matrices_dict['all_transit_matrices']['joint'] has the following sizes-
    #    if time_from in [geometric, t_array_from_file] and score_indels: (T,S,S)
    #    elif time_from == 't_per_sample' and score_indels: (B,S,S)
    #    elif not score_indels: (2,)
    #
    # joint_transit_score has the following sizes-
    #   if time_from in [geometric, t_array_from_file]: (T,B)
    #   elif time_from == t_per_sample or not score_indels: (B,)
    if score_indels:
        joint_transit_score = time_dep_score_fn(scoring_matrices_dict['all_transit_matrices']['joint'], 
                                                transCounts)
        
    elif not score_indels:
        align_lens = subCounts.sum(axis=(-1,-2))
        logprob_emit = scoring_matrices_dict['all_transit_matrices']['joint'][0]
        log_one_minus_prob_emit = scoring_matrices_dict['all_transit_matrices']['joint'][1]
        joint_transit_score = align_lens * logprob_emit + log_one_minus_prob_emit
    
    joint_transit_score = _selectively_add_time_dim( joint_transit_score,
                                                    expected_num_output_dims )
    joint_logprob_perSamp_maybePerTime = joint_transit_score + emission_score
    
    
    ################
    ### postproc   #
    ################
    # marginalize over times, if required
    if (expected_num_output_dims==2) and (t_array.shape[0] > 1):
        joint_neg_logP = -marginalize_over_times(logprob_perSamp_perTime = joint_logprob_perSamp_maybePerTime,
                                                 exponential_dist_param = exponential_dist_param,
                                                 t_array = t_array) #(B,)
         
    elif (expected_num_output_dims==2) and (t_array.shape[0] == 1):
        joint_neg_logP = -joint_logprob_perSamp_maybePerTime[0,:] #(B,)
    
    elif expected_num_output_dims == 1:
        joint_neg_logP = -joint_logprob_perSamp_maybePerTime #(B,)
    
    
    # normalize by some length    
    if (norm_reported_loss_by == 'desc_len') and score_indels:
        length_for_normalization = ( subCounts.sum(axis=(-2, -1)) + 
                                     insCounts.sum(axis=(-1))
                                     )
    
    elif (norm_reported_loss_by == 'align_len') and score_indels:
        length_for_normalization = ( subCounts.sum(axis=(-2, -1)) + 
                                     insCounts.sum(axis=(-1)) + 
                                     delCounts.sum(axis=(-1))
                                     ) 
    elif not score_indels:
        length_for_normalization = subCounts.sum(axis=(-2, -1))
    
    joint_neg_logP_length_normed = joint_neg_logP / length_for_normalization #(B,)
    
    out = {'joint_neg_logP': joint_neg_logP,  #(B,)
            'joint_neg_logP_length_normed': joint_neg_logP_length_normed,  #(B,)
            'align_length_for_normalization': length_for_normalization}  #(B,)
    
    if return_intermeds:
        out['joint_transit_score'] = joint_transit_score
        out['joint_emission_score'] = emission_score
    
    return out

def cond_prob_from_counts( batch: tuple[ArrayLike],
                            times_from: str,
                            score_indels: bool,
                            scoring_matrices_dict: dict,
                            t_array: ArrayLike or None,
                            exponential_dist_param: float,
                            norm_reported_loss_by: str or None,
                            return_intermeds: bool=False ):
    """
    score an alignment from summary counts
    
    B = batch; number of alignments
    A = alphabet size
    S = number of transition states; here, it's 4: M, I, D, [S or E]
    T = branch length; time
    
    
    Arguments
    ----------
    batch : Tuple (from pytorch dataloader)
        batch[0] : ArrayLike, (B,A,A)
            subCounts

        batch[1] : ArrayLike, (B,A)
            insCounts

        batch[2] : ArrayLike, (B,A)
            delCounts

        batch[3] : ArrayLike, (B,S,S)
            transCounts
        
        batch[4] : ArrayLike, (B,)
            branch length for each sample

    times_from :  {geometric, t_array_from_file, t_per_sample}
        STATIC ARGUEMENT FOR JIT COMPILATION
        how to handle time
    
    score_indels : bool
        STATIC ARGUEMENT FOR JIT COMPILATION
        whether or not to score indel positions

    scoring_matrices_dict : dict
        scoring_matrices_dict['logprob_emit_at_match'] : ArrayLike
            logprob of emissions at match sites
            if time_from in [geometric, t_array_from_file]: (T,A,A)
            elif time_from == 't_per_sample': (B,A,A)
        
        scoring_matrices_dict['logprob_emit_at_indel'] : ArrayLike, (A,)
            logprob of emissions at ins and del sites
        
        scoring_matrices_dict['all_transit_matrices']['joint'] : ArrayLike
            logprob transitions
            if time_from in [geometric, t_array_from_file] and score_indels: (T,S,S)
            elif time_from == 't_per_sample' and score_indels: (B,S,S)
            elif not score_indels: (2,)
            
    t_array : ArrayLike, (T,)
        branch lengths to apply to all samples
        if time_from in [geometric, t_array_from_file]: (T,)
        else t_array = None (never used) 
    
    exponential_dist_param : float or None
        when marginalizing over time, use an exponential prior; this is the 
        parameter for that exponential distribution
    
    norm_reported_loss_by : {desc_len, align_len}
        how to normalize the loss, if including indels (if not scoring indels, 
        normalization length is already decided)
        
    Returns
    -------
    out['joint_neg_logP'] : ArrayLike, (B,)
        raw loglikelihood of alignment
    
    out['joint_neg_logP_length_normed'] : ArrayLike, (B,)
        loglikelihood of alignment, after normalizing by length
    
    """
    ####################################################################
    ### static arguments that determine shape during jit compilation   #
    ####################################################################
    if times_from in ['geometric', 't_array_from_file']:
        time_dep_score_fn = partial( jnp.einsum, 'tij,bij->tb' )
        expected_num_output_dims = 2
    
    elif times_from == 't_per_sample':
        time_dep_score_fn = partial( jnp.einsum, 'bij,bij->b' )
        expected_num_output_dims = 1
    
    
    ####################
    ### unpack batch   #
    ####################
    subCounts = batch[0] #(B, A, A)
    insCounts = batch[1] #(B, A)
    delCounts = batch[2] #(B, A)
    transCounts = batch[3] #(B, S)
        
    
    #######################
    ### score emissions   #
    #######################
    ### emissions at match sites
    # subCounts is (B,A,A)
    #
    # scoring_matrices_dict['logprob_emit_at_match'] has following sizes-
    #   if time_from in [geometric, t_array_from_file]: (T,A,A)
    #   elif time_from == 't_per_sample': (B,A,A)
    #
    # emission_score has following sizes- 
    #   if time_from in [geometric, t_array_from_file]: (T,B)
    #   elif time_from == t_per_sample: (B,)
    cond_match_score = time_dep_score_fn(scoring_matrices_dict['cond_logprob_emit_at_match'], 
                                          subCounts)
    
    if score_indels:
        ### emissions at insert sites
        # insCounts is (B,A)
        # scoring_matrices_dict['logprob_emit_at_indel'] is (A,)
        # ins_emit_score is (B)
        ins_emit_score = jnp.einsum('i,bi->b',
                                    scoring_matrices_dict['logprob_emit_at_indel'], 
                                    insCounts) #(B,)
        ins_emit_score = _selectively_add_time_dim( ins_emit_score,
                                                   expected_num_output_dims )
        
        # add to emission score
        emission_score = ( cond_match_score +
                           ins_emit_score )
    
    elif not score_indels:
        emission_score = cond_match_score
        
    #########################
    ### score transitions   #
    #########################
    # transCounts is (B,S,S)
    #
    # scoring_matrices_dict['all_transit_matrices']['conditional'] has the following sizes-
    #    if time_from in [geometric, t_array_from_file] and score_indels: (T,S,S)
    #    elif time_from == 't_per_sample' and score_indels: (B,S,S)
    #    elif not score_indels: (2,)
    #
    # cond_transit_score has the following sizes-
    #   if time_from in [geometric, t_array_from_file]: (T,B)
    #   elif time_from == t_per_sample or not score_indels: (B,)
    if score_indels:
        cond_transit_score = time_dep_score_fn(scoring_matrices_dict['all_transit_matrices']['conditional'], 
                                                transCounts)
        mask = transCounts[:,3,1]
        corr = scoring_matrices_dict['all_transit_matrices']['log_corr']
        cond_transit_score = cond_transit_score - mask * corr
        
    elif not score_indels:
        align_lens = subCounts.sum(axis=(-1,-2))
        logprob_emit = scoring_matrices_dict['all_transit_matrices']['conditional'][0]
        log_one_minus_prob_emit = scoring_matrices_dict['all_transit_matrices']['conditional'][1]
        cond_transit_score = align_lens * logprob_emit + log_one_minus_prob_emit
    
    cond_transit_score = _selectively_add_time_dim( cond_transit_score,
                                                    expected_num_output_dims )
    cond_logprob_perSamp_maybePerTime = cond_transit_score + emission_score
    
    
    ################
    ### postproc   #
    ################
    # marginalize over times, if required
    if (expected_num_output_dims==2) and (t_array.shape[0] > 1):
        cond_neg_logP = -marginalize_over_times(logprob_perSamp_perTime = cond_logprob_perSamp_maybePerTime,
                                                 exponential_dist_param = exponential_dist_param,
                                                 t_array = t_array) #(B,)
         
    elif (expected_num_output_dims==2) and (t_array.shape[0] == 1):
        cond_neg_logP = -cond_logprob_perSamp_maybePerTime[0,:] #(B,)
    
    elif expected_num_output_dims == 1:
        cond_neg_logP = -cond_logprob_perSamp_maybePerTime #(B,)
    
    
    # normalize by some length    
    if (norm_reported_loss_by == 'desc_len') and score_indels:
        length_for_normalization = ( subCounts.sum(axis=(-2, -1)) + 
                                     insCounts.sum(axis=(-1))
                                     )
    
    elif (norm_reported_loss_by == 'align_len') and score_indels:
        length_for_normalization = ( subCounts.sum(axis=(-2, -1)) + 
                                     insCounts.sum(axis=(-1)) + 
                                     delCounts.sum(axis=(-1))
                                     ) 
    elif not score_indels:
        length_for_normalization = subCounts.sum(axis=(-2, -1))
    
    cond_neg_logP_length_normed = cond_neg_logP / length_for_normalization #(B,)
    
    out = {'cond_neg_logP': cond_neg_logP,  #(B,)
            'cond_neg_logP_length_normed': cond_neg_logP_length_normed}  #(B,)
    
    if return_intermeds:
        out['cond_transit_score'] = cond_transit_score
        out['cond_emission_score'] = emission_score
    
    return out
        

def anc_marginal_probs_from_counts( batch: tuple[ArrayLike],
                                    score_indels: bool,
                                    scoring_matrices_dict: dict,
                                    return_intermeds: bool=False ):
    """
    score single-sequence marginals of ANCESTOR SEQUENCE
    
    B = batch; number of alignments
    A = alphabet size
    S = number of transition states; here, it's 4: M, I, D, [S or E]
    
    
    Arguments
    ----------
    batch : Tuple (from pytorch dataloader)
        batch[0] : ArrayLike, (B,A,A)
            subCounts

        batch[2] : ArrayLike, (B,A)
            delCounts

        batch[3] : ArrayLike, (B,S,S)
            transCounts

    score_indels : bool
        STATIC ARGUEMENT FOR JIT COMPILATION
        whether or not to score indel positions
    
    which_seq : {anc_seq, desc_seq}
        which sequence you're scoring; size is the same 

    scoring_matrices_dict : dict
        scoring_matrices_dict['joint_logprob_emit_at_match'] : ArrayLike
            joint logprob of emissions at match sites
            if time_from in [geometric, t_array_from_file]: (T,A,A)
            elif time_from == 't_per_sample': (B,A,A)
        
        scoring_matrices_dict['logprob_emit_at_indel'] : ArrayLike, (A,)
            logprob of emissions at ins and del sites
        
        scoring_matrices_dict['all_transit_matrices']['marginal'] : ArrayLike
            if score_indels: (2,2)
            elif not score indels: (2,)
        
    Returns
    -------
    out['seq_neg_logP'] : ArrayLike, (B,)
        raw loglikelihood of sequence
    
    out['seq_neg_logP_length_normed'] : ArrayLike, (B,)
        loglikelihood of sequence, after normalizing by length
    
    """
    subCounts = batch[0] #(B, A, A)
    delCounts = batch[2] #(B, A)
    transCounts = batch[3] #(B, S)
    
    ### emissions
    anc_emitCounts = subCounts.sum(axis=2) #(B,A)
    if score_indels:
        anc_emitCounts = anc_emitCounts + delCounts #(B,A)
    anc_len = anc_emitCounts.sum(axis=(-1)) #(B,)
    
    anc_marg_emit_score = jnp.einsum('i,bi->b',
                                     scoring_matrices_dict['logprob_emit_at_indel'],
                                     anc_emitCounts) #(B,)
    
    ### transitions
    if score_indels:
        # use only transitions that end with match (0) and del (2)
        anc_emit_to_emit = ( transCounts[...,0].sum( axis=-1 ) + 
                              transCounts[...,2].sum( axis=-1 ) ) - 1  #(B,)
        
        anc_transCounts = jnp.stack( [jnp.stack( [anc_emit_to_emit, 
                                                  jnp.ones(anc_emit_to_emit.shape[0])], 
                                                axis=-1 ),
                                      jnp.stack( [jnp.ones(anc_emit_to_emit.shape[0]), 
                                                  jnp.zeros(anc_emit_to_emit.shape[0])], 
                                                axis=-1 )],
                                      axis = -2 ) #(B,2,2)
        
        anc_marg_transit_score = jnp.einsum( 'mn,bmn->b', 
                                              scoring_matrices_dict['all_transit_matrices']['marginal'] , 
                                              anc_transCounts ) #(B,)
        
    elif not score_indels:
        logprob_emit = scoring_matrices_dict['all_transit_matrices']['marginal'][0] #(1,)
        log_one_minus_prob_emit = scoring_matrices_dict['all_transit_matrices']['marginal'][1] #(1,)
        anc_marg_transit_score = anc_len * logprob_emit + log_one_minus_prob_emit #(B,)
    
    anc_neg_logP = -(anc_marg_emit_score + anc_marg_transit_score)
    anc_neg_logP_length_normed = anc_neg_logP / anc_len
    
    out = {'anc_neg_logP': anc_neg_logP,  #(B,)
            'anc_neg_logP_length_normed': anc_neg_logP_length_normed}  #(B,)
    
    if return_intermeds:
        out['anc_marg_transit_score'] = anc_marg_transit_score
        out['anc_marg_emit_score'] = anc_marg_emit_score
    
    return out


def desc_marginal_probs_from_counts( batch: tuple[ArrayLike],
                                     score_indels: bool,
                                     scoring_matrices_dict: dict,
                                     *args,
                                     **kwargs ):
    """
    score single-sequence marginals of DESCENDANT SEQUENCE
    
    B = batch; number of alignments
    A = alphabet size
    S = number of transition states; here, it's 4: M, I, D, [S or E]
    
    
    Arguments
    ----------
    batch : Tuple (from pytorch dataloader)
        batch[0] : ArrayLike, (B,A,A)
            subCounts

        batch[1] : ArrayLike, (B,A)
            insCounts

        batch[3] : ArrayLike, (B,S,S)
            transCounts

    score_indels : bool
        STATIC ARGUEMENT FOR JIT COMPILATION
        whether or not to score indel positions
    
    which_seq : {anc_seq, desc_seq}
        which sequence you're scoring; size is the same 

    scoring_matrices_dict : dict
        scoring_matrices_dict['joint_logprob_emit_at_match'] : ArrayLike
            joint logprob of emissions at match sites
            if time_from in [geometric, t_array_from_file]: (T,A,A)
            elif time_from == 't_per_sample': (B,A,A)
        
        scoring_matrices_dict['logprob_emit_at_indel'] : ArrayLike, (A,)
            logprob of emissions at ins and del sites
        
        scoring_matrices_dict['all_transit_matrices']['marginal'] : ArrayLike
            if score_indels: (2,2)
            elif not score indels: (2,)
        
    Returns
    -------
    out['seq_neg_logP'] : ArrayLike, (B,)
        raw loglikelihood of sequence
    
    out['seq_neg_logP_length_normed'] : ArrayLike, (B,)
        loglikelihood of sequence, after normalizing by length
    
    """
    subCounts = batch[0] #(B, A, A)
    insCounts = batch[1] #(B, A)
    transCounts = batch[3] #(B, S)
    
    ### emissions
    desc_emitCounts = subCounts.sum(axis=1) #(B,A)
    if score_indels:
        desc_emitCounts = desc_emitCounts + insCounts #(B,A)
    desc_len = desc_emitCounts.sum(axis=(-1)) #(B,)
    
    desc_marg_emit_score = jnp.einsum('i,bi->b',
                                     scoring_matrices_dict['logprob_emit_at_indel'],
                                     desc_emitCounts) #(B,)
    
    ### transitions
    if score_indels:
        # use only transitions that end with match (0) and ins(1)
        desc_emit_to_emit = ( transCounts[...,0].sum( axis=-1 ) + 
                              transCounts[...,1].sum( axis=-1 ) ) - 1  #(B,)
        
        desc_transCounts = jnp.stack( [jnp.stack( [desc_emit_to_emit, 
                                                   jnp.ones(desc_emit_to_emit.shape[0])], 
                                                 axis=-1 ),
                                       jnp.stack( [jnp.ones(desc_emit_to_emit.shape[0]), 
                                                   jnp.zeros(desc_emit_to_emit.shape[0])], 
                                                 axis=-1 )],
                                       axis = -2 ) #(B,2,2)
        
        desc_marg_transit_score = jnp.einsum( 'mn,bmn->b', 
                                              scoring_matrices_dict['all_transit_matrices']['marginal'] , 
                                              desc_transCounts ) #(B,)
    
    elif not score_indels:
        logprob_emit = scoring_matrices_dict['all_transit_matrices']['marginal'][0] #(1,)
        log_one_minus_prob_emit = scoring_matrices_dict['all_transit_matrices']['marginal'][1] #(1,)
        desc_marg_transit_score = desc_len * logprob_emit + log_one_minus_prob_emit #(B,)
    
    desc_neg_logP = -(desc_marg_emit_score + desc_marg_transit_score)
    desc_neg_logP_length_normed = desc_neg_logP / desc_len
    
    return {'desc_neg_logP': desc_neg_logP,  #(B,)
            'desc_neg_logP_length_normed': desc_neg_logP_length_normed}  #(B,)



###############################################################################
### logprob from forward/backward over site classes    ########################
###############################################################################
def get_joint_loglike_emission_time_grid(aligned_inputs,
                                         pos,
                                         joint_logprob_emit_at_match,
                                         logprob_emit_at_indel):
    """
    to use when MARGINALIZING over a grid of times; 
        joint_logprob_emit_at_match is (T, C, A, A)
    
    can use this function in forward and backward functions to find 
      emission probabilities (which are site independent)
    
    L: length of pairwise alignment
    T: number of timepoints
    B: batch size
    C: number of latent site clases
    A: alphabet size (20 for proteins, 4 for amino acids)
    
    
    Arguments
    ----------
    aligned_inputs : ArrayLike, (B, L, 3)
        dim2=0: ancestor
        dim2=1: descendant
        dim2=2: alignment state; M=1, I=2, D=3, S=4, E=5
    
    pos : int
        which alignment column you're at
    
    joint_logprob_emit_at_match : ArrayLike, (T, C, A, A)
        logP(anc, desc | c, t); log-probability of emission at match site
    
    logprob_emit_at_indel : ArrayLike, (C, A)
        logP(anc | c) or logP(desc | c); log-equilibrium distribution
        
    Returns
    -------
    joint_e : ArrayLike, (T, C, B)
        log-probability of emission at given column, across all possible 
        site classes
    """
    T = joint_logprob_emit_at_match.shape[0]
    C = joint_logprob_emit_at_match.shape[1]
    B = aligned_inputs.shape[0]
    
    # unpack
    anc_toks = aligned_inputs[:,pos,0]
    desc_toks = aligned_inputs[:,pos,1]
    state_at_pos = aligned_inputs[:,pos,2]
    
    # get all possible scores
    joint_emit_if_match = joint_logprob_emit_at_match[..., anc_toks - 3, desc_toks - 3] # (T, C, B) or (C, B)
    emit_if_indel_desc = logprob_emit_at_indel[:, desc_toks - 3] #(C, B)
    emit_if_indel_anc = logprob_emit_at_indel[:, anc_toks - 3] #(C, B)
    
    # stack all
    emit_if_indel_desc = jnp.broadcast_to( emit_if_indel_desc[None, :, :], 
                                           (T, C, B) ) #(T, C, B)
    emit_if_indel_anc = jnp.broadcast_to( emit_if_indel_anc[None, :, :], 
                                          (T, C, B) ) #(T, C, B)
    joint_emissions = jnp.stack([joint_emit_if_match, 
                                 emit_if_indel_desc, 
                                 emit_if_indel_anc], axis=0) #(3, T, C, B)

    # expand current state for take_along_axis operation
    state_at_pos_expanded = jnp.broadcast_to( state_at_pos[None, None, None, :]-1, 
                                              (1, T, C, B) )  #(1, T, C, B)

    # gather, remove temporary leading axis
    joint_e = jnp.take_along_axis( joint_emissions, 
                                   state_at_pos_expanded,
                                   axis=0 )[0, ...] # (T, C, B)
    
    return joint_e

def get_joint_loglike_emission_branch_len_per_samp(aligned_inputs,
                                                   pos,
                                                   joint_logprob_emit_at_match,
                                                   logprob_emit_at_indel):
    """
    to use when MARGINALIZING over a grid of times; 
        joint_logprob_emit_at_match is (B, C, A, A)
    
    can use this function in forward and backward functions to find 
      emission probabilities (which are site independent)
    
    L: length of pairwise alignment
    T: number of timepoints
    B: batch size
    C: number of latent site clases
    A: alphabet size (20 for proteins, 4 for amino acids)
    
    
    Arguments
    ----------
    aligned_inputs : ArrayLike, (B, L, 3)
        dim2=0: ancestor
        dim2=1: descendant
        dim2=2: alignment state; M=1, I=2, D=3, S=4, E=5
    
    pos : int
        which alignment column you're at
    
    joint_logprob_emit_at_match : ArrayLike, (B, C, A, A)
        logP(anc, desc | c, t); log-probability of emission at match site
    
    logprob_emit_at_indel : ArrayLike, (C, A)
        logP(anc | c) or logP(desc | c); log-equilibrium distribution
        
    Returns
    -------
    joint_e : ArrayLike, (C, B)
        log-probability of emission at given column, across all possible 
        site classes
    """
    # unpack
    anc_toks = aligned_inputs[:,pos,0]-3
    desc_toks = aligned_inputs[:,pos,1]-3
    state_at_pos = aligned_inputs[:,pos,2]-1
    
    # Gather over last two axes using (B, 1, 1) indices
    gather_idx_anc = anc_toks[:, None, None, None]  # (B, 1, 1)
    gather_idx_desc = desc_toks[:, None, None, None]  # (B, 1, 1)

    # Shape: (B, C, A, A) → (B, C, 1, 1) after indexing
    joint_emit_if_match = jnp.take_along_axis(
        joint_logprob_emit_at_match, gather_idx_anc, axis=2
    )
    joint_emit_if_match = jnp.take_along_axis(
        joint_emit_if_match, gather_idx_desc, axis=3
    )
    joint_emit_if_match = joint_emit_if_match[:, :, 0, 0].T  # (C, B)

    # Indels: (C, B)
    emit_if_indel_desc = logprob_emit_at_indel[:, desc_toks]
    emit_if_indel_anc = logprob_emit_at_indel[:, anc_toks]

    joint_emissions = jnp.stack([
        joint_emit_if_match,
        emit_if_indel_desc,
        emit_if_indel_anc
    ], axis=0)  # (3, C, B)

    state_exp = state_at_pos[None, None, :]  # (1, 1, B)
    joint_e = jnp.take_along_axis(joint_emissions, state_exp, axis=0)[0, ...]  # (C, B)

    return joint_e



def joint_only_forward(aligned_inputs,
                       joint_logprob_emit_at_match,
                       logprob_emit_at_indel,
                       joint_logprob_transit,
                       unique_time_per_sample: bool, 
                       return_all_intermeds: bool = False):
    """
    TODO: this should be ready for nested TKF92 model... but come back and 
          check this out later
    
    forward algo ONLY to find joint loglike
    
    L_align: length of pairwise alignment
    T: number of timepoints
    B: batch size
    C_trans = C: number of latent site clases
      > could be C_frag or C_dom * C_frag
    A: alphabet (20 for proteins, 4 for DNA)
    S: possible states; here, this is 4: M, I, D, start/end
    
    Arguments
    ----------
    aligned_inputs : ArrayLike, (B, L, 3)
        dim2=0: ancestor
        dim2=1: descendant
        dim2=2: alignment state; M=1, I=2, D=3, S=4, E=5
    
    joint_logprob_emit_at_match : ArrayLike, (T, C, A, A) or (B, C, A, A)
        logP(anc, desc | c, t); log-probability of emission at match site
    
    logprob_emit_at_indel : ArrayLike, (C, A)
        logP(anc | c) or P(desc | c); log-equilibrium distribution
    
    joint_logprob_transit : ArrayLike, (T, C, C, S, S) or (B, C, C, A, A)
        logP(new state, new class | prev state, prev class, t); the joint 
        transition matrix for finding logP(anc, desc, align | c, t)
    
    unique_time_per_sample : Bool 
        whether or not you have unqiue times per sample; affects indexing
    
    
    Returns:
    ---------
    stacked_outputs : ArrayLike, (L_align, T, C, B) or (L_align, C, B)
        the cache from the forward algorithm; this is the total log-probability 
        of ending at a given alignment column (l \in L_align) in class C, given
        the observed alignment
        
        to marginalize over all possible combinations of hidden site classes 
        for a given alignment: extract the final element of the length 
        dimension (i.e. stacked_outputs[-1,...]) and do logsumexp over all 
        classes C. This leaves you with the joint probability of the observed 
        alignment, at all branch lengths in T
    """
    ######################################################
    ### initialize with <start> -> any (curr pos is 1)   #
    ######################################################
    pos = 1
    L_align = aligned_inputs.shape[1]
    
    # decide which version of the functions you're going to use
    if not unique_time_per_sample:
        # output from this is (T, C, B)
        get_joint_loglike_emission = get_joint_loglike_emission_time_grid
        
    elif unique_time_per_sample:
        # output from this is (C, B)
        get_joint_loglike_emission = get_joint_loglike_emission_branch_len_per_samp
    
    # emissions;
    e = get_joint_loglike_emission( aligned_inputs=aligned_inputs,
                       pos=pos,
                       joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                       logprob_emit_at_indel=logprob_emit_at_indel ) # (T, C, B) or (C, B)
    
    # transitions; assume there's never start -> end
    # joint_logprob_transit is (T, C_prev, C_curr, S_prev, S_curr)
    # initial state is 4 (<start>); take the last row
    # use C_prev=0 for start class (but it doesn't matter, because the 
    # transition probability is the same for all C_prev)
    curr_state = aligned_inputs[:, pos, 2]  # (B,)
    curr_state_idx = curr_state - 1         # (B,)
    start_any = joint_logprob_transit[:, 0, :, -1, :] #(T, C_curr, S_curr) or (B, C_curr, S_curr)
    
    if not unique_time_per_sample:
        tr = start_any[...,curr_state-1] #(T, C_curr, B)
    
    elif unique_time_per_sample:
        # joint_logprob_transit: (B, C_curr, S_curr)
        # goal: (C_curr, B)
        tr = jnp.take_along_axis(
            start_any, 
            curr_state_idx[:, None, None],  # shape (B, 1)
            axis=-1
        ) 
        tr = jnp.squeeze(tr).T# (C_curr, B)

    # carry value; 
    init_alpha = e + tr # (T, C, B) or (C, B)
    del e, tr, start_any, curr_state, pos
    
    ######################################################
    ### scan down length dimension to end of alignment   #
    ######################################################
    def scan_fn(prev_alpha, pos):
        ### unpack
        anc_toks =   aligned_inputs[:,   pos, 0]
        desc_toks =  aligned_inputs[:,   pos, 1]

        prev_state = aligned_inputs[:, pos-1, 2]
        curr_state = aligned_inputs[:,   pos, 2]
        
        # remove invalid indexing tokens; this doesn't affect the actual '
        #   calculated loglike
        prev_state = jnp.where( prev_state!=5, prev_state, 4 )
        curr_state = jnp.where( curr_state!=5, curr_state, 4 )
        
        
        ### emissions
        e = get_joint_loglike_emission( aligned_inputs=aligned_inputs,
                           pos=pos,
                           joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                           logprob_emit_at_indel=logprob_emit_at_indel ) # (T, C, B) or (C, B)
        
        
        ### transition probabilities
        def main_body(in_carry, ps, cs):
            # replace padding idx with 1 to prevent NaN gradients; this doesn't
            #   affect the actual calculated loglike
            ps = jnp.maximum(ps, 1)
            cs = jnp.maximum(cs, 1)
            
            # like dot product with C_prev, C_curr
            if not unique_time_per_sample:
                # joint_logprob_transit is (T, C_prev, C_curr, S_prev, S_curr)
                tr_per_class = joint_logprob_transit[..., ps-1, cs-1] #(T, C_prev, C_curr, B)   
                to_add = logsumexp(in_carry[:, :, None, :] + tr_per_class, axis=1) #(T, C_curr, B)
            
            elif unique_time_per_sample:
                # joint_logprob_transit is (B, C_prev, C_curr, S_prev, S_curr)
                ps_idx = (ps-1)[:, None, None, None, None] #(B, 1, 1, 1, 1)
                cs_idx = (cs-1)[:, None, None, None, None] #(B, 1, 1, 1, 1)
                
                transit_ps = jnp.take_along_axis(joint_logprob_transit, ps_idx, axis=3)  # (B, C_prev, C_curr, 1, S_curr)
                transit_ps_cs = jnp.take_along_axis(transit_ps, cs_idx, axis=4)  # (B, C_prev, C_curr, 1, 1)
                tr_per_class = jnp.squeeze(transit_ps_cs, axis=(3,4)).transpose(1, 2, 0) #(C_prev, C_curr, B) 
                to_add = logsumexp(in_carry[:, None, :] + tr_per_class, axis=0) #(C_curr, B) # this line causes nan gradients
                
            return e + to_add #(T, C_curr, B) or (C_curr, B)
        
        def end(in_carry, ps, cs_not_used):
            # replace padding idx with 1 to prevent NaN gradients; this doesn't
            #   affect the actual calculated loglike
            ps = jnp.maximum(ps, 1)
            
            # if end, then curr_state = -1 (<end>)
            if not unique_time_per_sample:
                tr_per_class = joint_logprob_transit[..., -1, ps-1, -1] #(T, C_prev, B)    
            
            elif unique_time_per_sample:
                sliced = joint_logprob_transit[:, :, -1, :, -1]  # (B, C_prev, S_prev)
                ps_idx = (ps - 1)[:, None, None]  # (B, 1, 1)
                gathered = jnp.take_along_axis(sliced, ps_idx, axis=2)  # (B, C_prev, 1)
                tr_per_class = jnp.squeeze(gathered, axis=2).T  # (C_prev, B)
                 
            return tr_per_class + in_carry #(T, C, B) or (C, B)
        
        
        ### alpha update, in log space ONLY if curr_state is not pad
        new_alpha = jnp.where(curr_state != 0, 
                              jnp.where( curr_state != 4,
                                          main_body(prev_alpha, prev_state, curr_state),
                                          end(prev_alpha, prev_state, curr_state) ),
                              prev_alpha) #(T, C, B) or (C, B)
        
        return (new_alpha, new_alpha)
    
    ### end scan function definition, use scan
    # stacked_outputs is cumulative sum PER POSITION, PER TIME
    idx_arr = jnp.array( [ i for i in range(2, L_align) ] ) #(L_align)
    
    if not return_all_intermeds:
        last_alpha, _ = jax.lax.scan( f = scan_fn,
                                      init = init_alpha,
                                      xs = idx_arr,
                                      length = idx_arr.shape[0] )  #(T, C, B)  or (C, B)
        
        loglike = logsumexp(last_alpha,  # (T, C, B)  or (C, B)
                            axis = 1 if not unique_time_per_sample else 0)
        
        return loglike #(T, B)  or (B,)

        
    elif return_all_intermeds:
        _, stacked_outputs = jax.lax.scan( f = scan_fn,
                                            init = init_alpha,
                                            xs = idx_arr,
                                            length = idx_arr.shape[0] )  #(L_align-1, T, C, B)  or (L_align-1, C, B)
        
        # append the first return value (from sentinel -> first alignment column)
        stacked_outputs = jnp.concatenate( [ init_alpha[None,...], #(1, T, C, B) or (1, C, B)
                                             stacked_outputs ], #(L_align-1, T, C, B) or (L_align-1, C, B)
                                          axis=0) #(L_align, T, C, B) or (L_align, C, B)
        
        return stacked_outputs #(L_align, T, C, B) or or (L_align, C, B)
    
        
def _log_space_dot_prod_helper(alpha,
                              marginal_logprob_transit):
    """
    a helper used in all_loglikes_forward
    """
    alpha_reshaped = alpha[:,None,:] #(C_prev, 1, B)
    marginal_logprob_transit_reshaped = marginal_logprob_transit[...,0,0][...,None] #(C_prev, C_curr, 1)
    to_logsumexp = alpha_reshaped + marginal_logprob_transit_reshaped #(C_prev, C_curr, B)
    return logsumexp(to_logsumexp, axis=0) # (C_curr, B)


def all_loglikes_forward(aligned_inputs,
                         logprob_emit_at_indel,
                         joint_logprob_emit_at_match,
                         all_transit_matrices,
                         unique_time_per_sample: bool):
    """
    TODO: this should be ALMOST ready for nested TKF92 model... but come back  
          and check this out later
          > without domains: crude memory variables to remember if start -> emit
            has been seen yet; this handles alignments that start with a 
            start -> ins transitions
          > with domains: ??? 
    
    forward algo to find joint, conditional, and both single-sequence marginal 
        loglikeihoods
    
    IMPORANT: I never carry gradients through this!!!
    
    
    L_align: length of pairwise alignment
    T: number of timepoints
    B: batch size
    C_trans = C: number of latent site clases
      > could be C_frag or C_dom * C_frag
    A: alphabet (20 for proteins, 4 for DNA)
    S: possible states; here, this is 4: M, I, D, start/end
    
    Arguments
    ----------
    aligned_inputs : ArrayLike, (B, L, 3)
        dim2=0: ancestor
        dim2=1: descendant
        dim2=2: alignment state; M=1, I=2, D=3, S=4, E=5
    
    joint_logprob_emit_at_match : ArrayLike, (T, C, A, A)
        logP(anc, desc | c, t); log-probability of emission at match site
    
    logprob_emit_at_indel : ArrayLike, (C, A)
        logP(anc | c) or P(desc | c); log-equilibrium distribution
    
    all_transit_matrices : dict[ArrayLike]
        all_transit_matrices['joint'] : ArrayLike, (T, C, C, S, S)
            logP(new state, new class | prev state, prev class, t); the joint 
            transition matrix for finding logP(anc, desc, align | c, t)
        
        all_transit_matrices['marginal'] : ArrayLike, (C, C, 2, 2)
            logP(new state, new class | prev state, prev class, t); the marginal 
            transition matrix for finding logP(anc | c, t) or logP(desc | c, t)
    
    unique_time_per_sample : Bool 
        whether or not you have unqiue times per sample; affects indexing
        
    Returns:
    ---------
    
    """
    joint_logprob_transit = all_transit_matrices['joint']
    marginal_logprob_transit = all_transit_matrices['marginal'] 
    
    # decide which version of the functions you're going to use
    if not unique_time_per_sample:
        # output from this is (T, C, B)
        get_joint_loglike_emission = get_joint_loglike_emission_time_grid
        
    elif unique_time_per_sample:
        # output from this is (C, B)
        get_joint_loglike_emission = get_joint_loglike_emission_branch_len_per_samp
    
    B = aligned_inputs.shape[0]
    L_align = aligned_inputs.shape[1]
    C = logprob_emit_at_indel.shape[0]
    
    # memory for single-sequence marginals
    anc_alpha = jnp.zeros( (C, B) ) #(C, B)
    desc_alpha = jnp.zeros( (C, B) ) #(C, B)
    md_seen = jnp.zeros( B, ).astype(bool) #(B,)
    mi_seen = jnp.zeros( B, ).astype(bool) #(B,)
    
    ######################################################
    ### initialize with <start> -> any (curr pos is 1)   #
    ######################################################
    pos = 1
    anc_toks =   aligned_inputs[:, pos, 0] #(B,)
    desc_toks =  aligned_inputs[:, pos, 1] #(B,)
    curr_state = aligned_inputs[:, pos, 2] #(B,)

    
    ### joint: P(anc, desc, align)
    # emissions; 
    joint_e = get_joint_loglike_emission( aligned_inputs=aligned_inputs,
                                    pos=pos,
                                    joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                    logprob_emit_at_indel=logprob_emit_at_indel ) # (T, C, B)
    
    # transitions; assume there's never start -> end; 
    # joint_logprob_transit is (T, C_prev, C_curr, S_prev, S_curr)
    # initial state is 4 (<start>); take the last row
    # use C_prev=0 for start class (but it doesn't matter, because the 
    # transition probability is the same for all C_prev)
    curr_state_idx = curr_state - 1         # (B,)
    start_any = joint_logprob_transit[:, 0, :, -1, :] #(T, C_curr, S_curr) or (B, C_curr, S_curr)
    
    if not unique_time_per_sample:
        joint_tr = start_any[...,curr_state-1] #(T, C_curr, B)
    
    elif unique_time_per_sample:
        # joint_logprob_transit: (B, C_curr, S_curr)
        # goal: (C_curr, B)
        joint_tr = jnp.take_along_axis(
            start_any, 
            curr_state_idx[:, None, None],  # shape (B, 1)
            axis=-1
        ) 
        joint_tr = jnp.squeeze(joint_tr).T# (C_curr, B)
    
    # carry value
    init_joint_alpha = joint_e + joint_tr # (T, C, B) or (C, B)
    del joint_e, joint_tr, start_any
    
    
    ### logP(anc)
    # emissions; only valid if current position is match or delete
    anc_mask = (curr_state == 1) | (curr_state == 3)  # (B,)
    init_anc_e = logprob_emit_at_indel[:, anc_toks - 3] * anc_mask  # (C, B)
    
    # transitions
    # marginal_logprob_transit is (C_prev, C_curr, S_prev, S_curr), where:
    #   (S_prev=0, S_curr=0) is emit->emit
    #   (S_prev=1, S_curr=0) is <s>->emit
    #   (S_prev=0, S_curr=1) is emit-><e>
    # use C_prev=0 for start class (but it doesn't matter, because the 
    # transition probability is the same for all C_prev)
    # transition prob for <s>->emit
    first_anc_emission_flag = (~md_seen) & anc_mask  # (B,)
    anc_first_tr = marginal_logprob_transit[0,:,1,0][...,None] #(C_curr, 1)
    
    # transition prob for emit->emit
    continued_anc_emission_flag = md_seen & anc_mask  # (B,)
    anc_cont_tr = _log_space_dot_prod_helper(alpha = anc_alpha,
                                            marginal_logprob_transit = marginal_logprob_transit)  # (C_curr, B)
    
    # possibilities are: <s>->emit transition, emit->emit transition, or  
    #   nothing happened (at an indel site where ancestor was not emitted yet)
    init_anc_tr = ( anc_cont_tr * continued_anc_emission_flag +
                    anc_first_tr * first_anc_emission_flag ) # (C, B)
    
    # things to remember are:
    #   alpha: for forward algo
    #   md_seen: used to remember if <s> -> emit has been used yet
    #   (there could be gaps in between <s> and first emission)
    init_anc_alpha = init_anc_e + init_anc_tr # (C, B)
    del init_anc_e, init_anc_tr, anc_mask
    
    
    ### logP(desc); (C, B)
    # emissions; only valid if current position is match or ins
    desc_mask = (curr_state == 1) | (curr_state == 2) #(B,)
    init_desc_e = logprob_emit_at_indel[:, desc_toks - 3] * desc_mask # (C, B)
    
    # transitions
    first_desc_emission_flag = (~mi_seen) & desc_mask # (B,)
    desc_first_tr = marginal_logprob_transit[0,:,1,0][...,None] #(C_curr, 1)
    
    continued_desc_emission_flag = mi_seen & desc_mask # (B,)
    desc_cont_tr = _log_space_dot_prod_helper(alpha = desc_alpha,
                                             marginal_logprob_transit = marginal_logprob_transit)  # (C_curr, B)
    
    init_desc_tr = ( desc_cont_tr * continued_desc_emission_flag +
                     desc_first_tr * first_desc_emission_flag ) # (C, B)
    
    # things to remember are:
    #   alpha: for forward algo
    #   mi_seen: used to remember if <s> -> emit has been used yet
    #   (there could be gaps in between <s> and first emission)
    init_desc_alpha = init_desc_e + init_desc_tr  # (C, B)
    del init_desc_e, init_desc_tr, desc_mask, curr_state
    
    init_dict = {'joint_alpha': init_joint_alpha, # (T, C, B) or (C, B)
                 'anc_alpha': init_anc_alpha,  # (C, B)
                 'desc_alpha': init_desc_alpha,  # (C, B),
                 'md_seen': first_anc_emission_flag, # (B,)
                 'mi_seen': first_desc_emission_flag} # (B,)
    
    
    ######################################################
    ### scan down length dimension to end of alignment   #
    ######################################################
    def scan_fn(carry_dict, pos):
        ### unpack 
        # carry dict
        prev_joint_alpha = carry_dict['joint_alpha'] #(T, C, B) or (C, B)
        prev_anc_alpha = carry_dict['anc_alpha'] #(C, B)
        prev_desc_alpha = carry_dict['desc_alpha'] #(C, B)
        prev_md_seen = carry_dict['md_seen'] #(B,)
        prev_mi_seen = carry_dict['mi_seen'] #(B,)
        
        # batch
        anc_toks =   aligned_inputs[:,   pos, 0] #(B,)
        desc_toks =  aligned_inputs[:,   pos, 1] #(B,)
        prev_state = aligned_inputs[:, pos-1, 2] #(B,)
        curr_state = aligned_inputs[:,   pos, 2] #(B,)
        curr_state = jnp.where( curr_state!=5, curr_state, 4 ) #(B,)
        
        
        ### emissions
        joint_e = get_joint_loglike_emission( aligned_inputs=aligned_inputs,
                                              pos=pos,
                                              joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                              logprob_emit_at_indel=logprob_emit_at_indel ) #(T, C, B)
        
        anc_mask = (curr_state == 1) | (curr_state == 3) #(B,)
        anc_e = logprob_emit_at_indel[:, anc_toks - 3] * anc_mask  #(C,B)

        desc_mask = (curr_state == 1) | (curr_state == 2)  #(B,)
        desc_e = logprob_emit_at_indel[:, desc_toks - 3] * desc_mask #(C,B)
        
        
        ### flags needed for transitions
        # first_emission_flag: is the current position <s> -> emit?
        # continued_emission_flag: is the current postion emit -> emit?
        # need these because gaps happen in between single sequence 
        #   emissions...
        first_anc_emission_flag = (~prev_md_seen) & anc_mask  #(B,)
        continued_anc_emission_flag = prev_md_seen & anc_mask  #(B,)
        first_desc_emission_flag = (~prev_mi_seen) & desc_mask  #(B,)
        continued_desc_emission_flag = (prev_mi_seen) & desc_mask  #(B,)
        
        
        ### transition probabilities
        def main_body(joint_carry, anc_carry, desc_carry):
            # logP(anc, desc, align)
            if not unique_time_per_sample:
                # joint_logprob_transit is (T, C_prev, C_curr, S_prev, S_curr)
                joint_tr_per_class = joint_logprob_transit[..., prev_state-1, curr_state-1] #(T, C_prev, C_curr, B)   
                to_add = logsumexp(joint_carry[:, :, None, :] + joint_tr_per_class, axis=1) #(T, C_curr, B)
            
            elif unique_time_per_sample:
                # joint_logprob_transit is (B, C_prev, C_curr, S_prev, S_curr)
                ps_idx = (prev_state - 1)[:, None, None, None, None] #(B, 1, 1, 1, 1)
                cs_idx = (curr_state - 1)[:, None, None, None, None] #(B, 1, 1, 1, 1)
                transit_ps = jnp.take_along_axis(joint_logprob_transit, ps_idx, axis=3)  # (B, C_prev, C_curr, 1, S_curr)
                transit_ps_cs = jnp.take_along_axis(transit_ps, cs_idx, axis=4)  # (B, C_prev, C_curr, 1, 1)
                joint_tr_per_class = jnp.squeeze(transit_ps_cs, axis=(3,4)).transpose(1, 2, 0) #(C_prev, C_curr, B) 
                to_add = logsumexp(joint_carry[:, None, :] + joint_tr_per_class, axis=0) #(C_curr, B)
                         
            joint_out = joint_e + to_add #(T, C_curr, B) or (C_curr, B)
            
            
            # logP(anc)
            anc_first_tr = marginal_logprob_transit[0,:,1,0][...,None] #(C_curr, 1)
            anc_cont_tr = _log_space_dot_prod_helper(alpha = anc_carry,
                                                    marginal_logprob_transit = marginal_logprob_transit) #(C_curr, B)
            anc_tr = ( anc_cont_tr * continued_anc_emission_flag +
                       anc_first_tr * first_anc_emission_flag ) # (C_curr, B)
            anc_out = anc_e + anc_tr # (C, B)
            
            
            # logP(desc)
            desc_first_tr = marginal_logprob_transit[0,:,1,0][...,None] #(C_curr, 1)
            desc_cont_tr = _log_space_dot_prod_helper(alpha = desc_carry,
                                                    marginal_logprob_transit = marginal_logprob_transit) #(C_curr, B)
            desc_tr = ( desc_cont_tr * continued_desc_emission_flag +
                        desc_first_tr * first_desc_emission_flag ) # (C_curr, B)
            desc_out = desc_e + desc_tr # (C, B)
            
            return (joint_out, anc_out, desc_out)
        
        def end(joint_carry, anc_carry, desc_carry):
            # note for all: if end, then curr_state = -1 (<end>)
            # logP(anc, desc, align)
            if not unique_time_per_sample:
                joint_tr_per_class = joint_logprob_transit[..., -1, prev_state-1, -1] #(T, C_prev, B)    
            
            elif unique_time_per_sample:
                sliced = joint_logprob_transit[:, :, -1, :, -1]  # (B, C_prev, S_prev)
                ps_idx = (prev_state - 1)[:, None]  # (B, 1)
                gathered = jnp.take_along_axis(sliced, ps_idx[:, None, :], axis=2)  # (B, C_prev, 1)
                joint_tr_per_class = jnp.squeeze(gathered, axis=2).T  # (C_prev, B)
            
            joint_out = joint_tr_per_class + joint_carry #(T,C,B) or (C,B)
            
            
            # logP(anc)
            final_anc_tr = marginal_logprob_transit[:,-1,0,1] #(C,)
            final_anc_tr = jnp.broadcast_to( final_anc_tr[:,None], anc_carry.shape ) #(C, B)
            anc_out = anc_carry + final_anc_tr #(C, B)
            
            
            # logP(desc)
            final_desc_tr = marginal_logprob_transit[:,-1,0,1] #(C,)
            final_desc_tr = jnp.broadcast_to( final_desc_tr[:,None], desc_carry.shape ) #(C,B)
            desc_out = desc_carry + final_desc_tr #(C,B)
            
            return (joint_out, anc_out, desc_out)
        
        
        ### alpha updates, in log space 
        continued_out = main_body( prev_joint_alpha, 
                                   prev_anc_alpha, 
                                   prev_desc_alpha )
        end_out = end( prev_joint_alpha, 
                       prev_anc_alpha, 
                       prev_desc_alpha )
        
        # joint: update ONLY if curr_state is not pad
        new_joint_alpha = jnp.where( curr_state != 0,
                                     jnp.where( curr_state != 4,
                                                continued_out[0],
                                                end_out[0] ),
                                     prev_joint_alpha )
        
        # anc marginal; update ONLY if curr_state is not pad or ins
        new_anc_alpha = jnp.where( (curr_state != 0) & (curr_state != 2),
                                     jnp.where( curr_state != 4,
                                                continued_out[1],
                                                end_out[1] ),
                                     prev_anc_alpha )
        
        # desc margianl; update ONLY if curr_state is not pad or del
        new_desc_alpha = jnp.where( (curr_state != 0) & (curr_state != 3),
                                    jnp.where( curr_state != 4,
                                               continued_out[2],
                                               end_out[2] ),
                                    prev_desc_alpha )
        
        out_dict = { 'joint_alpha': new_joint_alpha, #(T, C, B) or (C, B)
                     'anc_alpha': new_anc_alpha, # (C, B)
                     'desc_alpha': new_desc_alpha, # (C, B)
                     'md_seen': (first_anc_emission_flag + prev_md_seen).astype(bool), #(B,)
                     'mi_seen': (first_desc_emission_flag + prev_mi_seen).astype(bool) } #(B,)
        
        return (out_dict, None)

    ### scan over remaining length
    idx_arr = jnp.array( [i for i in range(2, L_align)] )
    out_dict, _ = jax.lax.scan( f = scan_fn,
                                               init = init_dict,
                                               xs = idx_arr,
                                               length = idx_arr.shape[0] )
    final_joint_alpha = out_dict['joint_alpha'] #(T, C, B) or #(C, B)
    joint_neg_logP = -logsumexp(final_joint_alpha, 
                                axis = 1 if not unique_time_per_sample else 0) #(T, B) or (B,)
    
    final_anc_alpha = out_dict['anc_alpha'] #(C, B)
    anc_neg_logP = -logsumexp(final_anc_alpha, axis=0) # (B,)
    
    final_desc_alpha = out_dict['desc_alpha'] #(C, B)
    desc_neg_logP = -logsumexp(final_desc_alpha, axis=0) # (B,)
    
    loglike_dict = {'joint_neg_logP': joint_neg_logP,  #(T, B) or (B,)
                    'anc_neg_logP': anc_neg_logP, # (B,)
                    'desc_neg_logP': desc_neg_logP} # (B,)
    
    return loglike_dict


###############################################################################
### helpers for reporting   ###################################################
###############################################################################
def write_matrix_to_npy(out_folder,
                        mat,
                        key):
    with open(f'{out_folder}/PARAMS-MAT_{key}.npy', 'wb') as g:
        np.save( g, mat )

def maybe_write_matrix_to_ascii(out_folder,
                                mat,
                                key):
    mat = jnp.squeeze(mat)
    if len(mat.shape) <= 2:
        np.savetxt( f'{out_folder}/ASCII_{key}.tsv', 
                    np.array(mat), 
                    fmt = '%.8f',
                    delimiter= '\t' )
