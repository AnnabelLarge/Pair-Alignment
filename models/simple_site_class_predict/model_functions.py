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
 'CondTransitionLogprobs',
 'MargTKF91TransitionLogprobs',
 'MargTKF92TransitionLogprobs'
 'anc_marginal_probs_from_counts',
 'desc_marginal_probs_from_counts',
 'get_cond_logprob_emit_at_match_per_class',
 'get_joint_logprob_emit_at_match_per_class',
 'joint_prob_from_counts',
 'lse_over_equl_logprobs_per_class',
 'lse_over_match_logprobs_per_class',
 'marginalize_over_times',
 'rate_matrix_from_exch_equl',
 scale_rate_multipliers,
 'scale_rate_matrix',
 'upper_tri_vector_to_sym_matrix'


internal:
---------
'_selectively_add_time_dim'
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
                                norm: bool=True):
    """
    computes rate matrix Q = \chi * \pi_c; normalizes to substution 
      rate of one if desired
    
    only one exchangeability; rho and pi are properties of the class
    
    C = number of latent site classes
    A = alphabet size
    
    
    Arguments
    ----------
    exchangeabilities : ArrayLike, (A, A)
        symmetric exchangeability parameter matrix
        
    equilibrium_distributions : ArrayLike, (C, A)
        amino acid equilibriums per site
    
    norm : bool, optional; default is True

    Returns
    -------
    subst_rate_mat : ArrayLike, (C, A, A)
        rate matrix Q, for every class

    """
    C = equilibrium_distributions.shape[0]
    A = equilibrium_distributions.shape[1]

    # just in case, zero out the diagonals of exchangeabilities
    exchangeabilities_without_diags = exchangeabilities * ~jnp.eye(A, dtype=bool)

    # Q = chi * diag(pi); q_ij = chi_ij * pi_j
    rate_mat_without_diags = jnp.einsum('ij, cj -> cij', 
                                        exchangeabilities_without_diags, 
                                        equilibrium_distributions)   # (C, A, A)
    
    # put the row sums in the diagonals
    row_sums = rate_mat_without_diags.sum(axis=2)  # (C, A)
    ones_diag = jnp.eye( A, dtype=bool )[None,:,:]   # (1, A, A)
    ones_diag = jnp.broadcast_to( ones_diag, (C,
                                              ones_diag.shape[1],
                                              ones_diag.shape[2]) )
    diags_to_add = -jnp.einsum('ci,cij->cij', row_sums, ones_diag)  #(C, A, A)
    subst_rate_mat = rate_mat_without_diags + diags_to_add  #(C, A, A)
    
    # normalize (true by default)
    if norm:
        diag = jnp.einsum("cii->ci", subst_rate_mat)  # (C, A)
        norm_factor = -jnp.sum(diag * equilibrium_distributions, axis=1)[:,None,None]  #(C, 1, 1)
        subst_rate_mat = subst_rate_mat / norm_factor  # (C, A, A)
    
    return subst_rate_mat

def scale_rate_multipliers( unnormed_rate_multipliers: ArrayLike,
                            log_class_probs: ArrayLike ):
    """
    scale rate multipliers such rate sum_c rho_c * P(c) = 1
    
    C = number of latent site classes
    A = alphabet size
    
    
    Arguments
    ----------
    log_class_probs : ArrayLike, (C,)
        log-probability per class
    
    unnormed_rate_multipliers : ArrayLike, (C,)


    Returns
    -------
    ArrayLike, (C, )
        scaled rate multipliers

    """
    class_probs = jnp.exp(log_class_probs) #(C,)
    norm_factor = jnp.multiply(unnormed_rate_multipliers, class_probs) #one float
    norm_factor = norm_factor.sum()  #one float
    return unnormed_rate_multipliers / norm_factor #(C,)
    

def scale_rate_matrix(subst_rate_mat: ArrayLike,
                       rate_multiplier: ArrayLike):
    """
    Scale Q by rate multipliers, rho
    
    C = number of latent site classes
    A = alphabet size
    
    
    Arguments
    ----------
    subst_rate_mat : ArrayLike, (C, A, A)
    
    rate_multiplier : ArrayLike, (C,)

    Returns
    -------
    scaled rate matrix : ArrayLike, (C, A, A)

    """
    return jnp.einsum( 'c,cij->cij', 
                       rate_multiplier, 
                       subst_rate_mat )


###############################################################################
### functions used to calculate scoring matrix for substitution sites   #######
###############################################################################
def get_cond_logprob_emit_at_match_per_class( t_array: ArrayLike,
                                              scaled_rate_mat_per_class: ArrayLike):
    """
    P(y|x,c,t) = expm( rho_c * Q_c * t )

    C = number of latent site classes
    A = alphabet size
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    

    Arguments
    ----------
    t_array : ArrayLike, (T,)
        branch lengths
        
    scaled_rate_mat_per_class : ArrayLike, (C, A, A)
        rho_c * Q_c

    Returns
    -------
    to_expm : ArrayLike, (T, C, A, A)
        scaled rate matrix * t, for all classes, this is the input for the 
        matrix exponential function
        
    cond_logprob_emit_at_match_per_class :  ArrayLike, (T, C, A, A)
        final log-probability

    """
    to_expm = jnp.multiply( scaled_rate_mat_per_class[None,...],
                            t_array[:, None,None,None,] ) #(T, C, A, A)
    cond_prob_emit_at_match_per_class = expm(to_expm) #(T, C, A, A)
    cond_logprob_emit_at_match_per_class = safe_log( cond_prob_emit_at_match_per_class )  #(T, C, A, A)
    return cond_logprob_emit_at_match_per_class, to_expm


def get_joint_logprob_emit_at_match_per_class( cond_logprob_emit_at_match_per_class: ArrayLike,
                                              log_equl_dist_per_class: ArrayLike ):
    """
    P(x,y|c,t) = pi_c * expm( rho_c * Q_c * t )

    C = number of latent site classes
    A = alphabet size
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    

    Arguments
    ----------
    cond_logprob_emit_at_match_per_class : ArrayLike, (T, C, A, A)
        P(y|x,c,t), calculated before
    
    log_equl_dist_per_class : ArrayLike, (C, A, A)
        rho_c * Q_c

    Returns
    -------
    ArrayLike, (T, C, A, A)

    """
    return ( cond_logprob_emit_at_match_per_class +
             log_equl_dist_per_class[None,:,:,None] ) #(T, C, A, A)



###############################################################################
### for indp site classes, functions to marginalize over possible classes   ###
###############################################################################
def lse_over_match_logprobs_per_class(log_class_probs: ArrayLike,
                                      joint_logprob_emit_at_match_per_class: ArrayLike):
    """
    For indp sites emission model
    
    P(x,y|t) = \sum_c P(c) * P(x,y|c,t)
    
    C = number of latent site classes
    A = alphabet size
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    

    Arguments
    ----------
    log_class_probs : ArrayLike, (C,)
        log-transformed class probabilities (i.e. mixture weights)
    
    joint_logprob_emit_at_match_per_class : ArrayLike, (T, C, A, A)
        log-probability of emissions at match sites
        
    Returns
    -------
    ArrayLike, (T, A, A)

    """
    weighted_logprobs = ( log_class_probs[None,:,None,None] + 
                          joint_logprob_emit_at_match_per_class ) #(T, C, A, A)
    return logsumexp( weighted_logprobs, axis=1 ) #(T, A, A)


def lse_over_equl_logprobs_per_class(log_class_probs: ArrayLike,
                                     log_equl_dist_per_class: ArrayLike):
    """
    For indp sites emission model
    
    P(x) = \sum_c P(c) * P(x|c)
    
    C = number of latent site classes
    A = alphabet size
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    
    
    Arguments
    ----------
    log_class_probs : ArrayLike, (C,)
        log-transformed class probabilities (i.e. mixture weights)
    
    log_equl_dist_per_class : ArrayLike, (A,)
        log-transformed equilibrium distributions
        
    Returns
    -------
    ArrayLike, (T, A)
    
    """
    weighted_logprobs = log_equl_dist_per_class + log_class_probs[:, None] #(C, A)
    return logsumexp( weighted_logprobs, axis=0 )



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
    log_denom = logsumexp_with_arr_lst( [mu*offset*t, jnp.log1p(-offset)],
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

def stable_tkf( mu, offset, t_array ):
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
        
        approx_flags_dict = {'log_one_minus_alpha': ~(log_alpha_at_t < -SMALL_POSITIVE_NUM),
                             'log_beta': ~(mu*offset*t > SMALL_POSITIVE_NUM),
                             'log_one_minus_gamma': ~use_real_function,
                             'log_gamma': ~(log_one_minus_gamma < -SMALL_POSITIVE_NUM)}
        
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



###############################################################################
### for tkf91, tkf92: functions to get marginal and    ########################
### conditional transition matrices                    ########################
###############################################################################
def MargTKF91TransitionLogprobs(offset,
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


def MargTKF92TransitionLogprobs(offset,
                                class_probs,
                                r_ext_prob,
                                **kwargs):
    """
    For scoring single-sequence marginals under TKF92 model
    
    C = site classes
        
    
    Arguments
    ----------
    offset : ArrayLike, ()
        1 - (lam/mu)
    
    class_probs : ArrayLike, (C,)
        probability of being in latent site classes
    
    r_ext_prob : ArrayLike, (C,)
        TKF92 fragment extension probability per latent site class
    
        
    Returns
    -------
    log_arr : ArrayLike, (C,C,2,2)
        
        emit -> emit   |  emit -> end
        -------------------------------
        start -> emit  |  start -> end
        
    """
    C = class_probs.shape[-1]
    
    ### move values to log space
    log_class_prob = safe_log(class_probs) #(C,)
    log_r_ext_prob = safe_log(r_ext_prob) #(C,)
    log_one_minus_r = log_one_minus_x(log_r_ext_prob) #(C,)
    
    # lam / mu = 1 - offset
    # offset = 1 - (lam/mu)
    log_lam_div_mu = jnp.log1p(-offset) #float
    log_one_minus_lam_div_mu = jnp.log(offset) #float
    
    
    ### build cells
    # cell 1: emit -> emit 
    log_cell1 = (log_one_minus_r + log_lam_div_mu)[:, None] + log_class_prob[None, :] # (C,C)
    
    # cell 2: emit -> end 
    log_cell2 = ( log_one_minus_r + log_one_minus_lam_div_mu )[:,None] #(1,C)
    log_cell2 = jnp.broadcast_to( log_cell2, (C, C) ) # (C,C)
    
    # cell 3: start -> emit
    log_cell3 = ( log_lam_div_mu + log_class_prob )[None,:] # (1,C)
    log_cell3 = jnp.broadcast_to( log_cell3, (C, C) )   # (C,C)
    
    # cell 4: start -> end
    log_cell4 = jnp.broadcast_to( log_one_minus_lam_div_mu, (C,C) )  # (C,C)
    

    ### build matrix
    log_single_seq_tkf92 = jnp.stack( [jnp.stack( [log_cell1, log_cell2], axis=-1 ),
                                       jnp.stack( [log_cell3, log_cell4], axis=-1 )],
                                     axis = -2 ) #(C,C,2,2)
    
    # add fragment extension probability to transitions between same class
    i_idx = jnp.arange(C)
    prev_vals = log_single_seq_tkf92[i_idx, i_idx, 0, 0] #(C,C)
    new_vals = logsumexp_with_arr_lst([log_r_ext_prob, prev_vals]) #(C,C)
    log_single_seq_tkf92 = log_single_seq_tkf92.at[i_idx, i_idx, 0, 0].set(new_vals) #(C,C,2,2)
    
    return log_single_seq_tkf92 #(C,C,2,2)


def CondTransitionLogprobs(marg_matrix, 
                           joint_matrix):
    """
    obtain the conditional log probability by composing the joint with the marginal
    
    C = site classes
    S = number of states; 4 for {Match, Ins, Del, Start/End}
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    
    Arguments
    ----------
    marg_matrix : ArrayLike, (C,C,2,2) or (2,2)
        scoring matrix for marginal transition probabilities
        P(seq)
    
    joint_matrix : ArrayLike, (T,C,C,S,S) or (T,S,S)
        scoring matrix for joint transition probabilities
        P(desc, anc, align | t)
        
    Returns
    -------
    cond_matrix : ArrayLike, (T,C,C,S,S) or (T,S,S)
        scoring matrix for conditional transition probabilities
        P(desc, align | anc, t)
        
    """
    # cond_matrix is always  #(T,C,C,2,2)
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
                            norm_loss_by: str or None ):
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
    
    norm_loss_by : {desc_len, align_len}
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
    emission_score = time_dep_score_fn(scoring_matrices_dict['joint_logprob_emit_at_match'], 
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
        emission_score = ( emission_score +
                           ins_emit_score +
                           del_emit_score )
            
        
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
    if (norm_loss_by == 'desc_len') and score_indels:
        length_for_normalization = ( subCounts.sum(axis=(-2, -1)) + 
                                     insCounts.sum(axis=(-1))
                                     )
    
    elif (norm_loss_by == 'align_len') and score_indels:
        length_for_normalization = ( subCounts.sum(axis=(-2, -1)) + 
                                     insCounts.sum(axis=(-1)) + 
                                     delCounts.sum(axis=(-1))
                                     ) 
    elif not score_indels:
        length_for_normalization = subCounts.sum(axis=(-2, -1))
    
    joint_neg_logP_length_normed = joint_neg_logP / length_for_normalization #(B,)
    
    return {'joint_neg_logP': joint_neg_logP,  #(B,)
            'joint_neg_logP_length_normed': joint_neg_logP_length_normed,
            'align_length_for_normalization': length_for_normalization}  #(B,)


def anc_marginal_probs_from_counts( batch: tuple[ArrayLike],
                                    score_indels: bool,
                                    scoring_matrices_dict: dict):
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
        
    norm_loss_by : {desc_len, align_len}
        how to normalize the loss, if including indels (if not scoring indels, 
        normalization length is already decided)
        
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
    
    
    return {'anc_neg_logP': anc_neg_logP,  #(B,)
            'anc_neg_logP_length_normed': anc_neg_logP_length_normed}  #(B,)


def desc_marginal_probs_from_counts( batch: tuple[ArrayLike],
                                     score_indels: bool,
                                     scoring_matrices_dict: dict):
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
        
    norm_loss_by : {desc_len, align_len}
        how to normalize the loss, if including indels (if not scoring indels, 
        normalization length is already decided)
        
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
