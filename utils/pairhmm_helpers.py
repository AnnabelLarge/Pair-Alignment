#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 14:43:01 2025

@author: annabel
"""
from jax import numpy as jnp
from jax.scipy.special import logsumexp



def safe_log(x):
    return jnp.log( jnp.where( x>0, 
                               x, 
                               jnp.finfo('float32').smallest_normal ) )

def bounded_sigmoid(x, min_val, max_val):
    return min_val + (max_val - min_val) / (1 + jnp.exp(-x))

def concat_along_new_last_axis(arr_lst):
    return jnp.concatenate( [arr[...,None] for arr in arr_lst], 
                             axis = -1 )

def logsumexp_with_arr_lst(arr_lst, coeffs = None):
    """
    concatenate a list of arrays, then use logsumexp
    """
    a_for_logsumexp = concat_along_new_last_axis(arr_lst)
    
    out = logsumexp(a = a_for_logsumexp,
                    b = coeffs,
                    axis=-1)
    return out

def log_one_minus_x(x):
    """
    calculate log( exp(log(1)) - exp(log(x)) ),
      which is log( 1 - x )
    """
    a_for_logsumexp = concat_along_new_last_axis( [jnp.zeros(x.shape), x] )
    b_for_logsumexp = jnp.array([1.0, -1.0])
    out = logsumexp(a = a_for_logsumexp,
                    b = b_for_logsumexp,
                    axis = -1)
    
    return out

