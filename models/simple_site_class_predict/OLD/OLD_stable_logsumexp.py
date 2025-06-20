#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 20:22:09 2025

@author: annabel
"""

import jax
import jax.numpy as jnp

@jax.custom_vjp
def stable_logsumexp(x):
    axis=0
    x_max = jnp.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    sum_exp = jnp.sum(jnp.exp(shifted), axis=axis, keepdims=True)
    lse = jnp.log(sum_exp) + x_max
    return jnp.squeeze(lse, axis=axis)

def fwd(x):
    lse = stable_logsumexp(x)
    return lse, (x, lse)

def bwd(res, g):
    axis=0
    x, lse = res
    x_max = jnp.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    exp_shifted = jnp.exp(shifted)
    sum_exp = jnp.sum(exp_shifted, axis=axis, keepdims=True)

    # If sum_exp == 0, set gradient to zero (prevent 0/0 NaNs)
    safe_sum_exp = jnp.where(sum_exp == 0, 1.0, sum_exp)
    softmax = exp_shifted / safe_sum_exp

    grad_x = softmax * jnp.expand_dims(g, axis) 
    grad_x = jnp.where(sum_exp == 0, 0.0, grad_x)
    return (grad_x,)

stable_logsumexp.defvjp(fwd, bwd)

