#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 21:41:50 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
import numpy as np

def bounded_sigmoid(x, min_val, max_val, *args, **kwargs):
    return min_val + (max_val - min_val) / (1 + jnp.exp(-x))
val_grad_fn = jax.value_and_grad(bounded_sigmoid)


lam_val, lam_grad = val_grad_fn(-2., 
                                min_val = 1e-4, 
                                max_val = 1)
lam_val = lam_val.item()
lam_grad = lam_grad.item()


err_val, err_grad = val_grad_fn(-5., 
                                min_val = 1e-4, 
                                max_val = 0.333)
err_val = err_val.item()
err_grad = err_grad.item()


r_val = bounded_sigmoid(jnp.array([-i/10 for i in range(1, 11)]), 
                                min_val = 1e-4, 
                                max_val = 0.999)
r_val = np.array(r_val)


# r_val = bounded_sigmoid(jnp.array([-5, -4, -3, -2, -1]), 
#                                 min_val = 1e-4, 
#                                 max_val = 0.999)
# r_val = np.array(r_val)
