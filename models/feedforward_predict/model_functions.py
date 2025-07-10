#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 19:45:49 2025

@author: annabel
"""
import jax
from jax import numpy as jnp


def confusion_matrix( true, pred, mask, output_alph_with_pad ):
    B = true.shape[0]
    L = true.shape[1]
    A = output_alph_with_pad
    
    # flatten
    true_flat = true.reshape(-1) #(B * L)
    pred_flat = pred.reshape(-1) #(B * L)
    mask_flat = mask.reshape(-1) #(B * L)

    # mask
    batch_indices = jnp.repeat(jnp.arange(B), L) #(B * L)
    true_flat = jnp.where(mask_flat, true_flat, 0) #(B * L)
    pred_flat = jnp.where(mask_flat, pred_flat, 0) #(B * L)
    batch_indices = jnp.where(mask_flat, batch_indices, 0) #(B * L)

    combined_idx = A * true_flat + pred_flat #(B * L)
    combined_idx = A * A * batch_indices + combined_idx #(B * L)
    weights = mask_flat.astype(jnp.int32) #(B * L)

    counts = jnp.bincount(combined_idx, 
                          weights=weights, 
                          minlength=B * A * A,
                          length= B * A * A) #(B * A * A)
    cm = counts.reshape(B, A, A) #(B, A, A)
    return cm[:, 1:, 1:] #(B, A-1, A-1)
