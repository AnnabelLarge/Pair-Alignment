#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 17:25:00 2025

@author: annabel
"""
from typing import Callable

import flax
from flax import linen as nn
import jax
import jax.numpy as jnp

from models.BaseClasses import ModuleBase
from models.sequence_embedders.initial_embedding_blocks import FakeEmbeddingWithPadding


H = 10
causal = False

### fake seqs
seqs = jnp.array( [[1, 3, 4, 5, 2, 0],
                   [1, 6, 6, 6, 6, 2]] )


### true one-hot encoding
true_mask = (seqs != 0 )
true = jnp.array( nn.one_hot( seqs, num_classes=7 ) )
mask_exp = jnp.broadcast_to(mask[...,None], true.shape)
true = jnp.multiply(true, mask_exp)
true = jnp.pad(true,
               ( (0,0),
                 (0,0),
                 (0, H-true.shape[-1]) )
               )


### with fake layer
layer = FakeEmbeddingWithPadding( config = {'hidden_dim': H,
                                            'in_alph_size': 7},
                                 name = 'fake_conv' )

dummy_params = layer.init( rngs=jax.random.key(0),
                           datamat = jnp.zeros(seqs.shape, dtype=int) )

pred, _ = layer.apply(variables = dummy_params,
                      datamat = seqs )


### check values
assert jnp.allclose(true, pred)


