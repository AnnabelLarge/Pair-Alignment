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
from models.sequence_embedders.cnn.blocks_fns import FakeConvnetBlock


H = 10
causal = False

### fake seqs
seqs = jnp.array( [[1, 3, 4, 5, 2, 0],
                   [1, 6, 6, 6, 6, 2]] )
mask = (seqs != 0 )
seqs = jnp.array( nn.one_hot( seqs, num_classes=7 ) )
mask_exp = jnp.broadcast_to(mask[...,None], seqs.shape)
seqs = jnp.multiply(seqs, mask_exp)
seqs = jnp.pad(seqs,
               ( (0,0),
                 (0,0),
                 (0, H-seqs.shape[-1]) )
               )


### fake layer
layer = FakeConvnetBlock( config = {'hidden_dim': H},
                                kern_size = 4,
                                causal = True,
                                name = 'fake_conv' )

dummy_params = layer.init( rngs=jax.random.key(0),
                           datamat = jnp.zeros(seqs.shape),
                           padding_mask = mask,
                           sow_intermediates = False,
                           training = False )

out = layer.apply(variables = dummy_params,
                  datamat = seqs,
                  padding_mask = mask)


### check that input == output
assert jnp.allclose(out, seqs)


