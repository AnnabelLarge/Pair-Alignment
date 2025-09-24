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

import numpy as np
import numpy.testing as npt
import unittest

from models.BaseClasses import ModuleBase
from models.sequence_embedders.initial_embedding_blocks import FakeEmbeddingWithPadding

class TestFakeNNEmbed(unittest.TestCase):
    def setUp(self):
        H = 10
        
        # fake seqs
        seqs = jnp.array( [[1, 3, 4, 5, 2, 0],
                           [1, 6, 6, 6, 6, 2]] )

        # true one-hot encoding
        true_mask = (seqs != 0 )
        true = jnp.array( nn.one_hot( seqs, num_classes=7 ) )
        mask_exp = jnp.broadcast_to(true_mask[...,None], true.shape)
        true = jnp.multiply(true, mask_exp)
        true = jnp.pad(true,
                       ( (0,0),
                         (0,0),
                         (0, H-true.shape[-1]) )
                       )
        
        self.H = H
        self.true = true
        self.seqs = seqs
        
        
    def test_equal(self):
        H = self.H
        seqs = self.seqs
        true = self.true
        
        layer = FakeEmbeddingWithPadding( config = {'hidden_dim': H,
                                                    'in_alph_size': 7},
                                         name = 'fake_conv' )
        
        dummy_params = layer.init( rngs=jax.random.key(0),
                                   datamat = jnp.zeros(seqs.shape, dtype=int) )
        
        pred, _ = layer.apply(variables = dummy_params,
                              datamat = seqs )
        
        npt.assert_allclose(true, pred)


if __name__ == '__main__':
    unittest.main()
