#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:38:11 2024

@author: annabel
"""
from collections import OrderedDict


def create_config(block_name):
    if block_name in ['PlaceholderEmbedding', 
                      'EmbeddingWithPadding']:
        return OrderedDict({"hidden_dim": "[INT]"})
    
    
    elif block_name == 'TAPEEmbedding':
        return OrderedDict({"hidden_dim": "[INT]",
                            "max_len": "[INT=3000]",
                            "dropout": "[FLOAT=0.0]"})
    
    
    if block_name == 'ConvEmbedding':
        return OrderedDict({"hidden_dim": "[INT]",
                            "conv_emb_kernel_size": "[INT]"})
    
