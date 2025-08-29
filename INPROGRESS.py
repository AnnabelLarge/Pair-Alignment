#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 19:18:50 2025

@author: annabel
"""
import numpy as np

def load_counts(data_folder, prefix_lst, suffix):
    all_mats = []
    for pre in prefix_lst:
        all_mats.append( np.load(f'{data_folder}/{pre}_{suffix}.npy') )
    return np.stack(all_mats, axis=0).sum(axis=0)

desc_counts = load_counts( data_folder = 'DATA_cherries',
                           prefix_lst = ['FAMCLAN-CHERRIES_split8',
                                         'FAMCLAN-CHERRIES_split9'],
                           suffix = 'desc-align_counts')
desc_freqs = desc_counts / desc_counts.sum()

desc_given_anc_counts = load_counts( data_folder = 'DATA_cherries',
                                     prefix_lst = ['FAMCLAN-CHERRIES_split8',
                                                   'FAMCLAN-CHERRIES_split9'],
                                     suffix = 'desc-align_given_current_anc_counts')
desc_given_anc_freqs = desc_given_anc_counts / desc_given_anc_counts.sum(axis=1)[:,None]

