#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 19:21:42 2025

@author: annabel
"""
import flax
import pickle

def save_all_neural_trainstates( all_save_model_filenames,
                          all_trainstates,
                          suffix = None ):
    for i in range(len(all_trainstates)):
        new_outfile = all_save_model_filenames[i]
        
        if suffix is not None:
            new_outfile = new_outfile.replace('.pkl',f'_{suffix}.pkl')
        
        with open(new_outfile, 'wb') as g:
            model_state_dict = flax.serialization.to_state_dict(all_trainstates[i])
            pickle.dump(model_state_dict, g)