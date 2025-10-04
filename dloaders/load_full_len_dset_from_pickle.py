#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 16:22:23 2025

@author: annabel
"""
import pickle
from dloaders.init_dataloader import init_dataloader


def _validate(final_dim, 
              pred_model_type):
    if pred_model_type in ['pairhmm_frag_and_site_classes',
                           'pairhmm_nested_tkf']:
        assert final_dim == 3
    
    elif pred_model_type == 'neural_hmm':
        assert final_dim == 5
        
    elif pred_model_type == 'feedforward':
        assert final_dim == 4
    
    else:
        raise ValueError('Only preload full-length sequence datasets')

def _add_dload_obj_to_dict(prefix, 
                           dset_dict):
    # validate data structure using final dimension of the 
    #   Pytorch Dataset object
    _validate( final_dim = dset_dict[f'{prefix}_dset'].aligned_mat.shape[-1],
               pred_model_type = pred_model_type )
    
    # add dataloader objects
    dset_dict[f'{prefix}_dl'] = init_dataloader(args = args, 
                                                shuffle = False,
                                                pytorch_custom_dset = dset_dict[f'{prefix}_dset'],
                                                collate_fn = collate_fn)
    
    return dset_dict
    

def load_full_len_dset_from_pickle(pred_model_type,
                                   file_to_load, 
                                   args,
                                   collate_fn):
    # determine if classical mixture or neural model
    is_neural = pred_model_type in ['feedforward', 'neural_hmm']
    
    with open(file_to_load,'rb') as f:
        dset_dict = pickle.load(f)
        
    # test set
    dset_dict = _add_dload_obj_to_dict( prefix = 'test', 
                                        dset_dict = dset_dict )
    
    # dev dataset: if training a neural model, use this for early 
    #   stopping criteria
    if is_neural and ( 'training_dset' in dset_dict.keys() ):
        dset_dict = _add_dload_obj_to_dict( prefix = 'dev', 
                                            dset_dict = dset_dict )
    
    # Training set: if training any model
    if 'training_dset' in dset_dict.keys():
        dset_dict = _add_dload_obj_to_dict( prefix = 'train', 
                                            dset_dict = dset_dict )
        
    return dset_dict
