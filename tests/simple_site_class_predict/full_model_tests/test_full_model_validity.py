#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 18:40:55 2025

@author: annabel
"""
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

import numpy as np

import numpy.testing as npt
import unittest


from tests.data_processing import (str_aligns_to_tensor,
                                   summarize_alignment)
from models.simple_site_class_predict.IndpSites import IndpSites

THRESHOLD = 1e-6


class TestFullModelValidity(unittest.TestCase):
    """
    Make sure that joint = cond * marginal, with full models
    """
    def setUp(self):
        ### fake alignments, time array
        fake_aligns = [ ('AC-A','D-ED'),
                        ('D-ED','AC-A'),
                        ('ECDAD','-C-A-'),
                        ('-C-A-','ECDAD') ]

        fake_aligns =  str_aligns_to_tensor(fake_aligns) #(B, L, 3)
            
        vmapped_summarize_alignment = jax.vmap(summarize_alignment, 
                                               in_axes=0, 
                                               out_axes=0)
        counts =  vmapped_summarize_alignment( fake_aligns )
        self.batch = ( counts['match_counts'],
                       counts['ins_counts'],
                       counts['del_counts'],
                       counts['transit_counts'] )

        self.t_array = jnp.array([0.3, 0.5, 0.9])
        
        
        ### template for the config file
        self.config_template = {'load_all': False,
                    'norm_loss_by_length': False,
      
                    'subst_model_type': 'gtr',
                    "norm_rate_matrix": True,
                    "norm_rate_mults": True,
                    "random_init_exchanges": False,
      
                    "times_from": "t_array_from_file",
                    "exponential_dist_param": 1.0,
      
                    "filenames": {"exch": "tests/simple_site_class_predict/full_model_tests/req_files/LG08_exchangeability_vec.npy"},
                    
                    'training_dset_emit_counts': counts['emit_counts'].sum(axis=0),
                    'emission_alphabet_size': 20
                    }
        
    def _run_test(self, to_add):
        config = {**self.config_template, **to_add}
        
        my_model_instance = IndpSites(config=config,
                                      name='my_pairHMM')
        init_params = my_model_instance.init(rngs=jax.random.key(0),
                                             batch = self.batch,
                                             t_array = self.t_array,
                                             sow_intermediates = False)

        all_scores = my_model_instance.apply(variables=init_params,
                                             batch=self.batch,
                                             t_array=self.t_array,
                                             method='calculate_all_loglikes')

        frac = all_scores['joint_neg_logP'] - all_scores['anc_neg_logP']
        npt.assert_allclose( frac, all_scores['cond_neg_logP'] )
        
        
    
    def test_one_gtr(self):
        self._run_test({'num_mixtures': 1,
                        'indel_model_type': None,
                        'num_tkf_fragment_classes': None})
    
    def test_multiple_gtrs(self):
        self._run_test({'num_mixtures': 3,
                        'indel_model_type': None,
                        'num_tkf_fragment_classes': None})
        
    def test_one_tkf91(self):
        self._run_test({'num_mixtures': 1,
                        'indel_model_type': 'tkf91',
                        'num_tkf_fragment_classes': 1})
    
    def test_one_tkf92(self):
        self._run_test({'num_mixtures': 1,
                        'indel_model_type': 'tkf92',
                        'num_tkf_fragment_classes': 1})
    
    def test_one_tkf91_multiple_gtrs(self):
        self._run_test({'num_mixtures': 3,
                  'indel_model_type': 'tkf91',
                  'num_tkf_fragment_classes': 1})
    
    def test_one_tkf92_multiple_gtrs(self):
        self._run_test({'num_mixtures': 3,
                  'indel_model_type': 'tkf92',
                  'num_tkf_fragment_classes': 1})
    
if __name__ == '__main__':
    unittest.main()

