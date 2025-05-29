#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 18:40:55 2025

@author: annabel
"""
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
from jax.scipy.special import logsumexp

import numpy as np

import numpy.testing as npt
import unittest


from tests.data_processing import (str_aligns_to_tensor,
                                   summarize_alignment)
from models.simple_site_class_predict.transition_models import (TKF91TransitionLogprobs,
                                                                TKF92TransitionLogprobs)
from models.simple_site_class_predict.model_functions import (stable_tkf,
                                                              MargTKF91TransitionLogprobs,
                                                              MargTKF92TransitionLogprobs,
                                                              CondTransitionLogprobs)
from models.simple_site_class_predict.IndpSites import IndpSites

THRESHOLD = 1e-6

class TestTimeMarginalization(unittest.TestCase):
    """
    Individually evaluate over each possible branch length, and make sure
      this matches expected value when providing the full array of times
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

        self.t_array = jnp.array([0.3, 0.5, 0.9, 1.0, 1.1, 1.7])
        
        
        ### template for the config file
        self.config_template = {'load_all': False,
                    'norm_loss_by_length': False,
      
                    'subst_model_type': 'gtr',
                    "norm_rate_matrix": True,
                    "norm_rate_mults": True,
                    "random_init_exchanges": False,
      
                    "times_from": "t_array_from_file",
                    "exponential_dist_param": 1.0,
                    "filenames": {"exch": "tests/full_model_tests/req_files/LG08_exchangeability_vec.npy"},
                    
                    'training_dset_emit_counts': counts['emit_counts'].sum(axis=0),
                    'emission_alphabet_size': 20
                    }
        
    def _run_test(self, to_add):
        config = {**self.config_template, **to_add}
        
        ### init model
        my_model_instance = IndpSites(config=config,
                                      name='my_pairHMM')
        
        init_params = my_model_instance.init(rngs=jax.random.key(0),
                                             batch = self.batch,
                                             t_array = jnp.zeros(1,),
                                             sow_intermediates = False)
        
        ### score times across all times
        _, aux_dict = my_model_instance.apply(variables=init_params,
                                                 batch=self.batch,
                                                 t_array=self.t_array,
                                                 sow_intermediates = False)
        true = aux_dict['joint_neg_logP']
        
        
        ### manually logsumexp
        to_logsumexp_wo_constant = []
        for t in self.t_array:
            _, aux_dict = my_model_instance.apply(variables=init_params,
                                                  batch=self.batch,
                                                  t_array=jnp.array([t]),
                                                  sow_intermediates = False)
            logp = aux_dict['joint_neg_logP']
            to_logsumexp_wo_constant.append(-logp)
            
        to_logsumexp_wo_constant = jnp.stack(to_logsumexp_wo_constant, axis=0)
        
        # constants to add (multiply by)
        # logP(t_k) = exponential distribution
        exponential_dist_param = config['exponential_dist_param']
        logP_time = ( jnp.log(exponential_dist_param) - 
                      (exponential_dist_param * self.t_array) ) #(T,)
        log_t_grid = jnp.log( self.t_array[1:] - self.t_array[:-1] ) #(T-1,)
        
        # kind of a hack, but repeat the last time array value
        log_t_grid = jnp.concatenate( [log_t_grid, log_t_grid[-1][None] ], axis=0) #(T,)
        
        
        ### add in log space, multiply in probability space; logsumexp
        to_logsumexp = ( to_logsumexp_wo_constant +
                         logP_time[:,None] +
                         log_t_grid[:,None] ) #(T,B)
        
        pred = -logsumexp(to_logsumexp, axis=0)
        npt.assert_allclose( true, pred )
        
        
    def test_one_gtr(self):
        self._run_test({'num_mixtures': 1,
                             'indel_model_type': None,
                             'num_tkf_site_classes': None})
    
    def test_multiple_gtrs(self):
        self._run_test({'num_mixtures': 3,
                        'indel_model_type': None,
                        'num_tkf_site_classes': None})
        
    def test_one_tkf91(self):
        self._run_test({'num_mixtures': 1,
                        'indel_model_type': 'tkf91',
                        'num_tkf_site_classes': 1})
    
    def test_one_tkf92(self):
        self._run_test({'num_mixtures': 1,
                        'indel_model_type': 'tkf92',
                        'num_tkf_site_classes': 1})
    
    def test_one_tkf91_multiple_gtrs(self):
        self._run_test({'num_mixtures': 3,
                  'indel_model_type': 'tkf91',
                  'num_tkf_site_classes': 1})
    
    def test_one_tkf92_multiple_gtrs(self):
        self._run_test({'num_mixtures': 3,
                  'indel_model_type': 'tkf92',
                  'num_tkf_site_classes': 1})
    
if __name__ == '__main__':
    unittest.main()

