#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 18:36:05 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
import numpy as np
import pickle
from functools import partial
jax.config.update("jax_enable_x64", True)

import numpy.testing as npt
import unittest

from tests.data_processing import (str_aligns_to_tensor,
                                   summarize_alignment)
from models.simple_site_class_predict.IndpSites import IndpSitesLoadAll
from models.simple_site_class_predict.model_functions import (anc_marginal_probs_from_counts,
                                                              cond_prob_from_counts,
                                                              joint_prob_from_counts)

THRESHOLD = 1e-6


class TestJointCondMargWithScoringFuncs(unittest.TestCase):
    """
    Check that cond = joint / anc for indp sites model using a tkf model
    
    This isn't a problem for tkf91, but it's a bit fiddly with tkf92
    """
    def setUp(self):
        self.path = f'tests/pairhmm_site_class_mixtures'
        self.req_files_path = f'{self.path}/req_files'
        
        
        ###############################
        ### generate fake alignments  #
        ###############################
        fake_aligns = [ ('ECDAD','-C-A-'),
                        ('-C-A-','ECDAD') ]
        
        fake_aligns_pairhmm =  str_aligns_to_tensor(fake_aligns) #(B, L, 3)
        del fake_aligns
        
        # put together a batch for pairHMM
        vmapped_summarize_alignment = jax.vmap(summarize_alignment, 
                                               in_axes=0, 
                                               out_axes=0)
        counts =  vmapped_summarize_alignment( fake_aligns_pairhmm )
        
        self.pairhmm_batch = (counts['match_counts'],
                              counts['ins_counts'],
                              counts['del_counts'],
                              counts['transit_counts'],
                              None,
                              jnp.array( range(fake_aligns_pairhmm.shape[0]) )
                              )
        self.training_dset_emit_counts = counts['emit_counts'].sum(axis=0)
        
        # other params
        self.t_array = jnp.array([0.1, 0.2, 0.3])
        tkf_params = {'lambda': np.array(0.1),
                      'mu': np.array(0.11),
                      'r_extend': np.array([[0.9]])}
        
        # save params to files to load
        with open(f'{self.req_files_path}/equl_dist.npy','wb') as g:
            np.save(g, self.training_dset_emit_counts/self.training_dset_emit_counts.sum())
        
        with open(f'{self.req_files_path}/tkf_params_file.pkl','wb') as g:
            pickle.dump(tkf_params, g)
        
        # declare dims
        self.A = self.training_dset_emit_counts.shape[-1]
        
    
    def _check_validity(self,
                           subst_model_type,
                           indel_model_type):
        #############################################################
        ### use IndpSitesLoadAll to generate the scoring matrices   #
        #############################################################
        # init object
        pairhmm_config = {'num_domain_mixtures': 1,
                          'num_fragment_mixtures': 1,
                          'num_site_mixtures': 1,
                          'k_rate_mults':1,
                          'subst_model_type': subst_model_type,
                          'indel_model_type': indel_model_type,
                          'indp_rate_mults': False,
                          'times_from': 't_array_from_file',
                          'exponential_dist_param': 1.1,
                          'training_dset_emit_counts': self.training_dset_emit_counts,
                          'emission_alphabet_size': self.A,
                          'tkf_function': 'regular_tkf',
                          'filenames': {'exch': f'{self.req_files_path}/LG08_exchangeability_vec.npy',
                                        'equl_dist': f'{self.req_files_path}/equl_dist.npy',
                                        'tkf_params_file': f'{self.req_files_path}/tkf_params_file.pkl'}}
        pairhmm = IndpSitesLoadAll(config=pairhmm_config,
                            name='pairhmm')
        
        scoring_mat_dict = pairhmm.apply( variables={},
                                          t_array=self.t_array,
                                          return_intermeds=True,
                                          return_all_matrices=True,
                                          sow_intermediates=False,
                                          method = '_get_scoring_matrices')
        
        
        ### joint, anc, cond
        scores = joint_prob_from_counts( batch = self.pairhmm_batch,
                                            times_from = pairhmm_config['times_from'],
                                            score_indels = True,
                                            scoring_matrices_dict = scoring_mat_dict,
                                            t_array = self.t_array,
                                            exponential_dist_param = pairhmm_config['exponential_dist_param'],
                                            norm_reported_loss_by = 'desc_len',
                                            return_intermeds= True )
        
        to_add = anc_marginal_probs_from_counts( batch = self.pairhmm_batch,
                                            score_indels = True,
                                            scoring_matrices_dict = scoring_mat_dict,
                                            return_intermeds=True  )
        
        scores = {**scores, **to_add}
        del to_add
        
        
        to_add = cond_prob_from_counts( batch = self.pairhmm_batch,
                                        times_from = pairhmm_config['times_from'],
                                        score_indels = True,
                                        scoring_matrices_dict = scoring_mat_dict,
                                        t_array = self.t_array,
                                        exponential_dist_param = pairhmm_config['exponential_dist_param'],
                                        norm_reported_loss_by = 'desc_len',
                                        return_intermeds= True )
        
        scores = {**scores, **to_add}
        del to_add
        
        
        ##########################
        ### check for validity   #
        ##########################
        npt.assert_allclose( scores['cond_emission_score'], 
                              scores['joint_emission_score'] - scores['anc_marg_emit_score'] )
        
        npt.assert_allclose( scores['cond_transit_score'], 
                              scores['joint_transit_score'] - scores['anc_marg_transit_score'] )
        
        npt.assert_allclose( -scores['cond_neg_logP'], 
                              -scores['joint_neg_logP'] - -scores['anc_neg_logP'] )
        
    
    def test_tkf91_f81_validity(self):
        self._check_validity(indel_model_type = 'tkf91',
                                subst_model_type = 'f81')
    
    def test_tkf91_gtr_validity(self):
        self._check_validity(indel_model_type = 'tkf91',
                                subst_model_type = 'gtr')
    
    def test_tkf92_f81_validity(self):
        self._check_validity(indel_model_type = 'tkf92',
                                subst_model_type = 'f81')
    
    def test_tkf92_gtr_validity(self):
        self._check_validity(indel_model_type = 'tkf92',
                                subst_model_type = 'gtr')
        
        
if __name__ == '__main__':
    unittest.main()