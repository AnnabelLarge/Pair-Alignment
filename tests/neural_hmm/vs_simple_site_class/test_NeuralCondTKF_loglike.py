#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 12:50:39 2025

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
from models.neural_hmm_predict.NeuralCondTKF import NeuralCondTKF

THRESHOLD = 1e-6


class TestNeuralLoglikeVsPairhmmLoglike(unittest.TestCase):
    def setUp(self):
        ### fake alignments
        fake_aligns = [ ('AC-A','D-ED'),
                        ('D-ED','AC-A'),
                        ('ECDAD','-C-A-'),
                        ('-C-A-','ECDAD') ]
        
        fake_aligns_pairhmm =  str_aligns_to_tensor(fake_aligns) #(B, L, 3)
        
        seq_inputs_for_neural_code = fake_aligns_pairhmm[:, 1:, [0,1]]
        curr_pos = fake_aligns_pairhmm[:, 1:, 2][...,None]
        prev_pos = fake_aligns_pairhmm[:, :-1, 2][...,None]
        
        self.fake_aligns_neural = np.concatenate([seq_inputs_for_neural_code,
                                                  prev_pos,
                                                  curr_pos], axis=-1)
        
        # put together a batch for pairHMM
        vmapped_summarize_alignment = jax.vmap(summarize_alignment, 
                                               in_axes=0, 
                                               out_axes=0)
        counts =  vmapped_summarize_alignment( fake_aligns_pairhmm )
        
        self.pairhmm_batch = (counts['match_counts'],
                              counts['ins_counts'],
                              counts['del_counts'],
                              counts['transit_counts'],
                              jnp.array([0.1, 0.2, 0.3, 0.4]), #time per sample
                              jnp.array( range(fake_aligns_pairhmm.shape[0]) )
                              )
        training_dset_emit_counts = counts['emit_counts'].sum(axis=0)
        
        
        ### other params
        tkf91_params = {'lambda': np.array(0.2),
                        'mu': np.array(0.22)}
        
        self.tkf92_params = {'lambda': np.array(0.1),
                             'mu': np.array(0.11),
                             'r_extend': np.array([0.06])}
        
        # save params to files to load
        with open('./tests/neural_hmm/vs_simple_site_class/req_files/equl_dist.npy','wb') as g:
            np.save(g, training_dset_emit_counts/training_dset_emit_counts.sum())
        
        with open('./tests/neural_hmm/vs_simple_site_class/req_files/tkf91_params_file.pkl','wb') as g:
            pickle.dump(tkf91_params, g)
            
        with open('./tests/neural_hmm/vs_simple_site_class/req_files/tkf92_params_file.pkl','wb') as g:
            pickle.dump(self.tkf92_params, g)
        
        # declare dims
        self.B = self.fake_aligns_neural.shape[0]
        self.L = self.fake_aligns_neural.shape[1]
        self.A = training_dset_emit_counts.shape[0]
        
        
        ### make a common config
        self.common_config = {'training_dset_emit_counts': training_dset_emit_counts,
                              'emission_alphabet_size': self.A,
                              'tkf_function_name': 'regular',
                              'filenames': {'exch': './tests/neural_hmm/vs_simple_site_class/req_files/LG08_exchangeability_vec.npy',
                                            'equl_dist': './tests/neural_hmm/vs_simple_site_class/req_files/equl_dist.npy'}}
    
    def _run_test(self,
                  subst_model_type,
                  indel_model_type,
                  t_array,
                  unique_t_per_samp,
                  jit_compiled):
        ##################
        ### preprocess   #
        ##################
        self.common_config['subst_model_type'] = subst_model_type
        self.common_config['indel_model_type'] = indel_model_type
        self.common_config['filenames']['tkf_params_file']= f'./tests/neural_hmm/vs_simple_site_class/req_files/{indel_model_type}_params_file.pkl'
    
        ### determine t_array
        if unique_t_per_samp:
            t_array = self.pairhmm_batch[-2]
            self.common_config['times_from'] = 't_per_sample'
            
        else:
            self.common_config['times_from'] = 't_array_from_file'
            self.common_config['exponential_dist_param'] = 1.1
    
    
        ### determine correction factors for indel model
        if indel_model_type == 'tkf92':
            lam = self.tkf92_params['lambda']
            mu = self.tkf92_params['mu']
            r = self.tkf92_params['r_extend']
            corr_start_to_ins = jnp.log(mu / lam)
            corr_ins_to_end = jnp.log(r + (1 - r) * (lam / mu))
    
            if unique_t_per_samp:
                corr_factors = (corr_start_to_ins[None], corr_ins_to_end)  # shape (1,) to broadcast to (B,)
            else:
                corr_factors = (corr_start_to_ins[None, None], corr_ins_to_end[None])  # (1,1) to broadcast to (T,B)
    
        elif indel_model_type == 'tkf91':
            shape = (1,) if unique_t_per_samp else (1, 1)
            zeros = jnp.zeros(shape)
            corr_factors = (zeros, zeros)
        
        
        ##########################
        ### score with pairhmm   #
        ##########################
        pairhmm_config = {'num_mixtures':1,
                          'num_tkf_fragment_classes': 1}
        pairhmm_config = {**self.common_config, **pairhmm_config}
        
        # init object
        pairhmm = IndpSitesLoadAll(config=pairhmm_config,
                                   name='pairhmm')
        
        # get scoring matrices
        if jit_compiled:
            jitted_apply = jax.jit(pairhmm.apply, static_argnames=('method',))
            scoring_mat_dict = jitted_apply( variables={},
                                             t_array=t_array,
                                             sow_intermediates=False,
                                             method = '_get_scoring_matrices' )
            
            # use calculate_all_loglikes method to get conditional loglike
            true_scores = jitted_apply( variables={},
                                        batch = self.pairhmm_batch,
                                        t_array = t_array,
                                        method = 'calculate_all_loglikes' )
            
        elif not jit_compiled:
            scoring_mat_dict = pairhmm.apply( variables={},
                                              t_array=t_array,
                                              sow_intermediates=False,
                                              method = '_get_scoring_matrices' )
            
            # use calculate_all_loglikes method to get conditional loglike
            true_scores = pairhmm.apply( variables={},
                                          batch = self.pairhmm_batch,
                                          t_array = t_array,
                                          method = 'calculate_all_loglikes' )
            
        true_cond_neg_logP = true_scores['cond_neg_logP'] #(B,)
        del pairhmm_config, pairhmm, true_scores
        
        
        #########################################
        ### Get scoring matrices from pairHMM   #
        #########################################
        cond_logprob_emit_match_square = scoring_mat_dict['cond_logprob_emit_at_match'] #(T,A,A) or (B,A,A)
        
        if subst_model_type.lower() == 'f81':
            # cond_logprob_emit_match_square: convert from square (...,A,A) matrix
            #   to (...,A,2) matrix for F81
            logprobs_if_match = np.diagonal(cond_logprob_emit_match_square,
                                            axis1=-1,
                                            axis2=-2) #(T,A) or (B,A)
            logprobs_if_not_match = cond_logprob_emit_match_square[:,0,:] #(T,A) or (B,A)
            logprobs_if_not_match = logprobs_if_not_match.at[:,0].set( cond_logprob_emit_match_square[:,1,0] ) #(T,A) or (B,A)
            cond_logprob_emit_match = jnp.stack( [logprobs_if_match, logprobs_if_not_match], axis=-1 ) #(B,A,2)
        
        elif subst_model_type.lower() == 'gtr':
            cond_logprob_emit_match = cond_logprob_emit_match_square
        
        logprob_emit_indel = scoring_mat_dict['logprob_emit_at_indel'][None,None,:] #(1, 1, A)
        cond_logprob_transits = scoring_mat_dict['all_transit_matrices']['conditional'] #(T,S,S) or (B,S,S)
        
        if not unique_t_per_samp:
            cond_logprob_emit_match = cond_logprob_emit_match[:,None,None,...]#(T,1,1,A,2) 
            cond_logprob_transits = cond_logprob_transits[:,None,None,...] #(T,1,1,S,S)
            
        elif unique_t_per_samp:
            cond_logprob_emit_match = cond_logprob_emit_match[:,None,...]#(B, 1, A, 2)
            cond_logprob_transits = cond_logprob_transits[:,None,...] #(B,1,S,S)
            
        
        #############################
        ### score with neural tkf   #
        #############################
        neural_config = {'global_or_local': {'equl_dist': 'global',
                                             'rate_mult': 'global',
                                             'exch': 'global',
                                             'tkf_rates': 'global',
                                             'tkf92_frag_size': 'global'},
                         'use_which_emb': {'preproc_equl': (False, False),
                                           'preproc_subs': (False, False),
                                           'preproc_trans': (False, False)}}
        neural_config = {**self.common_config, **neural_config}
        
        neural = NeuralCondTKF(config=neural_config,
                               name='neural')
        
        init_params = neural.init( rngs=jax.random.key(0),
                                   datamat_lst = [jnp.zeros(self.fake_aligns_neural.shape[0]),
                                                  jnp.zeros(self.fake_aligns_neural.shape[0])],
                                   padding_mask = [jnp.zeros(self.fake_aligns_neural.shape[0]),
                                                  jnp.zeros(self.fake_aligns_neural.shape[0])],
                                   t_array = t_array,
                                   training = False )
        
        if jit_compiled:
            jitted_apply = jax.jit(neural.apply, static_argnames=('method',))
            raw_scores = jitted_apply(variables=init_params,
                                        logprob_emit_match=cond_logprob_emit_match,
                                        logprob_emit_indel=logprob_emit_indel,
                                        logprob_transits=cond_logprob_transits,
                                        corr=corr_factors,
                                        true_out=self.fake_aligns_neural,
                                        method='neg_loglike_in_scan_fn')
            
            _, pred_scores = jitted_apply( variables=init_params,
                                           logprob_perSamp_perTime=raw_scores,
                                           length_for_normalization=jnp.ones(raw_scores.shape),
                                           t_array=t_array,
                                           method='evaluate_loss_after_scan')
        
        elif not jit_compiled:
            raw_scores = neural.apply( variables=init_params,
                                        logprob_emit_match=cond_logprob_emit_match,
                                        logprob_emit_indel=logprob_emit_indel,
                                        logprob_transits=cond_logprob_transits,
                                        corr=corr_factors,
                                        true_out=self.fake_aligns_neural,
                                        method='neg_loglike_in_scan_fn') #(T,B) or (B,L)
        
            _, pred_scores = neural.apply( variables=init_params,
                                           logprob_perSamp_perTime=raw_scores,
                                           length_for_normalization=jnp.ones(raw_scores.shape),
                                           t_array=t_array,
                                           method='evaluate_loss_after_scan')
        pred_cond_neg_logP = pred_scores['sum_neg_logP']
        del pred_scores
        
        npt.assert_allclose( pred_cond_neg_logP, true_cond_neg_logP )
    
    
    ###############################
    ### without jit-compilation   #
    ###############################
    # f81 substitution model, tkf91 indel model
    def test_with_time_grid_f81_tkf91(self):
        self._run_test( t_array = jnp.array([0.1, 0.2, 0.3]),
                        unique_t_per_samp = False,
                        subst_model_type = 'f81',
                        indel_model_type = 'tkf91',
                        jit_compiled = False )
    
    def test_with_t_per_samp_f81_tkf91(self):
        self._run_test( t_array = None,
                        unique_t_per_samp = True,
                        subst_model_type = 'f81',
                        indel_model_type = 'tkf91',
                        jit_compiled = False  )
        
    # GTR substitution model, tkf91 indel model
    def test_with_time_grid_gtr_tkf91(self):
        self._run_test( t_array = jnp.array([0.1, 0.2, 0.3]),
                        unique_t_per_samp = False,
                        subst_model_type = 'gtr',
                        indel_model_type = 'tkf91',
                        jit_compiled = False  )
    
    def test_with_t_per_samp_gtr_tkf91(self):
        self._run_test( t_array = None,
                        unique_t_per_samp = True,
                        subst_model_type = 'gtr',
                        indel_model_type = 'tkf91',
                        jit_compiled = False  )
    
    # f81 substitution model, tkf92 indel model
    def test_with_time_grid_f81_tkf92(self):
        self._run_test( t_array = jnp.array([0.1, 0.2, 0.3]),
                        unique_t_per_samp = False,
                        subst_model_type = 'f81',
                        indel_model_type = 'tkf92',
                        jit_compiled = False )
    
    def test_with_t_per_samp_f81_tkf92(self):
        self._run_test( t_array = None,
                        unique_t_per_samp = True,
                        subst_model_type = 'f81',
                        indel_model_type = 'tkf92',
                        jit_compiled = False  )
        
    # GTR substitution model, tkf92 indel model
    def test_with_time_grid_gtr_tkf92(self):
        self._run_test( t_array = jnp.array([0.1, 0.2, 0.3]),
                        unique_t_per_samp = False,
                        subst_model_type = 'gtr',
                        indel_model_type = 'tkf92',
                        jit_compiled = False  )
    
    def test_with_t_per_samp_gtr_tkf92(self):
        self._run_test( t_array = None,
                        unique_t_per_samp = True,
                        subst_model_type = 'gtr',
                        indel_model_type = 'tkf92',
                        jit_compiled = False  )
    
    
    ############################
    ### with jit-compilation   #
    ############################
    # f81 substitution model, tkf91 indel model
    def test_with_time_grid_f81_tkf91_jitted(self):
        self._run_test( t_array = jnp.array([0.1, 0.2, 0.3]),
                        unique_t_per_samp = False,
                        subst_model_type = 'f81',
                        indel_model_type = 'tkf91',
                        jit_compiled = True )
    
    def test_with_t_per_samp_f81_tkf91_jitted(self):
        self._run_test( t_array = None,
                        unique_t_per_samp = True,
                        subst_model_type = 'f81',
                        indel_model_type = 'tkf91',
                        jit_compiled = True  )
        
    # GTR substitution model, tkf91 indel model
    def test_with_time_grid_gtr_tkf91_jitted(self):
        self._run_test( t_array = jnp.array([0.1, 0.2, 0.3]),
                        unique_t_per_samp = False,
                        subst_model_type = 'gtr',
                        indel_model_type = 'tkf91',
                        jit_compiled = True  )
    
    def test_with_t_per_samp_gtr_tkf91_jitted(self):
        self._run_test( t_array = None,
                        unique_t_per_samp = True,
                        subst_model_type = 'gtr',
                        indel_model_type = 'tkf91',
                        jit_compiled = True  )
    
    # f81 substitution model, tkf92 indel model
    def test_with_time_grid_f81_tkf92_jitted(self):
        self._run_test( t_array = jnp.array([0.1, 0.2, 0.3]),
                        unique_t_per_samp = False,
                        subst_model_type = 'f81',
                        indel_model_type = 'tkf92',
                        jit_compiled = True )
    
    def test_with_t_per_samp_f81_tkf92_jitted(self):
        self._run_test( t_array = None,
                        unique_t_per_samp = True,
                        subst_model_type = 'f81',
                        indel_model_type = 'tkf92',
                        jit_compiled = True  )
        
    # GTR substitution model, tkf92 indel model
    def test_with_time_grid_gtr_tkf92_jitted(self):
        self._run_test( t_array = jnp.array([0.1, 0.2, 0.3]),
                        unique_t_per_samp = False,
                        subst_model_type = 'gtr',
                        indel_model_type = 'tkf92',
                        jit_compiled = True  )
    
    def test_with_t_per_samp_gtr_tkf92_jitted(self):
        self._run_test( t_array = None,
                        unique_t_per_samp = True,
                        subst_model_type = 'gtr',
                        indel_model_type = 'tkf92',
                        jit_compiled = True  )
    
        
if __name__ == '__main__':
    unittest.main()
