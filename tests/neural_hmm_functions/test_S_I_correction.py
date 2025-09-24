#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 19:15:59 2025

@author: annabel

when S -> I happens, the ancestor path gets treated as:
    em -> em -> ...

when it SHOULD be
    S -> em -> ...

that is, there's an extra (em) and a missing (S), so manually correct for that
  wherever there's S -> I

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
from models.latent_class_mixtures.IndpSites import IndpSitesLoadAll
from models.neural_hmm_predict.NeuralCondTKF import NeuralCondTKF
from models.latent_class_mixtures.model_functions import cond_prob_from_counts
from models.neural_hmm_predict.model_functions import (regular_tkf,
                                                       logprob_tkf92)

THRESHOLD = 1e-6


class TestSICorrection(unittest.TestCase):
    """
    TODO: format this more nicely
    """
    def test_this(self):
        path = '.'
                
        ###################
        ### fake inputs   #
        ###################
        ### neural function
        fake_aligns = [ ('-A', 'AA'), # S -> I -> M -> E
                        ('--A', 'AAA'), # S -> I -> I -> M -> E
                        ('-A', 'A-'), # S -> I -> D -> E
                        ('A-', 'AA') # S -> M -> I -> E
                       ]
        anc_lens = [len(s[0].replace('-','')) for s in fake_aligns]
        
        fake_aligns_pairhmm =  str_aligns_to_tensor(fake_aligns) #(B, L, 3)
        seq_inputs_for_neural_code = fake_aligns_pairhmm[:, 1:, [0,1]]
        curr_pos = fake_aligns_pairhmm[:, 1:, 2][...,None]
        prev_pos = fake_aligns_pairhmm[:, :-1, 2][...,None]
        
        fake_aligns_neural = np.concatenate([seq_inputs_for_neural_code,
                                                  prev_pos,
                                                  curr_pos], axis=-1)
        
        del seq_inputs_for_neural_code, curr_pos, prev_pos
        
        
        ### summarized counts
        vmapped_summarize_alignment = jax.vmap(summarize_alignment, 
                                               in_axes=0, 
                                               out_axes=0)
        counts =  vmapped_summarize_alignment( fake_aligns_pairhmm )
        transit_counts = counts['transit_counts']
        
        
        ### other inputs for pairhmm
        t_array = jnp.array([0.1]*fake_aligns_neural.shape[0])
        
        pairhmm_batch = (counts['match_counts'],
                         counts['ins_counts'],
                         counts['del_counts'],
                         counts['transit_counts'],
                         t_array, #time per sample
                         jnp.array( range(fake_aligns_pairhmm.shape[0]) ) )
        training_dset_emit_counts = counts['emit_counts'].sum(axis=0)
        
        
        ### transit params
        lam = np.array( 0.4 )
        mu = np.array( 0.5 )
        r_extend = np.array( 0.9 )[None]
        true_corr = jnp.log( (lam/mu) / (r_extend + (1-r_extend)*(lam/mu) ) )
        
        tkf92_params = {'lambda': lam,
                        'mu': mu,
                        'r_extend': r_extend}
        
        # save params to files to load
        with open(f'{path}/equl_dist.npy','wb') as g:
            np.save(g, training_dset_emit_counts/training_dset_emit_counts.sum())
        
        with open(f'{path}/tkf92_params_file.pkl','wb') as g:
            pickle.dump(tkf92_params, g)
        
        del g
        
        ### declare dims
        B = fake_aligns_neural.shape[0]
        L = fake_aligns_neural.shape[1]
        A = training_dset_emit_counts.shape[0]
        
        
        ### make a common config
        common_config = {'subst_model_type': 'f81',
                         'indel_model_type': 'tkf92',
                         'times_from': 't_per_sample',
                         
                         'training_dset_emit_counts': training_dset_emit_counts,
                         'emission_alphabet_size': A,
                         'tkf_function_name': 'regular',
                         'filenames': {'equl_dist': f'{path}/equl_dist.npy',
                                       'tkf_params_file': f'{path}/tkf92_params_file.pkl'}}
        
        unique_t_per_samp = True
        jit_compiled = False
        
        
        
        #############################################################
        ### get transition scoring matrices from pairHMM function   #
        #############################################################
        pairhmm_config = {'num_domain_mixtures':1,
                          'num_fragment_mixtures':1,
                          'num_site_mixtures':1,
                          'k_rate_mults':1,
                          'indp_rate_mults':False,
                          'tkf_function': 'regular_tkf'}
        pairhmm_config = {**common_config, **pairhmm_config}
        
        # init object
        pairhmm = IndpSitesLoadAll(config=pairhmm_config,
                                   name='pairhmm')
        
        # get scoring matrices
        scoring_mat_dict = pairhmm.apply( variables={},
                                          t_array=t_array,
                                          sow_intermediates=False,
                                          return_intermeds=True,
                                          return_all_matrices=True,
                                          method = '_get_scoring_matrices' )
        joint_transit = scoring_mat_dict['all_transit_matrices']['joint']
        cond_transit = scoring_mat_dict['all_transit_matrices']['conditional']
        marg_transit = scoring_mat_dict['all_transit_matrices']['marginal']
        
        
        ########################################################
        ### score using built-in method for pairHMM function   #
        ########################################################
        out = pairhmm.apply( variables={},
                             batch = pairhmm_batch,
                             t_array = t_array,
                             return_intermeds = True,
                             method = 'calculate_all_loglikes' )
        
        joint_minus_marg_transit = out['joint_transit_score'] - out['anc_marg_transit_score']
        
        
        # double check S -> I transition in joint matrix, which should be beta
        npt.assert_allclose( joint_transit[:,-1,1], cond_transit[:,-1,1] )
        
        beta_num = lam * ( np.exp(-lam*t_array) - np.exp(-mu*t_array) )
        beta_denom = mu * np.exp(-lam*t_array) - lam * np.exp(-mu*t_array)
        log_beta = np.log( beta_num / beta_denom )
        
        npt.assert_allclose(log_beta, joint_transit[:,-1,1])
        npt.assert_allclose(log_beta, cond_transit[:,-1,1])
        
        del beta_num, beta_denom
        
        
        # double check the ancestor marginal score
        hand_done_anc_marg = []
        start_to_emit = np.log( lam/mu )
        emit_to_emit = np.log( r_extend + (1 - r_extend)*(lam/mu) )
        emit_to_end = np.log( (1 - r_extend) * (1 - (lam/mu)) )
        for l in anc_lens:
            s = start_to_emit + (l-1)*emit_to_emit + emit_to_end
            hand_done_anc_marg.append(s.item())
        
        npt.assert_allclose(np.array(hand_done_anc_marg),
                           out['anc_marg_transit_score'])
        
        del out, hand_done_anc_marg, l, s
        
        
        ##################################################################
        ### score using (unused so far) cond_prob_from_counts function   #
        ##################################################################
        out = cond_prob_from_counts( batch = pairhmm_batch,
                                     times_from = 't_per_sample',
                                     score_indels = True,
                                     scoring_matrices_dict = scoring_mat_dict,
                                     t_array = t_array,
                                     exponential_dist_param = jnp.array(1.0),
                                     norm_reported_loss_by = 'desc_len',
                                     return_intermeds = True )
        manual_cond_transit = out['cond_transit_score'] 
        del out
        
        npt.assert_allclose( manual_cond_transit, joint_minus_marg_transit )
        
        
        ##########################################
        ### reshape inputs for neural cond tkf   #
        ##########################################
        corr_factors = scoring_mat_dict['all_transit_matrices']['log_corr']
        cond_logprob_emit_match_square = scoring_mat_dict['cond_logprob_emit_at_match'] #(T,A,A) or (B,A,A)
        
        # cond_logprob_emit_match_square: convert from square (...,A,A) matrix
        #   to (...,A,2) matrix for F81
        logprobs_if_match = np.diagonal(cond_logprob_emit_match_square,
                                        axis1=-1,
                                        axis2=-2) #(T,A) or (B,A)
        logprobs_if_not_match = cond_logprob_emit_match_square[:,0,:] #(T,A) or (B,A)
        logprobs_if_not_match = logprobs_if_not_match.at[:,0].set( cond_logprob_emit_match_square[:,1,0] ) #(T,A) or (B,A)
        cond_logprob_emit_match = jnp.stack( [logprobs_if_match, logprobs_if_not_match], axis=-1 ) #(B,A,2)
        del logprobs_if_match, logprobs_if_not_match
        
        # others
        logprob_emit_indel = scoring_mat_dict['logprob_emit_at_indel'][None,None,:] #(1, 1, A)
        cond_logprob_transits = scoring_mat_dict['all_transit_matrices']['conditional'] #(T,S,S) or (B,S,S)
        
        # reshape
        cond_logprob_emit_match = cond_logprob_emit_match[:,None,...]#(B, 1, A, 2)
        cond_logprob_transits = cond_logprob_transits[:,None,...] #(B,1,S,S)
        
        
        ###################################
        ### score using neural cond tkf   #
        ###################################
        neural_config = {'load_all': False,
                         
                          'global_or_local': {'equl_dist': 'global',
                                              'rate_mult': 'global',
                                              'exch': 'global',
                                              'tkf_rates': 'global',
                                              'tkf92_frag_size': 'global'},
                         
                          'emissions_postproc_model_type': None,
                          'emissions_postproc_config': {'norm_rate_mult': False,
                                                        'emission_alphabet_size': np.array(A),
                                                        'training_dset_emit_counts': common_config['training_dset_emit_counts']},
                         
                          'transitions_postproc_model_type': None,
                          'transitions_postproc_config': {'tkf_function': 'regular_tkf'} }
        neural_config = {**common_config, **neural_config}
        
        neural = NeuralCondTKF(config=neural_config,
                                name='neural')
        
        init_params = neural.init( rngs=jax.random.key(0),
                                    datamat_lst = [jnp.zeros(fake_aligns_neural.shape[0]),
                                                  jnp.zeros(fake_aligns_neural.shape[0]),
                                                  jnp.zeros(fake_aligns_neural.shape[0])],
                                    padding_mask = [jnp.zeros(fake_aligns_neural.shape[0]),
                                                  jnp.zeros(fake_aligns_neural.shape[0])],
                                    t_array = t_array,
                                    training = False )
        
        out = neural.apply( variables=init_params,
                                    logprob_emit_match=cond_logprob_emit_match,
                                    logprob_emit_indel=logprob_emit_indel,
                                    logprob_transits=cond_logprob_transits,
                                    corr=corr_factors,
                                    rate_multiplier = jnp.ones((1,1)),
                                    true_out=fake_aligns_neural,
                                    return_result_before_sum = True,
                                    return_transit_emit = True,
                                    method='neg_loglike_in_scan_fn') #(T,B) or (B,L)
        neural_cond_transit = out['logprob_perSamp_perPos_perTime']
        neural_cond_transit = neural_cond_transit.sum(axis=-1)
        
        npt.assert_allclose( neural_cond_transit, joint_minus_marg_transit )
        
        del out
        
        
        ###########################################################################
        ### score using neural cond tkf AND tkf92 function from neural codebase   #
        ###########################################################################
        tkf_dict,_ = regular_tkf( mu = mu[None,None], 
                                offset = 1 - (lam[None,None]/mu[None,None]), 
                                t_array = t_array,
                                unique_time_per_sample = True )
        
        # should be the exact same function
        different_scoring_mat = logprob_tkf92(tkf_params_dict = tkf_dict,
                                              r_extend = r_extend[None],
                                              offset = 1 - (lam[None,None]/mu[None,None]),
                                              unique_time_per_sample = True )
        
        npt.assert_allclose(cond_logprob_transits, different_scoring_mat)
        
        
        out = neural.apply( variables=init_params,
                                    logprob_emit_match=cond_logprob_emit_match,
                                    logprob_emit_indel=logprob_emit_indel,
                                    logprob_transits=different_scoring_mat,
                                    corr=corr_factors,
                                    rate_multiplier = jnp.ones((1,1)),
                                    true_out=fake_aligns_neural,
                                    return_result_before_sum = True,
                                    return_transit_emit = True,
                                    method='neg_loglike_in_scan_fn') #(T,B) or (B,L)
        
        neural_cond_transit_different_mat = out['logprob_perSamp_perPos_perTime']
        neural_cond_transit_different_mat = neural_cond_transit_different_mat.sum(axis=-1)
        
        npt.assert_allclose( neural_cond_transit_different_mat, joint_minus_marg_transit )
    
if __name__ == '__main__':
    unittest.main()
    