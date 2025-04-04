#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 04:33:00 2025

@author: annabel


models here:
============
'IndpHKY85FitAll'
'IndpHKY85FitIndelOnly'
'IndpHKY85LoadAll'

inherit most functions from IndpPairHMMFitBoth


main methods for all models:
============================
- setup (self-explanatory)

- __call__: calculate loss based on joint prob P(anc, desc, align);
           use this during training; is jit compatible

- calculate_all_loglikes: calculate joint prob P(anc, desc, align),
           conditional prob P(desc, align | anc), and both marginals
           P(desc) and P(anc); use this during final eval; is also
           jit compatible
"""
import numpy as np
import pickle

# jumping jax and leaping flax
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax.scipy.special import logsumexp

from models.simple_site_class_predict.emission_models import (LogEqulVecFromCounts,
                                                       LogEqulVecPerClass,
                                                       LogEqulVecFromFile,
                                                       HKY85,
                                                       HKY85FromFile,
                                                       SiteClassLogprobs,
                                                       SiteClassLogprobsFromFile)
from models.simple_site_class_predict.PairHMM_indp_sites import IndpPairHMMFitBoth
from models.simple_site_class_predict.PairHMM_markovian_sites import MarkovPairHMM
from models.simple_site_class_predict.transition_models import (TKF91TransitionLogprobs,
                                                                TKF92TransitionLogprobs,
                                                                TKF91TransitionLogprobsFromFile,
                                                                TKF92TransitionLogprobsFromFile)
from utils.pairhmm_helpers import (bounded_sigmoid,
                                   safe_log)


class IndpHKY85FitAll(IndpPairHMMFitBoth):
    """
    uses HKY85 for susbtitution model
    
    
    unique methods
    ===============
        - setup    
        - write_params: write the parameters to files
    
    
    main methods inherited from IndpPairHMMFitBoth:
    ===============================================        
        - __call__: calculate loss based on joint prob P(anc, desc, align);
                   use this during training; is jit compatible
        
        - calculate_all_loglikes: calculate joint prob P(anc, desc, align),
                   conditional prob P(desc, align | anc), and both marginals
                   P(desc) and P(anc); use this during final eval; is also
                   jit compatible

    internal methods from IndpPairHMMFitBoth:
    ==========================================
        - _joint_logprob_align
        - _marginalize_over_times
        - _get_scoring_matrices
    """
    config: dict
    name: str
    
    def setup(self):
        """
        difference: use HKY85 for self.rate_matrix_module
        """
        
        num_emit_site_classes = self.config['num_emit_site_classes']
        self.indel_model_type = self.config['indel_model_type']
        self.norm_loss_by = self.config['norm_loss_by']
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        
        ### how to score emissions from indel sites
        if num_emit_site_classes == 1:
            self.indel_prob_module = LogEqulVecFromCounts(config = self.config,
                                                       name = f'get equilibrium')
        elif num_emit_site_classes > 1:
            self.indel_prob_module = LogEqulVecPerClass(config = self.config,
                                                     name = f'get equilibrium')
        
        
        ### rate matrix to score emissions from match sites
        self.rate_matrix_module = HKY85(config = self.config,
                                                 name = f'get rate matrix')
        
        
        ### probabilities for rate classes themselves
        self.site_class_probability_module = SiteClassLogprobs(config = self.config,
                                                  name = f'get site class probabilities')
        
        
        ### TKF91 or TKF92
        if self.indel_model_type == 'tkf91':
            self.transitions_module = TKF91TransitionLogprobs(config = self.config,
                                                     name = f'tkf91 indel model')
        
        elif self.indel_model_type == 'tkf92':
            self.transitions_module = TKF92TransitionLogprobs(config = self.config,
                                                     name = f'tkf92 indel model')
    
    
    def write_params(self,
                     t_array,
                     out_folder: str):
        """
        difference: explicitly write "ti" and "tv" to text files
        """
        out = self._get_scoring_matrices(t_array=t_array,
                                        sow_intermediates=False)
        
        # normalized rate matrix
        normalized_rate_matrix = out['normalized_rate_matrix']
        
        with open(f'{out_folder}/normalized_rate_matrix.npy', 'wb') as g:
            np.save(g, normalized_rate_matrix)
        
        np.savetxt( f'{out_folder}/ASCII_normalized_rate_matrix.tsv', 
                    np.array(normalized_rate_matrix), 
                    fmt = '%.4f',
                    delimiter= '\t' )
        
        # matrix that you apply expm() to
        to_expm = np.squeeze( out['to_expm'] )
        
        with open(f'{out_folder}/to_expm.npy', 'wb') as g:
            np.save(g, to_expm)
        
        if len(to_expm.shape) <= 2:
            np.savetxt( f'{out_folder}/ASCII_to_expm.tsv', 
                        to_expm, 
                        fmt = '%.4f',
                        delimiter= '\t' )
        
        del to_expm, g
        
        
        # other emission matrices; exponentiate them first
        for key in ['logprob_emit_at_indel', 
                    'joint_logprob_emit_at_match']:
            mat = np.exp(out[key])
            new_key = key.replace('logprob','prob')
            
            with open(f'{out_folder}/{new_key}.npy', 'wb') as g:
                np.save(g, mat)
            
            mat = np.squeeze(mat)
            if len(mat.shape) <= 2:
                np.savetxt( f'{out_folder}/ASCII_{new_key}.tsv', 
                            np.array(mat), 
                            fmt = '%.4f',
                            delimiter= '\t' )
            
            del key, mat, g
            
        for key, mat in out['all_transit_matrices'].items():
            mat = np.exp(mat)
            new_key = key.replace('logprob','prob')
            
            with open(f'{out_folder}/{new_key}_transit_matrix.npy', 'wb') as g:
                np.save(g, mat)
            
            mat = np.squeeze(mat)
            if len(mat.shape) <= 2:
                np.savetxt( f'{out_folder}/ASCII_{new_key}_transit_matrix.tsv', 
                            np.array(mat), 
                            fmt = '%.4f',
                            delimiter= '\t' )
            
            del key, mat, g
        
        
        ###############
        ### extract   #
        ###############
        ### site class probs
        if 'class_logits' in dir(self.site_class_probability_module):
            class_probs = nn.softmax(self.site_class_probability_module.class_logits)
            with open(f'{out_folder}/PARAMS_class_probs.txt','w') as g:
                [g.write(f'{elem.item()}\n') for elem in class_probs]
        
        
        ### emissions
        # exchangeabilities
        if 'exchangeabilities_logits_vec' in dir(self.rate_matrix_module):
            exchange_min_val = self.rate_matrix_module.exchange_min_val
            exchange_max_val = self.rate_matrix_module.exchange_max_val
            tv_logits = self.rate_matrix_module.exchangeabilities_logits_vec[0]
            ti_logits = self.rate_matrix_module.exchangeabilities_logits_vec[1]
            
            tv = bounded_sigmoid( x = tv_logits, 
                                  min_val = exchange_min_val,
                                  max_val = exchange_max_val )
            
            ti = bounded_sigmoid( x = ti_logits, 
                                  min_val = exchange_min_val,
                                  max_val = exchange_max_val )
            
            with open(f'{out_folder}/PARAMS_HKY85_params.txt','w') as g:
                g.write('under BOUNDED SIGMOID activation')
                g.write(f'transition rate, ti: {ti}\n')
                g.write(f'transversion rate, tv: {tv}\n')
            
        
        # emissions: rate multipliers
        if 'rate_mult_logits' in dir(self.rate_matrix_module):
            rate_mult_min_val = self.rate_matrix_module.rate_mult_min_val
            rate_mult_max_val = self.rate_matrix_module.rate_mult_max_val
            rate_mult_logits = self.rate_matrix_module.rate_mult_logits
                
            rate_mult = bounded_sigmoid(x = rate_mult_logits, 
                                        min_val = rate_mult_min_val,
                                        max_val = rate_mult_max_val)

            with open(f'{out_folder}/PARAMS_rate_multipliers.txt','w') as g:
                [g.write(f'{elem.item()}\n') for elem in rate_mult]
        
        # emissions: equilibrium distribution
        if 'logits' in dir(self.indel_prob_module):
            equl_logits = self.indel_prob_module.logits
            equl_dist = nn.softmax( equl_logits, axis=1 )
            
            np.savetxt( f'{out_folder}/PARAMS_equilibriums.tsv', 
                        np.array(equl_dist), 
                        fmt = '%.4f',
                        delimiter= '\t' )
            
            with open(f'{out_folder}/PARAMS-ARR_equilibriums.npy','wb') as g:
                jnp.save(g, equl_dist)
                
                
        ### transitions
        # always write lambda and mu
        if 'tkf_lam_mu_logits' in dir(self.transitions_module):
            lam_min_val = self.transitions_module.lam_min_val
            lam_max_val = self.transitions_module.lam_max_val
            offs_min_val = self.transitions_module.offs_min_val
            offs_max_val = self.transitions_module.offs_max_val
            lam_mu_logits = self.transitions_module.tkf_lam_mu_logits
        
            lam = bounded_sigmoid(x = lam_mu_logits[0],
                                  min_val = lam_min_val,
                                  max_val = lam_max_val)
            
            offset = bounded_sigmoid(x = lam_mu_logits[1],
                                     min_val = offs_min_val,
                                     max_val = offs_max_val)
            mu = lam / ( 1 -  offset) 
            
            with open(f'{out_folder}/PARAMS_{self.indel_model_type}_indel_params.txt','w') as g:
                g.write(f'insert rate, lambda: {lam}\n')
                g.write(f'deletion rate, mu: {mu}\n')
        
        # if tkf92, have extra r_ext param
        if 'r_extend_logits' in dir(self.transitions_module):
            r_extend_min_val = self.transitions_module.r_extend_min_val
            r_extend_max_val = self.transitions_module.r_extend_max_val
            r_extend_logits = self.transitions_module.r_extend_logits
            
            r_extend = bounded_sigmoid(x = r_extend_logits,
                                       min_val = r_extend_min_val,
                                       max_val = r_extend_max_val)
            
            mean_indel_lengths = 1 / (1 - r_extend)
            
            with open(f'{out_folder}/PARAMS_{self.indel_model_type}_indel_params.txt','a') as g:
                g.write(f'extension prob, r: ')
                [g.write(f'{elem}\t') for elem in r_extend]
                g.write('\n')
                g.write(f'mean indel length: ')
                [g.write(f'{elem}\t') for elem in mean_indel_lengths]
                g.write('\n')
    

class IndpHKY85LoadAll(IndpHKY85FitAll):
    """
    uses "FromFile" methods for all components
    
    
    
    unique methods
    ===============
        - setup    
        - write_params: make this a dummy function
    
    
    main methods inherited from IndpPairHMMFitBoth:
    ===============================================        
        - __call__: calculate loss based on joint prob P(anc, desc, align);
                   use this during training; is jit compatible
        
        - calculate_all_loglikes: calculate joint prob P(anc, desc, align),
                   conditional prob P(desc, align | anc), and both marginals
                   P(desc) and P(anc); use this during final eval; is also
                   jit compatible

    internal methods from IndpPairHMMFitBoth:
    ==========================================
        - _joint_logprob_align
        - _marginalize_over_times
        - _get_scoring_matrices
    """
    config: dict
    name: str
    
    def setup(self):
        num_emit_site_classes = self.config['num_emit_site_classes']
        self.indel_model_type = self.config['indel_model_type']
        self.norm_loss_by = self.config['norm_loss_by']
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        
        ### how to score emissions from indel sites
        self.indel_prob_module = LogEqulVecFromFile(config = self.config,
                                                   name = f'get equilibrium')
        
        
        ### rate matrix to score emissions from match sites
        self.rate_matrix_module = HKY85FromFile(config = self.config,
                                                 name = f'get rate matrix')
        
        
        ### probability of site classes
        self.site_class_probability_module = SiteClassLogprobsFromFile(config = self.config,
                                                 name = f'get site class probabilities')
        
        
        ### TKF91 or TKF92
        ### make sure you're loading from a model file here
        if self.indel_model_type == 'tkf91':
            self.transitions_module = TKF91TransitionLogprobsFromFile(config = self.config,
                                                     name = f'tkf91 indel model')
        
        elif self.indel_model_type == 'tkf92':
            self.transitions_module = TKF92TransitionLogprobsFromFile(config = self.config,
                                                     name = f'tkf92 indel model')
    
    def write_params(self,
                     **kwargs):
        pass


class OneClassMarkovHKY85FitAll(MarkovPairHMM):
    """
    uses HKY85 for susbtitution model; essentially the same as IndpHKY85FitAll
      if no hidden site classes
    
    
    unique methods
    ===============
        - setup    
        - write_params: write the parameters to files
    
    
    main methods inherited from MarkovPairHMM:
    ===============================================        
        - __call__: calculate loss based on joint prob P(anc, desc, align);
                   use this during training; is jit compatible
        
        - calculate_all_loglikes: calculate joint prob P(anc, desc, align),
                   conditional prob P(desc, align | anc), and both marginals
                   P(desc) and P(anc); use this during final eval; is also
                   jit compatible

    internal methods from MarkovPairHMM:
    ==========================================
        - _joint_logprob_align
        - _marginalize_over_times
        - _get_scoring_matrices
    """
    config: dict
    name: str
    
    def setup(self):
        """
        difference: use HKY85 for self.rate_matrix_module
        """
        assert self.config['num_emit_site_classes'] == 1
        assert self.config['num_tkf_site_classes'] == 1
        self.num_site_classes = 1
        
        self.indel_model_type = 'tkf92'
        self.norm_loss_by = self.config['norm_loss_by']
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        
        ### how to score emissions from indel sites
        self.indel_prob_module = LogEqulVecFromCounts(config = self.config,
                                                   name = f'get equilibrium')
            
        
        ### rate matrix to score emissions from match sites
        self.rate_matrix_module = HKY85(config = self.config,
                                        name = f'get rate matrix')
        
        
        ### probabilities for rate classes themselves
        self.site_class_probability_module = SiteClassLogprobs(config = self.config,
                                                  name = f'get site class probabilities')
        
       
        ## probabilities of transitions
        self.transitions_module = TKF92TransitionLogprobs(config = self.config,
                                                     name = f'tkf92 indel model')
    
    
    def write_params(self,
                     t_array,
                     out_folder: str):
        """
        difference: explicitly write "ti" and "tv" to text files
        """
        out = self._get_scoring_matrices(t_array=t_array,
                                        sow_intermediates=False)
        
        # normalized rate matrix
        normalized_rate_matrix = out['normalized_rate_matrix']
        
        with open(f'{out_folder}/normalized_rate_matrix.npy', 'wb') as g:
            np.save(g, normalized_rate_matrix)
        
        np.savetxt( f'{out_folder}/ASCII_normalized_rate_matrix.tsv', 
                    np.array(normalized_rate_matrix), 
                    fmt = '%.4f',
                    delimiter= '\t' )
        
        # matrix that you apply expm() to
        to_expm = np.squeeze( out['to_expm'] )
        
        with open(f'{out_folder}/to_expm.npy', 'wb') as g:
            np.save(g, to_expm)
        
        if len(to_expm.shape) <= 2:
            np.savetxt( f'{out_folder}/ASCII_to_expm.tsv', 
                        to_expm, 
                        fmt = '%.4f',
                        delimiter= '\t' )
        
        del to_expm, g
        
        
        # other emission matrices; exponentiate them first
        for key in ['logprob_emit_at_indel', 
                    'joint_logprob_emit_at_match']:
            mat = np.exp(out[key])
            new_key = key.replace('logprob','prob')
            
            with open(f'{out_folder}/{new_key}.npy', 'wb') as g:
                np.save(g, mat)
            
            mat = np.squeeze(mat)
            if len(mat.shape) <= 2:
                np.savetxt( f'{out_folder}/ASCII_{new_key}.tsv', 
                            np.array(mat), 
                            fmt = '%.4f',
                            delimiter= '\t' )
            
            del key, mat, g
            
        for key, mat in out['all_transit_matrices'].items():
            mat = np.exp(mat)
            new_key = key.replace('logprob','prob')
            
            with open(f'{out_folder}/{new_key}_transit_matrix.npy', 'wb') as g:
                np.save(g, mat)
            
            mat = np.squeeze(mat)
            if len(mat.shape) <= 2:
                np.savetxt( f'{out_folder}/ASCII_{new_key}_transit_matrix.tsv', 
                            np.array(mat), 
                            fmt = '%.4f',
                            delimiter= '\t' )
            
            del key, mat, g
        
        
        ###############
        ### extract   #
        ###############
        ### site class probs
        if 'class_logits' in dir(self.site_class_probability_module):
            class_probs = nn.softmax(self.site_class_probability_module.class_logits)
            with open(f'{out_folder}/PARAMS_class_probs.txt','w') as g:
                [g.write(f'{elem.item()}\n') for elem in class_probs]
        
        
        ### emissions
        # exchangeabilities
        if 'exchangeabilities_logits_vec' in dir(self.rate_matrix_module):
            exchange_min_val = self.rate_matrix_module.exchange_min_val
            exchange_max_val = self.rate_matrix_module.exchange_max_val
            tv_logits = self.rate_matrix_module.exchangeabilities_logits_vec[0]
            ti_logits = self.rate_matrix_module.exchangeabilities_logits_vec[1]
            
            tv = bounded_sigmoid( x = tv_logits, 
                                  min_val = exchange_min_val,
                                  max_val = exchange_max_val )
            
            ti = bounded_sigmoid( x = ti_logits, 
                                  min_val = exchange_min_val,
                                  max_val = exchange_max_val )
            
            with open(f'{out_folder}/PARAMS_HKY85_params.txt','w') as g:
                g.write('under BOUNDED SIGMOID activation')
                g.write(f'transition rate, ti: {ti}\n')
                g.write(f'transversion rate, tv: {tv}\n')
            
        
        # emissions: rate multipliers
        if 'rate_mult_logits' in dir(self.rate_matrix_module):
            rate_mult_min_val = self.rate_matrix_module.rate_mult_min_val
            rate_mult_max_val = self.rate_matrix_module.rate_mult_max_val
            rate_mult_logits = self.rate_matrix_module.rate_mult_logits
                
            rate_mult = bounded_sigmoid(x = rate_mult_logits, 
                                        min_val = rate_mult_min_val,
                                        max_val = rate_mult_max_val)

            with open(f'{out_folder}/PARAMS_rate_multipliers.txt','w') as g:
                [g.write(f'{elem.item()}\n') for elem in rate_mult]
        
        # emissions: equilibrium distribution
        if 'logits' in dir(self.indel_prob_module):
            equl_logits = self.indel_prob_module.logits
            equl_dist = nn.softmax( equl_logits, axis=1 )
            
            np.savetxt( f'{out_folder}/PARAMS_equilibriums.tsv', 
                        np.array(equl_dist), 
                        fmt = '%.4f',
                        delimiter= '\t' )
            
            with open(f'{out_folder}/PARAMS-ARR_equilibriums.npy','wb') as g:
                jnp.save(g, equl_dist)
                
                
        ### transitions
        # always write lambda and mu
        if 'tkf_lam_mu_logits' in dir(self.transitions_module):
            lam_min_val = self.transitions_module.lam_min_val
            lam_max_val = self.transitions_module.lam_max_val
            offs_min_val = self.transitions_module.offs_min_val
            offs_max_val = self.transitions_module.offs_max_val
            lam_mu_logits = self.transitions_module.tkf_lam_mu_logits
        
            lam = bounded_sigmoid(x = lam_mu_logits[0],
                                  min_val = lam_min_val,
                                  max_val = lam_max_val)
            
            offset = bounded_sigmoid(x = lam_mu_logits[1],
                                     min_val = offs_min_val,
                                     max_val = offs_max_val)
            mu = lam / ( 1 -  offset) 
            
            with open(f'{out_folder}/PARAMS_{self.indel_model_type}_indel_params.txt','w') as g:
                g.write(f'insert rate, lambda: {lam}\n')
                g.write(f'deletion rate, mu: {mu}\n')
        
        # if tkf92, have extra r_ext param
        if 'r_extend_logits' in dir(self.transitions_module):
            r_extend_min_val = self.transitions_module.r_extend_min_val
            r_extend_max_val = self.transitions_module.r_extend_max_val
            r_extend_logits = self.transitions_module.r_extend_logits
            
            r_extend = bounded_sigmoid(x = r_extend_logits,
                                       min_val = r_extend_min_val,
                                       max_val = r_extend_max_val)
            
            mean_indel_lengths = 1 / (1 - r_extend)
            
            with open(f'{out_folder}/PARAMS_{self.indel_model_type}_indel_params.txt','a') as g:
                g.write(f'extension prob, r: ')
                [g.write(f'{elem}\t') for elem in r_extend]
                g.write('\n')
                g.write(f'mean indel length: ')
                [g.write(f'{elem}\t') for elem in mean_indel_lengths]
                g.write('\n')







# def IndpHKY85FitIndelOnly(IndpHKY85FitAll):
#     """
#     uses "FromFile" parts for rate matrix, equilibrium distribution, and 
#       rate class multipliers
    
#     only find lambda, mu, and (if TKF92) r
    
    
#     unique methods
#     ===============
#         - setup    
    
    
#     methods from IndpHKY85FitAll:
#     =============================
#         - write_params: write the parameters to files
    
    
#     main methods inherited from IndpPairHMMFitBoth:
#     ===============================================        
#         - __call__: calculate loss based on joint prob P(anc, desc, align);
#                    use this during training; is jit compatible
        
#         - calculate_all_loglikes: calculate joint prob P(anc, desc, align),
#                    conditional prob P(desc, align | anc), and both marginals
#                    P(desc) and P(anc); use this during final eval; is also
#                    jit compatible

#     internal methods from IndpPairHMMFitBoth:
#     ==========================================
#         - _joint_logprob_align
#         - _marginalize_over_times
#         - _get_scoring_matrices
#     """
#     config: dict
#     name: str
    
#     def setup(self):
#         """
#         difference: use HKY85FromFile for self.rate_matrix_module
#         """
        
#         num_emit_site_classes = self.config['num_emit_site_classes']
#         self.indel_model_type = self.config['indel_model_type']
#         self.norm_loss_by = self.config['norm_loss_by']
#         self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        
#         ### how to score emissions from indel sites
#         self.indel_prob_module = LogEqulVecFromFile(config = self.config,
#                                                    name = f'get equilibrium')
        
        
#         ### rate matrix to score emissions from match sites
#         self.rate_matrix_module = HKY85FromFile(config = self.config,
#                                                  name = f'get rate matrix')
        
        
#         ### probabilities for rate classes themselves
#         self.site_class_probability_module = SiteClassLogprobs(config = self.config,
#                                                  name = f'get site class probabilities')
        
#         # this only makes sense for TKF92, where r is tied to a site class
#         assert self.indel_model_type == 'tkf92'
#         self.transitions_module = TKF92TransitionLogprobs(config = self.config,
#                                                  name = f'tkf92 indel model')
