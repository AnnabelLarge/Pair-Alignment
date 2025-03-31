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

from models.simple_site_class_predict.general_emission_models import (LogEqulVecFromCounts,
                                                       LogEqulVecPerClass,
                                                       LogEqulVecFromFile,
                                                       HKY85,
                                                       HKY85FromFile,
                                                       SiteClassLogprobs,
                                                       SiteClassLogprobsFromFile)
from models.simple_site_class_predict.PairHMM_indp_sites import IndpPairHMMFitBoth
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
                     pred_config,
                     tstate,
                     out_folder: str):
        """
        difference: explicitly write "ti" and "tv" to text files
        """
        
        params_dict = tstate.params['params']
        
        
        ##################################################
        ### use default values, if ranges aren't found   #
        ##################################################
        with open(f'{out_folder}/ranges_used.tsv','w') as g:
            g.write('Ranges used to convert params (if values were not provided, noted below)\n\n')
            
        def read_pred_config(key, default_tup):
            if key not in pred_config.keys():
                with open(f'{out_folder}/ranges_used.tsv','a') as g:
                    g.write(f'{key}: {default_tup} [NOT PROVIDED; USED DEFAULT VALUE]\n')                
                return default_tup
            
            else:
                with open(f'{out_folder}/ranges_used.tsv','a') as g:
                    g.write(f'{key}: {pred_config[key]}\n')
                return pred_config[key]
        
        
        out = read_pred_config( 'exchange_range', (1e-4, 10) )
        exchange_min_val, exchange_max_val = out
        del out
        
        out = read_pred_config( 'rate_mult_range', (0.01, 10) )
        rate_mult_min_val, rate_mult_max_val = out
        del out
        
        out = read_pred_config( 'lambda_range', (pred_config['tkf_err'], 3) )
        lam_min_val, lam_max_val = out
        del out
         
        out = read_pred_config( 'offset_range', (pred_config['tkf_err'], 0.333) )
        offs_min_val, offs_max_val = out
        del out
        
        
        
        ###############
        ### extract   #
        ###############
        ### site class probs
        if 'get site class probabilities' in params_dict.keys():
            class_logits = params_dict['get site class probabilities']['class_logits']
            class_probs = nn.softmax(class_logits)
            with open(f'{out_folder}/PARAMS_class_probs.txt','w') as g:
                [g.write(f'{elem.item()}\n') for elem in class_probs]
                
                
        ### emissions
        if 'get rate matrix' in params_dict.keys():
            
            if 'exchangeabilities' in params_dict['get rate matrix']:
                exch_logits = params_dict['get rate matrix']['exchangeabilities']
                tv_logits = exch_logits[0]
                ti_logits = exch_logits[1]
                
                tv = bounded_sigmoid( x = tv_logits, 
                                      min_val = exchange_min_val,
                                      max_val = exchange_max_val )
                
                ti = bounded_sigmoid( x = ti_logits, 
                                      min_val = exchange_min_val,
                                      max_val = exchange_max_val )
                
                with open(f'{out_folder}/PARAMS_HKY85_params.txt','w') as g:
                    g.write(f'transition rate, ti: {ti}\n')
                    g.write(f'transversion rate, tv: {tv}\n')
                
                
            if 'rate_multipliers' in params_dict['get rate matrix']:
                rate_mult_logits = params_dict['get rate matrix']['rate_multipliers']
                rate_mult = bounded_sigmoid(x = rate_mult_logits, 
                                            min_val = rate_mult_min_val,
                                            max_val = rate_mult_max_val)
    
                with open(f'{out_folder}/PARAMS_rate_multipliers.txt','w') as g:
                    [g.write(f'{elem.item()}\n') for elem in rate_mult]
        
        if 'get equilibrium' in params_dict.keys():
            equl_logits = params_dict['get equilibrium']['Equilibrium distr.']
            equl_dist = nn.softmax( equl_logits, axis=1 )
            
            np.savetxt( f'{out_folder}/PARAMS_equilibriums.tsv', 
                        np.array(equl_dist), 
                        fmt = '%.4f',
                        delimiter= '\t' )
            
            with open(f'{out_folder}/PARAMS-ARR_equilibriums.npy','wb') as g:
                jnp.save(g, equl_dist)
                
        ### transitions
        # tkf91
        if 'tkf91 indel model' in params_dict.keys():
            lam_mu_logits = params_dict['tkf91 indel model']['TKF91 lambda, mu']
            
            lam = bounded_sigmoid(x = lam_mu_logits[0],
                                  min_val = lam_min_val,
                                  max_val = lam_max_val)
            
            offset = bounded_sigmoid(x = lam_mu_logits[1],
                                     min_val = offs_min_val,
                                     max_val = offs_max_val)
            mu = lam / ( 1 -  offset) 
            
            with open(f'{out_folder}/PARAMS_tkf91_indel_params.txt','w') as g:
                g.write(f'insert rate, lambda: {lam}\n')
                g.write(f'deletion rate, mu: {mu}\n')
            
        # tkf92
        elif 'tkf92 indel model' in params_dict.keys():
            # also need range for r values
            out = read_pred_config( 'r_range', (pred_config['tkf_err'], 0.8) )
            r_extend_min_val, r_extend_max_val = out
            del out
        
            lam_mu_logits = params_dict['tkf92 indel model']['TKF92 lambda, mu']
        
            lam = bounded_sigmoid(x = lam_mu_logits[0],
                                  min_val = lam_min_val,
                                  max_val = lam_max_val)
            
            offset = bounded_sigmoid(x = lam_mu_logits[1],
                                     min_val = offs_min_val,
                                     max_val = offs_max_val)
            mu = lam / ( 1 -  offset) 
            
            r_extend_logits = params_dict['tkf92 indel model']['TKF92 r extension prob']
            r_extend = bounded_sigmoid(x = r_extend_logits,
                                       min_val = r_extend_min_val,
                                       max_val = r_extend_max_val)
            
            mean_indel_lengths = 1 / (1 - r_extend)
            
            with open(f'{out_folder}/PARAMS_tkf92_indel_params.txt','w') as g:
                g.write(f'insert rate, lambda: {lam}\n')
                g.write(f'deletion rate, mu: {mu}\n')
                g.write(f'extension prob, r: ')
                [g.write(f'{elem}\t') for elem in r_extend]
                g.write('\n')
                g.write(f'mean indel length: ')
                [g.write(f'{elem}\t') for elem in mean_indel_lengths]
                g.write('\n')
    

def IndpHKY85FitIndelOnly(IndpHKY85FitAll):
    """
    uses "FromFile" parts for rate matrix, equilibrium distribution, and 
      rate class multipliers
    
    only find lambda, mu, and (if TKF92) r
    
    
    unique methods
    ===============
        - setup    
    
    
    methods from IndpHKY85FitAll:
    =============================
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
        difference: use HKY85FromFile for self.rate_matrix_module
        """
        
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
        
        
        ### probabilities for rate classes themselves
        self.site_class_probability_module = SiteClassLogprobs(config = self.config,
                                                 name = f'get site class probabilities')
        
        # this only makes sense for TKF92, where r is tied to a site class
        assert self.indel_model_type == 'tkf92'
        self.transitions_module = TKF92TransitionLogprobs(config = self.config,
                                                 name = f'tkf92 indel model')


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