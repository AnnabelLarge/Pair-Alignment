#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 02:03:13 2025

@author: annabel


if loading from files, provide parameters in PROBABILITY SPACE 

most of these rate matrix functions are similar to protein_emission_models


modules:
========
'LogEqulVecFromCounts',
'LogEqulVecFromFile',
'LogEqulVecPerClass',

'HKY85',
'HKY85FromFile',
'RateMatFitBoth',
'RateMatFromFile',

'SiteClassLogprobs',
'SiteClassLogprobsFromFile'


not in use:
============
'RateMatFitRateMult',
'PerClassRateMat'

"""
# jumping jax and leaping flax
from flax import linen as nn
import jax
import jax.numpy as jnp

from functools import partial

from models.model_utils.BaseClasses import ModuleBase
from utils.pairhmm_helpers import (bounded_sigmoid,
                                   safe_log)


########################################
### helpers only for emission models   #
########################################
def bounded_sigmoid_inverse(y, min_val, max_val, eps=1e-4):
    """
    note: this is only for logit initialization; jnp.clip has bad 
          gradients at extremes
    """
    y = jnp.clip(y, min_val + eps, max_val - eps)
    return safe_log( (y - min_val) / (max_val - y) )

def save_interms(param_name, mat):
    with open(f'pred_{param_name}.npy','wb') as g:
        jnp.save(g, mat)



###############################################################################
### Probability of being in site classes   ####################################
###############################################################################
class SiteClassLogprobs(ModuleBase):
    """
    required in pred_config:
        - num_emit_site_classes: number of classes
    
    params:
        - class_logits: (C,)
    
    __call__ returns:
        - logP(site classes): (C,)
    """
    config: dict
    name: str
    
    def setup(self):
        self.n_classes = self.config['num_emit_site_classes']
        
        if self.n_classes > 1:
            self.class_logits = self.param('class_logits',
                                            nn.initializers.normal(),
                                            (self.n_classes,),
                                            jnp.float32)
        
    def __call__(self,
                 sow_intermediates):
        
        if self.n_classes > 1:
            log_class_probs = nn.log_softmax(self.class_logits)
        
            if sow_intermediates:
                for i in range(log_class_probs.shape[0]):
                    val_to_write = jnp.exp( log_class_probs[i] )
                    lab = f'{self.name}/prob of class {i}'
                    self.sow_histograms_scalars(mat= val_to_write, 
                                                label=lab, 
                                                which='scalars')
                    del lab
        
        else:
            log_class_probs = jnp.array([0])
            
        return log_class_probs


class SiteClassLogprobsFromFile(ModuleBase):
    """
    required in pred_config:
      - filenames: dictionary of files to load
          > pred_config["filenames"]["class_probs"]: file containing the 
            class probabilities to load
    
    params: None
    
    __call__ returns:
        - logP(site classes): (C,)
    """
    config: dict
    name: str
    
    def setup(self):
        in_file = self.config['filenames']['class_probs']
        with open(in_file,'rb') as f:
            class_probs = jnp.load(f)
        self.log_class_probs = safe_log(class_probs)
    
    def __call__(self,
                 **kwargs):
        return self.log_class_probs


###############################################################################
### GENERAL SUBSTITUTION RATE MATRIX MODELS   #################################
###############################################################################
class RateMatFromFile(ModuleBase):
    """
    ABOUT:
    ======
    return (rho * Q), to be directly used in matrix exponential

    load exchangeabilities and rate multiplier from files; exchangeabilties 
      could either be a vector of values (which will be transformed into a 
      square matrix), or the final square matrix
    
    return the normalized rate matrix, as well as the rate matrix after
      multiplying by rate multipliers
     
    out = (rate_mat_times_rho, rate_mat)
    
    
    tl;dr:
    =======
    required in pred_config:
        - num_emit_site_classes: number of classes
        - filenames: dictionary of files to load
          > pred_config["filenames"]["rate_mult"]
          > pred_config["filenames"]["exch"]
    
    __call__ returns:
        - rate matrix times rate multipliers: (C_curr, |\omega_Y|, |\omega_X|)
    """
    config: dict
    name: str
    
    def setup(self):
        # could still have multiple site classes
        self.num_emit_site_classes = self.config['num_emit_site_classes']
        rate_multiplier_file = self.config['filenames']['rate_mult']
        exchangeabilities_file = self.config['filenames']['exch']
        
        
        ### EXCHANGEABILITIES ALREADY TRANSFORMED:
        with open(exchangeabilities_file,'rb') as f:
            exch_from_file = jnp.load(f)
        
        # if providing a vector, need to prepare a square exchangeabilities matrix
        if len(exch_from_file.shape) == 2:
            self.exchangeabilities = exch_from_file
        
        # otherwise, use the matrix as-is
        elif len(exch_from_file.shape) == 1:
            self.exchangeabilities = self._upper_tri_vector_to_sym_matrix( exch_from_file )
                    
        
        ### RATE MULTIPLIERS: (c,)
        if self.num_emit_site_classes > 1:
            with open(rate_multiplier_file, 'rb') as f:
                self.rate_multiplier = jnp.load(f)
        else:
            self.rate_multiplier = jnp.array([1])
    
    
    def __call__(self,
                 logprob_equl,
                 sow_intermediates: bool,
                 *args,
                 **kwargs):
        # (C, alph)
        equl = jnp.exp(logprob_equl)
        
        out =  self._prepare_rate_matrix(exchangeabilities = self.exchangeabilities,
                                   equilibrium_distributions = equl,
                                   sow_intermediates = sow_intermediates,
                                   rate_multiplier = self.rate_multiplier)
        return out
    
    
    def _upper_tri_vector_to_sym_matrix( self, 
                                         vec ):
        # automatically detect emission alphabet size
        # 6 = DNA (alphabet size = 4)
        # 2016 = codons (alphabet size = 64)
        # 190 = proteins (alphabet size = 20)
        if vec.shape[-1] == 6:
            emission_alphabet_size = 4
        
        elif vec.shape[-1] == 2016:
            emission_alphabet_size = 64
        
        elif vec.shape[-1] == 190:
            emission_alphabet_size = 20
        
        # fill upper triangular part of matrix
        out_size = (emission_alphabet_size, emission_alphabet_size)
        upper_tri_exchang = jnp.zeros( out_size )
        idxes = jnp.triu_indices(emission_alphabet_size, k=1)  
        upper_tri_exchang = upper_tri_exchang.at[idxes].set(vec)
        
        # reflect across diagonal
        mat = (upper_tri_exchang + upper_tri_exchang.T)
        
        return mat
    
    
    def _prepare_rate_matrix(self,
                            exchangeabilities,
                            equilibrium_distributions,
                            rate_multiplier,
                            sow_intermediates: bool):
        C = equilibrium_distributions.shape[0]
        alphabet_size = equilibrium_distributions.shape[1]

        # just in case, zero out the diagonals of exchangeabilities
        exchangeabilities_without_diags = exchangeabilities * ~jnp.eye(alphabet_size, dtype=bool)

        # Q = chi * pi
        rate_mat_without_diags = jnp.einsum('ij, cj -> cij', 
                                            exchangeabilities_without_diags, 
                                            equilibrium_distributions)
    
        row_sums = rate_mat_without_diags.sum(axis=2) 
        ones_diag = jnp.eye( alphabet_size, dtype=bool )[None,:,:]
        ones_diag = jnp.broadcast_to( ones_diag, (C,
                                                  ones_diag.shape[1],
                                                  ones_diag.shape[2]) )
        diags_to_add = -jnp.einsum('ci,cij->cij', row_sums, ones_diag)
        subst_rate_mat = rate_mat_without_diags + diags_to_add
        
        # normalize by default
        diag = jnp.einsum("cii->ci", subst_rate_mat) 
        norm_factor = -jnp.sum(diag * equilibrium_distributions, axis=1)[:,None,None]
        subst_rate_mat = subst_rate_mat / norm_factor
        
        # after normalizing, multiply by a rate scalar
        rate_mat_times_rho = jnp.einsum( 'c,cij->cij', 
                                         rate_multiplier, 
                                         subst_rate_mat ) 
        
        return rate_mat_times_rho


class RateMatFitBoth(RateMatFromFile):
    """
    return (rho * Q), to be directly used in matrix exponential

    load initial values for exchangeabilities from file

    rate matrix is normalized to one substitution, THEN multiplied by a scalar 
      multiple; first hidden site class has rate of 1, then subsequent ones 
      are fit with gradient descent (rho = [1, rate2, rate3, ...])
    
    params: 
        - exch_raw; length of vector is ( alph * (alph - 1) ) / 2
        - rate_mult_logits( C, )
    
    valid ranges:
        - exchangeabilities: (0, inf); bound values with exchange_range
        - rate_mult: (0, inf); bound values with rate_mult_range
    
    inherit the following from RateMatFromFile:
        - _upper_tri_vector_to_sym_matrix
        - _prepare_rate_matrix
    
    
    tl;dr:
    =======
    required in pred_config:
        - num_emit_site_classes: number of classes
        - rate_mult_activation: what kind of activation to use for rate multipliers
          > if this is "bound_sigmoid", also need:
            - rate_mult_range: range for bounded sigmoid transformation of 
              rate multiplier logits 
        - exchange_range: range for bounded sigmoid transformation of 
          exchangeability logits 
        - filenames: dictionary of files to load
          > pred_config["filenames"]["exch"]
    
    __call__ returns:
        - rate matrix times rate multipliers: (C_curr, |\omega_Y|, |\omega_X|)
    """
    config: dict
    name: str
    
    def setup(self):
        ########################
        ### standard options   #
        ########################
        self.num_emit_site_classes = self.config['num_emit_site_classes']
        exchangeabilities_file = self.config['filenames']['exch']
        self.rate_mult_activation = self.config['rate_mult_activation']
        
        if self.rate_mult_activation not in ['bound_sigmoid', 'softplus']:
            raise ValueError('Pick either: bound_sigmoid, softplus')
            
        ########################
        ### RATE MULTIPLIERS   #
        ########################
        # activation
        if self.rate_mult_activation == 'bound_sigmoid':
            out  = self.config.get( 'rate_mult_range',
                                   (0.01, 10) )
            self.rate_mult_min_val, self.rate_mult_max_val = out
            del out
            
            self.rate_multiplier_activation = partial(bounded_sigmoid,
                                                      min_val = self.rate_mult_min_val,
                                                      max_val = self.rate_mult_max_val)
        
        elif self.rate_mult_activation == 'softplus':
            self.rate_multiplier_activation = jax.nn.softplus
        
        
        # initializers
        if self.num_emit_site_classes > 1:
            self.rate_mult_logits = self.param('rate_multipliers',
                                               nn.initializers.normal(),
                                               (self.num_emit_site_classes,),
                                               jnp.float32)
    
            
        #####################################
        ### EXCHANGEABILITIES AS A VECTOR   #
        #####################################
        with open(exchangeabilities_file,'rb') as f:
            vec = jnp.load(f)
            
        out  = self.config.get( 'exchange_range',
                               (1e-4, 12) )
        self.exchange_min_val, self.exchange_max_val = out
        del out
        
        transformed_vec = bounded_sigmoid_inverse(vec, 
                                                  min_val = self.exchange_min_val,
                                                  max_val = self.exchange_max_val)
        
        self.exchange_activation = partial(bounded_sigmoid,
                                           min_val = self.exchange_min_val,
                                           max_val = self.exchange_max_val)
        
        self.exchangeabilities_logits_vec = self.param("exchangeabilities", 
                                                       lambda rng, shape: transformed_vec,
                                                       transformed_vec.shape )
        
        
    def __call__(self,
                 logprob_equl,
                 sow_intermediates: bool,
                 *args,
                 **kwargs):
        # pi; one per class
        equl = jnp.exp(logprob_equl)
        
        #######################
        ### rate multiplier   #
        #######################
        if self.num_emit_site_classes > 1:
            
            ### apply activation of choice
            if sow_intermediates:
                for i in range(self.rate_mult_logits.shape[0]):
                    val_to_write = self.rate_mult_logits[i]
                    act = self.rate_mult_activation
                    lab = f'{self.name}/logit BEFORE {act} activation- rate multiplier for class {i}'
                    self.sow_histograms_scalars(mat= val_to_write, 
                                                label=lab, 
                                                which='scalars')
                    del lab
                    
            rate_multiplier = self.rate_multiplier_activation( self.rate_mult_logits )
            
            if sow_intermediates:
                for i in range(rate_multiplier.shape[0]):
                    val_to_write = rate_multiplier[i]
                    act = self.rate_mult_activation
                    lab = f'{self.name}/value AFTER {act} activation- rate multiplier for class {i}'
                    self.sow_histograms_scalars(mat= val_to_write, 
                                                label=lab, 
                                                which='scalars')
                    del lab
                    
        else:
            rate_multiplier = jnp.array([1])
        
        
        ###################################
        ### chi; one shared all classes   #
        ###################################
        ### apply activation of choice
        if sow_intermediates:
            self.sow_histograms_scalars(mat= self.exchangeabilities_logits_vec, 
                                        label= 'logit BEFORE bound_sigmoid activation- exchangeabilities', 
                                        which='scalars')
        
        upper_triag_values = self.exchange_activation( self.exchangeabilities_logits_vec )
    
        if sow_intermediates:
            self.sow_histograms_scalars(mat = upper_triag_values, 
                                        label = 'value AFTER bound_sigmoid activation- exchangeabilities', 
                                        which='scalars')
        
        ### update values
        exchangeabilities_mat = self._upper_tri_vector_to_sym_matrix( upper_triag_values )
        
        # output is (c, i, j)
        out = self._prepare_rate_matrix(exchangeabilities = exchangeabilities_mat,
                                   equilibrium_distributions = equl,
                                   sow_intermediates = sow_intermediates,
                                   rate_multiplier = rate_multiplier)
        return out
            
            

###############################################################################
### DNA SUBSTITUTION RATE MATRIX MODELS   #####################################
###############################################################################
class HKY85(RateMatFitBoth):
    """
    return (rho * Q), to be directly used in matrix exponential
    
    inherit the following from RateMatFromFile:
        - _upper_tri_vector_to_sym_matrix
        - _prepare_rate_matrix
    
    inherit __call__ from RateMatFitBoth

    params: 
        - ti, tv: each are floats; store in (2,) matrix
          > ti is first value, tv is second
        - rate_mult_logits( C, )
    
    valid ranges:
        - exchangeabilities: (0, inf); bound values with exchange_range
        - rate_mult: (0, inf); bound values with rate_mult_range
    
    
    tl;dr:
    =======
    required in pred_config:
        - num_emit_site_classes: number of classes
        - rate_mult_range: range for bounded sigmoid transformation of 
          rate multiplier logits 
        - exchange_range: range for bounded sigmoid transformation of 
          exchangeability logits 
    
    __call__ returns:
        - rate matrix times rate multipliers: (C_curr, |\omega_Y|, |\omega_X|)
    """
    config: dict
    name: str
    
    def setup(self):
        self.num_emit_site_classes = self.config['num_emit_site_classes']
        self.rate_mult_activation = self.config['rate_mult_activation']
        
        if self.rate_mult_activation not in ['bound_sigmoid', 'softplus']:
            raise ValueError('Pick either: bound_sigmoid, softplus')
            
            
        ########################
        ### RATE MULTIPLIERS   #
        ########################
        # activation
        if self.rate_mult_activation == 'bound_sigmoid':
            out  = self.config.get( 'rate_mult_range',
                                   (0.01, 10) )
            self.rate_mult_min_val, self.rate_mult_max_val = out
            del out
            
            self.rate_multiplier_activation = partial(bounded_sigmoid,
                                                      min_val = self.rate_mult_min_val,
                                                      max_val = self.rate_mult_max_val)
        
        elif self.rate_mult_activation == 'softplus':
            self.rate_multiplier_activation = jax.nn.softplus
        
        
        # initializers
        if self.num_emit_site_classes > 1:
            self.rate_mult_logits = self.param('rate_multipliers',
                                               nn.initializers.normal(),
                                               (self.num_emit_site_classes,),
                                               jnp.float32)
            
        
        #########################
        ### EXCHANGEABILITIES   #
        #########################
        ti_tv_vec = self.param('exchangeabilities',
                               nn.initializers.normal(),
                               (2,),
                               jnp.float32)
        
        # order should be: tv, ti, tv, tv, ti, tv
        self.exchangeabilities_logits_vec = jnp.stack( [ ti_tv_vec[1], 
                                                         ti_tv_vec[0], 
                                                         ti_tv_vec[1], 
                                                         ti_tv_vec[1], 
                                                         ti_tv_vec[0], 
                                                         ti_tv_vec[1] ] )
        
        out  = self.config.get( 'exchange_range',
                               (1e-4, 12) )
        
        self.exchange_min_val, self.exchange_max_val = out
        del out
        
        self.exchange_activation = partial(bounded_sigmoid,
                                           min_val = self.exchange_min_val,
                                           max_val = self.exchange_max_val)
        

class HKY85FromFile(RateMatFromFile):
    """
    return (rho * Q), to be directly used in matrix exponential

    load ti, tv, and rate multipliers from files
    
    return the normalized rate matrix, as well as the rate matrix after
      multiplying by rate multipliers
     
    out = (rate_mat_times_rho, rate_mat)
    
    inherit the following from RateMatFromFile:
        - _upper_tri_vector_to_sym_matrix
        - _prepare_rate_matrix
    
    inherit __call__ from RateMatFromFile
    
    
    tl;dr:
    =======
    required in pred_config:
        - num_emit_site_classes: number of classes
        - filenames: dictionary of files to load
          > pred_config["filenames"]["rate_mult"]
          > pred_config["filenames"]["exch"]
    
    __call__ returns:
        - rate matrix times rate multipliers: (C_curr, |\omega_Y|, |\omega_X|)
    """
    config: dict
    name: str
    
    def setup(self):
        self.num_emit_site_classes = self.config['num_emit_site_classes']
        rate_multiplier_file = self.config['filenames']['rate_mult']
        exchangeabilities_file = self.config['filenames']['exch']
        
        
        ### EXCHANGEABILITIES: (A_from, A_to)
        with open(exchangeabilities_file,'rb') as f:
            ti_tv_vec_from_file = jnp.load(f)
            
        # order should be: tv, ti, tv, tv, ti, tv
        hky85_raw_vec = jnp.stack( [ ti_tv_vec_from_file[1], 
                                     ti_tv_vec_from_file[0], 
                                     ti_tv_vec_from_file[1], 
                                     ti_tv_vec_from_file[1], 
                                     ti_tv_vec_from_file[0], 
                                     ti_tv_vec_from_file[1] ] )
        
        self.exchangeabilities = self._upper_tri_vector_to_sym_matrix( hky85_raw_vec )
        
        ### RATE MULTIPLIERS: (c,)
        if self.num_emit_site_classes > 1:
            with open(rate_multiplier_file, 'rb') as f:
                self.rate_multiplier = jnp.load(f)
        else:
            self.rate_multiplier = jnp.array([1])
        
            

###############################################################################
### LOGPROB (emit at indel sites)   ###########################################
###############################################################################
class LogEqulVecPerClass(ModuleBase):
    """
    required in pred_config:
        - emission_alphabet_size: alphabet size
        - num_emit_site_classes: number of classes
    
    params:
        - logits: (C, \omega)
    
    __call__ returns:
        - logP(emission from alphabet): (C, \omega)
    """
    config: dict
    name: str
    
    def setup(self):
        emission_alphabet_size = self.config['emission_alphabet_size']
        num_emit_site_classes = self.config['num_emit_site_classes']
        self.logits = self.param('Equilibrium distr.',
                                 nn.initializers.normal(),
                                 (num_emit_site_classes, emission_alphabet_size),
                                 jnp.float32)
        
    def __call__(self,
                 sow_intermediates: bool,
                 *args,
                 **kwargs):
        
        out = nn.log_softmax( self.logits, axis = 1 )

        if sow_intermediates:
            for c in range(out.shape[0]):
                lab = f'{self.name}/equilibrium dist for class {c}'
                self.sow_histograms_scalars(mat= jnp.exp(out[c,...]), 
                                            label=lab, 
                                            which='scalars')
                del lab
        
        return out


class LogEqulVecFromFile(ModuleBase):
    """
    required in pred_config:
      - filenames: dictionary of files to load
          > pred_config["filenames"]["equl_dist"]: file containing the 
            equilibrium distribution to load
    
    params: None
    
    __call__ returns:
        - logP(emission from alphabet): (C, \omega)
    """
    config: dict
    name: str
    
    def setup(self):
        # (C, alph)
        equl_file = self.config['filenames']['equl_dist']
        
        with open(equl_file,'rb') as f:
            prob_equilibr = jnp.load(f, allow_pickle=True)

        self.logprob_equilibr = safe_log(prob_equilibr)
        
        
    def __call__(self,
                 *args,
                 **kwargs):
        # (C, alpha)
        return self.logprob_equilibr
    

class LogEqulVecFromCounts(ModuleBase):
    """
    Generate equilibrium distribution based on observed counts (used when 
      only one class)
    
    required in pred_config:
      - training_dset_emit_counts: emission counts from training set
    
    params: None
    
    __call__ returns:
        - logP(emission from alphabet): (1, \omega)
    """
    config: dict
    name: str
    
    def setup(self):
        # (alph,)
        training_dset_emit_counts = self.config['training_dset_emit_counts']
        
        prob_equilibr = training_dset_emit_counts/training_dset_emit_counts.sum()
        logprob_equilibr = safe_log( prob_equilibr )
        
        # expand to to (C=1, alpha)
        self.logprob_equilibr = logprob_equilibr[None,...]
        
        
    def __call__(self,
                 *args,
                 **kwargs):
        # (C, alpha)
        return self.logprob_equilibr







# class PerClassRateMat(RateMatFitBoth):
#     """
#     return (rho * Q), to be directly used in matrix exponential
    
#     inherit the following from RateMatFromFile:
#         - _upper_tri_vector_to_sym_matrix
#         - _prepare_rate_matrix
    
#     inherit __call__ from RateMatFitBoth

#     params: 
#         - exch_raw; length of vector is ( alph * (alph - 1) ) / 2
#         - rate_mult_logits( C, )
    
#     valid ranges:
#         - exchangeabilities: (0, inf); bound values with exchange_range
#         - rate_mult: (0, inf); bound values with rate_mult_range
    
#     HAVE TO PROVIDE emission_alphabet_size IN PRED CONFIG!!!
    
#     tl;dr:
#     =======
#     required in pred_config:
#         - emission_alphabet_size
#         - num_emit_site_classes: number of classes
#         - rate_mult_range: range for bounded sigmoid transformation of 
#           rate multiplier logits 
    
#     __call__ returns:
#         - rate matrix times rate multipliers: (C_curr, |\omega_Y|, |\omega_X|)
#     """
#     config: dict
#     name: str
    
#     def setup(self):
#         emission_alphabet_size = self.config['emission_alphabet_size']
#         self.num_emit_site_classes = self.config['num_emit_site_classes']
#         out  = self.config.get( 'exchange_range',
#                                (1e-4, 10) )
#         self.exchange_min_val, self.exchange_max_val = out
#         del out
        
#         out  = self.config.get( 'rate_mult_range',
#                                (0.01, 10) )
#         self.rate_mult_min_val, self.rate_mult_max_val = out
#         del out
        
        
#         ### EXCHANGEABILITIES VECTOR
#         # init logits
#         num_vars = int( (emission_alphabet_size * (emission_alphabet_size-1))/2 )
#         self.exchangeabilities_logits_vec = self.param('exchangeabilities',
#                                                        nn.initializers.normal(),
#                                                        (num_vars,),
#                                                        jnp.float32)
        
        
#         ### RATE MULTIPLIERS: (c-1,)
#         if self.num_emit_site_classes > 1:
#             # first class automatically has rate multiplier of one
#             self.rate_mult_logits = self.param('rate_multipliers',
#                                                nn.initializers.normal(),
#                                                (self.num_emit_site_classes-1,),
#                                                jnp.float32)

