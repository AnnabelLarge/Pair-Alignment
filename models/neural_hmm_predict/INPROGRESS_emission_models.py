#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 13:46:56 2025

@author: annabel

TODO: 
    - add FromFile versions, for unit testing
    - for anything with local params, could add postprocessing network here? 
"""
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm 

from models.BaseClasses import (neuralTKFModuleBase, 
                                ModuleBase)
from models.neural_hmm_predict.model_functions import (bound_sigmoid,
                                                       logprob_f81,
                                                       logprob_gtr)


###############################################################################
### Substitution models: F81   ################################################
###############################################################################
class GlobalF81(neuralTKFModuleBase):
    """
    One F81 model for all positions; normalize by default
    
    No reason for this to be a flax module, other than consistency I guess
    """
    config: dict
    name: str
    
    def setup(self):
        """
        set substitution rate to 1 by default
        
        Flax Module Parameters
        -----------------------
        None
        """
        self.global_rate_multiplier = jnp.ones( (1,1) ) #(1,1)
    
    def __call__(self,
                 log_equl,
                 t_array,
                 unique_time_per_sample,
                 sow_intermediates: bool,
                 *args,
                 **kwargs):
        """
        B: batch size
        L_align: length of alignment
        T: number of times in the grid
        A: alphabet size
        
        Arguments
        ----------
        log_equl : ArrayLike
            > if global:  (1, 1, A) 
            > if per-sample, per-position: (B, L_align, A)
            log-transformed equilibrium distribution
        
        t_array : ArrayLike, (T,) or (B,)
            branch lengths
            
        unique_time_per_sample : Bool
            whether there's one time per sample, or a grid of times you'll 
            marginalize over
        
        sow_intermediates : bool
            switch for tensorboard logging
        
        Returns
        --------
        ArrayLike
          > if unique time per sample: (1, 1, A, 2)
          > if not unique time per sample: (T, 1, 1, A, 2)
          log-probability matrix for emissions from match sites, which includes 
            placeholder dimensions for B and L
        """
        equl = jnp.exp(log_equl) #(B, L_align, A) or (1, 1, A)
        
        # shape of output
        #   if unique time per sample: (1, 1, A, 2)
        #   if not unique time per sample: (T, 1, 1, A, 2)
        return logprob_f81(equl = equl,
                           rate_multiplier = self.global_rate_multiplier,
                           t_array = t_array,
                           unique_time_per_sample = unique_time_per_sample)
    
class LocalF81(neuralTKFModuleBase):
    """
    Decide F81 model for each sample, each position; normalize then scale by
      rate multiplier for the site
     
    inherited from GlobalF81:
    --------------------------
    get_scoring_matrix
    """
    config: dict
    name: str
    
    def setup(self):
        """
        in_feats = number of features of input matrix
        
        Flax Module Parameters
        -----------------------
        self.final_project : Flax module
          > kernel : ArrayLike, (in_feats, 1)
          > bias : ArrayLike, (1)  
        """
        self.rate_mult_min_val, self.rate_mult_max_val  = self.config.get( 'rate_mult_range', 
                                                                           (0.01, 10) )
        
        name = f'{self.name}/Project to rate multipliers'
        self.final_project = nn.Dense(features = 1,
                                      use_bias = True,
                                      name = name)
    
    def __call__(self,
                 datamat,
                 log_equl,
                 t_array,
                 unique_time_per_sample,
                 sow_intermediates: bool,
                 *args,
                 **kwargs):
        """
        apply final linear projection and bound_sigmoid activation to get final 
          log-probability at match sites
        
        T: number of times in the grid
        B: batch size
        L_align: length of alignment
        H: input hidden dim
        A: alphabet size
        
        Arguments
        ----------
        datamat : ArrayLike, (B, L_align, in_feats)
        
        log_equl : ArrayLike
            > if global: (1, 1, A)
            > if per-sample, per-position: (B, L_align, A); this is usually
              what I'll provide to this class
            log-transformed equilibrium distribution
        
        t_array : ArrayLike, (T,) or (B,)
            branch lengths; eventually, marginalize over this
        
        unique_time_per_sample : Bool
            whether there's one time per sample, or a grid of times you'll 
            marginalize over
        
        sow_intermediates : bool
            switch for tensorboard logging
        
        Returns
        --------
        ArrayLike
          > if unique time per sample: (B, L_align, A, 2)
          > if not unique time per sample: (T, B, L_align, A, 2)
            log-probability matrix for emissions from match sites
        """
        # (B, L_align, H) -> (B, L_align, 1) -> (B, L_align)
        rate_mult_logits = self.final_project(datamat)[...,0] # (B, L_align)
        rate_multiplier = self.apply_bound_sigmoid_activation(logits = rate_mult_logits,
                                           min_val = self.rate_mult_min_val,
                                           max_val = self.rate_mult_max_val,
                                           param_name = 'rate mult.',
                                           sow_intermediates = sow_intermediates) # (B, L_align)
        
        # return substitution probabilities directly
        equl = jnp.exp(log_equl) # (B, L_align, A)
        
        # shape of output
        #   if unique time per sample: (B, L_align, A, 2)
        #   if not unique time per sample: (T, B, L_align, A, 2)
        return logprob_f81(equl = equl,
                           rate_multiplier = rate_multiplier,
                           t_array = t_array,
                           unique_time_per_sample = unique_time_per_sample)


###############################################################################
### Substitution models: GTR   ################################################
###############################################################################
# these are very parameter hungry; try them later
class GTRGlobalExchGlobalRateMult(neuralTKFModuleBase):
    """
    One set of GTR exchangeabilities for all positions; normalize by default,
    
    Regardless of dimensions of equilibrium distribution, total substitution
      rate is equal to one (no rate multipliers)
    """
    config: dict
    name: str
    
    def setup(self):
        emission_alphabet_size = self.config['emission_alphabet_size']
        self.exchange_min_val, self.exchange_max_val  = self.config.get( 'exchange_range', (1e-4, 12) )
        
        # initialize exchangeabilities
        if emission_alphabet_size == 20:
            num_exchange = 190
        
        elif emission_alphabet_size == 4:
            num_exchange = 6
        
        self.exch_logits = self.param("exchangeabilities", 
                                      nn.initializers.normal(),
                                      (1, 1, num_exchange,),
                                      jnp.float32 ) #(1, 1, n)
        
        # rate multiplier is one
        self.global_rate_multiplier = jnp.ones( (1,1) ) #(1,1)
    
    def __call__(self, 
                 datamat,
                 log_equl,
                 t_array,
                 unique_time_per_sample,
                 sow_intermediates: bool,
                 *args,
                 **kwargs):
    
    exch_upper_triag_values = self.apply_bound_sigmoid_activation( logits = self.exch_logits,
                                               min_val = self.exchange_min_val,
                                               max_val = self.exchange_max_val,
                                               param_name = 'exchangeabilities',
                                               sow_intermediates = sow_intermediates)
        
    return logprob_gtr( exch_upper_triag_values = exch_upper_triag_values,
                        equilibrium_distributions = jnp.exp(log_equl),
                        rate_multiplier = self.global_rate_multiplier,
                        t_array = t_array,
                        unique_time_per_sample = unique_time_per_sample )
    

class GTRGlobalExchLocalRateMult(neuralTKFModuleBase):
    """
    One set of GTR exchangeabilities for all positions, but a substitution
      rate multiplier per-sample, and per alignment column
    """
    config: dict
    name: str
    
    def setup(self):
        emission_alphabet_size = self.config['emission_alphabet_size']
        self.exchange_min_val, self.exchange_max_val  = self.config.get( 'exchange_range', (1e-4, 12) )
        self.rate_mult_min_val, self.rate_mult_max_val  = self.config.get( 'rate_mult_range', 
                                                                           (0.01, 10) )
        
        # initialize exchangeabilities
        if emission_alphabet_size == 20:
            num_exchange = 190
        
        elif emission_alphabet_size == 4:
            num_exchange = 6
        
        self.exch_logits = self.param("exchangeabilities", 
                                      nn.initializers.normal(),
                                      (1, 1, num_exchange,),
                                      jnp.float32 ) #(1, 1, n)
        
        # final projection for rate multipliers
        name = f'{self.name}/Project to rate multipliers'
        self.rate_mult_final_project = nn.Dense(features = 1,
                                                use_bias = True,
                                                name = name)
        
    def __call__(self, 
                 datamat,
                 log_equl,
                 t_array,
                 unique_time_per_sample,
                 sow_intermediates: bool,
                 *args,
                 **kwargs):
        ### rate multipliers come from input
        # (B, L_align, H) -> (B, L_align, 1) -> (B, L_align)
        rate_mult_logits = self.rate_mult_final_project(datamat)[...,0] # (B, L_align)
        rate_multiplier = self.apply_bound_sigmoid_activation(logits = rate_mult_logits,
                                           min_val = self.rate_mult_min_val,
                                           max_val = self.rate_mult_max_val,
                                           param_name = 'rate mult.',
                                           sow_intermediates = sow_intermediates) # (B, L_align)
        
        
        ### exchangeabilities are global
        exch_upper_triag_values = self.apply_bound_sigmoid_activation( logits = self.exch_logits,
                                                   min_val = self.exchange_min_val,
                                                   max_val = self.exchange_max_val,
                                                   param_name = 'exchangeabilities',
                                                   sow_intermediates = sow_intermediates)
        
        return logprob_gtr( exch_upper_triag_values = exch_upper_triag_values,
                            equilibrium_distributions = jnp.exp(log_equl),
                            rate_multiplier = rate_multiplier,
                            t_array = t_array,
                            unique_time_per_sample = unique_time_per_sample )


class GTRLocalExchLocalRateMult(neuralTKFModuleBase):
    """
    GTR with exchangeabilities and substitution rate multipliers per-sample, 
      and per alignment column
    """
    config: dict
    name: str
    
    def setup(self):
        emission_alphabet_size = self.config['emission_alphabet_size']
        self.exchange_min_val, self.exchange_max_val  = self.config.get( 'exchange_range', (1e-4, 12) )
        self.rate_mult_min_val, self.rate_mult_max_val  = self.config.get( 'rate_mult_range', 
                                                                           (0.01, 10) )
        
        # final projection for exchangeabilities
        if emission_alphabet_size == 20:
            num_exchange = 190
        
        elif emission_alphabet_size == 4:
            num_exchange = 6
        
        name = f'{self.name}/Project to exchangeabilties'
        self.exch_final_project = nn.Dense(features = num_exchange,
                                           use_bias = True,
                                           name = name)
        del name
        
        # final projection for rate multipliers
        name = f'{self.name}/Project to rate multipliers'
        self.rate_mult_final_project = nn.Dense(features = 1,
                                                use_bias = True,
                                                name = name)
        
    def __call__(self, 
                 datamat,
                 log_equl,
                 t_array,
                 unique_time_per_sample,
                 sow_intermediates: bool,
                 *args,
                 **kwargs):
        ### rate multipliers come from input
        # (B, L_align, H) -> (B, L_align, 1) -> (B, L_align)
        rate_mult_logits = self.rate_mult_final_project(datamat)[...,0] # (B, L_align)
        rate_multiplier = self.apply_bound_sigmoid_activation(logits = rate_mult_logits,
                                           min_val = self.rate_mult_min_val,
                                           max_val = self.rate_mult_max_val,
                                           param_name = 'rate mult.',
                                           sow_intermediates = sow_intermediates) # (B, L_align)
        
        
        ### exchangeabilities also come from input
        # (B, L_align, H) -> (B, L_align, n)
        exch_logits = self.exch_final_project(datamat)
        exch_upper_triag_values = self.apply_bound_sigmoid_activation( logits = exch_logits,
                                                   min_val = self.exchange_min_val,
                                                   max_val = self.exchange_max_val,
                                                   param_name = 'exchangeabilities',
                                                   sow_intermediates = sow_intermediates)
        
        return logprob_gtr( exch_upper_triag_values = exch_upper_triag_values,
                            equilibrium_distributions = jnp.exp(log_equl),
                            rate_multiplier = rate_multiplier,
                            t_array = t_array,
                            unique_time_per_sample = unique_time_per_sample )
        
    
###############################################################################
### Equilibrium distribution models   #########################################
###############################################################################
class GlobalEqul(neuralTKFModuleBase):
    """
    Set of logits for equilibrium distribution for all positions
    """
    config: dict
    name: str
    
    def setup(self):
        """
        A = alphabet size
        
        Flax Module Parameters
        -----------------------
        logits : ArrayLike (A,)
            initialize logits from unit normal
        
        """
        emission_alphabet_size = self.config['emission_alphabet_size']
        
        self.logits = self.param('Equilibrium distr.',
                                  nn.initializers.normal(),
                                  (emission_alphabet_size,),
                                  jnp.float32)
        
    def __call__(self,
                 sow_intermediates: bool,
                 *args,
                 **kwargs): 
        """
        Arguments
        ----------
        sow_intermediates : bool
            switch for tensorboard logging
         
        Returns
        --------
        ArrayLike, (1, 1, A) 
            scoring matrix for emissions from indels, which includes 
            placeholder dimensions for B and L
        """
        out = self.get_scoring_matrix(logits = self.logits,
                                      sow_intermediates = sow_intermediates)
        return out[None, None, :] #(1, 1, A)
    
    
    

class LocalEqul(GlobalEqul):
    """
    Set of logits for equilibrium distribution for each sample, each position
      like GlobalEqul, but you do a linear projection first
    """
    config: dict
    name: str
    
    def setup(self):
        """
        in_feats = number of features of input matrix
        A = alphabet size
        
        Flax Module Parameters
        -----------------------
        self.final_project : Flax module
          > kernel : ArrayLike, (in_feats, A)
          > bias : ArrayLike, (A)  
        """
        emission_alphabet_size = self.config['emission_alphabet_size']
        
        name = f'{self.name}/Project to equilibriums'
        self.final_project = nn.Dense(features = emission_alphabet_size,
                                      use_bias = True,
                                      name = name)
    
    def _call__(self,
                datamat,
                sow_intermediates: bool, 
                *args,
                **kwargs):
        """
        apply final linear projection and log_softmax to get final 
          equilibrium distribution
        
        B: batch size
        L_align: length of alignment
        H: input hidden dim
        A: final alphabet size
        
        Arguments
        ----------
        datamat : ArrayLike, (B, L_align, in_feats)
        
        sow_intermediates : bool
            switch for tensorboard logging
          
        Returns
        --------
        ArrayLike, (B, L_align, A) 
            scoring matrix for emissions from indels
        """
        # (B, L_align, H) -> (B, L_align, A)
        logits = self.final_project(datamat)
        
        return self.apply_log_softmax_activation(logits = logits,
                                                 sow_intermediates = sow_intermediates) # (B, L_align, A)
        