#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 13:46:56 2025

@author: annabel

equilibrium dist, pred_config entries:
----------------------------------------
'GlobalEqul',
- pred_config['emission_alphabet_size']

'LocalEqul',
- pred_config['emission_alphabet_size']

'EqulFromFile',
- pred_config['filenames']['equl_dist']


F81, pred_config entries:
--------------------------
'GlobalF81', 
(no config entries)

'LocalF81',
- (OPTIONAL) pred_config['rate_mult_range']

'F81FromFile',
(no config entries)


GTR, pred_config entries:
-------------------------
'GTRGlobalExchGlobalRateMult',
- pred_config['emission_alphabet_size']
- (OPTIONAL) pred_config['exchange_range']

'GTRGlobalExchLocalRateMult',
- pred_config['emission_alphabet_size']
- (OPTIONAL) pred_config['exchange_range']
- (OPTIONAL) pred_config['rate_mult_range']

'GTRLocalExchLocalRateMult',
- pred_config['emission_alphabet_size']
- (OPTIONAL) pred_config['exchange_range']
- (OPTIONAL) pred_config['rate_mult_range']

'GTRFromFile',
- pred_config['filenames']['exch']

"""
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm 

from models.BaseClasses import (neuralTKFModuleBase, 
                                ModuleBase)
from models.neural_hmm_predict.model_functions import (safe_log,
                                                       bound_sigmoid,
                                                       logprob_f81,
                                                       logprob_gtr)



###############################################################################
### Equilibrium distribution models   #########################################
###############################################################################
class GlobalEqul(neuralTKFModuleBase):
    """
    Use the observed amino acid frequencies as the equilibrium distribution
    
    Doesn't have to be a module, but make it one for consistency, I guess
    """
    config: dict
    name: str
    
    def setup(self):
        """
        Flax Module Parameters
        -----------------------
        None
        """
        training_dset_emit_counts = self.config['training_dset_emit_counts'] #(A,)
        prob_equilibr = training_dset_emit_counts/training_dset_emit_counts.sum() #(A,)
        logprob_equilibr = safe_log( prob_equilibr ) #(A,)
        self.logprob_equilibr = logprob_equilibr[None,None,:] #(1,1,A)
        
    def __call__(self,
                 *args,
                 **kwargs): 
        """
        Arguments
        ----------
        None
        
        Returns
        --------
        ArrayLike, (1, 1, A) 
            scoring matrix for emissions from indels, from observed frequencies
        """
        return self.logprob_equilibr  #(1,1,A)

    
class LocalEqul(GlobalEqul):
    """
    Use a set of logits to find equilibrium distribution for each position, 
      each sample
    """
    config: dict
    name: str
    
    def setup(self):
        """
        H = number of features of input matrix
        A = alphabet size
        
        Flax Module Parameters
        -----------------------
        self.final_project : Flax module
          > kernel : ArrayLike, (H, A)
          > bias : ArrayLike, (A)  
        """
        emission_alphabet_size = self.config['emission_alphabet_size']
        self.use_bias = self.config.get('use_bias', True)
        
        name = f'{self.name}/Project to equilibriums'
        self.final_project = nn.Dense(features = emission_alphabet_size,
                                      use_bias = self.use_bias,
                                      name = name)
    
    def __call__(self,
                datamat: jnp.array,
                sow_intermediates: bool):
        """
        apply final linear projection and log_softmax to get final 
          equilibrium distribution
        
        B: batch size
        L_align: length of alignment
        H: input hidden dim
        A: final alphabet size
        
        Arguments
        ----------
        datamat : ArrayLike, (B, L_align, H)
        
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
                                                 sow_intermediates = sow_intermediates,
                                                 param_name = 'to equilibriums') # (B, L_align, A)


class EqulFromFile(neuralTKFModuleBase):
    """
    read one equilibrium distribution from numpy array file
    """
    config: dict
    name: str
    
    def setup(self):
        """
        Flax Module Parameters
        -----------------------
        None
        """
        equl_file = self.config['filenames']['equl_dist']
        
        with open(equl_file,'rb') as f:
            prob_equilibr = jnp.load(f, allow_pickle=True) #(A,) or (1,A) or (1,1,A)
        
        # if only one equlibrium distribution loaded, and it doesn't have
        #   the correct number of dimensions, fix that to (1,1,A)
        if len(prob_equilibr.shape) < 3:
            prob_equilibr = jnp.reshape(prob_equilibr,
                                        (1, 1, prob_equilibr.shape[-1])) #(1,1,A)
        
        self.logprob_equilibr = safe_log(prob_equilibr)
        
    def __call__(self,
                  *args,
                  **kwargs): 
        """
        Returns
        --------
        ArrayLike, (1, 1, A) 
            log-probability matrix for emissions from indels, which includes 
            placeholder dimensions for B and L_align
        """
        return self.logprob_equilibr #(1, 1, A) 
    

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
        cond_logprobs = logprob_f81(equl = equl,
                           rate_multiplier = self.global_rate_multiplier,
                           t_array = t_array,
                           unique_time_per_sample = unique_time_per_sample)
    
        intermed_params_dict = {'rate_multiplier': self.global_rate_multiplier}
        
        return cond_logprobs, intermed_params_dict
    
    
class LocalF81(neuralTKFModuleBase):
    """
    Decide F81 model for each sample, each position; normalize then scale by
      rate multiplier for the site
     
    """
    config: dict
    name: str
    
    def setup(self):
        """
        H = number of features of input matrix
        
        Flax Module Parameters
        -----------------------
        self.final_project : Flax module
          > kernel : ArrayLike, (H, 1)
          > bias : ArrayLike, (1)  
        """
        self.rate_mult_min_val, self.rate_mult_max_val  = self.config.get( 'rate_mult_range', 
                                                                           (0.01, 10) )
        
        # only change these when debugging
        self.use_bias = self.config.get('use_bias', True)
        self.force_unit_rate_multiplier = self.config.get( 'force_unit_rate_multiplier',
                                                            False )
        
        if not self.force_unit_rate_multiplier:
            name = f'{self.name}/Project to rate multipliers'
            self.final_project = nn.Dense(features = 1,
                                          use_bias = self.use_bias,
                                          name = name)
    
    def __call__(self,
                 datamat: jnp.array,
                 padding_mask: jnp.array,
                 log_equl: jnp.array,
                 t_array: jnp.array,
                 unique_time_per_sample: bool,
                 sow_intermediates: bool):
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
        datamat : ArrayLike, (B, L_align, H)
        
        pading_mask : ArrayLike, (B, L_align)
        
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
        ### rate multiplier
        if not self.force_unit_rate_multiplier:
            # (B, L_align, H) -> (B, L_align, 1) -> (B, L_align)
            rate_mult_logits = self.final_project(datamat)[...,0] # (B, L_align)
            rate_multiplier = self.apply_bound_sigmoid_activation(logits = rate_mult_logits,
                                               min_val = self.rate_mult_min_val,
                                               max_val = self.rate_mult_max_val,
                                               param_name = 'rate mult.',
                                               sow_intermediates = sow_intermediates) # (B, L_align)
                
        elif self.force_unit_rate_multiplier:
            final_shape = ( datamat.shape[0], datamat.shape[1] )
            rate_multiplier = jnp.ones( final_shape ) # (B, L_align)
        
        
        ### equilibrium distribution
        # return equilibrium probabilities directly
        equl = jnp.exp(log_equl) # (B, L_align, A)
        
        
        ### output
        #   if unique time per sample: (B, L_align, A, 2)
        #   if not unique time per sample: (T, B, L_align, A, 2)
        cond_logprobs = logprob_f81(equl = equl,
                                    rate_multiplier = rate_multiplier,
                                    t_array = t_array,
                                    unique_time_per_sample = unique_time_per_sample)
    
        intermed_params_dict = {'rate_multiplier': rate_multiplier}
    
        return cond_logprobs, intermed_params_dict


# alias for GlobalF81; never loading local parameters, so whenever this is 
#   invoked, just use GlobalF81 instead
F81FromFile = GlobalF81


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
        A = emission_alphabet_size
        num_exchange = ( ( A * (A-1) ) / 2 ).astype(int)
        self.exch_logits = self.param("exchangeabilities", 
                                      nn.initializers.normal(),
                                      (1, 1, num_exchange,),
                                      jnp.float32 ) #(1, 1, n)
        
        # rate multiplier is one
        self.global_rate_multiplier = jnp.ones( (1,1) ) #(1,1)
    
    def __call__(self, 
                 log_equl,
                 t_array,
                 unique_time_per_sample,
                 sow_intermediates: bool,
                 *args,
                 **kwargs):
        ### exchangeabilities 
        # exch_upper_triag_values is: 
        #   (B, L_align=1, 190) if proteins; n=190
        #   (B, L_align=1, 6) if dna; n=6
        exch_upper_triag_values = self.apply_bound_sigmoid_activation( logits = self.exch_logits,
                                                   min_val = self.exchange_min_val,
                                                   max_val = self.exchange_max_val,
                                                   param_name = 'exchangeabilities',
                                                   sow_intermediates = sow_intermediates) #(B, L_align, n)


        ### logP(desc | anc, t)
        # cond_logprobs is:
        # (B, L_align=1, A, A) if unique_time_per_sample
        # (T, B=1, L_align=1, A, A) if not unique_time_per_sample
        cond_logprobs = logprob_gtr( exch_upper_triag_values = exch_upper_triag_values,
                            equilibrium_distributions = jnp.exp(log_equl),
                            rate_multiplier = self.global_rate_multiplier,
                            t_array = t_array,
                            unique_time_per_sample = unique_time_per_sample )
    
        intermed_params_dict = {'rate_multiplier': self.global_rate_multiplier,
                                'exch_upper_triag_values': exch_upper_triag_values}
    
        return cond_logprobs, intermed_params_dict
    

class GTRGlobalExchLocalRateMult(neuralTKFModuleBase):
    """
    One set of GTR exchangeabilities for all positions, but a substitution
      rate multiplier per-sample, and per alignment column
    """
    config: dict
    name: str
    
    def setup(self):
        ### read config
        emission_alphabet_size = self.config['emission_alphabet_size']
        self.exchange_min_val, self.exchange_max_val  = self.config.get( 'exchange_range', (1e-4, 12) )
        self.rate_mult_min_val, self.rate_mult_max_val  = self.config.get( 'rate_mult_range', 
                                                                           (0.01, 10) )
        
        # for debugging, tests
        self.use_bias = self.config.get('use_bias', True)
        self.force_unit_rate_multiplier = self.config.get( 'force_unit_rate_multiplier',
                                                            False )
        
        
        ### exchangeabilities
        A = emission_alphabet_size
        num_exchange = ( ( A * (A-1) ) / 2 ).astype(int)
        self.exch_logits = self.param("exchangeabilities", 
                                      nn.initializers.normal(),
                                      (1, 1, num_exchange,),
                                      jnp.float32 ) #(1, 1, n)
        
        
        ### rate multipliers
        if not self.force_unit_rate_multiplier:
            name = f'{self.name}/Project to rate multipliers'
            self.rate_mult_final_project = nn.Dense(features = 1,
                                                    use_bias = self.use_bias,
                                                    name = name)
        
    def __call__(self, 
                 datamat: jnp.array,
                 padding_mask: jnp.array,
                 log_equl: jnp.array,
                 t_array: jnp.array,
                 unique_time_per_sample: bool,
                 sow_intermediates: bool):
        ### rate multipliers: (B, L_align, H) -> (B, L_align, 1) -> (B, L_align)
        if not self.force_unit_rate_multiplier:
            rate_mult_logits = self.rate_mult_final_project(datamat)[...,0] # (B, L_align)
            rate_multiplier = self.apply_bound_sigmoid_activation(logits = rate_mult_logits,
                                               min_val = self.rate_mult_min_val,
                                               max_val = self.rate_mult_max_val,
                                               param_name = 'rate mult.',
                                               sow_intermediates = sow_intermediates) # (B, L_align)
            
        elif self.force_unit_rate_multiplier:
            final_shape = ( datamat.shape[0], datamat.shape[1] )
            rate_multiplier = jnp.ones( final_shape ) # (B, L_align)
        
        
        ### exchangeabilities 
        # exch_upper_triag_values is: 
        #   (B, L_align, 190) if proteins
        #   (B, L_align, 6) if dna
        exch_upper_triag_values = self.apply_bound_sigmoid_activation( logits = self.exch_logits,
                                                   min_val = self.exchange_min_val,
                                                   max_val = self.exchange_max_val,
                                                   param_name = 'exchangeabilities',
                                                   sow_intermediates = sow_intermediates)
        
        ### logP(desc | anc, t)
        # cond_logprobs is:
        # (B, L_align, A, A) if unique_time_per_sample
        # (T, B, L_align, A, A) if not unique_time_per_sample
        cond_logprobs = logprob_gtr( exch_upper_triag_values = exch_upper_triag_values,
                            equilibrium_distributions = jnp.exp(log_equl),
                            rate_multiplier = rate_multiplier,
                            t_array = t_array,
                            unique_time_per_sample = unique_time_per_sample )
        
        intermed_params_dict = {'rate_multiplier': rate_multiplier,
                         'exch_upper_triag_values': exch_upper_triag_values}
    
        return cond_logprobs, intermed_params_dict


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
        
        
        # for debugging, tests
        self.use_bias = self.config.get('use_bias', True)
        self.force_unit_rate_multiplier = self.config.get( 'force_unit_rate_multiplier',
                                                            False )
       
        
        ### exchangeabilities
        A = emission_alphabet_size
        num_exchange = ( ( A * (A-1) ) / 2 ).astype(int)
        name = f'{self.name}/Project to exchangeabilties'
        self.exch_final_project = nn.Dense(features = num_exchange,
                                           use_bias = self.use_bias,
                                           name = name)
        del name
        
        ### rate multipliers
        if not self.force_unit_rate_multiplier:
            name = f'{self.name}/Project to rate multipliers'
            self.rate_mult_final_project = nn.Dense(features = 1,
                                                    use_bias = self.use_bias,
                                                    name = name)
        
    def __call__(self, 
                 datamat: jnp.array,
                 padding_mask: jnp.array,
                 log_equl: jnp.array,
                 t_array: jnp.array,
                 unique_time_per_sample: bool,
                 sow_intermediates: bool):
        ### rate multipliers: (B, L_align, H) -> (B, L_align, 1) -> (B, L_align)
        if not self.force_unit_rate_multiplier:
            rate_mult_logits = self.rate_mult_final_project(datamat)[...,0] # (B, L_align)
            rate_multiplier = self.apply_bound_sigmoid_activation(logits = rate_mult_logits,
                                               min_val = self.rate_mult_min_val,
                                               max_val = self.rate_mult_max_val,
                                               param_name = 'rate mult.',
                                               sow_intermediates = sow_intermediates) # (B, L_align)
            
        elif self.force_unit_rate_multiplier:
            final_shape = ( datamat.shape[0], datamat.shape[1] )
            rate_multiplier = jnp.ones( final_shape ) # (B, L_align)
        
        
        ### exchangeabilities: (B, L_align, H) -> (B, L_align, n) 
        # exch_upper_triag_values is: 
        #   (B, L_align, 190) if proteins; n=190
        #   (B, L_align, 6) if dna; n=6
        exch_logits = self.exch_final_project(datamat) # (B, L_align, n) 
        exch_upper_triag_values = self.apply_bound_sigmoid_activation( logits = exch_logits,
                                                   min_val = self.exchange_min_val,
                                                   max_val = self.exchange_max_val,
                                                   param_name = 'exchangeabilities',
                                                   sow_intermediates = sow_intermediates) # (B, L_align, n) 
        
        
        ### logP(desc | anc, t)
        # cond_logprobs is:
        # (B, L_align, A, A) if unique_time_per_sample
        # (T, B, L_align, A, A) if not unique_time_per_sample
        cond_logprobs = logprob_gtr( exch_upper_triag_values = exch_upper_triag_values,
                                     equilibrium_distributions = jnp.exp(log_equl),
                                     rate_multiplier = rate_multiplier,
                                     t_array = t_array,
                                     unique_time_per_sample = unique_time_per_sample )
        
        intermed_params_dict = {'rate_multiplier': rate_multiplier,
                         'exch_upper_triag_values': exch_upper_triag_values}
    
        return cond_logprobs, intermed_params_dict


class GTRFromFile(neuralTKFModuleBase):
    """
    GTR, but load exchangeabilities from file
    
    Rate multiplier is automatically one (i.e. global)
    """
    config: dict
    name: str
    
    def setup(self):
        ### exchangeabilities
        exchangeabilities_file = self.config['filenames']['exch']
        with open(exchangeabilities_file,'rb') as f:
            exch_from_file = jnp.load(f)
        
        shape = exch_from_file.shape
        correct = (
            shape == (shape[0],) or
            shape == (1, shape[1]) or
            shape == (1, 1, shape[2])
        )

        # if exch_from_file is (n,), (1,n), or (1,1,n), then it's the upper 
        #   triangular values
        if correct:
            self.exch_upper_triag_values = exch_from_file
            if len(self.exch_upper_triag_values.shape) < 3:
                final_shape = (1, 1, self.exch_upper_triag_values.shape[-1])
                self.exch_upper_triag_values = jnp.reshape( self.exch_upper_triag_values, 
                                                            final_shape ) #(1,1,n)
                del final_shape
        
        # if final two dims of exch_from_file are the same, then it's the 
        #   symmetric exchangeabilities matrix as-is
        elif not correct:
            exch_mat = jnp.squeeze(exch_from_file) #(A,A)
            exch_upper_triag_values = exch_mat[jnp.triu_indices_from(exch_mat, k=1)] #(n,)
            self.exch_upper_triag_values = exch_upper_triag_values[None,None,:] #(1,1,n)
            
        
        ### rate multiplier: automatically set to one
        self.global_rate_multiplier = jnp.ones((1,1)) #(1,1)
    
    def __call__(self, 
                 log_equl,
                 t_array,
                 unique_time_per_sample,
                 sow_intermediates: bool,
                 *args,
                 **kwargs):
        
        cond_logprobs = logprob_gtr( exch_upper_triag_values = self.exch_upper_triag_values,
                            equilibrium_distributions = jnp.exp(log_equl),
                            rate_multiplier = self.global_rate_multiplier,
                            t_array = t_array,
                            unique_time_per_sample = unique_time_per_sample )
    
        return cond_logprobs, None
    
    