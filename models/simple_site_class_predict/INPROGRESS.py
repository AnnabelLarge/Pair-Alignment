#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 02:03:13 2025

@author: annabel


ABOUT:
======
Flax Modules needed for scoring emissions; may have their own parameters, and 
    could write to tensorboard


modules:
=========
 'EqulDistLogprobsFromCounts',
 'EqulDistLogprobsFromFile',
 'EqulDistLogprobsPerClass',
 'F81Logprobs',
 'F81LogprobsFromFile',
 'GTRRateMat',
 'GTRRateMatFromFile',
 'HKY85RateMat',
 'HKY85RateMatFromFile',
 'ModuleBase',
 'SiteClassLogprobs',
 'SiteClassLogprobsFromFile',
"""
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax.scipy.special import logsumexp
from jax._src.typing import Array, ArrayLike

from functools import partial
from typing import Optional

from models.BaseClasses import ModuleBase
from models.simple_site_class_predict.model_functions import (bound_sigmoid,
                                                              bound_sigmoid_inverse,
                                                              safe_log,
                                                              rate_matrix_from_exch_equl,
                                                              scale_rate_multipliers,
                                                              scale_rate_matrix,
                                                              upper_tri_vector_to_sym_matrix,
                                                              get_cond_logprob_emit_at_match_per_class,
                                                              get_joint_logprob_emit_at_match_per_class,
                                                              lse_over_match_logprobs_per_class,
                                                              lse_over_equl_logprobs_per_class)


###############################################################################
### Probability of being in site classes   ####################################
###############################################################################
class SiteClassLogprobs(ModuleBase):
    """
    Probability of being in site class, P(c)
    
    
    Initialize with
    ----------------
    config : dict
        config['num_mixtures'] : int
            number of emission site classes
    
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    __call__
    
    Methods inherited from models.model_utils.BaseClasses.ModuleBase
    ----------------------------------------------------------------
    sow_histograms_scalars
        for tensorboard logging
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        C = number of site classes
        
        
        Flax Module Parameters
        -----------------------
        class_logits : ArrayLike (C,)
            initialize logits from unit normal
        
        """
        self.n_classes = self.config['num_mixtures']
        
        if self.n_classes > 1:
            self.class_logits = self.param('class_logits',
                                            nn.initializers.normal(),
                                            (self.n_classes,),
                                            jnp.float32) #(C,)
        
    def __call__(self,
                 sow_intermediates: bool):
        """
        C: number of site classes
        
        
        Arguments
        ----------
        sow_intermediates : bool
            switch for tensorboard logging
          
        Returns
        -------
        log_class_probs : ArrayLike, (C,)
            log-probability of being in each site class, P(c); if only one
            site class, then logP(c) = 0
          
        """
        if self.n_classes > 1:
            log_class_probs = nn.log_softmax(self.class_logits) #(C,)
        
            # tensorboard logging
            if sow_intermediates:
                for i in range(log_class_probs.shape[0]):
                    val_to_write = jnp.exp( log_class_probs[i] )
                    lab = f'{self.name}/prob of class {i}'
                    self.sow_histograms_scalars(mat= val_to_write, 
                                                label=lab, 
                                                which='scalars')
                    del lab
        
        else:
            log_class_probs = jnp.array([0]) #(1,)
            
        return log_class_probs


class SiteClassLogprobsFromFile(ModuleBase):
    """
    load probabilities of being in site class, P(c)
    
    If only using one model (no mixtures), then this does NOT read the file;
      it just returns log(1) = 0
    
    
    Initialize with
    ----------------
    config : dict
        config['filenames']['class_probs'] :  str
            file containing the class probabilities to load
        
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    __call__
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        
        Flax Module Parameters
        -----------------------
        none
        
        """
        self.n_classes = self.config['num_mixtures']
        
        if self.n_classes > 1:
            in_file = self.config['filenames']['class_probs']
            with open(in_file,'rb') as f:
                class_probs = jnp.load(f)
            self.log_class_probs = safe_log(class_probs) #(C,)
        
        else:
            self.log_class_probs = jnp.array([0])
    
    def __call__(self,
                 **kwargs):
        """
        Returns
        -------
        log_class_probs : ArrayLike, (C,)
            log-probability of being in each site class, P(c)
          
        """
        return self.log_class_probs #(C,)


###############################################################################
### Rate multipliers   ########################################################
###############################################################################
class RateMultipliersPerClass(ModuleBase):
    """
    C: number of latent site classes
    K: numer of rate multipliers
    
    
    Generate c * k rate multipliers, and probabilty of rate multiplier k, 
      given class c: P(k|c)
    
    
    Initialize with
    ----------------
    config : dict
        config['num_mixtures'] : int
            number of mixtures
    
        config['k_rate_mults'] : int
            number of rate multipliers
            
        config['rate_mult_range'] : (float, float)
            min and max rate multiplier; default is (0.01, 10)

        config['norm_rate_mults'] : bool
            flag to normalize \sum_k P(k)*\rho_k = 1 or 
            \sum_c \sum_k P(c,k) * \rho_{c,k} = 1
    
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    __call__
    
    Methods inherited from models.model_utils.BaseClasses.ModuleBase
    ----------------------------------------------------------------
    sow_histograms_scalars
        for tensorboard logging
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        K = number of site classes
        C: number of site classes
        
        
        Flax Module Parameters
        -----------------------
        rate_mult_prob_logits : ArrayLike (K,)
            initialize logits for P(k) from unit normal
        
        rate_mult_logits : ArrayLike (K,)
            initialize logits for each rate k from unit normal
        
        """
        ### read config file
        self.num_mixtures = self.config['num_mixtures']
        self.k_rate_mults = self.config['k_rate_mults']
        
        # optional
        self.rate_mult_min_val, self.rate_mult_max_val  = self.config.get( 'rate_mult_range', (0.01, 10) )
        self.norm_rate_mults = self.config.get('norm_rate_mults', True)
        
        
        ### parameters
        # probability P(k | c)
        out_size = ( self.num_mixtures, self.k_rate_mults )
        self.rate_mult_prob_logits = self.param('rate_mult_prob_logits',
                                        nn.initializers.normal(),
                                        out_size,
                                        jnp.float32) #(C,K)
        
        
        # rate multipliers, \rho_{c, k}
        self.rate_mult_logits = self.param('rate_mult_logits',
                                           nn.initializers.normal(),
                                           out_size,
                                           jnp.float32) #(C,K)
        
        self.exchange_activation = partial(bound_sigmoid,
                                           min_val = self.exchange_min_val,
                                           max_val = self.exchange_max_val)
        
        
    def __call__(self,
                 log_class_probs,
                 sow_intermediates: bool):
        """
        K = number of site classes
        C: number of site classes
        
        
        Arguments
        ----------
        sow_intermediates : bool
            switch for tensorboard logging
          
        Returns
        -------
        log_rate_mult_probs : ArrayLike, (C, K)
        
        rate_multiplier : ArrayLike, (C, K)
          
        """
        ### get params
        # P(K|C)
        log_rate_mult_probs = nn.log_softmax( self.rate_mult_prob_logits, axis=-1 ) #(C, K)
        
        if sow_intermediates:
            lab = (f'{self.name}/prob of rate multipliers')
            self.sow_histograms_scalars(mat= jnp.exp(logprob_rate_mult), 
                                        label=lab, 
                                        which='scalars')
            del lab

        # \rho_{c,k}
        rate_multiplier = self.rate_multiplier_activation( self.rate_mult_logits ) #(C, K)

        if sow_intermediates:
            lab = (f'{self.name}/rate multipliers')
            self.sow_histograms_scalars(mat= rate_multiplier, 
                                        label=lab, 
                                        which='scalars')
            del lab
           
            
        ### possibly normalize
        if self.norm_rate_mults:
            joint_logprob_class_rate_mult = log_class_probs[:, None] + log_rate_mult_probs
            joint_prob_class_rate_mult = jnp.exp(joint_logprob_class_rate_mult)
            norm_factor = jnp.multiply(joint_prob_class_rate_mult, rate_multiplier).sum()
            rate_multiplier = rate_multiplier / norm_factor
            
        return (log_rate_mult_probs, rate_multiplier)
        
    
class IndpRateMultipliers(ModuleBase):
    """
    C: number of latent site classes
    K: numer of rate multipliers
    
    
    Generate c * k rate multipliers, and probabilty of rate multiplier k, P(k)
    
    THIS ASSUMES K IS INDEPENDENT OF C
    
    
    Initialize with
    ----------------
    config : dict
        config['num_mixtures'] : int
            number of mixtures
    
        config['k_rate_mults'] : int
            number of rate multipliers
            
        config['rate_mult_range'] : (float, float)
            min and max rate multiplier; default is (0.01, 10)

        config['norm_rate_mults'] : bool
            flag to normalize \sum_k P(k)*\rho_k = 1 or 
            \sum_c \sum_k P(c,k) * \rho_{c,k} = 1
    
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    __call__
    
    Methods inherited from models.model_utils.BaseClasses.ModuleBase
    ----------------------------------------------------------------
    sow_histograms_scalars
        for tensorboard logging
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        K = number of site classes
        C: number of site classes
        
        
        Flax Module Parameters
        -----------------------
        rate_mult_prob_logits : ArrayLike (K,)
            initialize logits for P(k) from unit normal
        
        rate_mult_logits : ArrayLike (K,)
            initialize logits for each rate k from unit normal
        
        """
        ### read config file
        self.num_mixtures = self.config['num_mixtures']
        self.k_rate_mults = self.config['k_rate_mults']
        
        # optional
        self.rate_mult_min_val, self.rate_mult_max_val  = self.config.get( 'rate_mult_range', (0.01, 10) )
        self.norm_rate_mults = self.config.get('norm_rate_mults', True)
        
        
        ### parameters; broadcast all to (C,K)
        # probability P(k)
        out_size = ( self.num_mixtures, self.k_rate_mults )
        rate_mult_prob_logits = self.param( 'rate_mult_prob_logits',
                                            nn.initializers.normal(),
                                            self.k_rate_mults,
                                            jnp.float32 ) #(K,)
        self.rate_mult_prob_logits = jnp.broadcast_to( rate_mult_prob_logits[None,:],
                                                       out_size ) #(C, K)
        
        
        # rate multipliers, \rho_{c, k}
        rate_mult_logits = self.param( 'rate_mult_logits',
                                       nn.initializers.normal(),
                                       self.k_rate_mults,
                                       jnp.float32 ) #(K,)
        self.rate_mult_logits = jnp.broadcast_to( rate_mult_logits[None,:],
                                                  out_size ) #(C, K)
        
        self.exchange_activation = partial( bound_sigmoid,
                                            min_val = self.exchange_min_val,
                                            max_val = self.exchange_max_val )
        
        
    def __call__(self,
                 sow_intermediates: bool):
        """
        K = number of site classes
        
        
        Arguments
        ----------
        sow_intermediates : bool
            switch for tensorboard logging
          
        Returns
        -------
        log_rate_mult_probs : ArrayLike, (C, K)
        
        rate_multiplier : ArrayLike, (C, K)
          
        """
        ### get params
        # P(K)
        log_rate_mult_probs = nn.log_softmax( self.rate_mult_prob_logits, axis=-1 ) #(C, K)
        
        if sow_intermediates:
            lab = (f'{self.name}/prob of rate multipliers')
            val = log_rate_mult_probs[0,:]
            self.sow_histograms_scalars(mat= jnp.exp(val), 
                                        label=lab, 
                                        which='scalars')
            del lab

        # \rho_{c,k}
        rate_multiplier = self.rate_multiplier_activation( self.rate_mult_logits ) #(C, K)

        if sow_intermediates:
            lab = (f'{self.name}/rate multipliers')
            val = rate_multiplier[0,:]
            self.sow_histograms_scalars(mat= val, 
                                        label=lab, 
                                        which='scalars')
            del lab
           
            
        ### possibly normalize
        if self.norm_rate_mults:
            one_row_rate_mult = rate_multiplier[0,:] #(K,)
            one_row_rate_mult_prob = jnp.exp( log_rate_mult_probs[0,:] ) #(K,)
            norm_factor = jnp.multiply( one_row_rate_mult, one_row_rate_mult_prob ).sum()
            rate_multiplier = rate_multiplier / norm_factor #(C, K)
            
        return (log_rate_mult_probs, rate_multiplier)
        


###############################################################################
### RATE MATRICES: Generate time reversible   #################################
###############################################################################
class GTRRateMat(ModuleBase):
    """
    return (rho * Q), to be directly used in matrix exponential

    
    Initialize with
    ----------------
    config : dict
        config['num_mixtures'] :  int
            number of emission site classes
        
        config['random_init_exchanges'] : bool
            whether or not to initialize exchangeabilities from random; if 
            not random, initialize with LG08 values
            
        config['norm_rate_mults'] : bool
            flag to normalize rate multipliers times class probabilites to 1
        
        config['norm_rate_matrix'] : bool
            flag to normalize rate matrix to t = one substitution
        
        config['rate_mult_activation'] : {'bound_sigmoid', 'softplus'}, optional
            what activation to use for logits of rate multiplier
            Default is 'bound_sigmoid'
        
        config['rate_mult_range'] : List[float, float], optional
            only needed when using bound_sigmoid for rate multipliers
            first value is min, second value is max
            Default is (0.01, 10)
            
        config['exchange_range'] : List[float, float], optional
            exchangeabilities undergo bound_sigmoid transformation, this
            specifies the min and max
            Default is (1e-4, 12)
        
        config['filenames']['exch'] : str, optional
            name of the exchangeabilities to intiialize with, if desired
            Default is None
        
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    __call__
    _prepare_rate_matrix
        function to prepare rate matrix (defined in helpers above)
    
    Methods inherited from models.model_utils.BaseClasses.ModuleBase
    ----------------------------------
    sow_histograms_scalars
        for tensorboard logging
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        C: number of site classes
        
        
        Flax Module Parameters
        -----------------------
        rate_mult_logits : ArrayLike, (C,)
            rate multiplier per class; ONLY present if C > 1
        
        exchangeabilities_logits_vec : ArrayLike, (n,)
            upper triangular values for exchangeability matrix
            190 for proteins, 6 for DNA
            Usually initialize logits from LG08 exchangeabilities
        
        """
        ###################
        ### read config   #
        ###################
        # required
        self.num_mixtures = self.config['num_mixtures']
        
        # have defaults; may or may not be used
        self.rate_mult_min_val, self.rate_mult_max_val  = self.config.get( 'rate_mult_range', (0.01, 10) )
        self.exchange_min_val, self.exchange_max_val  = self.config.get( 'exchange_range', (1e-4, 12) )
        
        # tested on GTR; decided that these defaults are best
        #   change if you're comparing against XRATE or running previous tests
        self.rate_mult_activation = self.config.get('rate_mult_activation', 
                                                    'bound_sigmoid')
        self.random_init_exchanges = self.config.get('random_init_exchanges', 
                                                     True)
        self.norm_rate_mults = self.config.get('norm_rate_mults', True)
        self.norm_rate_matrix = self.config.get('norm_rate_matrix', True)
        
        
        # provided upon model initialization; guaranteed to be here
        emission_alphabet_size = self.config['emission_alphabet_size']
        
        # validate
        if self.rate_mult_activation not in ['bound_sigmoid', 'softplus']:
            raise ValueError('Pick either: bound_sigmoid, softplus')
            
            
        ########################
        ### RATE MULTIPLIERS   #
        ########################
        ### only really matters if there's more than one mixture; otherwise,
        ###   rate multiplier is set to one
        if self.num_mixtures > 1:
            # activation
            if self.rate_mult_activation == 'bound_sigmoid':
                self.rate_multiplier_activation = partial(bound_sigmoid,
                                                          min_val = self.rate_mult_min_val,
                                                          max_val = self.rate_mult_max_val)
            
            elif self.rate_mult_activation == 'softplus':
                self.rate_multiplier_activation = jax.nn.softplus
        
        
            # initializers
            self.rate_mult_logits = self.param('rate_multipliers',
                                               nn.initializers.normal(),
                                               (self.num_mixtures,),
                                               jnp.float32) #(C,)
    
            
        #####################################
        ### EXCHANGEABILITIES AS A VECTOR   #
        #####################################
        ### activation is bound sigmoid; setup the activation function with 
        ###   min/max values
        self.exchange_activation = partial(bound_sigmoid,
                                           min_val = self.exchange_min_val,
                                           max_val = self.exchange_max_val)
        
        
        ### initialization
        # init from file
        if not self.random_init_exchanges:
            exchangeabilities_file = self.config['filenames']['exch']
            with open(exchangeabilities_file,'rb') as f:
                vec = jnp.load(f)
        
            transformed_vec = bound_sigmoid_inverse(vec, 
                                                      min_val = self.exchange_min_val,
                                                      max_val = self.exchange_max_val)
        
        
            self.exchangeabilities_logits_vec = self.param("exchangeabilities", 
                                                           lambda rng, shape: transformed_vec,
                                                           transformed_vec.shape ) #(n,)
        
        # init from random
        elif self.random_init_exchanges:
            if emission_alphabet_size == 20:
                num_exchange = 190
            
            elif emission_alphabet_size == 4:
                num_exchange = 6
            
            self.exchangeabilities_logits_vec = self.param("exchangeabilities", 
                                                           nn.initializers.normal(),
                                                           (num_exchange,),
                                                           jnp.float32 ) #(n,)
        
        
    def __call__(self,
                 logprob_equl,
                 log_class_probs,
                 sow_intermediates: bool,
                 *args,
                 **kwargs):
        """
        C = number of latent site classes
        A = alphabet size
        
        
        Arguments
        ----------
        logprob_equl : ArrayLike, (C, A)
            log-transformed equilibrium distribution
        
        sow_intermediates : bool
            switch for tensorboard logging
          
        Returns
        -------
        rate_mat_times_rho : ArrayLike, (C, A, A)
            scaled rate matrix
        """
        # undo log transform on equilibrium
        equl = jnp.exp(logprob_equl) #(C, A)
        
        
        #######################
        ### rate multiplier   #
        #######################
        if self.num_mixtures > 1:
            
            ### apply activation of choice
            if sow_intermediates:
                for i in range(self.rate_mult_logits.shape[0]):
                    val_to_write = self.rate_mult_logits[i]
                    act = self.rate_mult_activation
                    lab = (f'{self.name}/logit BEFORE {act} activation- '+
                           f'rate multiplier for class {i}')
                    self.sow_histograms_scalars(mat= val_to_write, 
                                                label=lab, 
                                                which='scalars')
                    del lab
                    
            rate_multiplier = self.rate_multiplier_activation( self.rate_mult_logits ) #(C,)
            
            if sow_intermediates:
                for i in range(rate_multiplier.shape[0]):
                    val_to_write = rate_multiplier[i]
                    act = self.rate_mult_activation
                    lab = (f'{self.name}/rate AFTER {act} activation- '+
                           f'rate multiplier for class {i}')
                    self.sow_histograms_scalars(mat= val_to_write, 
                                                label=lab, 
                                                which='scalars')
                    del lab
            
            ### normalize
            if self.norm_rate_mults:
                rate_multiplier = scale_rate_multipliers( unnormed_rate_multipliers = rate_multiplier,
                                        log_class_probs = log_class_probs )
            
                if sow_intermediates:
                    for i in range(rate_multiplier.shape[0]):
                        val_to_write = rate_multiplier[i]
                        lab = (f'{self.name}/rate AFTER normalization- '+
                               f'rate multiplier for class {i}')
                        self.sow_histograms_scalars(mat= val_to_write, 
                                                    label=lab, 
                                                    which='scalars')
                        del lab
                    
        else:
            rate_multiplier = jnp.array([1]) #(1,)
        
        
        #################################################
        ### exchangeabilities; one shared all classes   #
        #################################################
        # apply activation of choice
        if sow_intermediates:
            self.sow_histograms_scalars(mat= self.exchangeabilities_logits_vec, 
                                        label= 'logit BEFORE bound_sigmoid activation- exchangeabilities', 
                                        which='scalars')
            
        # number of upper triangular values for alphabet size A is (A * A-1)/2
        # upper_triag_values will have this many values
        # for proteins: upper_triag_values is (190,)
        # for DNA: upper_triag_values is (6,)
        upper_triag_values = self.exchange_activation( self.exchangeabilities_logits_vec ) 
    
        if sow_intermediates:
            self.sow_histograms_scalars(mat = upper_triag_values, 
                                        label = 'value AFTER bound_sigmoid activation- exchangeabilities', 
                                        which='scalars')
        
        # create square matrix
        exchangeabilities_mat = upper_tri_vector_to_sym_matrix( upper_triag_values ) #(A, A)
        
        # scale rate matrix
        rate_mat_times_rho = self._prepare_rate_matrix(exchangeabilities = exchangeabilities_mat,
                                                       equilibrium_distributions = equl,
                                                       rate_multiplier = rate_multiplier,
                                                       norm = self.norm_rate_matrix) #(C, A, A)
        return rate_mat_times_rho
    
    def _prepare_rate_matrix(self,
                             exchangeabilities,
                             equilibrium_distributions,
                             rate_multiplier,
                             norm: bool):
        """
        Returns scaled rate matrix, Q = rho * chi * diag(pi)
            q_{ijc} = rho_c * chi_{ij} * pi{j}
        

        Arguments
        ----------
        exchangeabilities : ArrayLike, (C, A, A)
            square exchangeability matrix per clas
            
        equilibrium_distributions : ArrayLike, (C, A)
            equilibrium distribution
            
        rate_multiplier : ArrayLike, (C,)
            scaling factor

        Returns
        -------
        rate_mat_times_rho : ArrayLike, (C, A, A)
            Q = rho * chi * diag(pi)

        """
        # get normalized rate matrix per class
        subst_rate_mat = rate_matrix_from_exch_equl( exchangeabilities = exchangeabilities,
                                                      equilibrium_distributions = equilibrium_distributions,
                                                      norm=norm ) #(C, A, A)
        
        # scale it
        rate_mat_times_rho = scale_rate_matrix(subst_rate_mat = subst_rate_mat,
                                                rate_multiplier = rate_multiplier) #(C, A, A)
        
        return rate_mat_times_rho 


class GTRRateMatFromFile(GTRRateMat):
    """
    Like GTRRateMat, but load rate multipliers and exchangeabilities from 
        files as-is
    
    If only one model (no mixtures), then the rate multiplier is automatically 
        set to one
        
        
    Initialize with
    ----------------
    config : dict
        config['num_mixtures'] :  int
            number of emission site classes
            
        config['filenames']['rate_mult'] :  str
            name of the rate multipliers to load
        
        config['filenames']['exch'] : str
            name of the exchangeabilities to load
        
        config['norm_rate_matrix'] : bool
            flag to normalize rate matrix to t = one substitution
            
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    __call__
    
    Methods inheried from GTRRateMat
    ----------------------------------
    _prepare_rate_matrix
        function to prepare rate matrix 
    """
    config: dict
    name: str
    
    def setup(self):
        """
        Flax Module Parameters
        -----------------------
        None
        
        """
        ###################
        ### read config   #
        ###################
        self.num_mixtures = self.config['num_mixtures']
        self.norm_rate_mults = self.config.get('norm_rate_mults',True)
        self.norm_rate_matrix = self.config.get('norm_rate_matrix',True)

        exchangeabilities_file = self.config['filenames']['exch']
        rate_multiplier_file = self.config['filenames'].get('rate_mult', None)
        
        
        ########################
        ### RATE MULTIPLIERS   #
        ########################
        if self.num_mixtures > 1:
            with open(rate_multiplier_file, 'rb') as f:
                self.rate_multiplier = jnp.load(f)
        else:
            self.rate_multiplier = jnp.array([1])
            
            
        #########################
        ### EXCHANGEABILITIES   #
        #########################
        with open(exchangeabilities_file,'rb') as f:
            exch_from_file = jnp.load(f)
        
        # if providing a vector, need to prepare a square exchangeabilities matrix
        if len(exch_from_file.shape) == 1:
            self.exchangeabilities = upper_tri_vector_to_sym_matrix( exch_from_file ) #(A, A)
            
        # otherwise, use the matrix as-is
        else:
            self.exchangeabilities = exch_from_file #(A, A)
        
    def __call__(self,
                 logprob_equl,
                 log_class_probs,
                 *args,
                 **kwargs):
        """
        C = number of latent site classes
        A = alphabet size
        
        
        Arguments
        ----------
        logprob_equl : ArrayLike, (C, A)
            log-transformed equilibrium distribution
        
        Returns
        -------
        rate_mat_times_rho : ArrayLike, (C, A, A)
            scaled rate matrix
        """
        # undo log transform on equilibrium
        equl = jnp.exp(logprob_equl) #(C, A)
        
        # possibly rescale rate multiplier
        if (self.norm_rate_mults) and (self.num_mixtures > 1):
            rate_multiplier = scale_rate_multipliers( unnormed_rate_multipliers = self.rate_multiplier,
                                    log_class_probs = log_class_probs )
        else:
            rate_multiplier = self.rate_multiplier
        
        # return scaled rate matrix
        rate_mat_times_rho =  self._prepare_rate_matrix(exchangeabilities = self.exchangeabilities,
                                                        equilibrium_distributions = equl,
                                                        rate_multiplier = rate_multiplier,
                                                        norm = self.norm_rate_matrix) #(C, A, A)
        return rate_mat_times_rho
    


###############################################################################
### RATE MATRICES: HKY85   ####################################################
###############################################################################
class HKY85RateMat(GTRRateMat):
    """
    use the HKY85 rate matrix
    
    
    with ti = transition rate and tv = transversion rate, 
        exchangeabilities are:
            
        [[ 0, tv, ti, tv],
         [tv,  0, tv, ti],
         [ti, tv,  0, tv],
         [tv, ti, tv,  0]]
    
    
    Initialize with
    ----------------
    config : dict
        config['num_mixtures'] :  int
            number of emission site classes
        
        config['rate_mult_activation'] : {'bound_sigmoid', 'softplus'}
            what activation to use for logits of rate multiplier
        
        config['rate_mult_range'] : List[float, float], optional
            only needed when using bound_sigmoid for rate multipliers
            first value is min, second value is max
            Default is (0.01, 10)
        
        config['exchange_range'] : List[float, float]
            exchangeabilities undergo bound_sigmoid transformation, this
            specifies the min and max
            Default is (1e-4, 12)
        
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    
    Methods inheried from GTRRateMat
    ----------------------------------
    __call__
    _prepare_rate_matrix
        function to prepare rate matrix
    
    Methods inherited from models.model_utils.BaseClasses.ModuleBase
    -----------------------------------------------------------------
    sow_histograms_scalars
        for tensorboard logging
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        C: number of site classes
        
        
        Flax Module Parameters
        -----------------------
        rate_mult_logits : ArrayLike, (C,)
            rate multiplier per class; ONLY present if C > 1
            initialized from unit normal
        
        ti_tv_vec : ArrayLike, (2,)
            first value is transition rate, second value is transversion rate
            initialized from unit normal
        
        """
        # required
        self.num_mixtures = self.config['num_mixtures']
        
        # have defaults; may or may not be used
        self.rate_mult_min_val, self.rate_mult_max_val  = self.config.get( 'rate_mult_range', (0.01, 10) )
        self.exchange_min_val, self.exchange_max_val  = self.config.get( 'exchange_range', (1e-4, 12) )
        
        # tested on GTR for proteins; decided that these defaults are best
        self.rate_mult_activation = self.config.get('rate_mult_activation', 
                                                    'bound_sigmoid')
        self.norm_rate_mults = self.config.get('norm_rate_mults', True)
        self.norm_rate_matrix = self.config.get('norm_rate_matrix', True)
        
        # validate
        if self.rate_mult_activation not in ['bound_sigmoid', 'softplus']:
            raise ValueError('Pick either: bound_sigmoid, softplus')
            
            
        ########################
        ### RATE MULTIPLIERS   #
        ########################
        if self.num_mixtures > 1:
            # activation
            if self.rate_mult_activation == 'bound_sigmoid':
                self.rate_multiplier_activation = partial(bound_sigmoid,
                                                          min_val = self.rate_mult_min_val,
                                                          max_val = self.rate_mult_max_val)
            
            elif self.rate_mult_activation == 'softplus':
                self.rate_multiplier_activation = jax.nn.softplus
        
            # initializers
            self.rate_mult_logits = self.param('rate_multipliers',
                                               nn.initializers.normal(),
                                               (self.num_mixtures,),
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
        
        self.exchange_activation = partial(bound_sigmoid,
                                           min_val = self.exchange_min_val,
                                           max_val = self.exchange_max_val)
        

class HKY85RateMatFromFile(GTRRateMatFromFile):
    """
    Like HKY85RateMat, but load parameters from file
        
        
    Initialize with
    ----------------
    config : dict
        config['num_mixtures'] :  int
            number of emission site classes
            
        config['filenames']['rate_mult'] :  str
            name of the rate multipliers to load
        
        config['filenames']['exch'] : str
            name of the exchangeabilities to load
        
        config['norm_rate_matrix'] : bool
            flag to normalize rate matrix to t = one substitution
            
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    
    Methods inherited from GTRRateMatFromFile
    ------------------------------------------
    __call__
    
    Methods inheried from GTRRateMat
    ----------------------------------
    _prepare_rate_matrix
        function to prepare rate matrix 
    """
    config: dict
    name: str
    
    def setup(self):
        self.num_mixtures = self.config['num_mixtures']
        self.norm_rate_mults = self.config.get('norm_rate_mults',True)
        self.norm_rate_matrix = self.config.get('norm_rate_matrix',True)
        
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
        
        self.exchangeabilities = upper_tri_vector_to_sym_matrix( hky85_raw_vec )
        
        ### RATE MULTIPLIERS: (c,)
        if self.num_mixtures > 1:
            with open(rate_multiplier_file, 'rb') as f:
                self.rate_multiplier = jnp.load(f)
        else:
            self.rate_multiplier = jnp.array([1])
 
    
###############################################################################
### PROBABILITY MATRICES: F81   ###############################################
###############################################################################
class F81Logprobs(ModuleBase):
    """
    Get the conditional and joint logprobs for an F81 model
    
    If only one model (no mixtures), then the rate multiplier is automatically 
        set to one
    
    
    Initialize with
    ----------------
    config : dict
        config['num_mixtures'] :  int
            number of emission site classes
        
        config['rate_mult_activation'] : {'bound_sigmoid', 'softplus'}
            what activation to use for logits of rate multiplier
        
        config['rate_mult_range'] : List[float, float], optional
            only needed when using bound_sigmoid for rate multipliers
            first value is min, second value is max
            Default is (0.01, 10)
        
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    __call__
    
    Methods inherited from models.model_utils.BaseClasses.ModuleBase
    -----------------------------------------------------------------
    sow_histograms_scalars
        for tensorboard logging
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        C: number of site classes
        
        
        Flax Module Parameters
        -----------------------
        rate_mult_logits : ArrayLike, (C,)
            rate multiplier per class; ONLY present if C > 1
            initialized from unit normal
        
        """
        # required
        self.num_mixtures = self.config['num_mixtures']
        
        # have defaults
        self.norm_rate_mults = self.config.get('norm_rate_mults',True)
        self.norm_rate_matrix = self.config.get('norm_rate_matrix',True)
        self.rate_mult_min_val, self.rate_mult_max_val  = self.config.get( 'rate_mult_range', (0.01, 10) )
        self.rate_mult_activation = self.config.get('rate_mult_activation', 'bound_sigmoid')
        
        # validate
        if self.rate_mult_activation not in ['bound_sigmoid', 'softplus']:
            raise ValueError('Pick either: bound_sigmoid, softplus')
            
            
        ### RATE MULTIPLIERS
        if self.num_mixtures > 1:
            # activation
            if self.rate_mult_activation == 'bound_sigmoid':
                self.rate_multiplier_activation = partial(bound_sigmoid,
                                                          min_val = self.rate_mult_min_val,
                                                          max_val = self.rate_mult_max_val)
            
            elif self.rate_mult_activation == 'softplus':
                self.rate_multiplier_activation = jax.nn.softplus
        
            # initializers
            self.rate_mult_logits = self.param('rate_multipliers',
                                               nn.initializers.normal(),
                                               (self.num_mixtures,),
                                               jnp.float32)
        
    def __call__(self,
                 logprob_equl: jnp.array,
                 log_class_probs: jnp.array,
                 t_array: jnp.array,
                 return_cond: bool,
                 sow_intermediates: bool,
                 *args,
                 **kwargs):
        """
        B = batch size
        T = times
        C = number of latent site classes
        A = alphabet size
        
        Arguments
        ----------
        logprob_equl : ArrayLike, (C, A)
            log-transformed equilibrium distribution
        
        log_class_probs : ArrayLike, (C,)
            log-transformed class probabilties
        
        t_array : ArrayLike, (T,) or (B,)
            times at which to calculate F81 probability matrix
        
        
        Returns
        -------
        ArrayLike, (T, C, A, A)
            log-probability of emission at match sites, according to F81
        """
        prob_equl = jnp.exp(logprob_equl) #(C, A)
        C = prob_equl.shape[0]
        
        
        ### rate multiplier
        if self.num_mixtures > 1:
            # apply activation of choice
            if sow_intermediates:
                for i in range(self.rate_mult_logits.shape[0]):
                    val_to_write = self.rate_mult_logits[i]
                    act = self.rate_mult_activation
                    lab = (f'{self.name}/logit BEFORE {act} activation- '+
                           f'rate multiplier for class {i}')
                    self.sow_histograms_scalars(mat= val_to_write, 
                                                label=lab, 
                                                which='scalars')
                    del lab
                    
            rate_multiplier = self.rate_multiplier_activation( self.rate_mult_logits ) #(C,)
            
            if sow_intermediates:
                for i in range(rate_multiplier.shape[0]):
                    val_to_write = rate_multiplier[i]
                    act = self.rate_mult_activation
                    lab = (f'{self.name}/rate AFTER {act} activation- '+
                           f'rate multiplier for class {i}')
                    self.sow_histograms_scalars(mat= val_to_write, 
                                                label=lab, 
                                                which='scalars')
                    del lab
            
            # normalize
            if self.norm_rate_mults:
                rate_multiplier = scale_rate_multipliers( unnormed_rate_multipliers = rate_multiplier,
                                        log_class_probs = log_class_probs )
            
                if sow_intermediates:
                    for i in range(rate_multiplier.shape[0]):
                        val_to_write = rate_multiplier[i]
                        lab = (f'{self.name}/rate AFTER normalization- '+
                               f'rate multiplier for class {i}')
                        self.sow_histograms_scalars(mat= val_to_write, 
                                                    label=lab, 
                                                    which='scalars')
                        del lab
            
            
        elif self.num_mixtures == 1:
            rate_multiplier = jnp.ones( (C,) ) #(C,) but C=1
        
        joint_logprob = self._fill_f81(equl = prob_equl, 
                                      rate_multiplier = rate_multiplier, 
                                      t_array = t_array,
                                      return_cond = return_cond) #(T,C,A,A)
        return joint_logprob
        
    
    def _fill_f81(self, 
                  equl,
                  rate_multiplier, 
                  t_array, 
                  return_cond = False):
        """
        return logP(emission at match) directly
        """
        T = t_array.shape[0]
        C = rate_multiplier.shape[0]
        A = equl.shape[-1]
        
        # possibly normalize to one substitution per time t
        if self.norm_rate_matrix:
            # \sum_i pi_i chi_{ii} = \sum_i pi_i (1-\pi_i) = 1 - \sum_i pi_i^2
            norm_factor = 1 / ( 1 - jnp.square(equl).sum(axis=(-1)) ) # (C,)
        elif not self.norm_rate_matrix:
            norm_factor = jnp.ones( (C,) ) #(C,)
        
        # the exponential operand
        oper = -( rate_multiplier[None,:] * norm_factor[None,:] * t_array[:,None] ) #(T, C)
        exp_oper = jnp.exp(oper)[...,None] #(T, C, 1)
        equl = equl[None,:] #(1, C, A)
        
        # all off-diagonal entries, i != j
        # pi_j * ( 1 - exp(-rate*t) )
        row = equl * ( 1 - exp_oper ) #(T, C, A)
        cond_probs_raw = jnp.broadcast_to( row[:, :, None, :], (T, C, A, A) )  # (T, C, A, A)
        
        # diagonal entries, i = j
        #   pi_j + (1-pi_j) * exp(-rate*t)
        diags = equl + (1-equl) * exp_oper #(T, C, A)
        diag_indices = jnp.arange(A)  # (A,)
        cond_probs = cond_probs_raw.at[:, :, diag_indices, diag_indices].set(diags) # (T, C, A, A)
        
        if not return_cond:
            # P(x) P(y|x,t) for all T, C
            equl_reshaped = equl[..., None] #(1, C, A, 1)
            joint_probs = cond_probs * equl_reshaped # (T, C, A, A)
            return safe_log( joint_probs ) # (T, C, A, A)
        
        elif return_cond:
            return safe_log( cond_probs ) # (T, C, A, A)
        

class F81LogprobsFromFile(F81Logprobs):
    """
    like F81Logprobs, but load rates from file as-is
    
    If only one model (no mixtures), then the rate multiplier is automatically 
        set to one
    
    
    Initialize with
    ----------------
    config : dict
        config['num_mixtures'] :  int
            number of emission site classes
            
        config['filenames']['rate_mult'] :  str
            name of the rate multipliers to load
            
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    __call__
    
    Inherited from F81Logprobs
    --------------------------
    _fill_f81
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        Flax Module Parameters
        -----------------------
        None
        
        """
        ### read config
        self.num_mixtures = self.config['num_mixtures']
        self.norm_rate_mults = self.config.get('norm_rate_mults',True)
        self.norm_rate_matrix = self.config.get('norm_rate_matrix',True)
        rate_multiplier_file = self.config['filenames'].get('rate_mult', None)
        
        
        ### RATE MULTIPLIERS
        if self.num_mixtures > 1:
            with open(rate_multiplier_file, 'rb') as f:
                self.rate_multiplier = jnp.load(f)
        else:
            self.rate_multiplier = jnp.array([1])
    
    def __call__(self,
                 logprob_equl: jnp.array,
                 log_class_probs: jnp.array,
                 t_array: jnp.array,
                 return_cond: bool,
                 *args,
                 **kwargs):
        """
        B = batch size
        T = times
        C = number of latent site classes
        A = alphabet size
        
        Arguments
        ----------
        logprob_equl : ArrayLike, (C, A)
            log-transformed equilibrium distribution
        
        log_class_probs : ArrayLike, (C,)
            log-transformed class probabilties
        
        t_array : ArrayLike, (T,) or (B,)
            times at which to calculate F81 probability matrix
        
        
        Returns
        -------
        ArrayLike, (T, C, A, A)
            log-probability of emission at match sites, according to F81
        """
        equl = jnp.exp(logprob_equl) #(C, A)
        
        # possibly rescale rate multiplier
        if (self.norm_rate_mults) and (self.num_mixtures > 1):
            rate_multiplier = scale_rate_multipliers( unnormed_rate_multipliers = self.rate_multiplier,
                                    log_class_probs = log_class_probs )
        else:
            rate_multiplier = self.rate_multiplier
        
        return self._fill_f81(equl = equl, 
                              rate_multiplier = rate_multiplier, 
                              t_array = t_array,
                              return_cond = return_cond)

    
###############################################################################
### EQUILIBRIUM DISTRIBUTION MODELS   #########################################
###############################################################################
class EqulDistLogprobsPerClass(ModuleBase):
    """
    Equilibrium distribution of emissions
    
    
    Initialize with
    ----------------
    config : dict
        config['emission_alphabet_size'] : int
            size of emission alphabet; 20 for proteins, 4 for DNA
            
        config['num_mixtures'] : int
            number of emission site classes
    
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    __call__
    
    Methods inherited from models.model_utils.BaseClasses.ModuleBase
    ----------------------------------------------------------------
    sow_histograms_scalars
        for tensorboard logging
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        C = number of site classes
        A = alphabet size
        
        
        Flax Module Parameters
        -----------------------
        logits : ArrayLike (C,)
            initialize logits from unit normal
        
        """
        emission_alphabet_size = self.config['emission_alphabet_size']
        num_mixtures = self.config['num_mixtures']
        
        self.logits = self.param('Equilibrium distr.',
                                  nn.initializers.normal(),
                                  (num_mixtures, emission_alphabet_size),
                                  jnp.float32) #(C, A)
        
    def __call__(self,
                 sow_intermediates: bool,
                 *args,
                 **kwargs):
        """
        C: number of site classes
        
        
        Arguments
        ----------
        sow_intermediates : bool
            switch for tensorboard logging
          
        Returns
        -------
        log_equl_dist : ArrayLike, (C, A)
            log-transformed equilibrium distribution
        """
        log_equl_dist = nn.log_softmax( self.logits, axis = 1 ) #(C, A)

        if sow_intermediates:
            for c in range(log_equl_dist.shape[0]):
                lab = f'{self.name}/equilibrium dist for class {c}'
                self.sow_histograms_scalars(mat= jnp.exp(log_equl_dist[c,...]), 
                                            label=lab, 
                                            which='scalars')
                del lab
        
        return log_equl_dist


class EqulDistLogprobsFromFile(ModuleBase):
    """
    Load equilibrium distribution from file
    
    
    Initialize with
    ----------------
    config : dict
        config["filenames"]["equl_dist"]: str
              file of equilibrium distributions to load
            
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    __call__
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        
        Flax Module Parameters
        -----------------------
        none
        
        """
        equl_file = self.config['filenames']['equl_dist']
        
        with open(equl_file,'rb') as f:
            prob_equilibr = jnp.load(f, allow_pickle=True)
        
        # if there's no dim for class, add it
        if len(prob_equilibr.shape) == 1:
            prob_equilibr = prob_equilibr[None,:] #(C, A)
        
        self.logprob_equilibr = safe_log(prob_equilibr)
        
        
    def __call__(self,
                 *args,
                 **kwargs):
        """
        Returns log-transformed equilibrium distribution
        """
        return self.logprob_equilibr #(C, A)
    

class EqulDistLogprobsFromCounts(ModuleBase):
    """
    If there's only one class, construct an equilibrium distribution 
        from observed frequencies
    
    A = alphabet size
    
    
    Initialize with
    ----------------
    config : dict
        config["training_dset_emit_counts"] : ArrayLike, (A,)
            observed counts to turn into frequencies
            
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    __call__
    """
    config: dict
    name: str
    
    def setup(self):
        """
        
        Flax Module Parameters
        -----------------------
        none
        
        """
        training_dset_emit_counts = self.config['training_dset_emit_counts'] #(A,)
        prob_equilibr = training_dset_emit_counts/training_dset_emit_counts.sum() #(A,)
        logprob_equilibr = safe_log( prob_equilibr ) #(A,)
        
        # C=1
        self.logprob_equilibr = logprob_equilibr[None,...] #(C, A)
        
        
    def __call__(self,
                 *args,
                 **kwargs):
        """
        Returns log-transformed equilibrium distribution
        """
        return self.logprob_equilibr #(C, A)
