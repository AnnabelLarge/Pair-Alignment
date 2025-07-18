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

originals:
------------
'EqulDistLogprobsFromCounts',
'EqulDistLogprobsPerClass',
'F81Logprobs',
'GTRLogprobs',
'HKY85Logprobs',
'IndpRateMultipliers',
'RateMultipliersPerClass',
'SiteClassLogprobs',

loading from files:
--------------------
'EqulDistLogprobsFromFile',
'F81LogprobsFromFile',
'GTRLogprobsFromFile',
'HKY85LogprobsFromFile',
'IndpRateMultipliersFromFile',
'RateMultipliersPerClassFromFile',
'SiteClassLogprobsFromFile'
    

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
                                                              scale_rate_matrix,
                                                              upper_tri_vector_to_sym_matrix,
                                                              cond_logprob_emit_at_match_per_mixture,
                                                              joint_logprob_emit_at_match_per_mixture,
                                                              fill_f81_logprob_matrix)


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
            if true, enforce constraint: \sum_c \sum_k P(c,k) * \rho_{c,k} = 1
    
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    __call__
    _set_model_simplification_flags
    _init_prob_logits
    _init_rate_logits
    _get_norm_factor
    
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
        C = number of site classes
        
        a shared method; shapes vary depending on how this module is used
        
        
        Flax Module Parameters
        -----------------------
        rate_mult_prob_logits : ArrayLike 
            if fitting P(k|c): (C, K)
            if fitting P(k): (K,)
        
        rate_mult_logits : ArrayLike (K,)
            if rate multipliers are per class: (C, K)
            if rate multipliers are independent: (K,)
        
        """
        ### read config file
        self.num_mixtures = self.config['num_mixtures']
        self.k_rate_mults = self.config['k_rate_mults']
        
        # optional
        self.rate_mult_min_val, self.rate_mult_max_val  = self.config.get( 'rate_mult_range', (0.01, 10) )
        
        # sometimes, might use model simplifications
        # also decide norm_rate_mult here; sometimes I read this from config
        #   file, but at other times, I never have to normalize my rate 
        #   multipliers
        self._set_model_simplification_flags
        self._set_model_simplification_flags()
        
            
        ### rate multipliers
        if not self.use_unit_rate_mult:
            self.rate_multiplier_activation = partial(bound_sigmoid,
                                               min_val = self.rate_mult_min_val,
                                               max_val = self.rate_mult_max_val)
        
            # if \rho_{c, k}, then logits are (C, K)
            # if \rho_{k}, then logits are (K,)
            self._init_rate_logits()
        
        
        ### probability of choosing a specific rate multiplier
        if not self.prob_rate_mult_is_one:
            # if P(k | c), then logits are (C, K)
            # if P(k), then logits are (K,)
            self._init_prob_logits()            
    
        
    def __call__(self,
                 sow_intermediates: bool,
                 log_class_probs: jnp.array):
        """
        K = number of site classes
        C: number of site classes
        
        a shared method; shapes vary depending on how this module is used
        
        
        Arguments
        ----------
        sow_intermediates : bool
            switch for tensorboard logging
        
        log_class_probs : ArrayLike, (C,)
            (from a different module); the log-probability of latent class 
            assignment
        
        Returns
        -------
        log_rate_mult_probs : ArrayLike
            the log-probability of having rate class k, given that the column 
            is assigned to latent class c
            > if fitting P(k|c): (C, K)
            > if fitting P(k): (K,)
        
        rate_multipliers : ArrayLike, (C, K)
            the actual rate multiplier for rate class k and latent class c
            > if rate multipliers are per class: (C, K)
            > if rate multipliers are independent: (K,)
          
        """
        # all outputs must be this size
        out_size = ( self.num_mixtures, self.k_rate_mults )
            
        ### P(K|C) or P(K)
        if not self.prob_rate_mult_is_one:
            log_rate_mult_probs = nn.log_softmax( self.rate_mult_prob_logits, axis=-1 ) #(C, K) or (K,)
            
            if sow_intermediates:
                lab = (f'{self.name}/prob of rate multipliers')
                self.sow_histograms_scalars(mat= jnp.exp(log_rate_mult_probs), 
                                            label=lab, 
                                            which='scalars')
                del lab

            # possibly broadcast up to (C, K)
            log_rate_mult_probs = self._maybe_duplicate(log_rate_mult_probs) #(C, K)
        
        elif self.prob_rate_mult_is_one:
            log_rate_mult_probs = jnp.zeros( out_size ) #(C, K) 
        
        
        ### \rho_{c,k}
        if not self.use_unit_rate_mult:
            rate_multipliers = self.rate_multiplier_activation( self.rate_mult_logits ) #(C, K) or (K,)
    
            if sow_intermediates:
                lab = (f'{self.name}/rate multipliers')
                self.sow_histograms_scalars(mat= rate_multipliers, 
                                            label=lab, 
                                            which='scalars')
                del lab
        
            # possibly normalize to enforce one of these constraints:
            # \sum_c \sum_k P(c, k) * \rho_{c,k} = 1
            # \sum_k P(k) * \rho_{k} = 1
            if self.norm_rate_mults:
                norm_factor = self._get_norm_factor(rate_multipliers = rate_multipliers,
                                                    log_rate_mult_probs = log_rate_mult_probs,
                                                    log_class_probs = log_class_probs) #float
                rate_multipliers = rate_multipliers / norm_factor #(C, K) or (K,)
            
            # possibly broadcast up to (C, K)
            rate_multipliers = self._maybe_duplicate(rate_multipliers) #(C, K)
        
        elif self.use_unit_rate_mult:
            rate_multipliers = jnp.ones( out_size ) #(C, K) 
                
        return (log_rate_mult_probs, rate_multipliers)
        
    
    def _set_model_simplification_flags(self):
        """
        If C = 1 and K = 1: no mixtures
            > prob_rate_mult_is_one = True
            > use_unit_rate_mult = True
            > don't ever have to normalize
        
        If C > 1 and K = 1: then there's one unique rate per site class 
            (as was done in previous results); for each class, the probability
            of selecting the single possible rate multiplier is 1
            > prob_rate_mult_is_one = True
            > use_unit_rate_mult = False
            > may have to normalize rates by log_class_probs
        
        If C = 1 and K > 1: no mixtures over site classes, but still
            have a mixture over rate multipliers
            > prob_rate_mult_is_one = False
            > use_unit_rate_mult = False
            > may have to normalize rates by log_rate_mult_probs
        
        If C > 1 and K > 1: mixtures over both
            > prob_rate_mult_is_one = False
            > use_unit_rate_mult = False
            > may have to normalize rates by log_rate_mult_probs AND log_class_probs
        
        """
        # defaults
        prob_rate_mult_is_one = False
        use_unit_rate_mult = False
        norm_rate_mults = self.config.get('norm_rate_mults', True)
        
        # if k_rate_mults = 1, then P(k|c) = 1 ( or P(k)=1 )
        if self.k_rate_mults == 1:
            prob_rate_mult_is_one = True
            
            # if num_mixtures = 1 AND k_rate_mults = 1, then just use unit rate
            # also NEVER normalize the rate multiplier (since its just one)
            if self.num_mixtures == 1:
                use_unit_rate_mult = True
                norm_rate_mults = False
                
        self.prob_rate_mult_is_one = prob_rate_mult_is_one
        self.use_unit_rate_mult = use_unit_rate_mult
        self.norm_rate_mults = norm_rate_mults
        
    
    def _init_prob_logits(self):
        """
        initialize the (C, K) logits for P(K|C)
        
        self.rate_mult_prob_logits is a flax parameter
        """
        out_size = ( self.num_mixtures, self.k_rate_mults )
        self.rate_mult_prob_logits = self.param('rate_mult_prob_logits',
                                        nn.initializers.normal(),
                                        out_size,
                                        jnp.float32) #(C,K)
    
    def _init_rate_logits(self):
        """
        initialize the (C, K) logits for rate multiplier \rho_{c,k}
        self.rate_mult_logits is a flax parameter
        """
        out_size = ( self.num_mixtures, self.k_rate_mults )
        self.rate_mult_logits = self.param('rate_mult_logits',
                                           nn.initializers.normal(),
                                           out_size,
                                           jnp.float32) #(C,K)
    
    def _get_norm_factor(self,
                         rate_multipliers: jnp.array,
                         log_rate_mult_probs: jnp.array,
                         log_class_probs: jnp.array,
                         *args,
                         **kwargs):
        """
        return the normalization factor needed for constraint:
            \sum_c \sum_k P(c, k) * \rho_{c, k} = 1
        
        Arguments:
        ----------
        rate_multipliers : ArrayLike, (C,K)
            \rho_{c, k}; the un-normalized rate multipliers
            
        log_rate_mult_probs : ArrayLike, (C,K)
            P(k | c); probability of having rate class k, given that the 
            latent class assignment is c
        
        log_class_probs : ArrayLike, (C,)
            P(c); marginal probability of latent class assignment c
        
        
        Returns:
        --------
        norm_factor : float
        
        """
        # logP(C) + logP(K|C) = logP(C, K)
        joint_logprob_class_rate_mult = log_class_probs[:, None] + log_rate_mult_probs #(C, K)
        
        # exp( logP(C, K) ) = P(C, K)
        joint_prob_class_rate_mult = jnp.exp(joint_logprob_class_rate_mult) #(C, K)
        
        # \sum_c \sum_k P(c, k) * \rho_{c, k}
        norm_factor = jnp.multiply(joint_prob_class_rate_mult, rate_multipliers).sum() #float
        
        return norm_factor
    
    def _maybe_duplicate(self, 
                         matrix):
        """
        this is a placeholder function, for now
        """
        return matrix #(C,K)
        
    
class IndpRateMultipliers(RateMultipliersPerClass):
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
            if true, enforce constraint: \sum_k P(k)*\rho_k = 1
    
    name : str
        class name, for flax
    
    Methods here
    ------------
    _set_model_simplification_flags
    _init_prob_logits
    _init_rate_logits
    _get_norm_factor
    
    Methods inherited from RateMultipliersPerClass
    ------------------------------------------------
    __call__
    setup
    
    Methods inherited from models.model_utils.BaseClasses.ModuleBase
    ----------------------------------------------------------------
    sow_histograms_scalars
        for tensorboard logging
    
    """
    config: dict
    name: str
      
    def _set_model_simplification_flags(self):
        """
        If C = 1 and K = 1: no mixtures
            > prob_rate_mult_is_one = True
            > use_unit_rate_mult = True
            > dont ever have to normalize
        
        If C > 1 and K = 1: no mixtures over rate multipliers; all classes 
            use rate multiplier of 1
            > prob_rate_mult_is_one = True
            > use_unit_rate_mult = True
            > also dont ever have to normalize, since rate multipliers are
              independent of class label
        
        If C = 1 and K > 1: no mixtures over site classes, but still
            have a mixture over rate multipliers
            > prob_rate_mult_is_one = False
            > use_unit_rate_mult = False
            > might normalize over log_rate_mult_probs
        
        If C > 1 and K > 1: mixtures over both, but the same mixture over rate 
            multipliers is used for all possible latent site class labels
            > prob_rate_mult_is_one = False
            > use_unit_rate_mult = False
            > might normalize over log_rate_mult_probs (but not log_class_probs,
              since this is independent of class label)
        """
        if self.k_rate_mults == 1:
            self.prob_rate_mult_is_one = True
            self.use_unit_rate_mult = True
            self.norm_rate_mults = False
        
        elif self.k_rate_mults > 1:
            self.prob_rate_mult_is_one = False
            self.use_unit_rate_mult = False
            self.norm_rate_mults = self.config.get('norm_rate_mults', True)
        
        
    def _init_prob_logits(self):
        """
        initialize the (K,) logits for P(K)
        
        self.rate_mult_prob_logits is a flax parameter
        """
        self.rate_mult_prob_logits = self.param( 'rate_mult_prob_logits',
                                            nn.initializers.normal(),
                                            self.k_rate_mults,
                                            jnp.float32 ) #(K,)
    
    def _init_rate_logits(self):
        """
        initialize the (K,) logits for rate multiplier \rho_{k}
        
        self.rate_mult_logits is a flax parameter
        """
        self.rate_mult_logits = self.param( 'rate_mult_logits',
                                       nn.initializers.normal(),
                                       self.k_rate_mults,
                                       jnp.float32 ) #(K,)
        
    def _get_norm_factor(self,
                         rate_multipliers: jnp.array,
                         log_rate_mult_probs: jnp.array,
                         *args,
                         **kwargs):
        """
        return the normalization factor needed for constraint:
            \sum_k P(k) * \rho_{k} = 1
        
        Arguments:
        ----------
        rate_multipliers : ArrayLike, (K,)
            \rho_{k}; the un-normalized rate multipliers
            
        log_rate_mult_probs : ArrayLike, (K,)
            P(k); probability of having rate class k
        
        
        Returns:
        --------
        norm_factor : float
        
        """
        # exp( logP(K) ) = P(K)
        rate_mult_probs = jnp.exp(log_rate_mult_probs) #(K)
        
        # \sum_k P(K) \rho_{k} 
        norm_factor = jnp.multiply( rate_mult_probs, rate_multipliers ).sum()
        
        return norm_factor
    
    def _maybe_duplicate(self, 
                         matrix):
        """
        matrix is technically (K,), and I need to duplicate to (C, K)
        """
        out_size = ( self.num_mixtures, self.k_rate_mults )
        return jnp.broadcast_to( matrix[None,:], out_size )


class RateMultipliersPerClassFromFile(RateMultipliersPerClass):
    """
    C: number of latent site classes
    K: numer of rate multipliers
    
    load probabilities and rate multipliers from files
    
    
    Initialize with
    ----------------
    config : dict
        config['num_mixtures'] : int
            number of mixtures
            
        config['k_rate_mults'] : int
            number of rate multipliers
    
        config['filenames']['rate_mults'] : str
            file of rate multipliers
            
        config['filenames']['rate_mult_probs'] : str
            file of probabilities
            
        config['norm_rate_mults'] : bool
            if true, enforce constraint: \sum_c \sum_k P(c,k) * \rho_{c,k} = 1
    
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    __call__
    
    Methods inherited from RateMultipliersPerClass
    ------------------------------------------------
    _set_model_simplification_flags
    _get_norm_factor
    
    """
    def setup(self):
        ### read config file
        self.num_mixtures = self.config['num_mixtures']
        self.k_rate_mults = self.config['k_rate_mults']
        out_size = ( self.num_mixtures, self.k_rate_mults )
        
        # possibly simplify model setup
        self._set_model_simplification_flags()
        
        
        ### rate files: rate multipliers
        if not self.use_unit_rate_mult:
            in_file = self.config['filenames']['rate_mults']
            with open(in_file,'rb') as f:
                self.rate_multipliers = jnp.load(f) #(C, K)
        
        elif self.use_unit_rate_mult:
            self.rate_multipliers = jnp.ones( out_size ) #(C, K) 
            
            
        ### read files: P(k|c)
        if not self.prob_rate_mult_is_one:
            in_file =  self.config['filenames']['rate_mult_probs']
            with open(in_file,'rb') as f:
                rate_mult_probs = jnp.load(f) #(C, K)
            self.log_rate_mult_probs = safe_log( rate_mult_probs )
        
        elif self.prob_rate_mult_is_one:
            self.log_rate_mult_probs = jnp.zeros( out_size ) #(C, K) 
            
        
    def __call__(self,
                 sow_intermediates: bool,
                 log_class_probs: jnp.array):
        # P(K|C), \rho_{c,k}
        log_rate_mult_probs = self.log_rate_mult_probs #(C,K)
        rate_multipliers = self.rate_multipliers #(C,K)
        
        # possibly normalize
        if self.norm_rate_mults:
            norm_factor = self._get_norm_factor(rate_multipliers = rate_multipliers,
                                                log_rate_mult_probs = log_rate_mult_probs,
                                                log_class_probs = log_class_probs) #float
            rate_multipliers = rate_multipliers / norm_factor #(C, K)
            
        return (log_rate_mult_probs, rate_multipliers)


class IndpRateMultipliersFromFile(IndpRateMultipliers):
    """
    C: number of latent site classes
    K: numer of rate multipliers
    
    load probabilities and rate multipliers from files
    
    Initialize with
    ----------------
    config : dict
        config['num_mixtures'] : int
            number of mixtures
            
        config['k_rate_mults'] : int
            number of rate multipliers
    
        config['filenames']['rate_mults'] : str
            file of rate multipliers
            
        config['filenames']['rate_mult_probs'] : str
            file of probabilities
            
        config['norm_rate_mults'] : bool
            if true, enforce constraint: \sum_c \sum_k P(c,k) * \rho_{c,k} = 1
    
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    __call__
    _maybe_dedup
    
    Methods inherited from IndpRateMultipliers
    -------------------------------------------
    _set_model_simplification_flags
    _get_norm_factor
    _maybe_duplicate
    
    """
    def setup(self):
        ### read config file
        self.num_mixtures = self.config['num_mixtures']
        self.k_rate_mults = self.config['k_rate_mults']
        
        # possibly simplify model setup
        self._set_model_simplification_flags()
        
        
        ### rate files: rate multipliers
        if not self.use_unit_rate_mult:
            in_file = self.config['filenames']['rate_mults']
            with open(in_file,'rb') as f:
                rate_multipliers = jnp.load(f) #(C, K) or (K,)
            self.rate_multipliers = self._maybe_dedup(rate_multipliers) #(K,)
            
        elif self.use_unit_rate_mult:
            self.rate_multipliers = jnp.ones( (self.k_rate_mults,) ) #(K,)
            
            
        ### read files: P(k|c)
        if not self.prob_rate_mult_is_one:
            in_file =  self.config['filenames']['rate_mult_probs']
            with open(in_file,'rb') as f:
                rate_mult_probs = jnp.load(f) #(C, K) or (K,)
            rate_mult_probs = self._maybe_dedup(rate_mult_probs) #(K,)
            self.log_rate_mult_probs = safe_log( rate_mult_probs ) #(K,)
        
        elif self.prob_rate_mult_is_one:
            self.log_rate_mult_probs = jnp.zeros( (self.k_rate_mults,) ) #(K,)
    
    
    def __call__(self,
                 sow_intermediates: bool,
                 log_class_probs: Optional[jnp.array] = None):
        # P(K|C), \rho_{c,k}
        log_rate_mult_probs = self.log_rate_mult_probs #(K,)
        rate_multipliers = self.rate_multipliers #(K,)
        
        # possibly normalize
        if self.norm_rate_mults:
            norm_factor = self._get_norm_factor(rate_multipliers = rate_multipliers,
                                                log_rate_mult_probs = log_rate_mult_probs ) #float
            rate_multipliers = rate_multipliers / norm_factor #(K,)
            
        # broadcast both back up to (C, K)
        log_rate_mult_probs = self._maybe_duplicate(log_rate_mult_probs) #(C, K) 
        rate_multipliers = self._maybe_duplicate(rate_multipliers) #(C, K) 
            
        return (log_rate_mult_probs, rate_multipliers)
    
    
    def _maybe_dedup(self, 
                     matrix: jnp.array):
        """
        if matrix is (C,K), reduce to (K,) by taking values from C=0
        """
        duplicated_shape = (self.num_mixtures, self.k_rate_mults)
        
        if matrix.shape == duplicated_shape:
            return matrix[0,:]
        
        elif matrix.shape == (duplicated_shape[-1],):
            return matrix


###############################################################################
### Substitution Models: Generate time reversible   ###########################
###############################################################################
class GTRLogprobs(ModuleBase):
    """
    Get the conditional and joint logprobs for a GTR model
    
    
    Initialize with
    ----------------
    config : dict
        config['random_init_exchanges'] : bool
            whether or not to initialize exchangeabilities from random; if 
            not random, initialize with LG08 values
            
        config['norm_rate_matrix'] : bool
            flag to normalize rate matrix to t = one substitution
            
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
    _get_square_exchangeabilities_matrix
    
    Methods inherited from models.model_utils.BaseClasses.ModuleBase
    ----------------------------------
    sow_histograms_scalars
        for tensorboard logging
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        Flax Module Parameters
        -----------------------
        exchangeabilities_logits_vec : ArrayLike, (n,)
            upper triangular values for exchangeability matrix
            190 for proteins, 6 for DNA
            Usually initialize logits from LG08 exchangeabilities
        
        """
        ###################
        ### read config   #
        ###################
        # required
        emission_alphabet_size = self.config['emission_alphabet_size']
        
        # optional
        self.exchange_min_val, self.exchange_max_val  = self.config.get( 'exchange_range', (1e-4, 12) )
        self.random_init_exchanges = self.config.get('random_init_exchanges', True)
        self.norm_rate_matrix = self.config.get('norm_rate_matrix', True)
        
        # only needed if loading random_init_exchanges is False
        exchangeabilities_file = self.config.get('filenames', {}).get('exch', None)
        
        
        ##########################################################
        ### Parameter: exchangeabilities as a flattened vector   # 
        ### of upper triangular values                           #
        ##########################################################
        # N = 190 for proteins
        # N =  6 for DNA
        
        ### activation is bound sigmoid; setup the activation function with 
        ###   min/max values
        self.exchange_activation = partial(bound_sigmoid,
                                           min_val = self.exchange_min_val,
                                           max_val = self.exchange_max_val)
        
        ### initialization
        # init from file
        if not self.random_init_exchanges:
            with open(exchangeabilities_file,'rb') as f:
                vec = jnp.load(f)
        
            transformed_vec = bound_sigmoid_inverse(vec, 
                                                    min_val = self.exchange_min_val,
                                                    max_val = self.exchange_max_val)
        
            self.exchangeabilities_logits_vec = self.param("exchangeabilities", 
                                                           lambda rng, shape: transformed_vec,
                                                           transformed_vec.shape ) #(N,)
        
        # init from random
        elif self.random_init_exchanges:
            A = emission_alphabet_size
            num_exchange = int( (A * (A-1)) / 2 )
            self.exchangeabilities_logits_vec = self.param("exchangeabilities", 
                                                           nn.initializers.normal(),
                                                           (num_exchange,),
                                                           jnp.float32 ) #(N,)
        
        
    def __call__(self,
                 logprob_equl: jnp.array,
                 rate_multipliers: jnp.array,
                 t_array: jnp.array, 
                 sow_intermediates: bool,
                 return_cond: bool,
                 return_intermeds: bool = False,
                 *args,
                 **kwargs):
        """
        C = number of latent site classes
        K = number of site rates
        A = alphabet size
        
        
        Arguments
        ----------
        logprob_equl : ArrayLike, (C, A)
            log-transformed equilibrium distribution
        
        rate_multipliers : ArrayLike, (C, K)
            rate multiplier k for site class c; \rho_{c,k}
        
        t_array : ArrayLike, (T,) or (B,)
            either one time grid for all samples (T,) or unique branch
            length for each sample (B,)
        
        sow_intermediates : bool
            switch for tensorboard logging
        
        return_cond : bool
            whether or not to return conditional logprob
        
        return_intermeds : bool
            whether or not to return intermediate values:
                > exchangeabilities: (C, A, A)
                > rate_matrix: (A, A)
          
        Returns
        -------
        ArrayLike, (T, C, K, A, A)
            either joint or conditional logprob of emissions at match sites
        """
        # undo log transform on equilibrium
        equl = jnp.exp(logprob_equl) #(C, A)
        
        # 1.) fill in square matrix, \chi
        exchangeabilities_mat = self._get_square_exchangeabilities_matrix(sow_intermediates) #(A, A)
        
        # 2.) prepare rate matrix Q_c = \chi * \diag(\pi_c); normalize if desired
        rate_matrix_Q = rate_matrix_from_exch_equl( exchangeabilities = exchangeabilities_mat,
                                                    equilibrium_distributions = equl,
                                                    norm=self.norm_rate_matrix ) #(C, A, A)
        
        # 3.) scale by rate multipliers, \rho_{c,k}
        rate_matrix_times_rho = scale_rate_matrix(subst_rate_mat = rate_matrix_Q,
                                                  rate_multipliers = rate_multipliers) #(C, K, A, A)
        
        # 4.) apply matrix exponential to get conditional logprob
        # cond_logprobs is either (T, C, K, A, A) or (B, C, K, A, A)
        cond_logprobs = cond_logprob_emit_at_match_per_mixture( t_array = t_array,
                                                                scaled_rate_mat_per_mixture = rate_matrix_times_rho )
        
        if return_cond:
            logprobs = cond_logprobs
        
        # 5.) multiply by equilibrium distributions to get joint logprob
        elif not return_cond:
            # joint_logprobs is either (T, C, K, A, A) or (B, C, K, A, A)
            # NOTE: this uses original logprob_equl (before exp() operation)
            logprobs = joint_logprob_emit_at_match_per_mixture( cond_logprob_emit_at_match_per_mixture = cond_logprobs,
                                                                log_equl_dist_per_mixture = logprob_equl )
        
        # optionally, return intermediates too; useful for final eval or debugging
        if return_intermeds:
            intermeds_dict = {'exchangeabilities': exchangeabilities_mat,
                              'rate_matrix': rate_matrix_Q}
        
        elif not return_intermeds:
            intermeds_dict = {}
        
        return logprobs, intermeds_dict
    
    
    def _get_square_exchangeabilities_matrix(self,
                                             sow_intermediates: bool):
        ### apply activation of choice
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
        
        
        ### fill in square matrix
        exchangeabilities_mat = upper_tri_vector_to_sym_matrix( upper_triag_values ) #(A, A)
        
        return exchangeabilities_mat
        

class GTRLogprobsFromFile(GTRLogprobs):
    """
    Like GTRLogprobs, but load parameters as-is
        
        
    Initialize with
    ----------------
    config : dict
        config['filenames']['exch'] : str
            name of the exchangeabilities to load
        
        config['norm_rate_matrix'] : bool
            flag to normalize rate matrix to t = one substitution
            
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    _get_square_exchangeabilities_matrix
    
    Methods inheried from GTRLogprobs
    ----------------------------------
    __call__
    
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
        self.norm_rate_matrix = self.config.get('norm_rate_matrix',True)
        exchangeabilities_file = self.config['filenames']['exch']
        
        
        ###################################
        ### Read exchangeabilities file   #
        ###################################
        with open(exchangeabilities_file,'rb') as f:
            exch_from_file = jnp.load(f)
        
        # if providing a vector, need to prepare a square exchangeabilities matrix
        if len(exch_from_file.shape) == 1:
            self.exchangeabilities_mat = upper_tri_vector_to_sym_matrix( exch_from_file ) #(A, A)
            
        # otherwise, use the matrix as-is
        else:
            self.exchangeabilities_mat = exch_from_file #(A, A)
        
        
    def _get_square_exchangeabilities_matrix(self,
                                             *args,
                                             **kwargs):
        return self.exchangeabilities_mat
    


###############################################################################
### RATE MATRICES: HKY85   ####################################################
###############################################################################
class HKY85Logprobs(GTRLogprobs):
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
        config['norm_rate_matrix'] : bool
            flag to normalize rate matrix to t = one substitution
        
        config['exchange_range'] : List[float, float]
            exchangeabilities undergo bound_sigmoid transformation, this
            specifies the min and max
            Default is (1e-4, 12)
        
    name : str
        class name, for flax
    
    Methods here
    ------------
    setup
    _get_square_exchangeabilities_matrix
    
    Methods inheried from GTRLogprobs
    ----------------------------------
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
        Flax Module Parameters
        -----------------------
        ti_tv_vec : ArrayLike, (2,)
            first value is transition rate, second value is transversion rate
            initialized from unit normal
        
        """
        # optional
        self.exchange_min_val, self.exchange_max_val  = self.config.get( 'exchange_range', (1e-4, 12) )
        self.norm_rate_matrix = self.config.get('norm_rate_matrix', True)
        
          
        ##########################################################
        ### Parameter: exchangeabilities as a flattened vector   # 
        ### of upper triangular values                           #
        ##########################################################
        # [ti, tv]; need to stack these values later
        self.exchangeabilities_logits_vec = self.param('exchangeabilities',
                                                       nn.initializers.normal(),
                                                       (2,),
                                                       jnp.float32) #(2)
        
        self.exchange_activation = partial(bound_sigmoid,
                                           min_val = self.exchange_min_val,
                                           max_val = self.exchange_max_val)
        
        
    def _get_square_exchangeabilities_matrix(self,
                                             sow_intermediates: bool):
        ### apply activation of choice
        if sow_intermediates:
            self.sow_histograms_scalars(mat= self.exchangeabilities_logits_vec[0], 
                                        label= 'logit for transitions BEFORE bound_sigmoid activation', 
                                        which='scalars')
            
            self.sow_histograms_scalars(mat= self.exchangeabilities_logits_vec[1], 
                                        label= 'logit for transversions BEFORE bound_sigmoid activation', 
                                        which='scalars')
            
        # transitions is first value, transversions is second value
        ti_tv_vec = self.exchange_activation( self.exchangeabilities_logits_vec ) #(2,)
    
        if sow_intermediates:
            self.sow_histograms_scalars(mat= ti_tv_vec[0], 
                                        label= 'transition rate AFTER bound_sigmoid activation', 
                                        which='scalars')
            
            self.sow_histograms_scalars(mat= ti_tv_vec[1], 
                                        label= 'transversions rate AFTER bound_sigmoid activation', 
                                        which='scalars')
        
        
        ### stack these values
        # order should be: tv, ti, tv, tv, ti, tv
        upper_triag_values = jnp.stack( [ ti_tv_vec[1], 
                                          ti_tv_vec[0], 
                                          ti_tv_vec[1], 
                                          ti_tv_vec[1], 
                                          ti_tv_vec[0], 
                                          ti_tv_vec[1] ] ) #(6)
        
        ### fill in square matrix
        exchangeabilities_mat = upper_tri_vector_to_sym_matrix( upper_triag_values ) #(4, 4)
        
        return exchangeabilities_mat
        

class HKY85LogprobsFromFile(GTRLogprobsFromFile):
    """
    Like HKY85Logprobs, but load parameters from file
        
        
    Initialize with
    ----------------
    config : dict
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
    _get_square_exchangeabilities_matrix

    """
    config: dict
    name: str
    
    def setup(self):
        self.norm_rate_matrix = self.config.get('norm_rate_matrix',True)
        exchangeabilities_file = self.config['filenames']['exch']
        
        
        ### EXCHANGEABILITIES: (A_from, A_to)
        # transitions is first, transversions is second
        with open(exchangeabilities_file,'rb') as f:
            ti_tv_vec_from_file = jnp.load(f) #(2,)
            
        # order should be: tv, ti, tv, tv, ti, tv
        upper_triag_values = jnp.stack( [ ti_tv_vec_from_file[1], 
                                     ti_tv_vec_from_file[0], 
                                     ti_tv_vec_from_file[1], 
                                     ti_tv_vec_from_file[1], 
                                     ti_tv_vec_from_file[0], 
                                     ti_tv_vec_from_file[1] ] ) #(6,)
        
        self.exchangeabilities_mat = upper_tri_vector_to_sym_matrix( upper_triag_values ) #(4, 4)
        
    
    
###############################################################################
### PROBABILITY MATRICES: F81   ###############################################
###############################################################################
class F81Logprobs(ModuleBase):
    """
    Get the conditional and joint logprobs for an F81 model; doesn't
        really need to be a flax module, but keep for consistency with
        GTR and HKY85 models
    
    
    Initialize with
    ----------------
    config : dict
        config['num_mixtures'] :  int
            number of emission site classes
        
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
        None
        
        """
        self.norm_rate_matrix = self.config.get('norm_rate_matrix', True)
        
        
    def __call__(self,
                 logprob_equl: jnp.array,
                 rate_multipliers: jnp.array,
                 t_array: jnp.array,
                 sow_intermediates: bool,
                 return_cond: bool,
                 *args,
                 **kwargs):
        """
        C = number of latent site classes
        K = number of site rates
        A = alphabet size
        
        
        Arguments
        ----------
        logprob_equl : ArrayLike, (C, A)
            log-transformed equilibrium distribution
        
        rate_multipliers : ArrayLike, (C, K)
            rate multiplier k for site class c; \rho_{c,k}
        
        t_array : ArrayLike, (T,) or (B,)
            either one time grid for all samples (T,) or unique branch
            length for each sample (B,)
        
        sow_intermediates : bool
            switch for tensorboard logging
        
        return_cond : bool
            whether or not to return conditional logprob
        
        
        Returns
        -------
        ArrayLike, (T, C, K, A, A)
            log-probability of emission at match sites, according to F81
        """
        # undo log transform on equilibrium
        equl = jnp.exp(logprob_equl) #(C, A)
        
        logprobs = fill_f81_logprob_matrix( equl = equl, 
                                        rate_multipliers = rate_multipliers, 
                                        norm_rate_matrix = self.norm_rate_matrix,
                                        t_array = t_array,
                                        return_cond = return_cond ) #(T,C,K,A,A)
        
        intermeds_dict = {}
        return logprobs, intermeds_dict
        
# alias
F81LogprobsFromFile = F81Logprobs

    
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
