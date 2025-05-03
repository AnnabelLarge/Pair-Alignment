#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 02:03:13 2025

@author: annabel


ABOUT:
======
Functions and Flax Modules needed for scoring emissions


modules:
=========
'EqulDistLogprobsFromCounts',
'EqulDistLogprobsFromFile',
'EqulDistLogprobsPerClass',
'GTRRateMat',
'GTRRateMatFromFile',
'HKY85RateMat',
'HKY85RateMatFromFile',
'SiteClassLogprobs',
'SiteClassLogprobsFromFile',


functions:
===========
'get_cond_logprob_emit_at_match_per_class',
'get_joint_logprob_emit_at_match_per_class',
'_rate_matrix_from_exch_equl',
'_scale_rate_matrix',
'_upper_tri_vector_to_sym_matrix',
"""
from flax import linen as nn
import jax
from jax.scipy.linalg import expm
from jax._src.typing import Array, ArrayLike
import jax.numpy as jnp

from functools import partial

from models.model_utils.BaseClasses import ModuleBase
from utils.pairhmm_helpers import (bound_sigmoid,
                                   bound_sigmoid_inverse,
                                   safe_log)


###############################################################################
### Probability of being in site classes   ####################################
###############################################################################
class SiteClassLogprobs(ModuleBase):
    """
    Probability of being in site class, P(c)
    
    
    Initialize with
    ----------------
    config : dict
        config['num_emit_site_classes'] : int
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
        self.n_classes = self.config['num_emit_site_classes']
        
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
        in_file = self.config['filenames']['class_probs']
        with open(in_file,'rb') as f:
            class_probs = jnp.load(f)
        self.log_class_probs = safe_log(class_probs) #(C,)
    
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
### RATE MATRICES   ###########################################################
###############################################################################

########################
### helper functions   #
########################
def _upper_tri_vector_to_sym_matrix( vec: ArrayLike ):
    """
    Given upper triangular values, fill in a symmetric matrix


    Arguments
    ----------
    vec : ArrayLike, (n,)
        upper triangular values
    
    Returns
    -------
    mat : ArrayLike, (A, A)
        final matrix; A = ( n * (n-1) ) / 2
    
    Example
    -------
    vec = [a, b, c, d, e, f]
    
    _upper_tri_vector_to_sym_matrix(vec) = [[0, a, b, c],
                                            [a, 0, d, e],
                                            [b, d, 0, f],
                                            [c, e, f, 0]]

    """
    ### automatically detect emission alphabet size
    # 6 = DNA (alphabet size = 4)
    # 190 = proteins (alphabet size = 20)
    # 2016 = codons (alphabet size = 64)
    if vec.shape[-1] == 6:
        emission_alphabet_size = 4
    
    elif vec.shape[-1] == 190:
        emission_alphabet_size = 20
    
    elif vec.shape[-1] == 2016:
        emission_alphabet_size = 64
    
    else:
        raise ValueError(f'input dimensions are: {vec.shape}')
    
    
    ### fill upper triangular part of matrix
    out_size = (emission_alphabet_size, emission_alphabet_size)
    upper_tri_exchang = jnp.zeros( out_size )
    idxes = jnp.triu_indices(emission_alphabet_size, k=1)  
    upper_tri_exchang = upper_tri_exchang.at[idxes].set(vec) # (A, A)
    
    
    ### reflect across diagonal
    mat = (upper_tri_exchang + upper_tri_exchang.T) # (A, A)
    
    return mat


def _rate_matrix_from_exch_equl(exchangeabilities: ArrayLike,
                                equilibrium_distributions: ArrayLike,
                                norm: bool=True):
    """
    computes rate matrix Q = \chi * \pi_c; normalizes to substution 
      rate of one if desired
    
    only one exchangeability; rho and pi are properties of the class
    
    C = number of latent site classes
    A = alphabet size
    
    
    Arguments
    ----------
    exchangeabilities : ArrayLike, (A, A)
        symmetric exchangeability parameter matrix
        
    equilibrium_distributions : ArrayLike, (C, A)
        amino acid equilibriums per site
    
    norm : bool, optional; default is True

    Returns
    -------
    subst_rate_mat : ArrayLike, (C, A, A)
        rate matrix Q, for every class

    """
    C = equilibrium_distributions.shape[0]
    A = equilibrium_distributions.shape[1]

    # just in case, zero out the diagonals of exchangeabilities
    exchangeabilities_without_diags = exchangeabilities * ~jnp.eye(A, dtype=bool)

    # Q = chi * diag(pi); q_ij = chi_ij * pi_j
    rate_mat_without_diags = jnp.einsum('ij, cj -> cij', 
                                        exchangeabilities_without_diags, 
                                        equilibrium_distributions)   # (C, A, A)
    
    # put the row sums in the diagonals
    row_sums = rate_mat_without_diags.sum(axis=2)  # (C, A)
    ones_diag = jnp.eye( A, dtype=bool )[None,:,:]   # (1, A, A)
    ones_diag = jnp.broadcast_to( ones_diag, (C,
                                              ones_diag.shape[1],
                                              ones_diag.shape[2]) )
    diags_to_add = -jnp.einsum('ci,cij->cij', row_sums, ones_diag)  #(C, A, A)
    subst_rate_mat = rate_mat_without_diags + diags_to_add  #(C, A, A)
    
    # normalize (true by default)
    if norm:
        diag = jnp.einsum("cii->ci", subst_rate_mat)  # (C, A)
        norm_factor = -jnp.sum(diag * equilibrium_distributions, axis=1)[:,None,None]  #(C, 1, 1)
        subst_rate_mat = subst_rate_mat / norm_factor  # (C, A, A)
    
    return subst_rate_mat


def _scale_rate_matrix(subst_rate_mat: ArrayLike,
                       rate_multiplier: ArrayLike):
    """
    Scale Q by rate multipliers, rho
    
    C = number of latent site classes
    A = alphabet size
    
    
    Arguments
    ----------
    subst_rate_mat : ArrayLike, (C, A, A)
    
    rate_multiplier : ArrayLike, (C,)

    Returns
    -------
    scaled rate matrix : ArrayLike, (C, A, A)

    """
    return jnp.einsum( 'c,cij->cij', 
                       rate_multiplier, 
                       subst_rate_mat )
 

###############################
### General time reversible   #
###############################
class GTRRateMat(ModuleBase):
    """
    return (rho * Q), to be directly used in matrix exponential

    
    Initialize with
    ----------------
    config : dict
        config['num_emit_site_classes'] :  int
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
        
        config['filenames']['exch'] : str
            name of the exchangeabilities to intiialize with
        
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
            
            self.rate_multiplier_activation = partial(bound_sigmoid,
                                                      min_val = self.rate_mult_min_val,
                                                      max_val = self.rate_mult_max_val)
        
        elif self.rate_mult_activation == 'softplus':
            self.rate_multiplier_activation = jax.nn.softplus
        
        
        # initializers
        if self.num_emit_site_classes > 1:
            self.rate_mult_logits = self.param('rate_multipliers',
                                               nn.initializers.normal(),
                                               (self.num_emit_site_classes,),
                                               jnp.float32) #(C,)
    
            
        #####################################
        ### EXCHANGEABILITIES AS A VECTOR   #
        #####################################
        # get initial values from file
        with open(exchangeabilities_file,'rb') as f:
            vec = jnp.load(f)
            
        out  = self.config.get( 'exchange_range',
                               (1e-4, 12) )
        self.exchange_min_val, self.exchange_max_val = out
        del out
        
        transformed_vec = bound_sigmoid_inverse(vec, 
                                                  min_val = self.exchange_min_val,
                                                  max_val = self.exchange_max_val)
        
        self.exchange_activation = partial(bound_sigmoid,
                                           min_val = self.exchange_min_val,
                                           max_val = self.exchange_max_val)
        
        self.exchangeabilities_logits_vec = self.param("exchangeabilities", 
                                                       lambda rng, shape: transformed_vec,
                                                       transformed_vec.shape ) #(n,)
        
    def __call__(self,
                 logprob_equl,
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
        if self.num_emit_site_classes > 1:
            
            ### apply activation of choice
            if sow_intermediates:
                for i in range(self.rate_mult_logits.shape[0]):
                    val_to_write = self.rate_mult_logits[i]
                    act = self.rate_mult_activation
                    lab = (f'{self.name}/logit BEFORE {act} activation- '+
                           'rate multiplier for class {i}')
                    self.sow_histograms_scalars(mat= val_to_write, 
                                                label=lab, 
                                                which='scalars')
                    del lab
                    
            rate_multiplier = self.rate_multiplier_activation( self.rate_mult_logits ) #(C,)
            
            if sow_intermediates:
                for i in range(rate_multiplier.shape[0]):
                    val_to_write = rate_multiplier[i]
                    act = self.rate_mult_activation
                    lab = (f'{self.name}/logit AFTER {act} activation- '+
                           'rate multiplier for class {i}')
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
        
        upper_triag_values = self.exchange_activation( self.exchangeabilities_logits_vec ) #(A, A)
    
        if sow_intermediates:
            self.sow_histograms_scalars(mat = upper_triag_values, 
                                        label = 'value AFTER bound_sigmoid activation- exchangeabilities', 
                                        which='scalars')
        
        # create square matrix
        exchangeabilities_mat = _upper_tri_vector_to_sym_matrix( upper_triag_values ) #(A, A)
        
        # scale rate matrix
        rate_mat_times_rho = self._prepare_rate_matrix(exchangeabilities = exchangeabilities_mat,
                                                       equilibrium_distributions = equl,
                                                       rate_multiplier = rate_multiplier) #(C, A, A)
        return rate_mat_times_rho
    
    def _prepare_rate_matrix(self,
                             exchangeabilities,
                             equilibrium_distributions,
                             rate_multiplier):
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
        subst_rate_mat = _rate_matrix_from_exch_equl( exchangeabilities = exchangeabilities,
                                                      equilibrium_distributions = equilibrium_distributions,
                                                      norm=True ) #(C, A, A)
        
        # scale it
        rate_mat_times_rho = _scale_rate_matrix(subst_rate_mat = subst_rate_mat,
                                                rate_multiplier = rate_multiplier) #(C, A, A)
        
        return rate_mat_times_rho 


class GTRRateMatFromFile(GTRRateMat):
    """
    Like GTRRateMat, but load rate multipliers and exchangeabilities from 
        files as-is
        
        
    Initialize with
    ----------------
    config : dict
        config['num_emit_site_classes'] :  int
            number of emission site classes
            
        config['filenames']['rate_mult'] :  str
            name of the rate multipliers to load
        
        config['filenames']['exch'] : str
            name of the exchangeabilities to load
            
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
        self.num_emit_site_classes = self.config['num_emit_site_classes']
        rate_multiplier_file = self.config['filenames']['rate_mult']
        exchangeabilities_file = self.config['filenames']['exch']
        
        
        ########################
        ### RATE MULTIPLIERS   #
        ########################
        if self.num_emit_site_classes > 1:
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
            self.exchangeabilities = _upper_tri_vector_to_sym_matrix( exch_from_file ) #(A, A)
            
        # otherwise, use the matrix as-is
        else:
            self.exchangeabilities = exch_from_file #(A, A)
        
    def __call__(self,
                 logprob_equl,
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
        
        rate_mat_times_rho =  self._prepare_rate_matrix(exchangeabilities = self.exchangeabilities,
                                                        equilibrium_distributions = equl,
                                                        rate_multiplier = self.rate_multiplier) #(C, A, A)
        return rate_mat_times_rho
    
    
#############
### HKY85   #
#############
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
        config['num_emit_site_classes'] :  int
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
            
            self.rate_multiplier_activation = partial(bound_sigmoid,
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
        
        self.exchange_activation = partial(bound_sigmoid,
                                           min_val = self.exchange_min_val,
                                           max_val = self.exchange_max_val)
        

class HKY85RateMatFromFile(GTRRateMatFromFile):
    """
    Like v, but load parameters from file
        
        
    Initialize with
    ----------------
    config : dict
        config['num_emit_site_classes'] :  int
            number of emission site classes
            
        config['filenames']['rate_mult'] :  str
            name of the rate multipliers to load
        
        config['filenames']['exch'] : str
            name of the exchangeabilities to load
            
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
        
        self.exchangeabilities = _upper_tri_vector_to_sym_matrix( hky85_raw_vec )
        
        ### RATE MULTIPLIERS: (c,)
        if self.num_emit_site_classes > 1:
            with open(rate_multiplier_file, 'rb') as f:
                self.rate_multiplier = jnp.load(f)
        else:
            self.rate_multiplier = jnp.array([1])
        

###############################################################################
### GET SUBSTITUTION LOGPROBS   ###############################################
###############################################################################
def get_cond_logprob_emit_at_match_per_class( t_array: ArrayLike,
                                              scaled_rate_mat_per_class: ArrayLike):
    """
    P(y|x,c,t) = expm( rho_c * Q_c * t )

    C = number of latent site classes
    A = alphabet size
    T = number of branch lengths
    

    Arguments
    ----------
    t_array : ArrayLike, (T,)
        branch lengths
        
    scaled_rate_mat_per_class : ArrayLike, (C, A, A)
        rho_c * Q_c

    Returns
    -------
    to_expm : ArrayLike, (T, C, A, A)
        scaled rate matrix * t, for all classes, this is the input for the 
        matrix exponential function
        
    cond_logprob_emit_at_match_per_class :  ArrayLike, (T, C, A, A)
        final log-probability

    """
    to_expm = jnp.multiply( scaled_rate_mat_per_class[None,...],
                            t_array[:, None,None,None,] ) #(T, C, A, A)
    cond_prob_emit_at_match_per_class = expm(to_expm) #(T, C, A, A)
    cond_logprob_emit_at_match_per_class = safe_log( cond_prob_emit_at_match_per_class )  #(T, C, A, A)
    return cond_logprob_emit_at_match_per_class, to_expm


def get_joint_logprob_emit_at_match_per_class( cond_logprob_emit_at_match_per_class: ArrayLike,
                                              log_equl_dist_per_class: ArrayLike ):
    """
    P(x,y|c,t) = pi_c * expm( rho_c * Q_c * t )

    C = number of latent site classes
    A = alphabet size
    T = number of branch lengths
    

    Arguments
    ----------
    cond_logprob_emit_at_match_per_class : ArrayLike, (T, C, A, A)
        P(y|x,c,t), calculated before
    
    log_equl_dist_per_class : ArrayLike, (C, A, A)
        rho_c * Q_c

    Returns
    -------
    ArrayLike, (T, C, A, A)

    """
    return ( cond_logprob_emit_at_match_per_class +
             log_equl_dist_per_class[None,:,:,None] ) #(T, C, A, A)


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
            
        config['num_emit_site_classes'] : int
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
        num_emit_site_classes = self.config['num_emit_site_classes']
        
        self.logits = self.param('Equilibrium distr.',
                                  nn.initializers.normal(),
                                  (num_emit_site_classes, emission_alphabet_size),
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
            for c in range(out.shape[0]):
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
