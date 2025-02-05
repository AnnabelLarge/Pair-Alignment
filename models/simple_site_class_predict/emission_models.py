#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 02:03:13 2025

@author: annabel

modules:
========
'EqulVecFromCounts',
 'EqulVecPerClass',
 'LG08RateMat',
 'PerClassRateMat'

"""
# jumping jax and leaping flax
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

from models.model_utils.BaseClasses import ModuleBase

def bounded_sigmoid(x, min_val, max_val):
    return min_val + (max_val - min_val) / (1 + jnp.exp(-x))

def safe_log(x):
    return jnp.log( jnp.where( x>0, 
                               x, 
                               jnp.finfo('float32').smallest_normal ) )



###############################################################################
### SUBSTITUTION RATE MATRICES   ##############################################
###############################################################################
class LG08RateMat(ModuleBase):
    """
    return (rho * Q), to be directly used in matrix exponential

    exchanegabilities come from LG08 substitution model
    """
    config: dict
    name: str
    
    def setup(self):
        # could still have multiple site classes
        self.num_emit_site_classes = self.config['num_emit_site_classes']
        
        # LG08 exchangeabilities (for unit testing); (20, 20)
        with open(f'LG08_exchangeability_r.npy','rb') as f:
            self.lg08_exch = jnp.load(f)
        
        # RATE MULTIPLIERS: (c,)
        if self.num_emit_site_classes != 1:
            self.rate_mult_logits = self.param('rate_multipliers',
                                               nn.initializers.glorot_uniform(),
                                               (num_emit_site_classes,),
                                               jnp.float32)

    def __call__(self,
                 logprob_equl,
                 sow_intermediates: bool,
                 *args,
                 **kwargs):
        # pi; one per class
        equl = jnp.exp(logprob_equl)
        
        # chi; one shared all classes
        exchangeabilities = self.lg08_exch
        
        return prepare_rate_matrix(exchangeabilities = exchangeabilities,
                                   equilibrium_distributions = equl,
                                   sow_intermediates = sow_intermediates)
    
    
    def prepare_rate_matrix(exchangeabilities,
                            equilibrium_distributions,
                            sow_intermediates: bool):
        # Q = chi * pi
        rate_mat_without_diags = jnp.einsum('ij, cj -> cij', 
                                            exchangeabilities, 
                                            equilibrium_distributions)
    
        row_sums = rate_mat_without_diags.sum(axis=2) 
        ones_diag = jnp.eye( alphabet_size, dtype=bool )[None,:,:]
        ones_diag = jnp.broadcast_to( ones_diag, (C,
                                                  ones_diag.shape[1],
                                                  ones_diag.shape[2]) )
        diags_to_add = -jnp.einsum('ci,cij->cij', row_sums, ones_diag)
        subst_rate_mat = rate_mat_without_diags + diags_to_add
        
        # for one site class
        if self.num_emit_site_classes == 1:
            diag = jnp.einsum("cii->ci", subst_rate_mat) 
            norm_factor = -jnp.sum(diag * equl, axis=1)[:,None,None]
            subst_rate_mat = subst_rate_mat / norm_factor
            rate_multipilers = jnp.ones( (1,) )
            
        # for many site classes
        elif self.num_emit_site_classes > 1:
            rate_multipilers = bounded_sigmoid(x = self.rate_mult_logits, 
                                                min_val = self.rate_mult_min_val,
                                                max_val = self.rate_mult_max_val)
        
            if (sow_intermediates):
                self.sow_histograms_scalars(mat = rate_multipliers, 
                                            label = 'rate_multipliers', 
                                            which='scalars')
        
        return jnp.multiply( 'c,cij->cij', 
                             rate_multipliers, 
                             subst_rate_mat ) 
        

class PerClassRateMat(LG08RateMat):
    """
    return (rho * Q), to be directly used in matrix exponential

    inherit rate matrix calculation from LG08RateMat

    params: 
        - exchangeabilities_logits ( alph, alph )
        - rate_mult_logits( C, )
    
    valid ranges:
        - exchangeabilities: (0, inf); bound values with exchange_range
        - rate_mult: (0, inf); bound values with rate_mult_range
        
    """
    config: dict
    name: str
    
    def setup(self):
        emission_alphabet_size = self.config['emission_alphabet_size']
        self.num_emit_site_classes = self.config['num_emit_site_classes']
        
        out  = self.config.get( 'exchange_range',
                               (1e-4, 10) )
        self.exchange_min_val, self.exchange_max_val = out
        del out
        
        out  = self.config.get( 'rate_mult_range',
                               (0.01, 10) )
        self.rate_mult_min_val, self.rate_mult_max_val = out
        del out
        
        
        ### EXCHANGEABILITIES: (i,j)
        # init logits
        num_vars = int( (emission_alphabet_size * (emission_alphabet_size-1))/2 )
        exch_raw = self.param('exchangeabilities',
                               nn.initializers.glorot_uniform(),
                               (num_vars,),
                               jnp.float32)
        
        # fill upper triangular part of matrix
        out_size = (emission_alphabet_size, emission_alphabet_size)
        upper_tri_exchang = jnp.zeros( out_size )
        idxes = jnp.triu_indices(emission_alphabet_size, k=1)  
        upper_tri_exchang = upper_tri_exchang.at[idxes].set(exch_raw)
        
        # reflect across diagonal
        self.exchangeabilities_logits = (upper_tri_exchang + upper_tri_exchang.T)
        
        
        ### RATE MULTIPLIERS: (c,)
        if self.num_emit_site_classes != 1:
            self.rate_mult_logits = self.param('rate_multipliers',
                                               nn.initializers.glorot_uniform(),
                                               (num_emit_site_classes,),
                                               jnp.float32)
        
    def __call__(self,
                 logprob_equl,
                 sow_intermediates: bool,
                 *args,
                 **kwargs):
        # pi; one per class
        equl = jnp.exp(logprob_equl)
        
        # chi; one shared all classes
        exchangeabilities = bounded_sigmoid(x = self.exchangeabilities_logits, 
                                            min_val = self.exchange_min_val,
                                            max_val = self.exchange_max_val)
        
        if (sow_intermediates):
            self.sow_histograms_scalars(mat = exchangeabilities, 
                                        label = 'exchangeabilities', 
                                        which='scalars')
        
        # output is (c, i, j)
        return prepare_rate_matrix(exchangeabilities = exchangeabilities,
                                   equilibrium_distributions = equl,
                                   sow_intermediates = sow_intermediates)
    
    

###############################################################################
### LOGPROB (emit at indel sites)   ###########################################
###############################################################################
class EqulVecPerClass(ModuleBase):
    """
    generate equilibrium distribution; (num_site_clases, features) matrix
    """
    config: dict
    name: str
    
    def setup(self):
        emission_alphabet_size = self.config['emission_alphabet_size']
        num_emit_site_classes = self.config['num_emit_site_classes']
        self.logits = self.param('Equilibrium distr.',
                                 init_func,
                                 (num_emit_site_classes, emission_alphabet_size),
                                 jnp.float32)
        
    def __call__(self,
                 *args,
                 **kwargs):
        return nn.log_softmax( self.logits, axis = 1 )



class EqulVecFromCounts(ModuleBase):
    """
    A (1, faetures) matrix from counts
    """
    config: dict
    name: str
    
    def setup(self):
        # (alph,)
        training_dset_aa_counts = self.config['training_dset_aa_counts']
        
        prob_equilibr = training_dset_aa_counts/training_dset_aa_counts.sum()
        logprob_equilibr = jnp.log( jnp.where( prob_equilibr != 0,
                                              prob_equilibr,
                                              jnp.finfo('float32').smallest_normal
                                              )
                                   )
        
        # expand to to (C=1, alpha)
        self.logprob_equilibr = logprob_equilibr[None,...]
        
        
    def __call__(self,
                 *args,
                 **kwargs):
        # (C, alpha)
        return self.logprob_equilibr
        
    
        