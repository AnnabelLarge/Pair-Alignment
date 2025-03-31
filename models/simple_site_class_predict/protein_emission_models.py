#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 02:03:13 2025

@author: annabel

specifically set up for protein models; replace with general_emission_models
  at some point...

IN THE FUTURE: if you want to let protein models use generalizable 
emission functions:
	'protein_emission_models.LG08RateMatFromFile' ->
        replace with 'general_emission_models.RateMatFromFile'
        
	'protein_emission_models.LG08RateMatFitRateMult' ->
        replace with 'general_emission_models.RateMatFitRateMult'
        
	'protein_emission_models.LG08RateMatFitBoth' ->
        replace with 'general_emission_models.RateMatFitBoth'



modules:
========
 'LG08RateMatFromFile',
 'LG08RateMatFitRateMult',
 'LG08RateMatFitBoth',
 'PerClassRateMat',
 
 'LogEqulVecFromFile',
 'LogEqulVecFromCounts',
 'LogEqulVecPerClass',

 'SiteClassLogprobs',
 'SiteClassLogprobsFromFile'

"""
# jumping jax and leaping flax
from flax import linen as nn
import jax
import jax.numpy as jnp

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
### For joint loss functions: probability of classes   ########################
###############################################################################
class SiteClassLogprobs(ModuleBase):
    config: dict
    name: str
    
    def setup(self):
        self.n_classes = self.config['num_emit_site_classes']
        
        self.class_logits = self.param('class_logits',
                                        nn.initializers.normal(),
                                        (self.n_classes,),
                                        jnp.float32)
    
    def __call__(self,
                 sow_intermediates):
        log_class_probs = nn.log_softmax(self.class_logits)
        
        if sow_intermediates:
            for i in range(log_class_probs.shape[0]):
                val_to_write = jnp.exp( log_class_probs[i] )
                lab = f'{self.name}/prob of class {i}'
                self.sow_histograms_scalars(mat= val_to_write, 
                                            label=lab, 
                                            which='scalars')
                del lab
        
        return log_class_probs


class SiteClassLogprobsFromFile(ModuleBase):
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
### SUBSTITUTION RATE MATRICES   ##############################################
###############################################################################
class LG08RateMatFromFile(ModuleBase):
    """
    return (rho * Q), to be directly used in matrix exponential

    exchanegabilities come from LG08 substitution model
    rate multipliers directly loaded from separate file
    """
    config: dict
    name: str
    
    def setup(self):
        # could still have multiple site classes
        self.num_emit_site_classes = self.config['num_emit_site_classes']
        rate_multiplier_file = self.config['filenames']['rate_mult']
        exchangeabilities_file = self.config['filenames']['exch']
        
        # LG08 exchangeabilities; (20, 20)
        with open(exchangeabilities_file,'rb') as f:
            self.lg08_exch = jnp.load(f)
        
        # RATE MULTIPLIERS: (c,)
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
        
        # chi; one shared all classes
        exchangeabilities = self.lg08_exch
        
        return self.prepare_rate_matrix(exchangeabilities = exchangeabilities,
                                   equilibrium_distributions = equl,
                                   sow_intermediates = sow_intermediates,
                                   rate_multiplier = self.rate_multiplier)
    
    
    def prepare_rate_matrix(self,
                            exchangeabilities,
                            equilibrium_distributions,
                            rate_multiplier,
                            sow_intermediates: bool):
        
        C = equilibrium_distributions.shape[0]
        alphabet_size = equilibrium_distributions.shape[1]

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
        
        # normalize by default
        diag = jnp.einsum("cii->ci", subst_rate_mat) 
        norm_factor = -jnp.sum(diag * equilibrium_distributions, axis=1)[:,None,None]
        subst_rate_mat = subst_rate_mat / norm_factor
            
        final = jnp.einsum( 'c,cij->cij', 
                            rate_multiplier, 
                            subst_rate_mat ) 
        return final


class LG08RateMatFitRateMult(LG08RateMatFromFile):
    """
    return (rho * Q), to be directly used in matrix exponential
    inherit prepare_rate_matrix from LG08RateMatFromFile

    exchanegabilities come from LG08 substitution model
    rate multipliers fit with gradient updates
    
    rate matrix is normalized to one substitution, THEN multiplied by a scalar 
      multiple; first hidden site class has rate of 1, then subsequent ones 
      are fit with gradient descent (rho = [1, rate2, rate3, ...])
    
    params: 
        - rate_mult_logits( C, )
    
    valid ranges:
        - rate_mult: (0, inf); bound values with rate_mult_range
        
    """
    config: dict
    name: str
    
    def setup(self):
        self.num_emit_site_classes = self.config['num_emit_site_classes']
        exchangeabilities_file = self.config['filenames']['exch']
        
        out  = self.config.get( 'rate_mult_range',
                               (0.01, 10) )
        self.rate_mult_min_val, self.rate_mult_max_val = out
        del out
        
        ### LG08 exchangeabilities; (20, 20)
        with open(exchangeabilities_file,'rb') as f:
            self.lg08_exch = jnp.load(f)
        

        ### RATE MULTIPLIERS: (c-1,)
        if self.num_emit_site_classes > 1:
            # first class automatically has rate multiplier of one
            self.rate_mult_logits = self.param('rate_multipliers',
                                               nn.initializers.normal(),
                                               (self.num_emit_site_classes-1,),
                                               jnp.float32)
        
    def __call__(self,
                 logprob_equl,
                 sow_intermediates: bool,
                 *args,
                 **kwargs):
        # (C, alph)
        equl = jnp.exp(logprob_equl)
        
        # rate multiplier
        if self.num_emit_site_classes > 1:
            subsequent_rate_multipliers = bounded_sigmoid(self.rate_mult_logits,
                                                          min_val = self.rate_mult_min_val,
                                                          max_val = self.rate_mult_max_val)
            
            if sow_intermediates:
                for i in range(subsequent_rate_multipliers.shape[0]):
                    val_to_write = subsequent_rate_multipliers[i]
                    lab = f'{self.name}/rate multiplier for class {i+1}'
                    self.sow_histograms_scalars(mat= val_to_write, 
                                                label=lab, 
                                                which='scalars')
                    del lab
            
            rate_multiplier = jnp.concatenate( [jnp.array([1]), 
                                                subsequent_rate_multipliers],
                                               axis=0 )

        else:
            rate_multiplier = jnp.array([1])
        
        # chi; one shared all classes
        exchangeabilities = self.lg08_exch
        
        return self.prepare_rate_matrix(exchangeabilities = exchangeabilities,
                                   equilibrium_distributions = equl,
                                   sow_intermediates = sow_intermediates,
                                   rate_multiplier = rate_multiplier)


class LG08RateMatFitBoth(LG08RateMatFitRateMult):
    """
    return (rho * Q), to be directly used in matrix exponential
    inherit prepare_rate_matrix from LG08RateMatFromFile

    exchanegabilities come from LG08 substitution model, but are updated with
      gradient updates
    rate multipliers fit with gradient updates
    
    rate matrix is normalized to one substitution, THEN multiplied by a scalar 
      multiple; first hidden site class has rate of 1, then subsequent ones 
      are fit with gradient descent (rho = [1, rate2, rate3, ...])
    
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
        emission_alphabet_size = 20
        self.num_emit_site_classes = self.config['num_emit_site_classes']
        exchangeabilities_file = self.config['filenames']['exch']
        
        out  = self.config.get( 'rate_mult_range',
                               (0.01, 10) )
        self.rate_mult_min_val, self.rate_mult_max_val = out
        del out

        out  = self.config.get( 'exchange_range',
                               (1e-4, 10) )
        self.exchange_min_val, self.exchange_max_val = out
        del out
        
        
        ### initialize with LG08 upper triangular matrix
        # (190,)
        with open(exchangeabilities_file,'rb') as f:
            vec = jnp.load(f)
        transformed_vec = bounded_sigmoid_inverse(vec, 
                                                  min_val = self.exchange_min_val,
                                                  max_val = self.exchange_max_val)
        
        exch_raw = self.param("exchangeabilities", 
                              lambda rng, shape: transformed_vec,
                              transformed_vec.shape )
        
        # fill upper triangular part of matrix
        out_size = (emission_alphabet_size, emission_alphabet_size)
        upper_tri_exchang = jnp.zeros( out_size )
        idxes = jnp.triu_indices(emission_alphabet_size, k=1)  
        upper_tri_exchang = upper_tri_exchang.at[idxes].set(exch_raw)
        
        # reflect across diagonal
        self.exchangeabilities_logits = (upper_tri_exchang + upper_tri_exchang.T)
        
            
        ### RATE MULTIPLIERS: (c-1,)
        if self.num_emit_site_classes > 1:
            # first class automatically has rate multiplier of one
            self.rate_mult_logits = self.param('rate_multipliers',
                                               nn.initializers.normal(),
                                               (self.num_emit_site_classes-1,),
                                               jnp.float32)
        
    def __call__(self,
                 logprob_equl,
                 sow_intermediates: bool,
                 *args,
                 **kwargs):
        # pi; one per class
        equl = jnp.exp(logprob_equl)
        
        # rate multiplier
        if self.num_emit_site_classes > 1:
            subsequent_rate_multipliers = bounded_sigmoid(self.rate_mult_logits,
                                                          min_val = self.rate_mult_min_val,
                                                          max_val = self.rate_mult_max_val)
            
            if sow_intermediates:
                for i in range(subsequent_rate_multipliers.shape[0]):
                    val_to_write = subsequent_rate_multipliers[i]
                    lab = f'{self.name}/rate multiplier for class {i}'
                    self.sow_histograms_scalars(mat= val_to_write, 
                                                label=lab, 
                                                which='scalars')
                    del lab
                
            rate_multiplier = jnp.concatenate( [jnp.array([1]), 
                                                subsequent_rate_multipliers],
                                               axis=0 )
                    
        else:
            rate_multiplier = jnp.array([1])
        
        # chi; one shared all classes
        exchangeabilities = bounded_sigmoid(x = self.exchangeabilities_logits, 
                                            min_val = self.exchange_min_val,
                                            max_val = self.exchange_max_val)
        
        if sow_intermediates:
            self.sow_histograms_scalars(mat = exchangeabilities, 
                                        label = 'exchangeabilities', 
                                        which='scalars')
        
        # output is (c, i, j)
        return self.prepare_rate_matrix(exchangeabilities = exchangeabilities,
                                   equilibrium_distributions = equl,
                                   sow_intermediates = sow_intermediates,
                                   rate_multiplier = rate_multiplier)
            
            
class PerClassRateMat(LG08RateMatFitBoth):
    """
    return (rho * Q), to be directly used in matrix exponential
    inherit prepare_rate_matrix from LG08RateMatFromFile
    inherit call from LG08RateMatFitBoth

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
                               nn.initializers.normal(),
                               (num_vars,),
                               jnp.float32)
        
        # fill upper triangular part of matrix
        out_size = (emission_alphabet_size, emission_alphabet_size)
        upper_tri_exchang = jnp.zeros( out_size )
        idxes = jnp.triu_indices(emission_alphabet_size, k=1)  
        upper_tri_exchang = upper_tri_exchang.at[idxes].set(exch_raw)
        
        # reflect across diagonal
        self.exchangeabilities_logits = (upper_tri_exchang + upper_tri_exchang.T)
        
        
        ### RATE MULTIPLIERS: (c-1,)
        if self.num_emit_site_classes > 1:
            # first class automatically has rate multiplier of one
            self.rate_mult_logits = self.param('rate_multipliers',
                                               nn.initializers.normal(),
                                               (self.num_emit_site_classes-1,),
                                               jnp.float32)
        
    
    
    

###############################################################################
### LOGPROB (emit at indel sites)   ###########################################
###############################################################################
class LogEqulVecPerClass(ModuleBase):
    """
    generate equilibrium distribution; (num_site_clases, features) matrix
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
    A (1, features) matrix from counts
    """
    config: dict
    name: str
    
    def setup(self):
        # (alph,)
        training_dset_emit_counts = self.config['training_dset_aa_counts']
        
        prob_equilibr = training_dset_emit_counts/training_dset_emit_counts.sum()
        logprob_equilibr = safe_log( prob_equilibr )
        
        # expand to to (C=1, alpha)
        self.logprob_equilibr = logprob_equilibr[None,...]
        
        
    def __call__(self,
                 *args,
                 **kwargs):
        # (C, alpha)
        return self.logprob_equilibr
        
    
        