#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:43:12 2025

@author: annabel

About:
=======

Take (potentially post-processed) concatenated outputs from both sequence 
  embedders and output logits


classes available:
==================

misc:
----------
Placeholder
EvoparamsFromFile
LamMuExtendFromFile
RExtendFromFile

Global (one parameter set for all positions, all samples):
-----------------------------------------------------------
GlobalExchMat
GlobalEqulVec
GlobalEqulVecFromCounts
GlobalEqulVecFromFile
GlobalTKFLamMuRates
GlobalTKF92ExtProb

Local (unique params for each position, each sample):
-----------------------------------------------------
(all follow the pattern: norm -> dense -> act -> optional avg pool
 across length of sequence)
LocalExchMat
LocalEqulVec
LocalTKFLamMuRates
LocalTKF92ExtProb

"""
# jumping jax and leaping flax
from flax import linen as nn
import jax
import jax.numpy as jnp

from models.model_utils.BaseClasses import ModuleBase


def bounded_sigmoid(x, min_val, max_val):
    return min_val + (max_val - min_val) / (1 + jnp.exp(-x))


###############################################################################
### MISC   ####################################################################
###############################################################################
class Placeholder(ModuleBase):
    """
    to ignore parameter set entirely
    """
    config: None
    name: str
    
    @nn.compact
    def __call__(self, 
                 *args,
                 **kwargs):
        return None
    
class EvoparamsFromFile(ModuleBase):
    """
    load parameter set from file, and apply it to all samples, 
      at all positions
    
    Give it dummy dimensions at B and L
    """
    config: dict
    name: str
    
    def setup(self):
        load_from_file = self.config['load_from_file']

        with open(load_from_file, 'rb') as f:
            self.mat = jnp.load(f)
        
        # give B and L dimensions
        self.mat = self.mat[None, None, ...]
    
    def __call__(self, 
             *args,
             **kwargs):
        
        return self.mat


class LamMuFromFile(EvoparamsFromFile):
    """
    same as above, but I have to specify the key name :(
    """
    config: dict
    name: str
    
    def setup(self):
        load_from_file = self.config['lam_mu_file']

        with open(load_from_file, 'rb') as f:
            self.mat = jnp.load(f)
        
        # output is (B=1, L=1, 2)
        self.mat = self.mat[None, None, ...]


class RExtendFromFile(EvoparamsFromFile):
    """
    same as above, but I have to specify the key name :(
    """
    config: dict
    name: str
    
    def setup(self):
        load_from_file = self.config['r_extend_file']

        with open(load_from_file, 'rb') as f:
            self.mat = jnp.load(f)
        
        # output is (B=1, L=1)
        self.mat = self.mat[None, ...]

    

###############################################################################
### GLOBAL   ##################################################################
###############################################################################
class GlobalExchMat(ModuleBase):
    """
    generate a symetric matrix of exchangeabilities; one 
      (1, 1, alph_size, alph_size) matrix for all samples, at all positions
    
    constrained such that the only model parameters are the elements of the 
      upper triangle (lower triangle is just a copy)
      
    valid range: (min_val, max_val); canonically (0,inf)
    """
    config: dict
    name: str
    
    def setup(self):
        emission_alphabet_size = self.config['emission_alphabet_size']
        manual_init = self.config['manual_init']
        self.min_val, self.max_val = self.config.get( 'exchange_range',
                                                      (1e-4, 10) )
        filename = self.config.get('load_from_file', None)
        
        
        ### decide initialization
        # manually init from file of initial guesses
        if manual_init:
            with open(filename, 'rb') as f:
                mat = jnp.load(f)
            
            init_func = lambda key, shape, dtype: mat
        
        # init from xavier uniform
        elif not manual_init:
            init_func = nn.initializers.glorot_uniform()
        
        
        ### init matrix of logits
        n = emission_alphabet_size 
        num_vars = int( (n * (n-1))/2 )
        evo_param = self.param('exchangeabilities',
                               init_func,
                               (self.num_vars,),
                               jnp.float32)
        
        # fill upper triangular part of matrix
        upp_triag = self.upper_tri_exchang_mat(evoparam_array = evo_param, 
                                               alph_size = emission_alphabet_size)
        
        # reflect across diagonal; add B and L dimensions
        self.logits = (upp_triag + upp_triag.T)[None, None, ...]
        
        
    def __call__(self,
                 *args,
                 **kwargs):
        return bounded_sigmoid(x = self.logits, 
                               min_val = self.min_val,
                               max_val = self.max_val)
    
    
    def upper_tri_exchang_mat(self, 
                              evoparam_array, 
                              alph_size = 20):
        upper_tri_exchang = jnp.zeros( (alph_size, alph_size) )
        idxes = jnp.triu_indices(alph_size, k=1)  
        upper_tri_exchang = upper_tri_exchang.at[idxes].set(evoparam_array)
        return upper_tri_exchang


class GlobalEqulVec(ModuleBase):
    """
    generate equilibrium distribution; one (1, 1, features) vector to use on
      all samples, at all positions
     
    valid range: (0, 1), where sum_i(x_i) = 1 (i.e. valid probability 
      distribution); do this with log_softmax
    """
    config: dict
    name: str
    
    def setup(self):
        emission_alphabet_size = self.config['emission_alphabet_size']
        manual_init = self.config['manual_init']
        filename = self.config.get('load_from_file', None)


        ### decide initialization
        # manually init from file of initial guesses
        if manual_init:
            with open(filename, 'rb') as f:
                mat = jnp.load(f)
            
            init_func = lambda key, shape, dtype: mat
        
        # init from xavier uniform
        elif not manual_init:
            init_func = nn.initializers.glorot_uniform()
        
        
        ### init vector of logits
        self.logits = self.param('Equilibrium distr.',
                                 init_func,
                                 (emission_alphabet_size,),
                                 jnp.float32)[None, None, ...]
        
    def __call__(self,
                 *args,
                 **kwargs):
        return nn.log_softmax( self.logits )


class GlobalEqulVecFromCounts(ModuleBase):
    """
    construct logprobs from the aa counts in the training set
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
        
        # expand to to (B=1, L=1, alph)
        self.logprob_equilibr = logprob_equilibr[None, None, :]
        
    def __call__(self,
                 *args,
                 **kwargs):
        return self.logprob_equilibr


class GlobalTKFLamMuRates(ModuleBase):
    """
    lambda (first param) range: (min_val, max_val); canonically (0, inf)
    offset (second param) range: (min_val, max_val); canonically (0,1)
    
    one set for all sequences; output is (1,1,2)
    """
    config: dict
    name: str
    
    def setup(self):
        ### unpack config
        manual_init = self.config['manual_init']
        filename = self.config.get('load_from_file', None)
        self.tkf_err = self.config.get('tkf_err', 1e-4)
        self.lam_min_val, self.lam_max_val = self.config.get( 'lambda_range', 
                                                               [self.tkf_err, 3] )
        self.offs_min_val, self.offs_max_val = self.config.get( 'offset_range', 
                                                                [self.tkf_err, 0.333] )


        ### decide initialization
        # manually init from file of initial guesses
        if manual_init:
            with open(filename, 'rb') as f:
                mat = jnp.load(f)[:2]
            init_func = lambda key, shape, dtype: mat
        
        # init from xavier uniform
        elif not manual_init:
            init_func = nn.initializers.glorot_uniform()
        
        
        ### init vector of logits
        self.logits = self.param('TKF lam_rate mu_rate',
                                 init_func,
                                 (2,),
                                 jnp.float32)[None, None, ...]
        
    def __call__(self,
                 *args,
                 **kwargs):
        lam_mu, use_approx = self.logits_to_indel_rates(self.logits)
        
        if sow_intermediates:
            self.sow_histograms_scalars(mat=lam_mu[...,0], 
                                        label=f'{self.name}/lambda_insertion_rate', 
                                        which='scalars')
            
            self.sow_histograms_scalars(mat=lam_mu[...,1], 
                                        label=f'{self.name}/mu_deletion_rate', 
                                        which='scalars')
        
        return (lam_mu, use_approx)
        
    def logits_to_indel_rates(self, 
                              indel_param_logits):
        """
        assumes dim2=0 is lambda, dim2=1 is mu
        
        NOTE: could use another condition for use_approx, if lambda and mu both 
          get too small... but this shouldn't happen if their both sufficiently 
          lower bounded
        """
        # lambda (1,1)
        lam = bounded_sigmoid(x = self.logits[:,:,0],
                              min_val = self.lam_min_val,
                              max_val = self.lam_max_val)
        
        # mu (1,1)
        offset = bounded_sigmoid(x = self.logits[:,:,1],
                                 min_val = self.offs_min_val,
                                 max_val = self.offs_max_val)
        mu = lam / ( 1 -  offset) 

        use_approx = (offset == self.tkf_err)
        
        out = jnp.concatenate( [ lam[...,None], mu[...,None] ], axis = -1 )
        
        return out, use_approx


class GlobalTKF92ExtProb(ModuleBase):
    """
    r (third param) range: (min_val, max_val); canonically (0,1)
    
    one for all positions: (1,1)
    """
    config: dict
    name: str
    
    def setup(self):
        ### unpack config
        manual_init = self.config['manual_init']
        self.tkf_err = self.config.get('tkf_err', 1e-4)
        self.r_extend_min_val, self.r_extend_max_val = self.config.get( 'r_range', 
                                                                [self.tkf_err, 0.8] )
        filename = self.config.get('load_from_file', None)


        ### decide initialization
        # manually init from file of initial guesses
        if manual_init:
            with open(filename, 'rb') as f:
                mat = jnp.load(f)[2]
            init_func = lambda key, shape, dtype: mat
        
        # init from xavier uniform
        elif not manual_init:
            init_func = nn.initializers.glorot_uniform()
        
        
        ### init vector of logits
        self.logits = self.param('TKF92 R Extend Prob',
                                 init_func,
                                 (1,1),
                                 jnp.float32)
        
    def __call__(self,
                 *args,
                 **kwargs):
        r_extend = bounded_sigmoid(x = self.logits,
                                   min_val = self.r_extend_min_val,
                                   max_val = self.r_extend_max_val)
        
        if sow_intermediates:
            self.sow_histograms_scalars(mat=r_extend, 
                                        label=f'{self.name}/r_extension_prob', 
                                        which='scalars')
        
        return r_extend



###############################################################################
### LOCAL   ###################################################################
###############################################################################
# feats -> norm -> dense -> activation -> optional pool
class LocalExchMat(GlobalExchMat):
    """
    inherit upper_tri_exchang_mat() from GlobalExchLogits
    
    generate a symetric matrix of exchangeabilties from hidden representations;
      (B, L, alph_size, alph_size) matrices
    
    constrained such that the only model parameters are the elements of the 
      upper triangle (lower triangle is just a copy)
      
    valid range: (min_val, max_val); canonically (0,inf)
    """
    config: dict
    name: str
    
    def setup(self):
        # !!! manually set these
        self.norm_type = 'layer'
        self.norm = nn.LayerNorm( reduction_axes=-1,
                                  feature_axes=-1 )
        self.avg_pool_window = 9
        self.use_bias = True
        
        # load from config
        self.emission_alphabet_size = self.config['emission_alphabet_size']
        self.min_val, self.max_val = self.config.get( 'exchange_range',
                                                      (1e-4, 10) )
        self.avg_pool = self.config.get('avg_pool', False)
        
        
        ### projection layer; initialize with xavier (default)
        n = self.emission_alphabet_size 
        self.num_vars = int( (n * (n-1))/2 )
        name = f'{self.name}/Project to exchangeabilities'
        self.project_to_evoparams = nn.Dense(features = self.num_vars,
                                             use_bias = self.use_bias,
                                             name = name)
        
        
        ### function to generate upper triangular matrix
        self.vmapped_upper_tri_exchang_mat = jax.vmap( self.upper_tri_exchang_mat,
                                                       in_axes = (0, None) )
    
    @nn.compact
    def __call__(self, 
             datamat,  #(B, L_align, H)
             padding_mask,
             sow_intermediates: bool, 
             training: bool):
        B = datamat.shape[0]
        L = datamat.shape[1]
        
        ### 1.) normalize before final projection
        datamat = self.norm(datamat, mask=padding_mask)
        
        if sow_intermediates and (self.norm_type is not None):
            label = (f'{self.name}/'+
                     f'after {self.norm_type}norm to final projection')
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
            
            
        ### 2.) projection
        # (B, L_align, H) -> (B, L_align, self.num_vars)
        logits = self.project_to_evoparams(datamat)
        
        
        ### 3.) apply activation
        # (B, L_align, self.num_vars)
        exchangeabilities = bounded_sigmoid(x = logits, 
                               min_val = self.min_val,
                               max_val = self.max_val)
        
        if sow_intermediates:
            label = (f'{self.name}/'+
                     f'final exchangeabilities')
            self.sow_histograms_scalars(mat = exchangeabilities, 
                                        label = label, 
                                        which=['scalars'])
            del label
        
        # sliding window average pool along length, to avoid overfitting
        # (B, L_align, self.num_vars)
        if self.avg_pool:
            window_shape = (1, self.avg_pool_window, 1)
            exchangeabilities = nn.avg_pool( exchangeabilities,
                                            window_shape=window_shape, 
                                            strides=(1, 1, 1),         
                                            padding="SAME", 
                                            count_include_pad = False
                                            )
            if sow_intermediates:
                label = (f'{self.name}/'+
                         f'final exchangeabilities after avg pool')
                self.sow_histograms_scalars(mat = exchangeabilities, 
                                            label = label, 
                                            which=['scalars'])
                del label
        
        
        ### 4.) create symmetric matrix
        # reshape to (B*L, self.num_vars) before using vmapped fn
        exchangeabilities = exchangeabilities.reshape( (B*L, self.num_vars) )
        
        upp_triag = self.vmapped_upper_tri_exchang_mat(exchangeabilities, 
                                           self.emission_alphabet_size)
        
        #  output is symmetric and non-negative
        #   (B, L_align, emission_alphabet_size, emission_alphabet_size)
        upp_triag = upp_triag.reshape( (B, 
                                        L, 
                                        self.emission_alphabet_size, 
                                        self.emission_alphabet_size ) )
        exchangeabilities = ( upp_triag + jnp.transpose( upp_triag, (0,1,3,2) ) )
        
        return exchangeabilities
        
    
class LocalEqulVec(GlobalEqulVec):
    """
    generate equilibrium distribution; a (B, L, features) tensor 
     
    valid range: (0, 1), where sum_i(x_i) = 1 (i.e. valid probability 
      distribution); do this with log_softmax
    """
    config: dict
    name: str
    
    def setup(self):
        # !!! manually set these
        self.norm_type = 'layer'
        self.norm = nn.LayerNorm( reduction_axes=-1,
                                  feature_axes=-1 )
        self.avg_pool_window = 9
        self.use_bias = True
        
        # load from config
        self.emission_alphabet_size = self.config['emission_alphabet_size']
        self.avg_pool = self.config['avg_pool']
        
        ### projection layer
        name = f'{self.name}/Project to equilibriums'
        self.project_to_evoparams = nn.Dense(features = self.emission_alphabet_size,
                                             use_bias = self.use_bias,
                                             name = name)
        del name
        
    def __call__(self, 
             datamat,  #(B, L_align, H)
             padding_mask,
             sow_intermediates: bool, 
             training: bool):
        
        ### 1.) normalize before final projection
        datamat = self.norm(datamat, mask=padding_mask)
        
        if sow_intermediates and (self.norm_type is not None):
            label = (f'{self.name}/'+
                     f'after {self.norm_type}norm to final projection')
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
            
            
        ### 2.) projection
        # (B, L_align, H) -> (B, L_align, self.num_vars)
        logits = self.project_to_evoparams(datamat)
        
        
        ### 3.) activation
        # (B, L_align, self.num_vars)
        equilibr_dist = nn.log_softmax(logits)
        
        if sow_intermediates:
            label = (f'{self.name}/'+
                     f'final equilibriums')
            self.sow_histograms_scalars(mat = equilibr_dist, 
                                        label = label, 
                                        which=['scalars'])
            del label
        
        # average pool across sequence length (if desired)
        if self.avg_pool:
            window_shape = (1, self.avg_pool_window, 1)
            equilibr_dist = nn.avg_pool( equilibr_dist,
                                         window_shape=window_shape, 
                                         strides=(1, 1, 1),         
                                         padding="SAME", 
                                         count_include_pad = False
                                         )
            
            if sow_intermediates:
                label = (f'{self.name}/'+
                         f'final equilibriums after avg pool')
                self.sow_histograms_scalars(mat = equilibr_dist, 
                                            label = label, 
                                            which=['scalars'])
                del label
            
        return equilibr_dist


class LocalTKFLamMuRates(GlobalTKFLamMuRates):
    """
    lambda (first param) range: (min_val, max_val); canonically (0, inf)
    offset (second param) range: (min_val, max_val); canonically (0,1)
    
    output is (B,L,2)
    """
    config: dict
    name: str
    
    def setup(self):
        # !!! manually set these
        self.norm_type = 'layer'
        self.norm = nn.LayerNorm( reduction_axes=-1,
                                  feature_axes=-1 )
        self.avg_pool_window = 9
        self.use_bias = True
        
        # read from config
        self.avg_pool = self.config['avg_pool']
        self.tkf_err = self.config.get('tkf_err', 1e-4)
        self.lam_min_val, self.lam_max_val = self.config.get( 'lambda_range', 
                                                               [self.tkf_err, 3] )
        self.offs_min_val, self.offs_max_val = self.config.get( 'offset_range', 
                                                                [self.tkf_err, 0.333] )
        
        
        ### projection layer
        name = f'{self.name}/Project to lam, mu'
        self.project_to_evoparams = nn.Dense(features = 2,
                                             use_bias = self.use_bias,
                                             name = name)
        del name
        
    def __call__(self, 
             datamat,  #(B, L_align, H)
             padding_mask,
             sow_intermediates: bool, 
             training: bool):
        
        ### 1.) normalize before final projection
        datamat = self.norm(datamat, mask=padding_mask)
        
        if sow_intermediates and (self.norm_type is not None):
            label = (f'{self.name}/'+
                     f'after {self.norm_type}norm to final projection')
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
            
            
        ### 2.) projection
        # (B, L_align, H) -> (B, L_align, 2)
        logits = self.project_to_evoparams(datamat)
        
        
        ### 3.) activation
        lam_mu, use_approx = self.logits_to_indel_rates(logits)
        
        if sow_intermediates:
            self.sow_histograms_scalars(mat=lam_mu[...,0], 
                                        label=f'{self.name}/lambda_insertion_rate', 
                                        which='scalars')
            
            self.sow_histograms_scalars(mat=lam_mu[...,1], 
                                        label=f'{self.name}/mu_deletion_rate', 
                                        which='scalars')
            
        # average pool across length, if desired
        if self.avg_pool:
            lam_mu = nn.avg_pool( lam_mu,
                                  window_shape=(1, self.avg_pool_window, 2), 
                                  strides=(1, 1, 1),         
                                  padding="SAME", 
                                  count_include_pad = False
                                  )
            
            if sow_intermediates:
                self.sow_histograms_scalars(mat=lam_mu[...,0], 
                                            label=f'{self.name}/lam after pool', 
                                            which='scalars')
                
                self.sow_histograms_scalars(mat=lam_mu[...,1], 
                                            label=f'{self.name}/mu after pool', 
                                            which='scalars')
                
        return lam_mu, use_approx


class LocalTKF92ExtProb(GlobalTKF92ExtProb):
    """
    r (third param) range: (min_val, max_val); canonically (0,1)
    
    output is (B,L)
    """
    config: dict
    name: str
    
    def setup(self):
        # !!! manually set these
        self.norm_type = 'layer'
        self.norm = nn.LayerNorm( reduction_axes=-1,
                                  feature_axes=-1 )
        self.avg_pool_window = 9
        self.use_bias = True
        
        # read from config
        self.avg_pool = self.config['avg_pool']
        self.tkf_err = self.config.get('tkf_err', 1e-4)
        self.r_extend_min_val, self.r_extend_max_val = self.config.get( 'r_range', 
                                                                [self.tkf_err, 0.8] )
        
        
        ### projection layer
        name = f'{self.name}/Project to TKF92 r'
        self.project_to_evoparams = nn.Dense(features = 1,
                                             use_bias = self.use_bias,
                                             name = name)
        del name
        
    def __call__(self, 
             datamat,  #(B, L_align, H)
             padding_mask,
             sow_intermediates: bool, 
             training: bool):
        
        ### 1.) normalize before final projection
        datamat = self.norm(datamat, mask=padding_mask)
        
        if sow_intermediates and (self.norm_type is not None):
            label = (f'{self.name}/'+
                     f'after {self.norm_type}norm to final projection')
            self.sow_histograms_scalars(mat = datamat, 
                                        label = label, 
                                        which=['scalars'])
            del label
            
            
        ### 2.) projection
        # (B, L_align, H) -> (B, L_align)
        logits = jnp.self.project_to_evoparams(datamat)[...,0]
        
        ### 3.) activation
        r_extend = bounded_sigmoid(x = logits,
                                   min_val = self.r_extend_min_val,
                                   max_val = self.r_extend_max_val)
        
        if sow_intermediates:
            self.sow_histograms_scalars(mat=r_extend, 
                                        label=f'{self.name}/TKF92 r', 
                                        which='scalars')
            
        # average pool across length, if desired
        if self.avg_pool:
            r_extend = nn.avg_pool( r_extend,
                                  window_shape=(1, self.avg_pool_window), 
                                  strides=(1, 1),
                                  padding="SAME", 
                                  count_include_pad = False
                                  )
            
            if sow_intermediates:
                label = f'{self.name}/TKF92 r after avg pool'
                self.sow_histograms_scalars(mat=r_extend, 
                                            label=label, 
                                            which='scalars')
                del label
            
        return r_extend
    
    
    