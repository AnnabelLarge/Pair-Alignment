#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:42:56 2024

@author: annabel
"""
# jumping jax and leaping flax
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm 

from models.model_utils.BaseClasses import ModuleBase
    


####################
### misc helpers   #
####################
# We replace zeroes and infinities with small numbers sometimes
SMALLEST_FLOAT32 = jnp.finfo('float32').smallest_normal

def bounded_sigmoid(x, min_val, max_val):
    return min_val + (max_val - min_val) / (1 + jnp.exp(-x))


class CondMatchEmissionsLogprobs(ModuleBase):
    config: dict
    name: str
    
    @nn.compact
    def __call__(self,
                 final_shape,
                 exchangeability_matrices,
                 log_equilibr_distrib,
                 t_array,
                 sow_intermediates: bool=False):
        """
        (no parameters to train, but ModuleBase allows writing to tensorboard)
        
        purpose:
        --------
        evolutionary parameters (from neural network) -> 
            logprob(emissions at match sites)
        
        input sizes:
        -------------
        exchangeability_matrices: (B, L, alph, alph) OR (1,1,alph,alph)
        log_equilibr_distrib: (B, L, alph) OR (1,1,alph)
        t_array: (T, B)
        
        output sizes:
        -------------
        logprob_subst: (T, B, L, alph, alph)
          > this code automatically broadcasts up to full (T, B, L, alph, alph)
        
        """
        # read config
        unit_norm_rate_matrix = self.config['unit_norm_rate_matrix']
        
        # unpack shape
        T, B, L, alph, _ = final_shape
        
        
        ### broadcast everythin up to full B, L (if needed)
        # equilibr_probs: (1, 1, alph) -> (B, L, alph)
        equilibr_distrib = jnp.exp(log_equilibr_distrib)
        if equilibr_distrib.shape[0] == 1:
            new_shape = (B, L, alph)
            equilibr_distrib = jnp.broadcast_to(equilibr_distrib, new_shape)
            del new_shape
        
        # exchangeability_matrices: (1, 1, alph, alph) -> (B, L, alph, alph)
        # diagonals are already zero
        if exchangeability_matrices.shape[0] == 1:
            new_shape = (B, L, alph, alph)
            exchangeability_matrices = jnp.broadcast_to(exchangeability_matrices, 
                                                        new_shape)
            del new_shape
        
        ### generate rate matrix
        # fill in values for i != j
        # exchangeability_matrices * diag(equilibr_distrib) for all b, l
        rate_mat_without_diags = jnp.einsum('blij, blj -> blij', 
                                            exchangeability_matrices, 
                                            equilibr_distrib)
        
        # find rowsums i.e. sum across columns j
        row_sums = rate_mat_without_diags.sum(axis=-1) #(B, L, base_alphabet_size)
        ones_diag = jnp.eye( alph, dtype=bool )
        diags_to_add = -jnp.einsum('bli,ij->blij', row_sums, ones_diag)
        
        # add both
        subst_rate_mat = rate_mat_without_diags + diags_to_add
        
        if sow_intermediates:
            self.sow_histograms_scalars(mat=subst_rate_mat, 
                                        label=f'{self.name}/subst_rate_matrices', 
                                        which='scalars')
        
        
        ### optionally, normalize rate matrix
        if unit_norm_rate_matrix:
            diag_vecs = jnp.diagonal(subst_rate_mat, 
                                     axis1=2, 
                                     axis2=3) #(B, L, base_alphabet_size)
            norm_factor = -jnp.einsum('bli, bli -> bl', 
                                      diag_vecs, 
                                      equilibr_distrib)
            
            # divide each subst_rate_mat with this vec
            subst_rate_mat = subst_rate_mat/(norm_factor[:,:,None,None])
            del norm_factor
        
            if sow_intermediates:
                self.sow_histograms_scalars(mat=subst_rate_mat, 
                                            label=f'{self.name}/UNIT-NORMED_subst_rate_matrices', 
                                            which='scalars')
                
        
        ### final logprobs per time obtained from log( mat_exp(Rt) )
        Qt = jnp.einsum('blij, tb -> tblij', 
                        subst_rate_mat,
                        t_array)
        
        # ####################################################################
        # ####### __vvv__ in debug mode (not jit), check these __vvv__ #######
        # ####################################################################
        # if Qt.sum() > 0:
        #     to_fill = jnp.zeros( (T, B, L, 20, 20) )
        #     for t in range(T):
        #         for b in range(B):
        #             for l in range(L):
        #                 q = subst_rate_mat[b,l,:,:]
                        
        #                 # do rows sum to zero
        #                 assert jnp.allclose( q.sum(axis=1), 
        #                                       jnp.zeros( (q.shape[0],)), 
        #                                       atol=1e-5 )
                        
        #                 # is matrix properly normalized
        #                 pi = equilibr_distrib[b,l,:]
        #                 checksum = 0
        #                 for i in range(20):
        #                     checksum += -( pi[i] * q[i,i] )
        #                 assert jnp.allclose(checksum, 1)
                        
        #                 # is einsum recipe for Q*t correct
        #                 t_val = t_array[t, b]
        #                 to_fill = to_fill.at[t, b, l, :, :].set( q * t_val )
        #     assert jnp.allclose(to_fill, Qt)
            
        #     print('Qt is calculated correctly!')
        # ####################################################################
        # ####### __^^^__ in debug mode (not jit), check these __^^__ ########
        # ####################################################################
        
        # use the MATRIX EXPONENTIAL
        prob_subst = expm(Qt)    
        logprob_subst = jnp.log(prob_subst)
        
        if sow_intermediates:
            self.sow_histograms_scalars(mat=logprob_subst, 
                                        label=f'{self.name}/logprob_subst', 
                                        which='scalars')
        
        ### final matrix sizes
        # logprob_subst: (T, B, L, base_alphabet_size, base_alphabet_size)
        # subst_rate_mat: (B, L, base_alphabet_size, base_alphabet_size)
        return (logprob_subst, subst_rate_mat)


class JointMatchEmissionsLogprobs(ModuleBase):
    """
    joint = cond * marginal(anc)
    """
    config: dict
    name: str
    
    def setup():
        self.conditional_model = CondMatchEmissionsLogprobs(config=self.config,
                                                            name=f'cond_logprob')
        
    @nn.compact
    def __call__(self,
                 final_shape,
                 exchangeability_matrices,
                 log_equilibr_distrib,
                 t_array,
                 sow_intermediates: bool=False):
        
        out = CondMatchEmissionsLogprobs(final_shape = final_shape,
                                         exchangeability_matrices = exchangeability_matrices,
                                         log_equilibr_distrib = log_equilibr_distrib,
                                         t_array = t_array,
                                         sow_intermediates = sow_intermediates )
        
        cond_logprob_subst, subst_rate_mat = out
        joint_logprob_subst = ( cond_logprob_subst + 
                                log_equilibr_distrib[...,None] )
        
        ### final matrix sizes
        # logprob_subst: (T, B, L, base_alphabet_size, base_alphabet_size)
        # subst_rate_mat: (B, L, base_alphabet_size, base_alphabet_size)
        return (logprob_subst, subst_rate_mat)


class MatchEmissionsLogprobsFromFile(ModuleBase):
    config: dict
    name: str
    
    def setup(self):
        load_from_file = self.config['load_from_file']
        
        with open(load_from_file, 'rb') as f:
            self.logprob_subst = jnp.load(f)
        
        if len(self.logprob_subst.shape) == 2:
            self.logprob_subst[None, None, None, :, :]
            self.expand_dims = True
        else:
            self.expand_dims = False
    
    def __call__(self,
                 t_array,
                 sow_intermediates: bool=False,
                 **kwargs):
        if self.expand_dims:
            new_shape = ( t_array.shape[0],
                          self.logprob_subst.shape[1],
                          self.logprob_subst.shape[2],
                          self.logprob_subst.shape[3],
                          self.logprob_subst.shape[4] )
    
            logprob_subst = jnp.broadcast_to( self.logprob_subst, new_shape )
        
        else:
            logprob_subst = self.logprob_subst

        placeholder_mat = jnp.zeros( (logprob_subst.shape[1],
                                      logprob_subst.shape[2],
                                      logprob_subst.shape[3],
                                      logprob_subst.shape[4]),
                                    dtype = bool)
        # (T, B, L, base_alphabet_size, base_alphabet_size), OR
        # (1, 1, 1, base_alphabet_size, base_alphabet_size)
        return (logprob_subst, placeholder_mat)



###############################################################################
### Retrieve logprob at insert sites   ########################################
###############################################################################
### this is handled in concat_feats_to_params
# class IndelEmissionsLogprobs(ModuleBase):
#     config: dict
#     name: str
    
#     @nn.compact
#     def __call__(self,
#                  equilibr_logits,
#                  sow_intermediates: bool,
#                  **kwargs):
#         """
#         (no parameters to train, but ModuleBase allows writing to tensorboard)
        
#         purpose:
#         --------
#         evolutionary parameters (from neural network) -> 
#             logprob(emissions at insert sites)
        
        
#         input sizes:
#         -------------
#         equilibr_logits: (B, L, alph) OR (1, 1, alph)
        
        
#         output sizes:
#         -------------
#         logprob_equilibr: (B, L, alph) OR (1, 1, alph)
#         prob_equilibr: (B, L, alph) OR (1, 1, alph)
        
#         """
#         logprob_equilibr = nn.log_softmax(equilibr_logits)
#         prob_equilibr = jnp.exp( logprob_equilibr )
         
#         if sow_intermediates:
#             self.sow_histograms_scalars(mat=logprob_equilibr, 
#                                         label=f'{self.name}/logprob_equilibr', 
#                                         which='scalars')
        
#         # final mats are both: 
#         # (B, L, base_alphabet_size), OR (1, 1, base_alphabet_size)
#         return prob_equilibr, logprob_equilibr


class IndelEmissionsLogprobsFromFile(ModuleBase):
    """
    read the logprobs from a file
    """
    config: dict
    name: str
    
    def setup(self):
        load_from_file = self.config['load_from_file']
        
        with open(load_from_file, 'rb') as f:
            self.logprob_equilibr = jnp.load(f)
        
        if len(self.logprob_equilibr.shape) == 1:
            self.logprob_equilibr = self.logprob_equilibr[None, None, :]
        
    
    def __call__(self,
                 sow_intermediates: bool=False,
                 **kwargs):
        logprob_equilibr = self.logprob_equilibr
        prob_equilibr = jnp.exp( logprob_equilibr )

        # (B, L, base_alphabet_size), OR (1, 1, base_alphabet_size)
        return prob_equilibr, logprob_equilibr


class IndelEmissionsLogprobsFromCounts(ModuleBase):
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
                                              SMALLEST_FLOAT32
                                              )
                                   )
        
        # expand to to (B=1, L=1, alph)
        self.logprob_equilibr = logprob_equilibr[None, None, :]
        self.prob_equilibr = prob_equilibr[None, None, :]
        
    def __call__(self,
                 sow_intermediates: bool=False,
                 **kwargs):
        # (1, 1, base_alphabet_size)
        return self.prob_equilibr, self.logprob_equilibr
    