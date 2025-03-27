#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:42:28 2024

@author: annabel

modules used for training:
==========================
 'TKF91TransitionLogprobs',
 'TKF91TransitionLogprobsFromFile',
 
 'TKF92TransitionLogprobs',
 'TKF92TransitionLogprobsFromFile',

functions:
===========
 'MargTKF91TransitionLogprobs'
 'MargTKF92TransitionLogprobs'
 'CondTKF92TransitionLogprobs'

"""
# jumping jax and leaping flax
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import pickle

from models.model_utils.BaseClasses import ModuleBase
from utils.pairhmm_helpers import (bounded_sigmoid,
                                   safe_log,
                                   concat_along_new_last_axis,
                                   logsumexp_with_arr_lst,
                                   log_one_minus_x)


###############################################################################
### single sequence and conditional FUNCTIONS   ###############################
###############################################################################
def MargTKF91TransitionLogprobs(lam,
                                mu,
                                **kwargs):
    """
    one (2,2) matrix
    
    
    emit -> emit   |  emit -> end
    -------------------------------
    start -> emit  |  start -> end
    """
    log_lam = safe_log( lam )
    log_mu = safe_log(mu)
    
    log_lam_div_mu = log_lam - log_mu
    log_one_minus_lam_div_mu = log_one_minus_x(log_lam_div_mu)
    
    log_arr = jnp.array( [[log_lam_div_mu, log_one_minus_lam_div_mu],
                          [log_lam_div_mu, log_one_minus_lam_div_mu]] )
    return log_arr


def MargTKF92TransitionLogprobs(lam,
                                mu,
                                class_probs,
                                r_ext_prob,
                                **kwargs):
    C = class_probs.shape[-1]
    
    ### move values to log space
    log_class_prob = safe_log(class_probs)
    
    log_r_ext_prob = safe_log(r_ext_prob)
    log_one_minus_r = log_one_minus_x(log_r_ext_prob)
    
    log_lam = safe_log(lam)
    log_mu = safe_log(mu)
    log_lam_div_mu = log_lam - log_mu
    log_one_minus_lam_div_mu = log_one_minus_x(log_lam_div_mu)
    
    
    ### build cells
    # cell 1: emit -> emit (C,C)
    log_cell1 = (log_one_minus_r + log_lam_div_mu)[:, None] + log_class_prob[None, :]
    
    # cell 2: emit -> end (C,1)
    log_cell2 = ( log_one_minus_r + log_one_minus_lam_div_mu )[:,None]
    log_cell2 = jnp.broadcast_to( log_cell2, (C, C) )
    
    # cell 3: start -> emit
    log_cell3 = ( log_lam_div_mu + log_class_prob )[None,:]
    log_cell3 = jnp.broadcast_to( log_cell3, (C, C) ) 
    
    # cell 4: start -> end
    log_cell4 = jnp.broadcast_to( log_one_minus_lam_div_mu, (C,C) )
    

    ### build matrix
    log_single_seq_tkf92 = jnp.stack( [jnp.stack( [log_cell1, log_cell2], axis=-1 ),
                                       jnp.stack( [log_cell3, log_cell4], axis=-1 )],
                                     axis = -2 )
    
    i_idx = jnp.arange(C)
    prev_vals = log_single_seq_tkf92[i_idx, i_idx, 0, 0]
    new_vals = logsumexp_with_arr_lst([log_r_ext_prob, prev_vals])
    log_single_seq_tkf92 = log_single_seq_tkf92.at[i_idx, i_idx, 0, 0].set(new_vals)
    
    return log_single_seq_tkf92


def CondTransitionLogprobs(marg_matrix, joint_matrix):
    """
    obtain the conditional log probability by composing the joint with the marginal
    """
    cond_matrix = joint_matrix.at[...,[0,1,2], 0].add(-marg_matrix[..., 0,0][None,...,None])
    cond_matrix = cond_matrix.at[...,[0,1,2], 2].add(-marg_matrix[..., 0,0][None,...,None])
    cond_matrix = cond_matrix.at[...,3,0].add(-marg_matrix[..., 1,0][None,...])
    cond_matrix = cond_matrix.at[...,3,2].add(-marg_matrix[..., 1,0][None,...])
    cond_matrix = cond_matrix.at[...,[0,1,2],3].add(-marg_matrix[..., 0,1][None,...,None])
    cond_matrix = cond_matrix.at[...,3,3].add(-marg_matrix[..., 1,1][None,...])
    return cond_matrix



###############################################################################
### TKF91   ###################################################################
###############################################################################
class TKF91TransitionLogprobs(ModuleBase):
    """
    Used for calculating P(anc, desc, align)
    
    Returns three matrices in a dictionary: 
        - "joint": (T, 4, 4)
        - "marginal": (4, 4)
        - "conditional": (T, 4, 4)
    """
    config: dict
    name: str
    
    def setup(self):
        ### unpack config
        self.tkf_err = self.config.get('tkf_err', 1e-4)
        
        # initializing lamda, offset
        init_lam_offset_logits = self.config.get( 'init_lambda_offset_logits',
                                                [-2, -5] )
        init_lam_offset_logits = jnp.array(init_lam_offset_logits, dtype=float)
        self.lam_min_val, self.lam_max_val = self.config.get( 'lambda_range', 
                                                               [self.tkf_err, 3] )
        self.offs_min_val, self.offs_max_val = self.config.get( 'offset_range', 
                                                                [self.tkf_err, 0.333] )
        
        
        ### initialize logits for lambda, offset
        # with default values:
        # init lam: ~0.35769683
        # init offset: ~0.02017788
        self.tkf_lam_mu_logits = self.param('TKF91 lambda, mu',
                                            lambda rng, shape, dtype: init_lam_offset_logits,
                                            init_lam_offset_logits.shape,
                                            jnp.float32)
   
    def __call__(self,
                 t_array,
                 sow_intermediates: bool):
        out = self.logits_to_indel_rates(lam_mu_logits = self.tkf_lam_mu_logits,
                                         lam_min_val = self.lam_min_val,
                                         lam_max_val = self.lam_max_val,
                                         offs_min_val = self.offs_min_val,
                                         offs_max_val = self.offs_max_val,
                                         tkf_err = self.tkf_err )
        lam, mu, use_approx = out
        del out
        
        # get alpha, beta, gamma
        # only one class for TKF91
        out_dict = self.tkf_params(lam = lam, 
                                   mu = mu, 
                                   t_array = t_array,
                                   use_approx = use_approx,
                                   tkf_err = self.tkf_err)
        
        # record values
        if sow_intermediates:
            self.sow_histograms_scalars(mat= jnp.exp(out_dict['log_alpha']), 
                                        label=f'{self.name}/tkf_alpha', 
                                        which='scalars')
            
            self.sow_histograms_scalars(mat= jnp.exp(out_dict['log_beta']), 
                                        label=f'{self.name}/tkf_beta', 
                                        which='scalars')
            
            self.sow_histograms_scalars(mat= jnp.exp(out_dict['log_gamma']), 
                                        label=f'{self.name}/tkf_gamma', 
                                        which='scalars')
            
            self.sow_histograms_scalars(mat= lam, 
                                        label=f'{self.name}/lam', 
                                        which='scalars')
            
            self.sow_histograms_scalars(mat= mu, 
                                        label=f'{self.name}/mu', 
                                        which='scalars')
        
        joint_matrix =  self.fill_joint_tkf91(out_dict)
        
        matrix_dict = self.return_all_matrices(lam=lam,
                                               mu=mu,
                                               joint_matrix=joint_matrix)
        return matrix_dict
        
    
    def fill_joint_tkf91(self, 
                         out_dict):
        ### entries in the matrix
        log_lam_div_mu = out_dict['log_lam'] - out_dict['log_mu']
        log_one_minus_lam_div_mu = log_one_minus_x(log_lam_div_mu)
        
        
        # a_f = (1-beta)*alpha*(lam/mu);     log(a_f) = log(1-beta) + log(alpha) + log_lam_div_mu
        # b_g = beta;                        log(b_g) = log(beta)
        # c_h = (1-beta)*(1-alpha)*(lam/mu); log(c_h) = log(1-beta) + log(1-alpha) + log_lam_div_mu
        log_a_f = (out_dict['log_one_minus_beta'] + 
                   out_dict['log_alpha'] + 
                   log_lam_div_mu)
        log_b_g = out_dict['log_beta']
        log_c_h = (out_dict['log_one_minus_beta'] + 
                   out_dict['log_one_minus_alpha'] + 
                   log_lam_div_mu)
        log_mis_e = out_dict['log_one_minus_beta'] + log_one_minus_lam_div_mu

        # p = (1-gamma)*alpha*(lam/mu);     log(p) = log(1-gamma) + log(alpha) + log_lam_div_mu
        # q = gamma;                        log(q) = log(gamma)
        # r = (1-gamma)*(1-alpha)*(lam/mu); log(r) = log(1-gamma) + log(1-alpha) + log_lam_div_mu
        log_p = (out_dict['log_one_minus_gamma'] + 
                 out_dict['log_alpha'] +
                 log_lam_div_mu)
        log_q = out_dict['log_gamma']
        log_r = (out_dict['log_one_minus_gamma'] + 
                 out_dict['log_one_minus_alpha'] +
                 log_lam_div_mu)
        log_d_e = out_dict['log_one_minus_gamma'] + log_one_minus_lam_div_mu
        
        #(T, 4, 4)
        out = jnp.stack([ jnp.stack([log_a_f, log_b_g, log_c_h, log_mis_e], axis=-1),
                           jnp.stack([log_a_f, log_b_g, log_c_h, log_mis_e], axis=-1),
                           jnp.stack([  log_p,   log_q,   log_r,   log_d_e], axis=-1),
                           jnp.stack([log_a_f, log_b_g, log_c_h, log_mis_e], axis=-1)
                          ], axis=-2)
        return out
    
    def logits_to_indel_rates(self, 
                              lam_mu_logits,
                              lam_min_val,
                              lam_max_val,
                              offs_min_val,
                              offs_max_val,
                              tkf_err):
        """
        assumes idx=0 is lambda, idx=1 is for calculating mu
        
        TODO:
        tkf_err: \epsilon = 1 - (lam/mu), so you're directly setting the 
          probability of no ancestor sequence... there should be a smarter
          way to initialize this
        """
        # lambda
        lam = bounded_sigmoid(x = lam_mu_logits[0],
                              min_val = lam_min_val,
                              max_val = lam_max_val)
        
        # mu
        offset = bounded_sigmoid(x = lam_mu_logits[1],
                                 min_val = offs_min_val,
                                 max_val = offs_max_val)
        mu = lam / ( 1 -  offset) 

        use_approx = (offset == tkf_err)
        
        return (lam, mu, use_approx)
    
    
    def tkf_params(self,
                   lam,
                   mu,
                   t_array,
                   use_approx,
                   tkf_err):
        """
        lam and mu are single integers
        
        output alpha, beta, gamma, etc. all have shape (T,)
        """
        ### lam * t, mu * t
        mu_per_t = mu * t_array
        lam_per_t = lam * t_array
        
        ### log(lam), log(mu); one value
        log_lam = safe_log(lam)
        log_mu = safe_log(mu)
        
        
        ### alpha and one minus alpha IN LOG SPACE
        # alpha = jnp.exp(-mu_per_t); log(alpha) = -mu_per_t
        # alpha = jnp.exp( -mu_per_t )
        # one_minus_alpha = 1 - alpha
        # log_alpha = jnp.log(alpha)
        # log_one_minus_alpha = jnp.log(one_minus_alpha)
        log_alpha = -mu_per_t
        log_one_minus_alpha = log_one_minus_x(log_alpha)
        
        
        ### beta
        def orig_beta():
            # log( exp(-lambda * t) - exp(-mu * t) )
            term2_logsumexp = logsumexp_with_arr_lst( [-lam_per_t, -mu_per_t],
                                                      coeffs = jnp.array([1, -1]) )
            
            # log( mu*exp(-lambda * t) - lambda*exp(-mu * t) )
            mixed_coeffs = concat_along_new_last_axis([mu, -lam])
            term3_logsumexp = logsumexp_with_arr_lst([-lam_per_t, -mu_per_t],
                                                     coeffs = mixed_coeffs)
            del mixed_coeffs
            
            # combine
            log_beta = log_lam + term2_logsumexp - term3_logsumexp
            
            return log_beta
            
        def approx_beta():
            return ( safe_log(1 - tkf_err) + 
                     safe_log(mu_per_t) - 
                     safe_log(mu_per_t + 1) )
        
        log_beta = jnp.where( use_approx,
                              approx_beta(),
                              orig_beta() )
        
        
        ### gamma
        # numerator = mu * beta; log(numerator) = log(mu) + log(beta)
        gamma_numerator = log_mu + log_beta
        
        # denom = lambda * (1-alpha); log(denom) = log(lam) + log(1-alpha)
        gamma_denom = log_lam + log_one_minus_alpha
        
        # 1 - gamma = num/denom; log(1 - gamma) = log(num) - log(denom)
        log_one_minus_gamma = gamma_numerator - gamma_denom
        log_gamma = log_one_minus_x(log_one_minus_gamma)
        log_one_minus_beta = log_one_minus_x(log_beta)
            
        
        ### final dictionary
        out_dict = {'log_lam': log_lam,
                    'log_mu':log_mu,
                    'log_alpha': log_alpha,
                    'log_beta': log_beta,
                    'log_gamma': log_gamma,
                    'log_one_minus_alpha': log_one_minus_alpha,
                    'log_one_minus_beta': log_one_minus_beta,
                    'log_one_minus_gamma': log_one_minus_gamma,
                    'used_tkf_approx': use_approx
                    }
        
        return out_dict
    
    
    def return_all_matrices(self,
                            lam,
                            mu,
                            joint_matrix):
        # output is: (4, 4)
        marginal_matrix = MargTKF91TransitionLogprobs(lam=lam,
                                                      mu=mu)
        
        # output is same as joint: (T, 4, 4)
        cond_matrix = CondTransitionLogprobs(marg_matrix=marginal_matrix, 
                                             joint_matrix=joint_matrix)
        
        return {'joint': joint_matrix,
                'marginal': marginal_matrix,
                'conditional': cond_matrix}
    


class TKF91TransitionLogprobsFromFile(TKF91TransitionLogprobs):
    """
    inherit fill_joint_tkf91, tkf_params from TKF91TransitionLogprobs
    
    Returns three matrices in a dictionary: 
        - "joint": (T, 4, 4)
        - "marginal": (4, 4)
        - "conditional": (T, 4, 4)
    """
    config: dict
    name: str
    
    def setup(self):
        ### unpack config
        self.tkf_err = self.config.get('tkf_err', 1e-4)
        in_file = self.config['filenames']['tkf_params_file']
        
        with open(in_file,'rb') as f:
            self.tkf_lam_mu = jnp.load(f)
    
    def __call__(self,
                 t_array,
                 sow_intermediates: bool):
        lam = self.tkf_lam_mu[...,0]
        mu = self.tkf_lam_mu[...,1]
        use_approx = False
        
        # get alpha, beta, gamma
        out_dict = self.tkf_params(lam = lam, 
                                   mu = mu, 
                                   t_array = t_array,
                                   use_approx = use_approx,
                                   tkf_err = self.tkf_err)
        
        joint_matrix =  self.fill_joint_tkf91(out_dict)
        
        matrix_dict = self.return_all_matrices(lam=lam,
                                               mu=mu,
                                               joint_matrix=joint_matrix)
        return matrix_dict
        
    
    
###############################################################################
### TKF92   ###################################################################
###############################################################################
class TKF92TransitionLogprobs(TKF91TransitionLogprobs):
    """
    inherit logits_to_indel_rates, tkf_params from TKF91TransitionLogprobs
    
    Returns three matrices in a dictionary: 
        - "joint": (T, C, C, 4, 4)
        - "marginal": (C, C, 4, 4)
        - "conditional": (T, C, C, 4, 4)
    """
    config: dict
    name: str
    
    def setup(self):
        ### unpack config
        self.tkf_err = self.config.get('tkf_err', 1e-4)
        self.num_tkf_site_classes = self.config['num_tkf_site_classes']
        
        ### initializing lamda, offset
        init_lam_offset_logits = self.config.get( 'init_lambda_offset_logits', 
                                                  [-2, -5] )
        init_lam_offset_logits = jnp.array(init_lam_offset_logits, dtype=float)
        self.lam_min_val, self.lam_max_val = self.config.get( 'lambda_range', 
                                                               [self.tkf_err, 3] )
        self.offs_min_val, self.offs_max_val = self.config.get( 'offset_range', 
                                                                [self.tkf_err, 0.333] )
        
        
        ### initializing r extension prob
        init_r_extend_logits = self.config.get( 'init_r_extend_logits',
                                               [-x/10 for x in 
                                                range(1, self.num_tkf_site_classes+1)] )
        init_r_extend_logits = jnp.array(init_r_extend_logits, dtype=float)
        self.r_extend_min_val, self.r_extend_max_val = self.config.get( 'r_range', 
                                                                [self.tkf_err, 0.8] )
        
        
        ### initialize logits for lambda, offset
        # with default values:
        # init lam: ~0.35769683
        # init offset: ~0.02017788
        self.tkf_lam_mu_logits = self.param('TKF92 lambda, mu',
                                            lambda rng, shape, dtype: init_lam_offset_logits,
                                            init_lam_offset_logits.shape,
                                            jnp.float32)
        
        # up to num_tkf_site_classes different r extension probabilities
        # with default first 10 values: 
        #   0.40004998, 0.38006914, 0.3601878, 0.34050342, 0.32110974
        #   0.30209476, 0.2835395, 0.26551658, 0.24808942, 0.2313115
        self.r_extend_logits = self.param('TKF92 r extension prob',
                                          lambda rng, shape, dtype: init_r_extend_logits,
                                          init_r_extend_logits.shape,
                                          jnp.float32)
    
    def __call__(self,
                 t_array,
                 class_probs,
                 sow_intermediates: bool):
        # lam, mu are of size () (i.e. just floats)
        out = self.logits_to_indel_rates(lam_mu_logits = self.tkf_lam_mu_logits,
                                         lam_min_val = self.lam_min_val,
                                         lam_max_val = self.lam_max_val,
                                         offs_min_val = self.offs_min_val,
                                         offs_max_val = self.offs_max_val,
                                         tkf_err = self.tkf_err )
        lam, mu, use_approx = out
        del out
        
        # r_extend is of size (C,)
        r_extend = bounded_sigmoid(x = self.r_extend_logits,
                                   min_val = self.r_extend_min_val,
                                   max_val = self.r_extend_max_val)
        
        # get alpha, beta, gamma; these are of size (T,)
        out_dict = self.tkf_params(lam = lam, 
                                   mu = mu, 
                                   t_array = t_array,
                                   use_approx = use_approx,
                                   tkf_err = self.tkf_err)
        
        # record values
        if sow_intermediates:
            self.sow_histograms_scalars(mat= lam, 
                                        label=f'{self.name}/lam', 
                                        which='scalars')
            
            self.sow_histograms_scalars(mat= mu, 
                                        label=f'{self.name}/mu', 
                                        which='scalars')
            
            self.sow_histograms_scalars(mat= jnp.exp(out_dict['log_alpha']), 
                                        label=f'{self.name}/tkf_alpha', 
                                        which='scalars')
            
            self.sow_histograms_scalars(mat= jnp.exp(out_dict['log_beta']), 
                                        label=f'{self.name}/tkf_beta', 
                                        which='scalars')
            
            self.sow_histograms_scalars(mat= jnp.exp(out_dict['log_gamma']), 
                                        label=f'{self.name}/tkf_gamma', 
                                        which='scalars')
            
            for c in range(r_extend.shape[0]):
                self.sow_histograms_scalars(mat= r_extend[c], 
                                            label=f'{self.name}/r_extend_class_{c}', 
                                            which='scalars')
    
        # (T, C_from, C_to, S_from=4, S_to=4)
        joint_matrix =  self.fill_joint_tkf92(out_dict=out_dict, 
                                              r_extend=r_extend,
                                              class_probs=class_probs)
        
        matrix_dict = self.return_all_matrices(lam=lam,
                                               mu=mu,
                                               class_probs=class_probs,
                                               r_ext_prob = r_extend,
                                               joint_matrix=joint_matrix)
        return matrix_dict
        
    
    def fill_joint_tkf92(self,
                        out_dict,
                        r_extend,
                        class_probs):
        """
        final output shape: (T, C_from, C_to, S_from=4, S_to=4)
        """
        ### need joint TKF91 for this (which already contains lam/mu terms)
        log_U = self.fill_joint_tkf91(out_dict)
        
        # dims
        T = out_dict['log_alpha'].shape[0]
        C = class_probs.shape[0] #site classes
        S = log_U.shape[-1] #hidden states (like start, M, I, D, and end)
        
        # converted log values and broadcast to (T,C)
        log_r_ext_prob = jnp.broadcast_to( safe_log(r_extend)[None,...], (T,C) )
        log_one_minus_r = log_one_minus_x(log_r_ext_prob)
        log_class_prob = jnp.broadcast_to( safe_log(class_probs)[None,...], (T,C) )
        
        
        ### entries in the matrix
        # (1-r_c) U(i,j) for all (MID -> MIDE transitions), 
        #   U(i,j) for all start -> MIDE transitions
        # (T, C, S_from, S_to)
        operation_mask = jnp.ones( log_U.shape, dtype=bool )
        operation_mask = operation_mask.at[:, -1, :].set(False)
        operation_mask = jnp.broadcast_to( operation_mask[:, None, :, :], 
                                           (T,C,S,S) )
        log_tkf92_rate_mat = jnp.where( operation_mask,
                                        log_one_minus_r[:,:,None,None] + log_U[:, None, :, :],
                                        jnp.broadcast_to( log_U[:, None, :, :], (T,C,S,S) ) )
        del operation_mask
    
        # duplicate each C times and stack across C_to dimension 
        # (T, C_from, C_to, S_from, S_to)
        log_tkf92_rate_mat = jnp.broadcast_to( log_tkf92_rate_mat[:,:,None,:,:], 
                                               (T,C,C,S,S) )
    
        # multiply by P(c) across all C_to (not including transitions that 
        #   end with <end>)
        operation_mask = jnp.ones( log_U.shape, dtype=bool )
        operation_mask = operation_mask.at[:, :, -1].set(False)
        operation_mask = jnp.broadcast_to( operation_mask[:, None, None, :, :], 
                                           (T,C,C,S,S) )
        log_tkf92_rate_mat = jnp.where( operation_mask,
                                        log_tkf92_rate_mat + log_class_prob[:, None, :, None, None],
                                        log_tkf92_rate_mat )
        del operation_mask
        
        # at MID: where class_from == class_to and state_from == state_to, 
        #   add factor of r
        #   THIS ASSUMES START AND END TRANSITIONS ARE AFTER MID
        #   that is, S_from=3 means start, and S_to=3 means end
        i_idx, j_idx = jnp.meshgrid(jnp.arange(C), jnp.arange(S-1), indexing="ij")
        i_idx = i_idx.flatten()
        j_idx = j_idx.flatten()

        # add r to specific locations
        prev_vals = log_tkf92_rate_mat[:, i_idx, i_idx, j_idx, j_idx].reshape( (T, C, S-1) )
        r_to_add = jnp.broadcast_to( log_r_ext_prob[...,None], prev_vals.shape)
        new_vals = logsumexp_with_arr_lst([r_to_add, prev_vals]).reshape(T, -1)
        del prev_vals, r_to_add

        # Now scatter these back in one shot
        #(T, C, 4, 4)
        log_tkf92_rate_mat = log_tkf92_rate_mat.at[:, i_idx, i_idx, j_idx, j_idx].set(new_vals)
        
        return log_tkf92_rate_mat
    
    
    def return_all_matrices(self,
                            lam,
                            mu,
                            class_probs,
                            r_ext_prob,
                            joint_matrix):
        # output is: (C, C, 4, 4)
        marginal_matrix = MargTKF92TransitionLogprobs(lam=lam,
                                                      mu=mu,
                                                      class_probs=class_probs,
                                                      r_ext_prob=r_ext_prob)
        
        # output is: (T, C, C, 4, 4)
        cond_matrix = CondTransitionLogprobs(marg_matrix=marginal_matrix, 
                                             joint_matrix=joint_matrix)
        
        return {'joint': joint_matrix,
                'marginal': marginal_matrix,
                'conditional': cond_matrix}
        

class TKF92TransitionLogprobsFromFile(TKF92TransitionLogprobs):
    """
    inherit logits_to_indel_rates, tkf_params from TKF91TransitionLogprobs
    inherit fill_joint_tkf92 from TKF92TransitionLogprobs
    
    Returns three matrices in a dictionary: 
        - "joint": (T, C, C, 4, 4)
        - "marginal": (C, C, 4, 4)
        - "conditional": (T, C, C, 4, 4)
    """
    config: dict
    name: str
    
    def setup(self):
        ### unpack config
        self.tkf_err = self.config.get('tkf_err', 1e-4)
        in_file = self.config['filenames']['tkf_params_file']
        
        with open(in_file,'rb') as f:
            in_dict = pickle.load(f)
            self.tkf_lam_mu = in_dict['lam_mu']
            self.r_extend = in_dict['r_extend']
    
    def __call__(self,
                 t_array,
                 class_probs,
                 sow_intermediates: bool):
        lam = self.tkf_lam_mu[0]
        mu = self.tkf_lam_mu[1]
        r_extend = self.r_extend
        num_site_classes = r_extend.shape[0]
        use_approx = False
        
        # get alpha, beta, gamma
        out_dict = self.tkf_params(lam = lam, 
                                   mu = mu, 
                                   t_array = t_array,
                                   use_approx = use_approx,
                                   tkf_err = self.tkf_err)
        
        # (T, C_from, C_to, S_from=4, S_to=4)
        joint_matrix =  self.fill_joint_tkf92(out_dict, 
                                              r_extend,
                                              class_probs)
        
        matrix_dict = self.return_all_matrices(lam=lam,
                                               mu=mu,
                                               class_probs=class_probs,
                                               r_ext_prob=r_extend,
                                               joint_matrix=joint_matrix)
        return matrix_dict
    
