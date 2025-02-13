#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:42:28 2024

@author: annabel

modules:
========
 'CondTKF91TransitionLogprobs',
 'CondTKF91TransitionLogprobsFromFile',
  
 'JointTKF91TransitionLogprobs',
 'JointTKF91TransitionLogprobsFromFile',
 
 'CondTKF92TransitionLogprobs',
 'CondTKF92TransitionLogprobsFromFile',

 'JointTKF92TransitionLogprobs',
 'JointTKF92TransitionLogprobsFromFile',
 
"""
# jumping jax and leaping flax
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import pickle

from models.model_utils.BaseClasses import ModuleBase


###############################################################################
### helper functions   ########################################################
###############################################################################
def bounded_sigmoid(x, min_val, max_val):
    return min_val + (max_val - min_val) / (1 + jnp.exp(-x))

def safe_log(x):
    return jnp.log( jnp.where( x>0, 
                               x, 
                               jnp.finfo('float32').smallest_normal ) )

def concat_along_new_last_axis(arr_lst):
    return jnp.concatenate( [arr[...,None] for arr in arr_lst], 
                             axis = -1 )

def logsumexp_with_arr_lst(arr_lst, coeffs = None):
    """
    concatenate a list of arrays, then use logsumexp
    """
    a_for_logsumexp = concat_along_new_last_axis(arr_lst)
    
    out = logsumexp(a = a_for_logsumexp,
                    b = coeffs,
                    axis=-1)
    return out

def log_one_minus_x(x):
    """
    calculate log( exp(log(1)) - exp(log(x)) ),
      which is log( 1 - x )
    """
    a_for_logsumexp = concat_along_new_last_axis( [jnp.zeros(x.shape), x] )
    b_for_logsumexp = jnp.array([1, -1])
    
    out = logsumexp(a = a_for_logsumexp,
                    b = b_for_logsumexp,
                    axis = -1)
    
    return out



###############################################################################
### TKF91 (conditional and joint)   ###########################################
###############################################################################
class CondTKF91TransitionLogprobs(ModuleBase):
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
                                   tkf_err = self.tkf_err,
                                   num_classes = 1)
        
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
        return self.fill_cond_tkf91(out_dict)
    
    
    def fill_cond_tkf91(self,
                        out_dict):
        ### entries in the matrix
        # a_f = (1-beta)*alpha;     log(a_f) = log(1-beta) + log(alpha)
        # b_g = beta;               log(b_g) = log(beta)
        # c_h = (1-beta)*(1-alpha); log(c_h) = log(1-beta) + log(1-alpha)
        log_a_f = out_dict['log_one_minus_beta'] + out_dict['log_alpha']
        log_b_g = out_dict['log_beta']
        log_c_h = out_dict['log_one_minus_beta'] + out_dict['log_one_minus_alpha']
        log_mis_e = out_dict['log_one_minus_beta']

        # p = (1-gamma)*alpha;     log(p) = log(1-gamma) + log(alpha)
        # q = gamma;               log(q) = log(gamma)
        # r = (1-gamma)*(1-alpha); log(r) = log(1-gamma) + log(1-alpha)
        log_p = out_dict['log_one_minus_gamma'] + out_dict['log_alpha']
        log_q = out_dict['log_gamma']
        log_r = out_dict['log_one_minus_gamma'] + out_dict['log_one_minus_alpha']
        log_d_e = out_dict['log_one_minus_gamma']
        
        # (T,4,4)
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
                   tkf_err,
                   num_classes):
        # T, C
        t_array = jnp.broadcast_to( t_array[:,None],
                                    (t_array.shape[0],
                                     num_classes) 
                                    )
        
        ### lam * t, mu * t
        mu_per_t = mu * t_array
        lam_per_t = lam * t_array
        
        ### log(lam), log(mu); one value
        log_lam = safe_log(lam)
        log_mu = safe_log(mu)
        
        
        ### alpha and one minus alpha IN LOG SPACE
        # alpha = jnp.exp(-mu_per_t); log(alpha) = -mu_per_t
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
    
        
    
class CondTKF91TransitionLogprobsFromFile(CondTKF91TransitionLogprobs):
    """
    inherit tkf_params and fill_cond_tkf91 from CondTKF91TransitionLogprobs
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
                                   tkf_err = self.tkf_err,
                                   num_classes = 1)
        
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
        
        return self.fill_cond_tkf91(out_dict)
        
    
    
class JointTKF91TransitionLogprobs(CondTKF91TransitionLogprobs):
    """
    inherit logits_to_indel_rates, tkf_params, and setup 
      from CondTKF91TransitionLogprobs
    """
    config: dict
    name: str
   
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
                                   tkf_err = self.tkf_err,
                                   num_classes = 1)
        
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
        
        return self.fill_joint_tkf91(out_dict)
        
    
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


class JointTKF91TransitionLogprobsFromFile(JointTKF91TransitionLogprobs):
    """
    inherit fill_joint_tkf91 from JointTKF91TransitionLogprobs
    inherit tkf_params from CondTKF91TransitionLogprobs
    
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
                                   tkf_err = self.tkf_err,
                                   num_classes = 1)
        
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
        
        return self.fill_joint_tkf91(out_dict)
        
    
    
###############################################################################
### TKF92 (conditional and joint)   ###########################################
###############################################################################
class CondTKF92TransitionLogprobs(CondTKF91TransitionLogprobs):
    """
    inherit logits_to_indel_rates and tkf_params from CondTKF91TransitionLogprobs
    
    """
    config: dict
    name: str
    
    def setup(self):
        ### unpack config
        self.tkf_err = self.config.get('tkf_err', 1e-4)
        self.num_tkf_site_classes = self.config['num_tkf_site_classes']
        
        # initializing lamda, offset
        init_lam_offset_logits = self.config.get( 'init_lambda_offset_logits', 
                                                  [-2, -5] )
        init_lam_offset_logits = jnp.array(init_lam_offset_logits, dtype=float)
        self.lam_min_val, self.lam_max_val = self.config.get( 'lambda_range', 
                                                               [self.tkf_err, 3] )
        self.offs_min_val, self.offs_max_val = self.config.get( 'offset_range', 
                                                                [self.tkf_err, 0.333] )
        
        # initializing r extension prob
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
                 sow_intermediates: bool):
        out = self.logits_to_indel_rates(lam_mu_logits = self.tkf_lam_mu_logits,
                                         lam_min_val = self.lam_min_val,
                                         lam_max_val = self.lam_max_val,
                                         offs_min_val = self.offs_min_val,
                                         offs_max_val = self.offs_max_val,
                                         tkf_err = self.tkf_err )
        lam, mu, use_approx = out
        del out
        
        r_extend = bounded_sigmoid(x = self.r_extend_logits,
                                   min_val = self.r_extend_min_val,
                                   max_val = self.r_extend_max_val)
        
        # get alpha, beta, gamma
        out_dict = self.tkf_params(lam = lam, 
                                   mu = mu, 
                                   t_array = t_array,
                                   use_approx = use_approx,
                                   tkf_err = self.tkf_err,
                                   num_classes = r_extend.shape[0])
        
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
    
        return self.fill_cond_tkf92(out_dict, r_extend)
    
    def fill_cond_tkf92(self,
                        out_dict,
                        r_extend):
        T = out_dict['log_alpha'].shape[0]
        
        ### entries in the matrix
        log_r_extend = jnp.broadcast_to( safe_log(r_extend)[None,...],
                                         (T,
                                          r_extend.shape[0])
                                         )
                                         
        log_one_minus_r_extend = log_one_minus_x(log_r_extend)
        
        # a = r_extend + (1-r_extend)*(1-beta)*alpha
        # log(a) = logsumexp([r_extend, 
        #                     log(1-r_extend) + log(1-beta) + log(alpha)
        #                     ]
        #                    )
        log_a_second_half = ( log_one_minus_r_extend + 
                              out_dict['log_one_minus_beta'] +
                              out_dict['log_alpha'] )
        log_a = logsumexp_with_arr_lst([log_r_extend, log_a_second_half])
        
        # b = (1-r_extend)*beta
        # log(b) = log(1-r_extend) + log(beta)
        log_b = log_one_minus_r_extend + out_dict['log_beta']
        
        # c_h = (1-r_extend)*(1-beta)*(1-alpha)
        # log(c_h) = log(1-r_extend) + log(1-beta) + log(1-alpha)
        log_c_h = ( log_one_minus_r_extend +
                    out_dict['log_one_minus_beta'] +
                    out_dict['log_one_minus_alpha'] )
        
        # m_e = (1-r_extend) * (1-beta)
        # log(mi_e) = log(1-r_extend) + log(1-beta)
        log_mi_e = log_one_minus_r_extend + out_dict['log_one_minus_beta']

        # f = (1-r_extend)*(1-beta)*alpha
        # log(f) = log(1-r_extend) +log(1-beta) +log(alpha)
        log_f = ( log_one_minus_r_extend +
                  out_dict['log_one_minus_beta'] +
                  out_dict['log_alpha'] )
        
        # g = r_extend + (1-r_extend)*beta
        # log(g) = logsumexp([r_extend, 
        #                     log(1-r_extend) + log(beta)
        #                     ]
        #                    )
        log_g_second_half = log_one_minus_r_extend + out_dict['log_beta']
        log_g = logsumexp_with_arr_lst([log_r_extend, log_g_second_half])
        
        # h and log(h) are the same as c and log(c) 

        # p = (1-r_extend)*(1-gamma)*alpha
        # log(p) = log(1-r_extend) + log(1-gamma) +log(alpha)
        log_p = ( log_one_minus_r_extend +
                  out_dict['log_one_minus_gamma'] +
                  out_dict['log_alpha'] )

        # q = (1-r_extend)*gamma
        # log(q) = log(1-r_extend) + log(gamma)
        log_q = log_one_minus_r_extend + out_dict['log_gamma']

        # r = r_extend + (1-r_extend)*(1-gamma)*(1-alpha)
        # log(r) = logsumexp([r_extend, 
        #                     log(1-r_extend) + log(1-gamma) + log(1-alpha)
        #                     ]
        #                    )
        log_r_second_half = ( log_one_minus_r_extend +
                              out_dict['log_one_minus_gamma'] +
                              out_dict['log_one_minus_alpha'] )
        log_r = logsumexp_with_arr_lst([log_r_extend, log_r_second_half])
        
        # d_e = (1-r_extend) * (1-gamma)
        # log(d_e) = log(1-r_extend) + log(1-gamma)
        log_d_e = log_one_minus_r_extend + out_dict['log_one_minus_gamma']
        
        # final row; same as TKF91 final row
        log_s_m = out_dict['log_one_minus_beta'] + out_dict['log_alpha']
        log_s_i = out_dict['log_beta']
        log_s_d = out_dict['log_one_minus_beta'] + out_dict['log_one_minus_alpha']
        log_s_e = out_dict['log_one_minus_beta']
        
        #(T, C, 4, 4)
        return jnp.stack([ jnp.stack([  log_a,   log_b, log_c_h, log_mi_e], axis=-1),
                           jnp.stack([  log_f,   log_g, log_c_h, log_mi_e], axis=-1),
                           jnp.stack([  log_p,   log_q,   log_r,  log_d_e], axis=-1),
                           jnp.stack([log_s_m, log_s_i, log_s_d,  log_s_e], axis=-1)
                          ], axis=-2)

    
class CondTKF92TransitionLogprobsFromFile(CondTKF92TransitionLogprobs):
    """
    inherit setup and fill_cond_tkf92 from CondTKF92TransitionLogprobs
    inherit tkf_params from CondTKF91TransitionLogprobs
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
                                   tkf_err = self.tkf_err,
                                   num_classes = num_site_classes)
        
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
        
        return self.fill_cond_tkf92(out_dict, r_extend)
    

class JointTKF92TransitionLogprobs(CondTKF92TransitionLogprobs):
    """
    inherit setup from CondTKF92TransitionLogprobs
    inherit logits_to_indel_rates, tkf_params from CondTKF91TransitionLogprobs
    
    """
    config: dict
    name: str
    
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
        
        r_extend = bounded_sigmoid(x = self.r_extend_logits,
                                   min_val = self.r_extend_min_val,
                                   max_val = self.r_extend_max_val)
        
        # get alpha, beta, gamma
        out_dict = self.tkf_params(lam = lam, 
                                   mu = mu, 
                                   t_array = t_array,
                                   use_approx = use_approx,
                                   tkf_err = self.tkf_err,
                                   num_classes = r_extend.shape[0])
        
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
    
        return self.fill_joint_tkf92(out_dict, r_extend)
    
    
    def fill_joint_tkf92(self,
                        out_dict,
                        r_extend):
        log_lam_div_mu = out_dict['log_lam'] - out_dict['log_mu']
        log_one_minus_lam_div_mu = log_one_minus_x(log_lam_div_mu)
        T = out_dict['log_alpha'].shape[0]
        
        ### entries in the matrix
        log_r_extend = jnp.broadcast_to( safe_log(r_extend)[None,...],
                                         (T,
                                          r_extend.shape[0])
                                         )
        log_one_minus_r_extend = log_one_minus_x(log_r_extend)
        
        # a = r_extend + (1-r_extend)*(1-beta)*alpha*(lam/mu)
        # log(a) = logsumexp([r_extend, 
        #                     log(1-r_extend) + log(1-beta) + log(alpha) + log(lam/mu)
        #                     ]
        #                    )
        log_a_second_half = ( log_one_minus_r_extend + 
                              out_dict['log_one_minus_beta'] +
                              out_dict['log_alpha'] +
                              log_lam_div_mu )
        log_a = logsumexp_with_arr_lst([log_r_extend, log_a_second_half])
        
        # b = (1-r_extend)*beta
        # log(b) = log(1-r_extend) + log(beta)
        log_b = log_one_minus_r_extend + out_dict['log_beta']
        
        # c_h = (1-r_extend)*(1-beta)*(1-alpha)*(lam/mu)
        # log(c_h) = log(1-r_extend) + log(1-beta) + log(1-alpha) + log(lam/mu)
        log_c_h = ( log_one_minus_r_extend +
                    out_dict['log_one_minus_beta'] +
                    out_dict['log_one_minus_alpha'] +
                    log_lam_div_mu )
        
        # m_e = (1-r_extend) * (1-beta) * (1- (lam/mu))
        # log(mi_e) = log(1-r_extend) + log(1-beta) + log(1- (lam/mu))
        log_mi_e = ( log_one_minus_r_extend + 
                     out_dict['log_one_minus_beta'] +
                     log_one_minus_lam_div_mu )

        # f = (1-r_extend)*(1-beta)*alpha*(lam/mu)
        # log(f) = log(1-r_extend) +log(1-beta) +log(alpha) + lam/mu
        log_f = ( log_one_minus_r_extend +
                  out_dict['log_one_minus_beta'] +
                  out_dict['log_alpha'] +
                  log_lam_div_mu )
        
        # g = r_extend + (1-r_extend)*beta
        # log(g) = logsumexp([r_extend, 
        #                     log(1-r_extend) + log(beta)
        #                     ]
        #                    )
        log_g_second_half = log_one_minus_r_extend + out_dict['log_beta']
        log_g = logsumexp_with_arr_lst([log_r_extend, log_g_second_half])
        
        # h and log(h) are the same as c and log(c) 

        # p = (1-r_extend)*(1-gamma)*alpha*(lam/mu)
        # log(p) = log(1-r_extend) + log(1-gamma) +log(alpha) + log(lam/mu)
        log_p = ( log_one_minus_r_extend +
                  out_dict['log_one_minus_gamma'] +
                  out_dict['log_alpha'] +
                  log_lam_div_mu )

        # q = (1-r_extend)*gamma
        # log(q) = log(1-r_extend) + log(gamma)
        log_q = log_one_minus_r_extend + out_dict['log_gamma']

        # r = r_extend + (1-r_extend)*(1-gamma)*(1-alpha)*(lam/mu)
        # log(r) = logsumexp([r_extend, 
        #                     log(1-r_extend) + log(1-gamma) + log(1-alpha) + log(lam/mu)
        #                     ]
        #                    )
        log_r_second_half = ( log_one_minus_r_extend +
                              out_dict['log_one_minus_gamma'] +
                              out_dict['log_one_minus_alpha'] +
                              log_lam_div_mu )
        log_r = logsumexp_with_arr_lst([log_r_extend, log_r_second_half])
        
        # d_e = (1-r_extend) * (1-gamma) * (1 - (lam/mu))
        # log(d_e) = log(1-r_extend) + log(1-gamma) +log(1 - (lam/mu))
        log_d_e = ( log_one_minus_r_extend + 
                    out_dict['log_one_minus_gamma'] +
                    log_one_minus_lam_div_mu )
        
        # final row 
        log_s_m = ( out_dict['log_one_minus_beta'] + 
                    out_dict['log_alpha'] + 
                    log_lam_div_mu )
        log_s_i = out_dict['log_beta']
        log_s_d = ( out_dict['log_one_minus_beta'] + 
                    out_dict['log_one_minus_alpha'] + 
                    log_lam_div_mu)
        log_s_e = out_dict['log_one_minus_beta'] + log_one_minus_lam_div_mu
        
        
        #(T, C, 4, 4)
        return jnp.stack([ jnp.stack([  log_a,   log_b, log_c_h, log_mi_e], axis=-1),
                           jnp.stack([  log_f,   log_g, log_c_h, log_mi_e], axis=-1),
                           jnp.stack([  log_p,   log_q,   log_r,  log_d_e], axis=-1),
                           jnp.stack([log_s_m, log_s_i, log_s_d,  log_s_e], axis=-1)
                          ], axis=-2)


class JointTKF92TransitionLogprobsFromFile(JointTKF92TransitionLogprobs):
    """
    inherit fill_joint_tkf92 from JointTKF92TransitionLogprobs
    inherit setup from CondTKF92TransitionLogprobs
    inherit logits_to_indel_rates, tkf_params from CondTKF91TransitionLogprobs
    
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
                                   tkf_err = self.tkf_err,
                                   num_classes = num_site_classes)
        
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
        
        return self.fill_joint_tkf92(out_dict, r_extend)
