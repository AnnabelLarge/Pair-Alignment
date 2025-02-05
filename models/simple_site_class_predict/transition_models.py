#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:42:28 2024

@author: annabel

modules:
========
CondTKF91TransitionLogprobs
CondTKF92TransitionLogprobs
JointTKF91TransitionLogprobs
JointTKF92TransitionLogprobs
 
"""
# jumping jax and leaping flax
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from models.model_utils.BaseClasses import ModuleBase


###############################################################################
### helper functions   ########################################################
###############################################################################
SMALLEST_FLOAT32 = jnp.finfo('float32').smallest_normal

def bounded_sigmoid(x, min_val, max_val):
    return min_val + (max_val - min_val) / (1 + jnp.exp(-x))

def safe_log(x):
    return jnp.log( jnp.where( x>0, x, SMALLEST_FLOAT32 ) )

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




class CondTKF91TransitionLogprobs(ModuleBase):
    config: dict
    name: str
    
    def setup(self):
        ### unpack config
        self.tkf_err = self.config.get('tkf_err', 1e-4)
        self.load_tkf_params = self.config['load_tkf_params']
        
        if self.load_tkf_params:
            self.tkf_params_file = self.config['tkf_params_file']
            with open(self.tkf_params_file, 'rb') as f:
                self.tkf_params = jnp.load(f)
                self.use_approx = False
        
        elif not self.load_tkf_params:
            self.lam_min_val, self.lam_max_val = self.config.get( 'lambda_range', 
                                                                   [self.tkf_err, 3] )
            self.offs_min_val, self.offs_max_val = self.config.get( 'offset_range', 
                                                                    [self.tkf_err, 0.333] )
            
            ### initialize logits
            self.tkf_logits = self.param('TKF91 lambda, mu',
                                            nn.initializers.glorot_uniform(),
                                            (2,),
                                            jnp.float32)
        
    def __call__(self,
                 t_array,
                 sow_intermediates: bool):
        if not self.load_tkf_params:
            out = self.logits_to_indel_rates(tkf_logits = self.tkf_logits,
                                             lam_min_val = self.lam_min_val,
                                             lam_max_val = self.lam_max_val,
                                             offs_min_val = self.offs_min_val,
                                             offs_max_val = self.offs_max_val,
                                             tkf_err = self.tkf_err )
            lam, mu, use_approx = out
            del out
        
        elif self.load_tkf_params:
            lam = self.tkf_params[0]
            mu = self.tkf_params[1]
            use_approx = False
        
        # get alpha, beta, gamma
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
        return out[:,None,...]
    
    
    def logits_to_indel_rates(self, 
                              tkf_logits,
                              lam_min_val,
                              lam_max_val,
                              offs_min_val,
                              offs_max_val,
                              tkf_err):
        """
        assumes idx=0 is lambda, idx=1 is for calculating mu
        """
        # lambda
        lam = bounded_sigmoid(x = tkf_logits[...,0],
                              min_val = lam_min_val,
                              max_val = lam_max_val)
        
        # mu
        offset = bounded_sigmoid(x = tkf_logits[...,1],
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
        ### lam * t, mu * t; (T,)
        mu_per_t = mu * t
        lam_per_t = lam * t
        
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
        
        # probably pretty rare that log(1 - gamma) is exactly zero,
        # but it does come up when overfitting to one sample
        log_one_minus_gamma = jnp.where( log_one_minus_gamma != 0.0,
                                         log_one_minus_gamma,
                                         jnp.log(SMALLEST_FLOAT32) )

        log_gamma = log_one_minus_x(log_one_minus_gamma)
        log_one_minus_beta = log_one_minus_x(log_beta)
            
        
        ### final dictionary
        out_dict = {'log_lam':log_lam,
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
    
    
class JointTKF91TransitionLogprobs(CondTKF91TransitionLogprobs):
    """
    inherit setup from CondTKF91TransitionLogprobs
    """
    config: dict
    name: str
    
    def __call__(self,
                 t_array,
                 sow_intermediates: bool):
        if not self.load_tkf_params:
            out = self.logits_to_indel_rates(tkf_logits = self.tkf_logits,
                                             lam_min_val = self.lam_min_val,
                                             lam_max_val = self.lam_max_val,
                                             offs_min_val = self.offs_min_val,
                                             offs_max_val = self.offs_max_val,
                                             tkf_err = self.tkf_err )
            lam, mu, use_approx = out
            del out
        
        elif self.load_tkf_params:
            lam = self.tkf_params[0]
            mu = self.tkf_params[0]
            use_approx = False
        
        # get alpha, beta, gamma
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
        
        
        
        # get alpha, beta, gamma
        out_dict = self.tkf_params(lam = lam_mu[...,0], 
                                   mu = lam_mu[...,1], 
                                   t_array = t_array,
                                   use_approx = use_approx)
        
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
        
        #(T, 1, 4, 4)
        out = jnp.stack([ jnp.stack([log_a_f, log_b_g, log_c_h, log_mis_e], axis=-1),
                           jnp.stack([log_a_f, log_b_g, log_c_h, log_mis_e], axis=-1),
                           jnp.stack([  log_p,   log_q,   log_r,   log_d_e], axis=-1),
                           jnp.stack([log_a_f, log_b_g, log_c_h, log_mis_e], axis=-1)
                          ], axis=-2)
        
        return out[:,None,...]
        
    
class CondTKF92TransitionLogprobs(CondTKF91TransitionLogprobs):
    """
    only one TKF extension probability for independent site classes
    """
    config: dict
    name: str
    
    def setup(self):
        ### unpack config
        self.tkf_err = self.config.get('tkf_err', 1e-4)
        self.load_tkf_params = self.config['load_tkf_params']
        self.num_tkf_site_classes = self.config['num_tkf_site_classes']
        
        if self.load_tkf_params:
            self.tkf_params_file = self.config['tkf_params_file']
            with open(self.tkf_params_file, 'rb') as f:
                self.tkf_params = jnp.load(f)
                self.use_approx = False
        
        elif not self.load_tkf_params:
            self.lam_min_val, self.lam_max_val = self.config.get( 'lambda_range', 
                                                                   [self.tkf_err, 3] )
            self.offs_min_val, self.offs_max_val = self.config.get( 'offset_range', 
                                                                    [self.tkf_err, 0.333] )
            self.r_extend_min_val, self.r_extend_max_val = self.config.get( 'r_range', 
                                                                    [self.tkf_err, 0.8] )
            
            ### initialize logits
            self.tkf_logits = self.param( 'TKF92 lambda, mu, r_extend',
                                          nn.initializers.glorot_uniform(),
                                          (self.num_tkf_site_classes, 3),
                                          jnp.float32 )
            
            
    def __call__(self,
                 t_array,
                 sow_intermediates: bool):
        if not self.load_tkf_params:
            out = self.logits_to_indel_rates(tkf_logits = self.tkf_logits,
                                             lam_min_val = self.lam_min_val,
                                             lam_max_val = self.lam_max_val,
                                             offs_min_val = self.offs_min_val,
                                             offs_max_val = self.offs_max_val,
                                             tkf_err = self.tkf_err )
            lam, mu, use_approx = out
            del out
            r_extend = bounded_sigmoid(x = self.tkf_logits[:,-1],
                                       min_val = r_extend_min_val,
                                       max_val = r_extend_max_val)
        
        elif self.load_tkf_params:
            lam = self.tkf_params[...,0][None,:]
            mu = self.tkf_params[...,1][None,:]
            r_extend = self.tkf_params[...,2][None,:]
            use_approx = False
        
        # get alpha, beta, gamma
        out_dict = self.tkf_params(lam = lam, 
                                   mu = mu, 
                                   t_array = t_array,
                                   use_approx = use_approx,
                                   tkf_err = self.tkf_err)
        out_dict = {key: val[...,None] for key, val in out_dict}
        
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
        
        ### entries in the matrix
        log_r_extend = safe_log(r_extend)[None,...]
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
    

class JointTKF92TransitionLogprobs(CondTKF92TransitionLogprobs):
    """
    only one TKF extension probability for independent site classes
    
    inherit setup from  CondTKF92TransitionLogprobs
    """
    config: dict
    name: str
    
    def __call__(self,
                 t_array,
                 sow_intermediates: bool):
        if not self.load_tkf_params:
            out = self.logits_to_indel_rates(tkf_logits = self.tkf_logits,
                                             lam_min_val = self.lam_min_val,
                                             lam_max_val = self.lam_max_val,
                                             offs_min_val = self.offs_min_val,
                                             offs_max_val = self.offs_max_val,
                                             tkf_err = self.tkf_err )
            lam, mu, use_approx = out
            del out
            r_extend = bounded_sigmoid(x = self.tkf_logits[:,-1],
                                       min_val = r_extend_min_val,
                                       max_val = r_extend_max_val)
        
        elif self.load_tkf_params:
            lam = self.tkf_params[...,0][None,:]
            mu = self.tkf_params[...,1][None,:]
            r_extend = self.tkf_params[...,2][None,:]
            use_approx = False
        
        # get alpha, beta, gamma
        out_dict = self.tkf_params(lam = lam, 
                                   mu = mu, 
                                   t_array = t_array,
                                   use_approx = use_approx,
                                   tkf_err = self.tkf_err)
        out_dict = {key: val[...,None] for key, val in out_dict}
        
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
        
        ### entries in the matrix
        log_r_extend = safe_log(r_extend)[None,...]
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
        