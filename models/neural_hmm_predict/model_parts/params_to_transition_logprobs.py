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
ModuleBase
NoIndels
 
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



###############################################################################
### START OF MODULE CLASSES   #################################################
###############################################################################
class NoIndels(ModuleBase):
    config: dict
    name: str
    
    @nn.compact
    def __call__(self,
                 indel_params,
                 t_array,
                 sow_intermediates: bool):
        """
        NoIndels assigns no logprob to indels (return a zeros matrix with 
            appropriate sizing)
        
        input sizes:
        -------------
        indel_params: (B, L, 2)
          > L is L_{align}
          
        t_array: (T, B)
        
        
        output sizes:
        -------------
        zeros matrix of size: (T, B, L, 4, 4)
          > L is L_{align}
        
        """
        T = t_array.shape[0]
        B = indel_param_logits.shape[0]
        L = indel_param_logits.shape[1]
        
        return ( jnp.zeros( (T, B, L, 4, 4) ),
                 dict() ) 
    
    
    def concat_transition_matrix(self, 
                                  m_m, m_i, m_d, m_e,
                                  i_m, i_i, i_d, i_e,
                                  d_m, d_i, d_d, d_e,
                                  s_m, s_i, s_d, s_e):
        """
        all parameters should be: (T, B, L)
        
        not used here, but is used in other indel function classes
        """
        return jnp.stack([ jnp.stack([m_m, m_i, m_d, m_e], axis=-1),
                           jnp.stack([i_m, i_i, i_d, i_e], axis=-1),
                           jnp.stack([d_m, d_i, d_d, d_e], axis=-1),
                           jnp.stack([s_m, s_i, s_d, s_e], axis=-1)
                          ], axis=-2)
    

class CondTKF91TransitionLogprobs(NoIndels):
    """
    inherit generate_log_transition_matrix() from NoIndels
    (no parameters to train, but ModuleBase allows writing to tensorboard)
    
    purpose:
    --------
    evolutionary parameters (from neural network) -> 
        logprob(transitions)
    
    use the TKF91 indel model
    
    
    input sizes:
    -------------
    lam_mu: (B, L, 2)
    t_array: (T, B)
    
    
    output sizes:
    -------------
    logprob_trans: (T, B, L_{align}, 4, 4)
    
    """
    config: dict
    name: str
    
    def setup(self):
        self.tkf_err = self.config.get('tkf_err', 1e-4)
        
        
    def __call__(self,
                 lam_mu,
                 use_approx,
                 t_array,
                 sow_intermediates: bool,
                 **kwargs):
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
        
        # logprob_trans is (T, B, L, 4, 4)
        logprob_trans = self.concat_transition_matrix(m_m = log_a_f, 
                                                      m_i = log_b_g,
                                                      m_d = log_c_h, 
                                                      m_e = log_mis_e,
                                                      
                                                      i_m = log_a_f, 
                                                      i_i = log_b_g, 
                                                      i_d = log_c_h, 
                                                      i_e = log_mis_e,
                                                      
                                                      d_m = log_p, 
                                                      d_i = log_q, 
                                                      d_d = log_r, 
                                                      d_e = log_d_e,
                                                      
                                                      s_m = log_a_f, 
                                                      s_i = log_b_g, 
                                                      s_d = log_c_h, 
                                                      s_e = log_mis_e)
        
        if sow_intermediates:
            self.sow_histograms_scalars(mat=logprob_trans, 
                                        label=f'{self.name}/logprob_trans', 
                                        which='scalars')
        
        
        ### when debugging, also output all intermediate calculations
        intermeds = {'FPO_lam': lam,
                     'FPO_mu': mu}
        for key, value in out_dict.items():
            intermeds[f'FPO_{key}'] = value
        return (logprob_trans, intermeds)
    
    
    def tkf_params(self,
                   lam,
                   mu,
                   t_array,
                   use_approx):
        ### lam * t, mu * t
        mu_per_t = jnp.einsum('bl,tb->tbl', mu, t_array) #(T, B, L)
        lam_per_t = jnp.einsum('bl,tb->tbl', lam, t_array) #(T, B, L)
        
        ### log(lam), log(mu)
        log_lam = jnp.broadcast_to( safe_log(lam)[None,:,:],
                                    lam_per_t.shape ) #(T, B, L)
        log_mu = jnp.broadcast_to( safe_log(mu)[None, :, :],
                                   mu_per_t.shape) #(T, B, L)
        
        
        ### alpha and one minus alpha IN LOG SPACE
        # alpha = jnp.exp(-mu_per_t); log(alpha) = -mu_per_t
        log_alpha = -mu_per_t
        log_one_minus_alpha = log_one_minus_x(log_alpha)
        
        
        ### beta
        # COME BACK HERE
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
            return ( safe_log(1 - self.tkf_err) + 
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
    
        # ### add these entries if debugging and want to see intermediate values
        # # 'gamma_numerator': gamma_numerator,
        # # 'gamma_denom': gamma_denom
        
        # return out_dict

class JointTKF91TransitionLogprobs(CondTKF91TransitionLogprobs):
    """
    CondTKF91TransitionLogprobs but now with probability of ancestor
    
    
    input sizes:
    -------------
    lam_mu: (B, L, 2)
    t_array: (T, B)
    
    
    output sizes:
    -------------
    logprob_trans: (T, B, L_{align}, 4, 4)
    
    """
    config: dict
    name: str
    
    def __call__(self,
                 lam_mu,
                 use_approx,
                 t_array,
                 sow_intermediates: bool,
                 **kwargs):
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
        
        # logprob_trans is (T, B, L, 4, 4)
        logprob_trans = self.concat_transition_matrix(m_m = log_a_f, 
                                                      m_i = log_b_g,
                                                      m_d = log_c_h, 
                                                      m_e = log_mis_e,
                                                      
                                                      i_m = log_a_f, 
                                                      i_i = log_b_g, 
                                                      i_d = log_c_h, 
                                                      i_e = log_mis_e,
                                                      
                                                      d_m = log_p, 
                                                      d_i = log_q, 
                                                      d_d = log_r, 
                                                      d_e = log_d_e,
                                                      
                                                      s_m = log_a_f, 
                                                      s_i = log_b_g, 
                                                      s_d = log_c_h, 
                                                      s_e = log_mis_e)
        
        if sow_intermediates:
            self.sow_histograms_scalars(mat=logprob_trans, 
                                        label=f'{self.name}/logprob_trans', 
                                        which='scalars')
        
        
        ### when debugging, also output all intermediate calculations
        intermeds = {'FPO_lam': lam_mu[...,0],
                     'FPO_mu': lam_mu[...,1]}
        for key, value in out_dict.items():
            intermeds[f'FPO_{key}'] = value
        return (logprob_trans, intermeds)        



    
class CondTKF92TransitionLogprobs(CondTKF91TransitionLogprobs):
    """
    inherit tkf_params() and logits_to_indel_rates() 
        from CondTKF91TransitionLogprobs 
    inherit generate_log_transition_matrix() from NoIndels 
    
    (no parameters to train, but ModuleBase allows writing to tensorboard)
    
    purpose:
    --------
    evolutionary parameters (from neural network) -> 
        logprob(transitions)
    
    use the TKF92 indel model
    
    
    input sizes:
    -------------
    lam_mu: (B, L, 2)
    r: (B,L)
      > L is L_{align}
      
    t_array: (T, B)
    
    
    output sizes:
    -------------
    logprob_trans: (T, B, L, 4, 4)
      > L is L_{align}
    
    """
    config: dict
    name: str
    
    def __call__(self,
                 lam_mu,
                 r_extend,
                 use_approx,
                 t_array,
                 sow_intermediates: bool):
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
        # broadcast up from (B,L) -> (T,B,L)
        T = t_array.shape[0]
        B = max( [ t_array.shape[1], r_extend.shape[0] ] )
        L = r_extend.shape[1]
        r_extend = jnp.broadcast_to( r_extend[None, :, :],
                                           (T,B,L) )
        del T, B, L
        
        # need log(r_extend) and log(1 - r_extend) for this
        log_r_extend = safe_log(r_extend)
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
        
        # final mat is (T, B, L, 4, 4)
        logprob_trans = self.concat_transition_matrix(m_m = log_a, 
                                                      m_i = log_b,
                                                      m_d = log_c_h, 
                                                      m_e = log_mi_e,
                                                      
                                                      i_m = log_f, 
                                                      i_i = log_g, 
                                                      i_d = log_c_h, 
                                                      i_e = log_mi_e,
                                                      
                                                      d_m = log_p, 
                                                      d_i = log_q, 
                                                      d_d = log_r, 
                                                      d_e = log_d_e,
                                                      
                                                      s_m = log_s_m, 
                                                      s_i = log_s_i, 
                                                      s_d = log_s_d, 
                                                      s_e = log_s_e)
        
        if sow_intermediates:
            self.sow_histograms_scalars(mat=logprob_trans, 
                                        label=f'{self.name}/logprob_trans', 
                                        which='scalars')
        
        ## when debugging, output more stuff
        intermeds = {'FPO_lam': lam,
                      'FPO_mu': mu}
        for key, value in out_dict.items():
            intermeds[f'FPO_{key}'] = value
        return (logprob_trans, intermeds)


class JointTKF92TransitionLogprobs(CondTKF92TransitionLogprobs):
    """
    TKF92 but with joint probability
    
    
    input sizes:
    -------------
    lam_mu: (B, L, 2)
    r: (B,L)
      > L is L_{align}
      
    t_array: (T, B)
    
    
    output sizes:
    -------------
    logprob_trans: (T, B, L, 4, 4)
      > L is L_{align}
    
    """
    config: dict
    name: str
    
    def __call__(self,
                 lam_mu,
                 r_extend,
                 use_approx,
                 t_array,
                 sow_intermediates: bool):
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
        # broadcast up from (B,L) -> (T,B,L)
        T = t_array.shape[0]
        B = max( [ t_array.shape[1], r_extend.shape[0] ] )
        L = r_extend.shape[1]
        r_extend = jnp.broadcast_to( r_extend[None, :, :],
                                           (T,B,L) )
        del T, B, L
        
        # need some extra values pre-computed
        log_r_extend = safe_log(r_extend)
        log_one_minus_r_extend = log_one_minus_x(log_r_extend)
        log_lam_div_mu = out_dict['log_lam'] - out_dict['log_mu']
        log_one_minus_lam_div_mu = log_one_minus_x(log_lam_div_mu)
        
        
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
        
        # final mat is (T, B, L, 4, 4)
        logprob_trans = self.concat_transition_matrix(m_m = log_a, 
                                                      m_i = log_b,
                                                      m_d = log_c_h, 
                                                      m_e = log_mi_e,
                                                      
                                                      i_m = log_f, 
                                                      i_i = log_g, 
                                                      i_d = log_c_h, 
                                                      i_e = log_mi_e,
                                                      
                                                      d_m = log_p, 
                                                      d_i = log_q, 
                                                      d_d = log_r, 
                                                      d_e = log_d_e,
                                                      
                                                      s_m = log_s_m, 
                                                      s_i = log_s_i, 
                                                      s_d = log_s_d, 
                                                      s_e = log_s_e)
        
        if sow_intermediates:
            self.sow_histograms_scalars(mat=logprob_trans, 
                                        label=f'{self.name}/logprob_trans', 
                                        which='scalars')
        
        ## when debugging, output more stuff
        intermeds = {'FPO_lam': lam,
                      'FPO_mu': mu}
        for key, value in out_dict.items():
            intermeds[f'FPO_{key}'] = value
        return (logprob_trans, intermeds)
    
    

class TransitionLogprobsFromFile(ModuleBase):
    config: dict
    name: str
    
    def setup(self):
        load_from_file = self.config['load_from_file']
        
        with open(load_from_file, 'rb') as f:
            self.logprob_trans = jnp.load(f)
        
        if len(self.logprob_trans.shape) == 2:
            self.logprob_trans = self.logprob_trans[None, None, ...]
            self.expand_dims = True
        
        else:
            self.expand_dims = False
    
    def __call__(self,
                 t_array,
                 sow_intermediates: bool=False,
                 **kwargs):
        logprob_trans = self.logprob_trans
        
        if self.expand_dims:
            new_shape = ( t_array.shape[0], #T
                          logprob_trans.shape[0], #B=1
                          logprob_trans.shape[1], #L=1
                          logprob_trans.shape[2], #num_transits (4 for tkf)
                          logprob_trans.shape[3] ) #num_transits (4 for tkf)
            logprob_trans = jnp.broadcast_to( logprob_trans, new_shape )
        
        return logprob_trans
