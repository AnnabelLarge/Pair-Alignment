#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:08:14 2025

@author: annabel

Todo: 
-----
- merge all vmap + jax.lax.conds into one vmap function

"""
import jax 
from jax import numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import rc
plt.rcParams["font.family"] = 'Optima'
plt.rcParams["font.size"] = 14
plt.rc('axes', unicode_minus=False)


# make this slightly more than true jnp.finfo(jnp.float32).eps, 
#  for numerical safety at REALLY small parameter values
SMALL_POSITIVE_NUM = 5e-7


### helpers
def logsumexp_with_arr_lst(array_of_log_vals, coeffs = None):
    """
    concatenate a list of arrays, then use logsumexp
    """
    a_for_logsumexp = jnp.stack(array_of_log_vals, axis=-1)
    out = logsumexp(a = a_for_logsumexp,
                    b = coeffs,
                    axis=-1)
    return out

def log_one_minus_x(log_x):
    """
    calculate log( exp(log(1)) - exp(log(x)) ), which is log( 1 - x )
    """
    return jnp.log1p( -jnp.exp(log_x) )

def log_x_minus_one(log_x):
    """
    calculate log( exp(log(x)) - exp(log(1)) ), which is log( x - 1 )
    """
    return jnp.log( jnp.expm1(log_x) )

# def clip_log_val(log_x):
#     """
#     this kills gradients, but hard-clip and log(x) values such that:
#         0.0 < x < 1.0
#         lower_lim < log(x) < upper_lim
#     """
#     upper_lim = -jnp.finfo(jnp.float32).eps
#     lower_lim = jnp.log(jnp.finfo(jnp.float32).eps)
    
#     log_x = jnp.minimum(log_x, upper_lim)
#     log_x = jnp.maximum(log_x, lower_lim)
    
#     return log_x


### reference function
def TKF_coeffs(lam, mu, t):
    alpha = np.exp(-mu*t)
    beta = (lam*(np.exp(-lam*t)-np.exp(-mu*t))) / (mu*np.exp(-lam*t)-lam*np.exp(-mu*t))
    gamma = 1 - ( (mu * beta) / ( lam * (1-alpha) ) )
    
    out = {'log_alpha': np.log(alpha),
           'log_beta': np.log(beta),
           'log_gamma': np.log(gamma),
           'log_one_minus_alpha': np.log(1 - alpha), 
           'log_one_minus_beta': np.log(1 - beta), 
           'log_one_minus_gamma': np.log(1 - gamma)}
    return out

def TKF_coeffs_float32 (lam, mu, t):
    alpha = jnp.exp(-mu*t)
    beta = (lam*(jnp.exp(-lam*t)-jnp.exp(-mu*t))) / (mu*jnp.exp(-lam*t)-lam*jnp.exp(-mu*t))
    gamma = 1 - ( (mu * beta) / ( lam * (1-alpha) ) )
    out = {'log_alpha': jnp.log(alpha),
           'log_beta': jnp.log(beta),
           'log_gamma': jnp.log(gamma),
           'log_one_minus_alpha': jnp.log(1 - alpha), 
           'log_one_minus_beta': jnp.log(1 - beta), 
           'log_one_minus_gamma': jnp.log(1 - gamma)}
    return out

def tkf_alpha( mu, t_array ):
    """
    alpha = exp(-mu*t)
    log(alpha) = -mu*t
    """
    
    def approx_one_minus_alpha(log_alpha):
        return jax.lax.cond( log_alpha < -SMALL_POSITIVE_NUM,
                             log_one_minus_x,
                             lambda x: jnp.log(-x),
                             log_alpha)
    vmapped_approx_one_minus_alpha = jax.vmap(approx_one_minus_alpha, in_axes=0)
    
    # original formula
    log_alpha = -mu*t_array
    
    # may or may not use approximation
    log_one_minus_alpha = vmapped_approx_one_minus_alpha( log_alpha )
    
    out = {'log_alpha': log_alpha,
           'log_one_minus_alpha': log_one_minus_alpha}
    
    return out

def tkf_beta( mu, 
              offset, 
              t_array ):
    def true_beta(oper):
        """
        the true formula for beta, assuming mu = lambda * (1 - offset)
        """
        mu, offset, t = oper
        
        # log( (1 - offset) * (exp(mu*offset*t) - 1) )
        log_num = jnp.log1p(-offset) + log_x_minus_one( mu*offset*t )
        
        # x = mu*offset*t
        # y = jnp.log( 1 - offset )
        # logsumexp with coeffs does: 
        #   log( exp(x) - exp(y) ) = log( exp(mu*offset*t) - (1-offset) )
        log_denom = logsumexp_with_arr_lst( [mu*offset*t, jnp.log1p(-offset)],
                                        coeffs = jnp.array([1.0, -1.0]) )
        
        log_beta = log_num - log_denom
        log_one_minus_beta = log_one_minus_x(log_x = log_beta)
        out = {'log_beta': log_beta,
                'log_one_minus_beta': log_one_minus_beta}
        return out

    def approx_beta(oper):
        """
        as lambda approaches mu (or as time shrinks to small values), use 
          first order taylor approximation
        """
        mu, offset, t = oper
        
        # log(  (1 - offset) * mu * t  )
        log_num = jnp.log1p(-offset) + jnp.log(mu) + jnp.log(t)
        
        # log( mu*t + 1 )
        log_denom = jnp.log1p( mu * t )
        
        log_beta = log_num - log_denom
        log_one_minus_beta = log_one_minus_x(log_x = log_beta)
        
        # return
        out = {'log_beta': log_beta,
               'log_one_minus_beta': log_one_minus_beta}
        return out

    def safe_calc_beta(mu, 
                       offset, 
                       t):
        """
        use jax.lax.cond to guard against infs and nans in the gradient
        """
        return jax.lax.cond( mu*offset*t > SMALL_POSITIVE_NUM ,
                             true_beta,
                             approx_beta,
                             (mu, offset, t) )

    vmapped_safe_calc_beta = jax.vmap( safe_calc_beta,
                                       in_axes=(None,None,0) )
    
    return vmapped_safe_calc_beta(mu, offset, t_array)
    

def tkf_gamma( log_one_minus_alpha,
               mu, 
               offset, 
               t_array ):
    def approx_gamma(oper):
        mu, offset, t = oper
        
        # log( 1 + 0.5*mu*offset*t )
        log_num = jnp.log1p( 0.5 * mu * offset * t )
        
        # log( (1 - 0.5*mu*t) (mu*t + 1) )
        # there's another squared term here:
        #   0.5 * offset * (mu*t)**2
        # but it's so small that it's negligible
        log_denom = jnp.log1p( -0.5*mu*t ) + jnp.log1p( mu*t )
        
        return log_num - log_denom
        
        ### uncomment for first-order approx, to show this is bad
        # return -jnp.log1p( mu*t )
        
    
    def approx_one_minus_gamma(log_one_minus_gamma):
        return jax.lax.cond( log_one_minus_gamma < -SMALL_POSITIVE_NUM,
                             log_one_minus_x,
                             lambda x: jnp.log(-x),
                             log_one_minus_gamma)

    def gamma_elementwise_safe_calc(log_n, 
                                    log_d, 
                                    t, 
                                    mu, 
                                    offset):
        """
        do standard num / denom in log space if num < denom   AND   one of the 
          following is true:
            1.) mu*offset*t > 1e-3
                > it's too large to be approximated with taylor expansion
        
            2.) mu*offset*t < 1e-3   AND   jnp.abs(log_n - log_d) > 0.1
                > it's a small operand, but the difference between log(num) 
                  and log(denom) is stable enough
        
            3.) mu*offset*t < 1e-3   AND   jnp.abs(log_n - log_d) < 0.1  AND  
                (0.5*mu*t) < 1.0
                > it's a small operand and the difference between log(num) 
                  and log(denom) isn't large enough, but if you try to use 
                  the approximation formula, it will fail. So don't.
        
        use second-order taylor approximation formula otherwise. I drop the 
          extra (mu*t)**2 term, because it's pretty insignificant.
        """
        ### complex conditions for using the real gamma function
        ### TODO: this is all kind of ad-hoc...
        valid_frac = log_n < log_d
        large_product = mu * offset * t > 1e-3
        log_diff_large = jnp.abs(log_n - log_d) > 0.1
        approx_formula_will_fail = (0.5*mu*t) > 1.0
        
        cond1 = large_product
        cond2 = ~large_product & log_diff_large
        cond3 = ~large_product & ~log_diff_large & approx_formula_will_fail
        use_real_function = valid_frac & ( cond1 | cond2 | cond3 )
        
        log_one_minus_gamma = jax.lax.cond( use_real_function,
                                            lambda _: log_n - log_d,
                                            approx_gamma,
                                            (mu, offset, t) )
            
        log_gamma = approx_one_minus_gamma(log_one_minus_gamma)
        
        out = {'log_gamma': log_gamma,
               'log_one_minus_gamma': log_one_minus_gamma,
               'gamma_approx': ~use_real_function}
        return out
    
    vmapped_gamma_elementwise_safe_calc = jax.vmap( gamma_elementwise_safe_calc,
                                              in_axes=(0, 0, 0, None, None) )

    # log( exp(mu*offset*t) - 1 )
    log_num = log_x_minus_one( log_x = mu*offset*t_array )
    
    # x = mu*offset*t
    # y = jnp.log( 1 - offset )
    # logsumexp with coeffs does: 
    #   log( exp(x) - exp(y) ) = log( exp(mu*offset*t) - (1-offset) )
    constant = jnp.broadcast_to(jnp.log1p(-offset), t_array.shape)
    log_denom_term1 = logsumexp_with_arr_lst( [mu*offset*t_array, constant],
                                              coeffs = jnp.array([1.0, -1.0]) )
    log_denom = log_denom_term1 + log_one_minus_alpha
    
    return vmapped_gamma_elementwise_safe_calc(log_num, 
                                                log_denom, 
                                                t_array, 
                                                mu, 
                                                offset)

def my_approx( mu, offset, t ):
    out = tkf_alpha(mu, t)
    log_one_minus_alpha = out['log_one_minus_alpha']
    
    to_add = tkf_beta( mu, offset, t )
    out = {**out, **to_add}
    
    to_add = tkf_gamma( log_one_minus_alpha, mu, offset, t )
    out = {**out, **to_add}
    
    return out
    

def main(lam, offset, times_file):
    ####################
    ### other params   #
    ####################
    mu = lam / (1-offset)

    with open(times_file,'rb') as f:
        t_array = np.load(f) #(T,)
    t_array = np.sort(t_array)
    del f
    
    out_plot_file = f'PLOT_lam_{lam}_offs_{offset}_{times_file.replace(".npy","")}.png'
    
    
    #################
    ### calculate   #
    #################
    # in float64
    true_float64_vals = TKF_coeffs( lam=lam, 
                                    mu=mu, 
                                    t=t_array )
    
    # float32 with original formula
    float32_values = TKF_coeffs_float32( lam=jnp.array(lam), 
                                         mu=jnp.array(mu), 
                                         t=jnp.array(t_array) )
    
    
    # float32, with approximations and safeguards
    approx_values = my_approx( mu=jnp.array(mu), 
                               offset=jnp.array(offset),
                               t=jnp.array(t_array) )
    
    
    ################
    ### validate   #
    ################
    for varname in ['alpha', 'beta', 'gamma']:
        # arrays should be valid
        assert jnp.isnan(approx_values[f'log_{varname}']).sum() == 0, f'log_{varname}'
        assert jnp.isinf(approx_values[f'log_{varname}']).sum() == 0, f'log_{varname}'
        assert (jnp.exp( approx_values[f'log_{varname}'] ) <= 1.0).all(), f'log_{varname}'
        assert (jnp.exp( approx_values[f'log_{varname}'] ) >= 0.0).all(), f'log_{varname}'
        
        assert jnp.isnan(approx_values[f'log_one_minus_{varname}']).sum() == 0, f'log_one_minus_{varname}'
        assert jnp.isinf(approx_values[f'log_one_minus_{varname}']).sum() == 0, f'log_one_minus_{varname}'
        assert (jnp.exp( approx_values[f'log_one_minus_{varname}'] ) <= 1.0).all(), f'log_one_minus_{varname}'
        assert (jnp.exp( approx_values[f'log_one_minus_{varname}'] ) >= 0.0).all(), f'log_one_minus_{varname}'
        
        # sums should be 1 (check in log space)
        log_checksum = logsumexp( jnp.stack([ approx_values[f'log_{varname}'],
                                              approx_values[f'log_one_minus_{varname}'] ],
                                            axis=-1),
                                  axis = -1)
        assert jnp.isclose(log_checksum, jnp.zeros(log_checksum.shape), atol=1e-6).all()
        
        
    ############
    ### plot   #
    ############
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 12), sharex='col')
    axs_lst = axs.flatten()
    name_mapping = {'log_alpha': 'log(alpha)',
                    'log_beta': 'log(beta)',
                    'log_gamma': 'log(gamma)',
                    'log_one_minus_alpha': 'log(1-alpha)',
                    'log_one_minus_beta': 'log(1-beta)',
                    'log_one_minus_gamma': 'log(1-gamma)'}
    
    for i, key in enumerate(true_float64_vals.keys()):
        ax = axs_lst[i]
        true = true_float64_vals[key]
        true_float32 = float32_values[key]
        approx = approx_values[key]
        
        ax.plot(t_array, 
                true, 
                '.', 
                label='True (float64)', 
                color='red', 
                zorder=3,
                alpha=0.75)
        
        ax.plot(t_array, 
                true_float32, 
                '.', 
                label='True (float32)', 
                color='salmon', 
                zorder=1,
                alpha=0.75)
        
        ax.plot(t_array, 
                approx, 
                '--', 
                linewidth=4, 
                color='darkcyan', 
                label='My Function (float32)', 
                zorder=2)
        
        ax.grid(zorder=0)
        ax.set_xscale('log')
        ax.set_title(name_mapping[key])
        ax.legend()
    
    
    # Add a shared title
    fig.suptitle( (f'TKF Parameter Approximation\n'+
                    f'lambda: {lam}, mu: {mu}\n'+
                    f'times from: {times_file}'), fontsize=20)
    axs[1,1].set_xlabel('log(time)')
    axs[0,0].set_ylabel('log(Parameter value)')
    axs[1,0].set_ylabel('log(Parameter value)')
    fig.tight_layout(rect=[0, 0, 1, 0.95])  
    fig.savefig(out_plot_file)


if __name__ == '__main__':
    ### params to change
    times_file = 'cherryML_times.npy'
    # times_file = 'observed_times.npy'
    
    for lam in [1e-4, 0.06, 1.1]:
        for offset in [1e-4, 0.333]:
            main(lam, offset, times_file)
            