#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:42:28 2024

@author: annabel

These are SITE AND FRAGMENT LEVEL transition models

todo: domain-level transition models (nested TKF92)


modules:
=========

originals:
------------
'GeomLenTransitionLogprobs',
'TKF91TransitionLogprobs',
'TKF92TransitionLogprobs',

loading from files:
--------------------
'GeomLenTransitionLogprobsFromFile',
'TKF91TransitionLogprobsFromFile',
'TKF92TransitionLogprobsFromFile',

"""
# jumping jax and leaping flax
from jax.core import Tracer
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import pickle

from models.BaseClasses import ModuleBase
from models.simple_site_class_predict.model_functions import (bound_sigmoid,
                                                              safe_log,
                                                              logsumexp_with_arr_lst,
                                                              log_one_minus_x,
                                                              switch_tkf,
                                                              regular_tkf,
                                                              approx_tkf,                                                              
                                                              get_tkf91_single_seq_marginal_transition_logprobs,
                                                              get_tkf92_single_seq_marginal_transition_logprobs,
                                                              get_cond_transition_logprobs)

###############################################################################
### Geometric sequence length (no indels)   ###################################
###############################################################################
class GeomLenTransitionLogprobs(ModuleBase):
    """
    Simply assume sequence has geometrically-distributed sequence length
    
    Used mostly for debugging; doesn't have mixtures
    
    
    Initialize with
    ----------------
    config : dict (but nothing used here)
            
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
        
        Flax Module Parameters
        -----------------------
        p_emit_logit: ArrayLike (1,)
            P(emit at alignment column) = 1 - P(end alignment)
        
        """
        # under standard sigmoid activation, initial value is: 0.95257413
        init_logit = jnp.array([3.0])
        self.p_emit_logit = self.param('p_emit_logit',
                                       lambda rng, shape, dtype: init_logit,
                                       init_logit.shape,
                                       jnp.float32)
        
        # no domain or fragment mixtures possible
        self.log_frag_class_probs = jnp.zeros( (1,1) ) #(C_dom, C_frag)
        
    def __call__(self,
                 return_all_matrices: bool,
                 sow_intermediates: bool,                 
                 *args,
                 **kwargs):
        """
        
        Arguments
        ----------
        return_all_matrices : bool
            if false, only return the joint (used for model training)
            if true, return joint, conditional, and marginal matrices
        
        sow_intermediates : bool
            switch for tensorboard logging
          
        Returns
        -------
        log_fragment_class_probs : (0, 0)
            placeholder values
            
        matrix_dict : dict
            matrix_dict["joint"]: (2,)
                score transitions in joint probability calculation
                
            (if return_all_matrices) matrix_dict["marginal"]: (2,)
                score transitions in marginal probability calculation
            
            (if return_all_matrices) matrix_dict["conditional"]: (2,)
                score transitions in conditional probability calculation
            
            (if return_all_matrices) matrix_dict["log_corr"]: 0
                placeholder
        
        (output tuple) :  None
            placeholder values
        """
        p_emit = nn.sigmoid(self.p_emit_logit) #(1,)
        
        if sow_intermediates:
            self.sow_histograms_scalars(mat= p_emit, 
                                        label=f'{self.name}/geom_prob_emit', 
                                        which='scalars')
        
        out_vec = safe_log( jnp.concatenate( [p_emit, 1-p_emit] ) ) #(2,)
        
        if not return_all_matrices:
            matrix_dict = {'joint': out_vec}
        
        elif return_all_matrices:
            matrix_dict = {'joint': out_vec,
                           'marginal': out_vec,
                           'conditional': out_vec,
                           'log_corr': 0}
            
        return (self.log_frag_class_probs, matrix_dict, None)
        

class GeomLenTransitionLogprobsFromFile(GeomLenTransitionLogprobs):
    """
    same as GeomLenTransitionLogprobs, but load parameter from file
    
    
    Initialize with
    ----------------
    config : dict 
        config["filenames"]["geom_length_params_file"] : str
            contains probability of emission; could either be a 1-element 
            numpy array or a flat text file
            
    name : str
        class name, for flax
    
    Methods here
    ------------
    __call__
    setup
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        
        Flax Module Parameters
        -----------------------
        None
        
        """
        file_with_transit_probs = self.config['filenames']['geom_length_params_file']
        
        if file_with_transit_probs.endswith('.npy'):
            with open(file_with_transit_probs,'rb') as f:
                vec = jnp.load(f) #(2,)
        p_emit = vec[0]
        one_minus_p_emit = vec[1]
        self.out_vec = safe_log( jnp.array( [p_emit, one_minus_p_emit] ) )
        
        # no domain or fragment mixtures possible
        self.log_frag_class_probs = jnp.zeros( (1,1) ) #(C_dom, C_frag)
        
    
    def __call__(self,
                 *args,
                 **kwargs):
        """
        
        Arguments
        ----------
        None
          
        Returns
        -------
        out_dict : dict
            out_dict["joint"]: (2,)
                score transitions in joint probability calculation
                
            out_dict["marginal"]: (2,)
                score transitions in marginal probability calculation
            
            out_dict["conditional"]: (2,)
                score transitions in conditional probability calculation
        
        (output tuple) : None
            placeholder values
        """
        if not return_all_matrices:
            matrix_dict = {'joint': self.out_vec}
        
        elif return_all_matrices:
            matrix_dict = {'joint': self.out_vec,
                           'marginal': self.out_vec,
                           'conditional': self.out_vec,
                           'log_corr': 0}
            
        return (self.log_frag_class_probs, matrix_dict, None)
    
    
###############################################################################
### TKF91   ###################################################################
###############################################################################
class TKF91TransitionLogprobs(ModuleBase):
    """
    TKF91 model; used for calculating transitions in model of
        P(anc, desc, align)
    
    C_dom: number of mixtures associated with nested TKF92 models (domain-level)
    C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
        > for tkf91, there can NOT be mixtures over transitions (i.e. C_frag=1)
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    S: number of transition states (4 here: M, I, D, start/end)
        
    Initialize with
    ----------------
    config : dict 
        config["tkf_function"] : {'regular_tkf','approx_tkf','switch_tkf'}
            which function to use to solve for tkf parameters

        config["mu_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines lamdba
            DEFAULT: -1e-4, 2
        
        config["offset_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines offset 
            (which determines mu)
            DEFAULT: -1e-4, 0.333
            
    name : str
        class name, for flax
    
    
    Methods here
    ------------
    setup
    
    __call__
    
    fill_joint_tkf91
        fills in joint TKF91 transition matrix
        
    _logits_to_indel_rates
        converts mu/offset logits to mu/offset values
    
    return_all_matrices
        return transition matrices used for joint, marginal, and conditional 
        log-likelihood transitions
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
            > for tkf91, there can NOT be mixtures over transitions (i.e. C_frag=1)
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
            
            
        Flax Module Parameters
        -----------------------
        tkf_mu_offset_logits: ArrayLike (2,)
            first value is logit for mu, second is for offset
        
        """
        ### unpack config
        # required
        tkf_function_name = self.config['tkf_function']
        
        # optional
        self.mu_min_val, self.mu_max_val = self.config.get( 'mu_range', [1e-4, 2] )
        self.offs_min_val, self.offs_max_val = self.config.get( 'offset_range', [1e-4, 0.333] )
        
        # no domain or fragment mixtures possible
        self.log_frag_class_probs = jnp.zeros( (1,1) ) #(C_dom, C_frag)
        
        
        ### initialize logits for mu, offset
        self.tkf_mu_offset_logits = self.param('tkf_mu_offset_logits',
                                               nn.initializers.normal(),
                                               (2,),
                                               jnp.float32)
        
        
        ### decide tkf function
        if tkf_function_name == 'regular_tkf':
            self.tkf_function = regular_tkf
        elif tkf_function_name == 'approx_tkf':
            self.tkf_function = approx_tkf
        elif tkf_function_name == 'switch_tkf':
            self.tkf_function = switch_tkf
    
    def __call__(self,
                 t_array,
                 return_all_matrices: bool,
                 sow_intermediates: bool):
        """
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
            > for tkf91, there can NOT be mixtures over transitions (i.e. C_frag=1)
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
           
        
        Arguments
        ----------
        t_array : ArrayLike
            branch lengths, times for marginalizing over
        
        return_all_matrices : bool
            if false, only return the joint (used for model training)
            if true, return joint, conditional, and marginal matrices
        
        sow_intermediates : bool
            switch for tensorboard logging
          
        Returns
        -------
        log_fragment_class_probs : (0, 0)
            placeholder tuple
        
        matrix_dict : dict
            matrix_dict["joint"]: (T,S,S)
                score transitions in joint probability calculation
            
            (if return_all_matrices) matrix_dict["marginal"]: (2,2)
                score transitions in marginal probability calculation
            
            (if return_all_matrices) matrix_dict["conditional"]: (T,S,S)
                score transitions in conditional probability calculation
        
            (if return_all_matrices) matrix_dict["corr"]: 0
                placeholder value
                
        approx_flags_dict : dict
            where approximations are used in tkf formulas
            
            out_dict['log_one_minus_alpha']: (T,)
            
            out_dict['log_beta']: (T,)
            
            out_dict['log_one_minus_gamma']: (T,)
            
            out_dict['log_gamma']: (T,)
            
        """
        # logits -> params
        mu, offset = self._logits_to_indel_rates(mu_offset_logits = self.tkf_mu_offset_logits,
                                         mu_min_val = self.mu_min_val,
                                         mu_max_val = self.mu_max_val,
                                         offs_min_val = self.offs_min_val,
                                         offs_max_val = self.offs_max_val)
        # get alpha, beta, gamma
        # contents of out_dict ( all ArrayLike[float32], (T,) ):
        #   out_dict['log_alpha']
        #   out_dict['log_one_minus_alpha']
        #   out_dict['log_beta']
        #   out_dict['log_one_minus_beta']
        #   out_dict['log_gamma']
        #   out_dict['log_one_minus_gamma']
        #
        # contents of approx_flags_dict ( all ArrayLike[bool], (T,) ):
        #   out_dict['log_one_minus_alpha']
        #   out_dict['log_beta']
        #   out_dict['log_one_minus_gamma']
        #   out_dict['log_gamma']
        out_dict, approx_flags_dict = self.tkf_function(mu = mu, 
                                                        offset = offset,
                                                        t_array = t_array)
        
        # add to these dictionaries before filling out matrix
        out_dict['log_offset'] = jnp.log(offset)
        out_dict['log_one_minus_offset'] = jnp.log1p(-offset)
        
        if approx_flags_dict is not None:
            approx_flags_dict['t_array'] = t_array

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
            
            lam = mu * (1-offset)
            self.sow_histograms_scalars(mat= lam, 
                                        label=f'{self.name}/lam', 
                                        which='scalars')
            del lam
            
            self.sow_histograms_scalars(mat= mu, 
                                        label=f'{self.name}/mu', 
                                        which='scalars')
        
        joint_matrix =  self.fill_joint_tkf91(out_dict) #(T, S, S)
        log_corr = 0
        
        if not return_all_matrices:
            matrix_dict = {'joint': joint_matrix}
        
        elif return_all_matrices:
            # contents of final matrix_dict:
            # matrix_dict['joint'] (T, S, S)
            # matrix_dict['conditional'] (T, S, S)
            # matrix_dict['marginal'] (2, 2)
            # matrix_dict['log_corr'] float
            matrix_dict = self.return_all_matrices(offset=offset,
                                                   joint_matrix=joint_matrix)
            matrix_dict['log_corr'] = 0
        
        return (self.log_frag_class_probs, matrix_dict, approx_flags_dict)
        
    
    def fill_joint_tkf91(self, out_dict):
        """
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
            > for tkf91, there can NOT be mixtures over transitions (i.e. C_frag=1)
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
        
        
        Arguments
        ----------
        out_dict : dict
            contains values for calculating matrix terms: (all in log space)
                - offset = 1 - (lam/mu)
                - alpha, beta, gamma, 1 - alpha, 1 - beta, 1 - gamma
                  
        Returns
        -------
        out : ArrayLike, (T,S,S)
            joint loglike of transitions
        
        """
        ### entries in the matrix
        # lam / mu = 1 - offset
        log_lam_div_mu = out_dict['log_one_minus_offset']
        log_one_minus_lam_div_mu = out_dict['log_offset']
        
        
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
        
        #(T, S, S)
        out = jnp.stack([ jnp.stack([log_a_f, log_b_g, log_c_h, log_mis_e], axis=-1),
                           jnp.stack([log_a_f, log_b_g, log_c_h, log_mis_e], axis=-1),
                           jnp.stack([  log_p,   log_q,   log_r,   log_d_e], axis=-1),
                           jnp.stack([log_a_f, log_b_g, log_c_h, log_mis_e], axis=-1)
                          ], axis=-2)
        return out
    
    
    def return_all_matrices(self,
                            offset,
                            joint_matrix):
        """
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
         
        
        Arguments
        ---------
        offset : float
            1 - (lam/mu)
        
        joint_matrix : ArrayLike, (T, S, S)
        
        
        Returns
        -------
        (returned_dictionary)["joint"]: (T, S, S)
        (returned_dictionary)["marginal"]: (2, 2)
        (returned_dictionary)["conditional"]: (T, S, S)
        
        """
        # output is: (S, S)
        marginal_matrix = get_tkf91_single_seq_marginal_transition_logprobs(offset=offset)
        
        # output is same as joint: (T, S, S)
        cond_matrix = get_cond_transition_logprobs(marg_matrix=marginal_matrix, 
                                             joint_matrix=joint_matrix)
        
        return {'joint': joint_matrix,
                'marginal': marginal_matrix,
                'conditional': cond_matrix}
    
    def _logits_to_indel_rates(self, 
                              mu_offset_logits,
                              mu_min_val,
                              mu_max_val,
                              offs_min_val,
                              offs_max_val):
        """
        Arguments
        ---------
        mu_offset_logits : ArrayLike, (2,)
            logits to transform with bound sigmoid activation
        
        mu_min_val : float
            minimum value for bound sigmoid, to get mu
        
        mu_max_val : float
            maximum value for bound sigmoid, to get mu
        
        offs_min_val : float
            minimum value for bound sigmoid, to get offset
        
        offs_max_val : float
            maximum value for bound sigmoid, to get offset
        
        Returns
        -------
        mu : ArrayLike, ()
            delete rate
        
        offset : ArrayLike, ()
            used to calculate lambda: lambda = mu * (1 - offset)
        
        """
        # mu
        mu = bound_sigmoid(x = mu_offset_logits[0],
                           min_val = mu_min_val,
                           max_val = mu_max_val)
        
        # mu
        offset = bound_sigmoid(x = mu_offset_logits[1],
                               min_val = offs_min_val,
                               max_val = offs_max_val)

        return (mu, offset)
    
    
class TKF91TransitionLogprobsFromFile(TKF91TransitionLogprobs):
    """
    like TKF91TransitionLogprobs, but load values from a file
    
    NOTE: lambda and mu are provided directly, no need for offset
    
    C_dom: number of mixtures associated with nested TKF92 models (domain-level)
    C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
        > for tkf91, there can NOT be mixtures over transitions (i.e. C_frag=1)
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    S: number of transition states (4 here: M, I, D, start/end)
        
        
    Initialize with
    ----------------
    config : dict
        config["tkf_function"] : {'regular_tkf','approx_tkf','switch_tkf'}
            which function to use to solve for tkf parameters
            
        config["filenames"]["tkf_params_file"]
            contains values for lambda, mu
            
    name : str
        class name, for flax
    
    
    Methods here
    ------------
    setup
    __call__
    
    
    Inherited from TKF91TransitionLogprobs
    ---------------------------------------
    fill_joint_tkf91
        fills in joint TKF91 transition matrix
        
    _logits_to_indel_rates
        converts mu/offset logits to mu/offset values
    
    return_all_matrices
        return transition matrices used for joint, marginal, and conditional 
        log-likelihood transitions
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        
        Flax Module Parameters
        -----------------------
        None
        
        """
        # unpack config
        in_file = self.config['filenames']['tkf_params_file']
        tkf_function_name = self.config['tkf_function']
        
        # no domain or fragment mixtures possible
        self.log_frag_class_probs = jnp.zeros( (1,1) ) #(C_dom, C_frag)
        
        # read file
        with open(in_file,'rb') as f:
            self.param_dict = pickle.load(f)
                
        err = f'KEYS SEEN: {self.param_dict.keys()}'
        assert 'lambda' in self.param_dict.keys(), err
        assert 'mu' in self.param_dict.keys(), err
        
        # pick tkf function
        if tkf_function_name == 'regular_tkf':
            self.tkf_function = regular_tkf
        elif tkf_function_name == 'approx_tkf':
            self.tkf_function = approx_tkf
        elif tkf_function_name == 'switch_tkf':
            self.tkf_function = switch_tkf
            
    
    def __call__(self,
                 t_array,
                 return_all_matrices: bool,
                 sow_intermediates: bool):
        """
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
            > for tkf91, there can NOT be mixtures over transitions (i.e. C_frag=1)
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
        
        
        Arguments
        ----------
        t_array : ArrayLike
            branch lengths, times for marginalizing over
        
        return_all_matrices : bool
            if false, only return the joint (used for model training)
            if true, return joint, conditional, and marginal matrices
            
        sow_intermediates : bool
            switch for tensorboard logging
          
        Returns
        -------
        log_fragment_class_probs : (0, 0)
            placeholder tuple
            
        matrix_dict : dict
            matrix_dict["joint"]: (T,S,S)
                score transitions in joint probability calculation
            
            (if return_all_matrices) matrix_dict["marginal"]: (2,2)
                score transitions in marginal probability calculation
            
            (if return_all_matrices) matrix_dict["conditional"]: (T,S,S)
                score transitions in conditional probability calculation
        
            (if return_all_matrices) matrix_dict["corr"]: 0
                placeholder value
        
        None
            placeholder value
        
        """
        lam = self.param_dict['lambda']
        mu = self.param_dict['mu']
        offset = 1 - (lam /mu)
        
        # get alpha, beta, gamma
        out_dict, _ = self.tkf_function(mu = mu, 
                                        offset = offset,
                                        t_array = t_array)
        out_dict['log_offset'] = jnp.log(offset)
        out_dict['log_one_minus_offset'] = jnp.log1p(-offset)
        
        joint_matrix =  self.fill_joint_tkf91(out_dict) #(T, S, S)
        log_corr = 0
        
        if not return_all_matrices:
            matrix_dict = {'joint': joint_matrix}
        
        elif return_all_matrices:
            # contents of final matrix_dict:
            # matrix_dict['joint'] (T, S, S)
            # matrix_dict['conditional'] (T, S, S)
            # matrix_dict['marginal'] (2, 2)
            # matrix_dict['log_corr'] float
            matrix_dict = self.return_all_matrices(offset=offset,
                                                   joint_matrix=joint_matrix)
            matrix_dict['log_corr'] = 0
            
        return (self.log_frag_class_probs, matrix_dict, None)
        
    
    
###############################################################################
### TKF92   ###################################################################
###############################################################################
class TKF92TransitionLogprobs(TKF91TransitionLogprobs):
    """
    TKF92 model; used for calculating transitions in model of
        P(anc, desc, align)
    
    C_dom: number of mixtures associated with nested TKF92 models (domain-level)
    C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
        > for tkf91, there can NOT be mixtures over transitions (i.e. C_frag=1)
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
        
        
    Initialize with
    ----------------
    config : dict
        config["tkf_function"] : {'regular_tkf','approx_tkf','switch_tkf'}
            which function to use to solve for tkf parameters
        
        config["num_domain_mixtures"] : int
            number of domain mixtures (for nested TKF model)
            
        config["num_fragment_mixtures"] : int
            number of tkf92 fragments

        config["mu_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines mu
            DEFAULT: -1e-4, 2
        
        config["offset_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines offset 
            (which determines mu)
            DEFAULT: -1e-4, 0.333
            
        config["r_range"]
            range for bound sigmoid activation that determines TKF r
            DEFAULT: -1e-4, 0.999
            
    name : str
        class name, for flax
    
    
    Methods here
    ------------
    setup
    
    __call__
    
    fill_joint_tkf92
        fills in joint TKF92 transition matrix
        
    return_all_matrices
        return transition matrices used for joint, marginal, and conditional 
        log-likelihood transitions
        
        
    Inherited from TKF91TransitionLogprobs
    ---------------------------------------
    fill_joint_tkf91
        fills in joint TKF91 transition matrix
    
    _logits_to_indel_rates
        converts mu/offset logits to mu/offset values
    """
    config: dict
    name: str
    
    def setup(self):
        """
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
            > for tkf91, there can NOT be mixtures over transitions (i.e. C_frag=1)
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
        
        
        Flax Module Parameters
        -----------------------
        tkf_mu_offset_logits : ArrayLike (2,)
            first value is logit for mu, second is for offset
        
        r_extend_logits : ArrayLike (C_frag,)
            logits for TKF fragment extension probability, r
            this is EXCLUSIVELY for the fragment-level tkf92 indel process
        
        frag_class_prob_logits : ArrayLike (C_dom, C_frag)
        
        """
        ### unpack config
        # required
        self.num_domain_mixtures = self.config['num_domain_mixtures']
        self.num_fragment_mixtures = self.config['num_fragment_mixtures']
        tkf_function_name = self.config['tkf_function']
        
        # optional inputs
        self.mu_min_val, self.mu_max_val = self.config.get( 'mu_range', [1e-4, 2] )
        self.offs_min_val, self.offs_max_val = self.config.get( 'offset_range', [1e-4, 0.333] )
        self.r_extend_min_val, self.r_extend_max_val = self.config.get( 'r_range', [1e-4, 0.999] )
        
        
        ### init flax parameters
        # initialize logits for mu, offset
        self.tkf_mu_offset_logits = self.param('tkf_mu_offset_logits',
                                               nn.initializers.normal(),
                                               (2,),
                                               jnp.float32) #(2,)
        
        # initializing r extension prob
        self.r_extend_logits = self.param('r_extend_logits',
                                          nn.initializers.normal(),
                                          (self.num_domain_mixtures, self.num_fragment_mixtures),
                                          jnp.float32) #(C_dom, C_frag)
        
        # initializing probability of fragment classes
        self.frag_class_prob_logits = self.param('frag_class_prob_logits',
                                          nn.initializers.normal(),
                                          (self.num_domain_mixtures, self.num_fragment_mixtures),
                                          jnp.float32) #(C_dom, C_frag)
        
        
        ### decide tkf function
        if tkf_function_name == 'regular_tkf':
            self.tkf_function = regular_tkf
        elif tkf_function_name == 'approx_tkf':
            self.tkf_function = approx_tkf
        elif tkf_function_name == 'switch_tkf':
            self.tkf_function = switch_tkf
        
        
    
    def __call__(self,
                 t_array,
                 return_all_matrices: bool,
                 sow_intermediates: bool):
        """
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
            > for tkf91, there can NOT be mixtures over transitions (i.e. C_frag=1)
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
           
        
        Arguments
        ----------
        t_array : ArrayLike, (T,)
            branch lengths, times for marginalizing over
        
        return_all_matrices : bool
            if false, only return the joint (used for model training)
            if true, return joint, conditional, and marginal matrices
        
        sow_intermediates : bool
            switch for tensorboard logging
          
        Returns
        -------
        log_frag_class_probs : ArrayLike, (C_dom, C_frag) 
            P(c_fragment | c_domain)
        
        matrix_dict : dict
            matrix_dict["joint"]: (T,C_dom,C_frag,C_frag,S,S)
                score transitions in joint probability calculation
                
            matrix_dict["marginal"]: (C_dom,C_frag,C_frag,2,2)
                score transitions in marginal probability calculation
            
            matrix_dict["conditional"]: (T,C_dom,C_frag,C_frag,S,S)
                score transitions in conditional probability calculation
            
            matrix_dict["log_corr"]: (C_dom, C_frag)
                correction factor in case alignment starts with start -> ins
        
        use_approx : Tuple( ArrayLike, ArrayLike ), ( (T,), (T,) )
            where tkf approximation formulas were used
            
        """
        ### P(C_fragment | C_domain)
        log_frag_class_probs = nn.log_softmax( self.frag_class_prob_logits, axis = -1 ) #(C_dom, C_fr)
        frag_class_probs = jnp.exp(log_frag_class_probs) #(C_dom, C_fr)
        
        if sow_intermediates:
            for c_dom in range(frag_class_probs.shape[0]):
                lab = f'{self.name}/fragment class probabilities, domain class {c_dom}'
                self.sow_histograms_scalars(mat= frag_class_probs[c_tr, ...], 
                                            label=lab, 
                                            which='scalars')
                del lab
                
        
        ### TKF92 model
        out = self._logits_to_indel_rates(mu_offset_logits = self.tkf_mu_offset_logits,
                                         mu_min_val = self.mu_min_val,
                                         mu_max_val = self.mu_max_val,
                                         offs_min_val = self.offs_min_val,
                                         offs_max_val = self.offs_max_val) #(2,)
        mu, offset = out # floats
        del out
        
        # r_extend
        r_extend = bound_sigmoid(x = self.r_extend_logits,
                                 min_val = self.r_extend_min_val,
                                 max_val = self.r_extend_max_val) # (C_dom, C_frag)
        
        # get alpha, beta, gamma
        # contents of out_dict ( all ArrayLike[float32], (T,) ):
        #   out_dict['log_alpha']
        #   out_dict['log_one_minus_alpha']
        #   out_dict['log_beta']
        #   out_dict['log_one_minus_beta']
        #   out_dict['log_gamma']
        #   out_dict['log_one_minus_gamma']
        #
        # contents of approx_flags_dict ( all ArrayLike[bool], (T,) ):
        #   out_dict['log_one_minus_alpha']
        #   out_dict['log_beta']
        #   out_dict['log_one_minus_gamma']
        #   out_dict['log_gamma']
        out_dict, approx_flags_dict = self.tkf_function(mu = mu, 
                                                        offset = offset,
                                                        t_array = t_array)
        
        # add to these dictionaries before filling out matrix
        out_dict['log_offset'] = jnp.log(offset)
        out_dict['log_one_minus_offset'] = jnp.log1p(-offset)
        
        if approx_flags_dict is not None:
            approx_flags_dict['t_array'] = t_array
        
        # record values
        if sow_intermediates:
            self.sow_histograms_scalars(mat= lam, 
                                        label=f'{self.name}/lam', 
                                        which='scalars')
            
            lam = mu * (1-offset)
            self.sow_histograms_scalars(mat= mu, 
                                        label=f'{self.name}/mu', 
                                        which='scalars')
            del lam
            
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
    
        # (T, C_dom, C_{frag_from}, C_{frag_to}, S_from=4, S_to=4)
        joint_matrix =  self.fill_joint_tkf92(out_dict=out_dict, 
                                              r_extend=r_extend,
                                              frag_class_probs=frag_class_probs)
        
        # since num_domain_mixtures=1, index away intermediates
        joint_matrix = joint_matrix[:,0,...] # (T, C_{frag_from}, C_{frag_to}, S_from=4, S_to=4)
        
        if not return_all_matrices:
            matrix_dict = {'joint': joint_matrix}
        
        elif return_all_matrices:
            matrix_dict = self.return_all_matrices(offset=offset,
                                                   class_probs=frag_class_probs,
                                                   r_ext_prob = r_extend,
                                                   joint_matrix=joint_matrix)
            
            # correction factors for S->I transition
            matrix_dict['log_corr'] = jnp.log(lam/mu) - jnp.log( r_extend + (1-r_extend)*(lam/mu) ) #(C_dom, C_fr)
        
        return (log_frag_class_probs, matrix_dict, approx_flags_dict)
        
    
    def fill_joint_tkf92(self,
                        out_dict,
                        r_extend,
                        frag_class_probs):
        """
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
            > for tkf91, there can NOT be mixtures over transitions (i.e. C_frag=1)
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
        
        
        Arguments
        ----------
        out_dict : dict
            contains values for calculating matrix terms: lambda, mu, 
            alpha, beta, gamma, 1 - alpha, 1 - beta, 1 - gamma
            (all in log space)
            all are (T,)
        
        r_extend : ArrayLike, (C_dom, C_frag)
            fragment extension probabilities
        
        frag_class_probs : ArrayLike, (C_dom, C_frag)
            support for the classes i.e. P(end at class c_frag)
          
        Returns
        -------
        out : ArrayLike, (T, C_dom, C_{frag_from}, C_{frag_to}, S_from=4, S_to=4)
            joint loglike of transitions
        
        """
        ### need joint TKF91 for this (which already contains lam/mu terms)
        log_U = self.fill_joint_tkf91(out_dict) #(T, S_from, S_to)
        
        # dims
        T = out_dict['log_alpha'].shape[0]
        C_dom = frag_class_probs.shape[0] #domain-level classes
        C_frag = frag_class_probs.shape[1] #fragment-level classes
        S = log_U.shape[-1] #number of hidden states (like M, I, D, and start/end)
        
        # converted log values; expand
        log_r_ext_prob = safe_log( r_extend ) #(C_dom, C_{frag_from})
        log_one_minus_r = log_one_minus_x(log_r_ext_prob) #(C_dom, C_{frag_from})
        log_one_minus_r = log_one_minus_r[None, ..., None, None] #(1, C_dom, C_{frag_from}, 1, 1)
        
        ### entries in the matrix
        # (1-r_c) U(i,j) for all (MID -> MIDE transitions), 
        #   U(i,j) for all start -> MIDE transitions
        log_U_exp = log_U[:, None, None, :, :]  # shape: (T, 1, 1, S, S)

        # Build a mask of shape (S_from,) where the last index is False
        s_mask = jnp.arange(S) != (S - 1)  # shape: (S_from,)
        s_mask_exp = s_mask[None, None, None, :, None]  # shape: (1, 1, 1, S_from, 1)
        
        # Apply log_one_minus_r only where S_from != S-1
        #                        log_U_exp: (T,     1,             1, S_from, S_to)
        # log_one_minus_r[..., None, None]: (1, C_dom, C_{frag_from},      1,    1)
        #            log_tkf92_rate_mat is: (T, C_dom, C_{frag_from}, S_from, S_to)
        log_tkf92_rate_mat = log_U_exp + jnp.where( s_mask_exp,
                                                    log_one_minus_r, 
                                                    0.0 ) 
        del s_mask_exp
        
        # expand again
        s_mask_exp = s_mask[None, None, None, None, None, :]  # shape: (1, 1, 1, 1, 1, S_to)
        log_tkf92_rate_mat = log_tkf92_rate_mat[:,:,:, None, ...] #(T, C_dom, C_{frag_from}, 1, S_from, S_to)
        log_frag_class_probs = safe_log(frag_class_probs) #(C_dom, C_{frag_to})
        log_frag_class_probs = log_frag_class_probs[None, :, None, :, None, None] #(1, C_dom, 1, C_{frag_to}, 1, 1)
        
        # multiply by P(c) across all C_to (not including transitions that 
        #   end with <end>
        # log_tkf92_rate_mat before: (T, C_dom, C_{frag_from},           1, S_from, S_to)
        #      log_frag_class_probs: (1, C_dom,             1, C_{frag_to},      1,    1)
        #  log_tkf92_rate_mat AFTER: (T, C_dom, C_{frag_from}, C_{frag_to}, S_from, S_to)        
        log_tkf92_rate_mat = log_tkf92_rate_mat + jnp.where( s_mask_exp,
                                                             log_frag_class_probs, 
                                                             0.0 ) 
        del s_mask_exp
        
        # at MID: where frag_class_from == frag_class_to and state_from == state_to, 
        #   add factor of r; S_from=3 means start, and S_to=3 means end
        i_idx, j_idx = jnp.meshgrid(jnp.arange(C_frag), jnp.arange(S-1), indexing="ij")
        i_idx = i_idx.flatten()
        j_idx = j_idx.flatten()

        # add r to specific locations
        # log_tkf92_rate_mat[:, :, i_idx, i_idx, j_idx, j_idx] (T, C_dom, S-1)
        prev_vals = log_tkf92_rate_mat[:, :, i_idx, i_idx, j_idx, j_idx].reshape( (T, C_dom, C_frag, S-1) ) #(T, C_dom, C_frag, S-1)
        r_to_add = jnp.broadcast_to( log_r_ext_prob[None,...,None], prev_vals.shape) #(T, C_dom, C_frag, S-1)
        new_vals = logsumexp_with_arr_lst([r_to_add, prev_vals]).reshape(T, C_dom, -1) #(T, C_dom, C_frag * S-1)
        del prev_vals, r_to_add

        # scatter back with advanced indexing
        #  log_tkf92_rate_mat is: (T, C_dom, C_{frag_from}, C_{frag_to}, S_from, S_to)   
        log_tkf92_rate_mat = log_tkf92_rate_mat.at[:, :, i_idx, i_idx, j_idx, j_idx].set(new_vals) 
        
        return log_tkf92_rate_mat
    
    
    def return_all_matrices(self,
                            offset,
                            r_ext_prob,
                            frag_class_probs,
                            joint_matrix):
        """
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
            > for tkf91, there can NOT be mixtures over transitions (i.e. C_frag=1)
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
         
        
        Arguments
        ---------
        offset : ArrayLike, ()
            1 - (lam/mu)
        
        r_ext_prob : ArrayLike, (C_dom, C_frag)
            fragment extension probabilities
        
        frag_class_probs : ArrayLike, (C_dom, C_frag)
            support for the classes i.e. P(end at class c_frag)
         
        joint_matrix : ArrayLike, (T, C_dom, C_{frag_from}, C_{frag_to}, S_from=4, S_to=4)
        
        
        Returns
        -------
        (returned_dictionary)["joint"]: (T, C_dom, C_{frag_from}, C_{frag_to}, S_from=4, S_to=4)
        (returned_dictionary)["marginal"]: (C_dom, C_{frag_from}, C_{frag_to}, 2, 2)
        (returned_dictionary)["conditional"]: (T, C_dom, C_{frag_from}, C_{frag_to}, S_from=4, S_to=4)
        
        """
        # output is: (C_dom, C_{frag_from}, C_{frag_to}, 2, 2)
        marginal_matrix = get_tkf92_single_seq_marginal_transition_logprobs(offset=offset,
                                                      frag_class_probs=frag_class_probs,
                                                      r_ext_prob=r_ext_prob)
        
        # output is: (T, C_dom, C_{frag_from}, C_{frag_to}, S_from=4, S_to=4)
        cond_matrix = get_cond_transition_logprobs(marg_matrix=marginal_matrix, 
                                             joint_matrix=joint_matrix)
        
        return {'joint': joint_matrix,
                'marginal': marginal_matrix,
                'conditional': cond_matrix}
        

class TKF92TransitionLogprobsFromFile(TKF92TransitionLogprobs):
    """
    like TKF91TransitionLogprobs, but load values from a file
    
    NOTE: mu is provided directly, no need for offset
    
    C_dom: number of mixtures associated with nested TKF92 models (domain-level)
    C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
        > for tkf91, there can NOT be mixtures over transitions (i.e. C_frag=1)
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
    
        
    Initialize with
    ----------------
    config : dict 
        config["tkf_function"] : {'regular_tkf','approx_tkf','switch_tkf'}
            which function to use to solve for tkf parameters
    
        config["filenames"]["tkf_params_file"] : str
            contains values for lambda, mu, r-extension
            
        config["filenames"]["frag_class_probs"]: str
              file of fragment class probabilities to load
                    
    name : str
        class name, for flax
    
    
    Methods here
    ------------
    setup
    __call__
    
    
    Inherited from TKF91TransitionLogprobs
    ---------------------------------------
    fill_joint_tkf91
        fills in joint TKF91 transition matrix
        
    _logits_to_indel_rates
        converts mu/offset logits to mu/offset values
    
    
    Inherited from TKF92TransitionLogprobs
    ---------------------------------------
    fill_joint_tkf92
        fills in joint TKF92 transition matrix
        
    return_all_matrices
        return transition matrices used for joint, marginal, and conditional 
        log-likelihood transitions
    """
    config: dict
    name: str
    
    def setup(self):
        """
        
        Flax Module Parameters
        -----------------------
        None
        
        """
        ### unpack config
        tkf_params_file = self.config['filenames']['tkf_params_file']
        frag_class_probs_file = self.config['filenames']['frag_class_probs']
        tkf_function_name = self.config['tkf_function']
        
        
        ### read files
        # tkf parameters
        with open(in_file,'rb') as f:
            self.param_dict = pickle.load(f)
    
        err = f'KEYS SEEN: {self.param_dict.keys()}'
        assert 'lambda' in self.param_dict.keys(), err
        assert 'mu' in self.param_dict.keys(), err
        assert 'r_extend' in self.param_dict.keys(), err
        
        # mixture probability of fragment classes
        with open(frag_class_probs_file,'rb') as f:
            frag_class_probs = jnp.load(f) #(C_dom, C_frag) or (C_frag,)
        
        if len(frag_class_probs.shape)==1:
            frag_class_probs = frag_class_probs[None, :] #(C_dom=1, C_frag)
        
        self.log_frag_class_probs = safe_log(frag_class_probs) #(C_dom, C_frag)
        
        # for now, don't allow domain mixtures
        assert self.param_dict.shape[0] == 1
        assert self.log_frag_class_probs.shape[0] == 1
        
        ### pick tkf function
        if tkf_function_name == 'regular_tkf':
            self.tkf_function = regular_tkf
        elif tkf_function_name == 'approx_tkf':
            self.tkf_function = approx_tkf
        elif tkf_function_name == 'switch_tkf':
            self.tkf_function = switch_tkf
                    
        
    def __call__(self,
                 t_array,
                 return_all_matrices: bool,
                 sow_intermediates: bool):
        """
        C_dom: number of mixtures associated with nested TKF92 models (domain-level)
        C_frag: number of mixtures associated with TKF92 fragments (fragment-level)
            > for tkf91, there can NOT be mixtures over transitions (i.e. C_frag=1)
        B = batch size; number of samples
        T = number of branch lengths; this could be: 
            > an array of times for all samples (T; marginalize over these later)
            > an array of time per sample (T=B)
            > a quantized array of times per sample (T = T', where T' <= T)
        S: number of transition states (4 here: M, I, D, start/end)
           
        
        Arguments
        ----------
        t_array : ArrayLike, (T,)
            branch lengths, times for marginalizing over
        
        return_all_matrices : bool
            if false, only return the joint (used for model training)
            if true, return joint, conditional, and marginal matrices
        
        sow_intermediates : bool
            switch for tensorboard logging
          
        Returns
        -------
        log_frag_class_probs : ArrayLike, (C_dom, C_frag) 
            P(c_fragment | c_domain)
        
        matrix_dict : dict
            matrix_dict["joint"]: (T,C_dom,C_frag,C_frag,S,S)
                score transitions in joint probability calculation
                
            matrix_dict["marginal"]: (C_dom,C_frag,C_frag,2,2)
                score transitions in marginal probability calculation
            
            matrix_dict["conditional"]: (T,C_dom,C_frag,C_frag,S,S)
                score transitions in conditional probability calculation
            
            matrix_dict["log_corr"]: (C_dom, C_frag)
                correction factor in case alignment starts with start -> ins
        
        use_approx : Tuple( ArrayLike, ArrayLike ), ( (T,), (T,) )
            where tkf approximation formulas were used
            
        """
        log_frag_class_probs = self.log_frag_class_probs #(C_dom, C_frag)
        frag_class_probs = jnp.exp(log_frag_class_probs) #(C_dom, C_frag)
        
        lam = self.param_dict['lambda'] #float
        mu = self.param_dict['mu'] #float
        offset = 1 - (lam /mu) #float
        r_extend = self.param_dict['r_extend'] #(C_dom, C_frag)
        
        num_dom_classes = r_extend.shape[0]
        num_frag_classes = r_extend.shape[1]
        
        # get alpha, beta, gamma
        # contents of out_dict ( all ArrayLike[float32], (T,) ):
        #   out_dict['log_alpha']
        #   out_dict['log_one_minus_alpha']
        #   out_dict['log_beta']
        #   out_dict['log_one_minus_beta']
        #   out_dict['log_gamma']
        #   out_dict['log_one_minus_gamma']
        #
        # contents of approx_flags_dict ( all ArrayLike[bool], (T,) ):
        #   out_dict['log_one_minus_alpha']
        #   out_dict['log_beta']
        #   out_dict['log_one_minus_gamma']
        #   out_dict['log_gamma']
        out_dict, _ = self.tkf_function(mu = mu, 
                                        offset = offset,
                                        t_array = t_array)
        
        # add to these dictionaries before filling out matrix
        out_dict['log_offset'] = jnp.log(offset)
        out_dict['log_one_minus_offset'] = jnp.log1p(-offset)
        
        # (T, C_dom, C_{frag_from}, C_{frag_to}, S_from=4, S_to=4)
        joint_matrix =  self.fill_joint_tkf92(out_dict, 
                                              r_extend,
                                              class_probs)
        
        # since num_domain_mixtures=1, index away intermediates
        joint_matrix = joint_matrix[:,0,...] # (T, C_{frag_from}, C_{frag_to}, S_from=4, S_to=4)
        
        if not return_all_matrices:
            matrix_dict = {'joint': joint_matrix}
        
        elif return_all_matrices:
            matrix_dict = self.return_all_matrices(offset=offset,
                                                   class_probs=frag_class_probs,
                                                   r_ext_prob = r_extend,
                                                   joint_matrix=joint_matrix)
            
            # correction factors for S->I transition
            matrix_dict['log_corr'] = jnp.log(lam/mu) - jnp.log( r_extend + (1-r_extend)*(lam/mu) ) #(C_dom, C_fr)
        
        return (log_frag_class_probs, matrix_dict, approx_flags_dict)
