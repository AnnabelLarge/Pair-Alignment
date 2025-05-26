#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:42:28 2024

@author: annabel

modules used for training:
==========================
  GeomLenTransitionLogprobs
  GeomLenTransitionLogprobsFromFile

 'TKF91TransitionLogprobs',
 'TKF91TransitionLogprobsFromFile',
 
 'TKF92TransitionLogprobs',
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
                                                              stable_tkf,
                                                              MargTKF91TransitionLogprobs,
                                                              MargTKF92TransitionLogprobs,
                                                              CondTransitionLogprobs)

###############################################################################
### Geometric sequence length (no indels)   ###################################
###############################################################################
class GeomLenTransitionLogprobs(ModuleBase):
    """
    Simply assume sequence has geometrically-distributed sequence length
    
    
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
        self.p_emit_logit = self.param('geom length p',
                                       lambda rng, shape, dtype: init_logit,
                                       init_logit.shape,
                                       jnp.float32)
    
    def __call__(self,
                 sow_intermediates: bool,
                 *args,
                 **kwargs):
        """
        
        Arguments
        ----------
        sow_intermediates : bool
            switch for tensorboard logging
          
        Returns
        -------
        out_dict : dict
            out_dict["joint"]: (2,)
                score transitions in joint probability calculation
                
            out_dict["marginal"]: (2,)
                score transitions in marginal probability calculation
            
            out_dict["conditional"]: (2,)
                score transitions in conditional probability calculation
        
        (output tuple) :  None
            placeholder values
        """
        p_emit = nn.sigmoid(self.p_emit_logit) #(1,)
        
        if sow_intermediates:
            self.sow_histograms_scalars(mat= p_emit, 
                                        label=f'{self.name}/geom_prob_emit', 
                                        which='scalars')
        
        out_vec = safe_log( jnp.array( [p_emit, 1-p_emit] ) ) #(2,)
        
        out_dict = {'joint': out_vec,
                    'marginal': out_vec,
                    'conditional': out_vec}
        
        return out_dict, None
        

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
        out_dict = {'joint': self.out_vec,
                    'marginal': self.out_vec,
                    'conditional': self.out_vec}
        
        return out_dict, None
        
        
    
    
###############################################################################
### TKF91   ###################################################################
###############################################################################
class TKF91TransitionLogprobs(ModuleBase):
    """
    TKF91 model; used for calculating transitions in model of
        P(anc, desc, align)
    
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
        
    Initialize with
    ----------------
    config : dict (but nothing used here)
        config["tkf_err"] : float
            error term for tkf approximation
            DEFAULT: 1e-4
            
        config["init_lambda_offset_logits"] : Tuple, (2,)
            initial values for logits that determine lambda, offset
            DEFAULT: -2, -5
        
        config["lambda_range"] : Tuple, (2,)
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
        
    logits_to_indel_rates
        converts lambda/offset logits to lambda/mu values
    
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
        tkf_lam_mu_logits: ArrayLike (2,)
            first value is logit for lambda, second is for offset
        
        """
        ### unpack config
        # initializing lamda, offset
        init_lam_offset_logits = self.config.get( 'init_lambda_offset_logits',
                                                [-2, -5] )
        init_lam_offset_logits = jnp.array(init_lam_offset_logits, dtype=float)
        self.lam_min_val, self.lam_max_val = self.config.get( 'lambda_range', 
                                                               [1e-4, 2] )
        self.offs_min_val, self.offs_max_val = self.config.get( 'offset_range', 
                                                                [1e-4, 0.333] )
        
        # were options at one point, but I'm fixing the values now
        self.sigmoid_temp = 1
        
        
        ### initialize logits for lambda, offset
        # with default values:
        # init lam: 0.11929100006818771
        # init offset: 0.0023280500900000334
        self.tkf_lam_mu_logits = self.param('TKF91 lambda, mu',
                                            lambda rng, shape, dtype: init_lam_offset_logits,
                                            init_lam_offset_logits.shape,
                                            jnp.float32)
   
    def __call__(self,
                 t_array,
                 sow_intermediates: bool):
        """
        Arguments
        ----------
        t_array : ArrayLike
            branch lengths, times for marginalizing over
        
        sow_intermediates : bool
            switch for tensorboard logging
          
        Returns
        -------
        out_dict : dict
            out_dict["joint"]: (T,4,4)
                score transitions in joint probability calculation
                
            out_dict["marginal"]: (2,2)
                score transitions in marginal probability calculation
            
            out_dict["conditional"]: (T,4,4)
                score transitions in conditional probability calculation
        
        approx_flags_dict : dict
            where approximations are used in tkf formulas
            
            out_dict['log_one_minus_alpha']: (T,)
            
            out_dict['log_beta']: (T,)
            
            out_dict['log_one_minus_gamma']: (T,)
            
            out_dict['log_gamma']: (T,)
            
        """
        out = self.logits_to_indel_rates(lam_mu_logits = self.tkf_lam_mu_logits,
                                         lam_min_val = self.lam_min_val,
                                         lam_max_val = self.lam_max_val,
                                         offs_min_val = self.offs_min_val,
                                         offs_max_val = self.offs_max_val)
        lam, mu, offset = out
        del out
        
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
        out_dict, approx_flags_dict = stable_tkf(mu = mu, 
                                                 offset = offset,
                                                 t_array = t_array)
        
        # add to these dictionaries before filling out matrix
        out_dict['log_offset'] = jnp.log(offset)
        out_dict['log_one_minus_offset'] = jnp.log1p(-offset)
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
            
            self.sow_histograms_scalars(mat= lam, 
                                        label=f'{self.name}/lam', 
                                        which='scalars')
            
            self.sow_histograms_scalars(mat= mu, 
                                        label=f'{self.name}/mu', 
                                        which='scalars')
        
        joint_matrix =  self.fill_joint_tkf91(out_dict)
        
        matrix_dict = self.return_all_matrices(offset=offset,
                                               joint_matrix=joint_matrix)
        return matrix_dict, approx_flags_dict
        
    
    def fill_joint_tkf91(self, 
                         out_dict):
        """
        Arguments
        ----------
        out_dict : dict
            contains values for calculating matrix terms: (all in log space)
                - offset = 1 - (lam/mu)
                - alpha, beta, gamma, 1 - alpha, 1 - beta, 1 - gamma
                  
        Returns
        -------
        out : ArrayLike, (T,4,4)
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
                              offs_max_val):
        """
        Arguments
        ---------
        lam_mu_logits : ArrayLike, (2,)
            logits to transform with bound sigmoid activation
        
        lam_min_val : float
            minimum value for bound sigmoid, to get lambda
        
        lam_max_val : float
            maximum value for bound sigmoid, to get lambda
        
        offs_min_val : float
            minimum value for bound sigmoid, to get offset
        
        offs_max_val : float
            maximum value for bound sigmoid, to get offset
        
        Returns
        -------
        lambda : ArrayLike, ()
            insert rate
        
        mu : ArrayLike, ()
            delete rate
        
        """
        # lambda
        lam = bound_sigmoid(x = lam_mu_logits[0],
                            min_val = lam_min_val,
                            max_val = lam_max_val,
                            temperature = self.sigmoid_temp)
        
        # mu
        offset = bound_sigmoid(x = lam_mu_logits[1],
                               min_val = offs_min_val,
                               max_val = offs_max_val,
                               temperature = self.sigmoid_temp)
        mu = lam / ( 1 -  offset) 

        return (lam, mu, offset)
    
    
    def return_all_matrices(self,
                            offset,
                            joint_matrix):
        """
        T = times
        
        
        Arguments
        ---------
        offset : ArrayLike, ()
            1 - (lam/mu)
        
        joint_matrix : ArrayLike, (T, 4, 4)
        
        
        Returns
        -------
        (returned_dictionary)["joint"]: (T, 4, 4)
        (returned_dictionary)["marginal"]: (2, 2)
        (returned_dictionary)["conditional"]: (T, 4, 4)
        
        """
        # output is: (4, 4)
        marginal_matrix = MargTKF91TransitionLogprobs(offset=offset)
        
        # output is same as joint: (T, 4, 4)
        cond_matrix = CondTransitionLogprobs(marg_matrix=marginal_matrix, 
                                             joint_matrix=joint_matrix)
        
        return {'joint': joint_matrix,
                'marginal': marginal_matrix,
                'conditional': cond_matrix}
    


class TKF91TransitionLogprobsFromFile(TKF91TransitionLogprobs):
    """
    like TKF91TransitionLogprobs, but load values from a file
    
    NOTE: mu is provided directly, no need for offset
    
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
        
        
    Initialize with
    ----------------
    config : dict (but nothing used here)
        config["tkf_err"]
            error term for tkf approximation
            DEFAULT: 1e-4
            
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
        
    logits_to_indel_rates
        converts lambda/offset logits to lambda/mu values
    
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
        in_file = self.config['filenames']['tkf_params_file']
        
        with open(in_file,'rb') as f:
            self.tkf_lam_mu = jnp.load(f)
    
    def __call__(self,
                 t_array,
                 sow_intermediates: bool):
        """
        Arguments
        ----------
        t_array : ArrayLike
            branch lengths, times for marginalizing over
        
        sow_intermediates : bool
            switch for tensorboard logging
          
        Returns
        -------
        out_dict : dict
            out_dict["joint"]: (T,4,4)
                score transitions in joint probability calculation
                
            out_dict["marginal"]: (2,2)
                score transitions in marginal probability calculation
            
            out_dict["conditional"]: (T,4,4)
                score transitions in conditional probability calculation
        
        """
        lam = self.tkf_lam_mu[...,0]
        mu = self.tkf_lam_mu[...,1]
        offset = 1 - (lam /mu)
        
        # get alpha, beta, gamma
        out_dict, _ = stable_tkf(mu = mu, 
                                 offset = offset,
                                 t_array = t_array)
        out_dict['log_offset'] = jnp.log(offset)
        out_dict['log_one_minus_offset'] = jnp.log1p(-offset)
        
        joint_matrix =  self.fill_joint_tkf91(out_dict)
        
        matrix_dict = self.return_all_matrices(offset=offset,
                                               joint_matrix=joint_matrix)
        return matrix_dict, None
        
    
    
###############################################################################
### TKF92   ###################################################################
###############################################################################
class TKF92TransitionLogprobs(TKF91TransitionLogprobs):
    """
    TKF92 model; used for calculating transitions in model of
        P(anc, desc, align)
    
    B = batch size; number of samples
    C = number of site classes 
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
        
    Initialize with
    ----------------
    config : dict (but nothing used here)
        config["tkf_err"] : float
            error term for tkf approximation
            DEFAULT: 1e-4
            
        config["init_lambda_offset_logits"] : Tuple, (2,)
            initial values for logits that determine lambda, offset
            DEFAULT: -2, -5
        
        config["lambda_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines lamdba
            DEFAULT: -1e-4, 2
        
        config["offset_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines offset 
            (which determines mu)
            DEFAULT: -1e-4, 0.333
            
        config["init_r_extend_logits"] : Tuple, (C,)
            initial values for logits that determine lambda, offset
            DEFAULT: -x/10 for x in range(1,C+1)
        
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
    
    logits_to_indel_rates
        converts lambda/offset logits to lambda/mu values
    
    tkf_params
        from lambda and mu, calculate TKF alpha, beta, gamma
    
    
    """
    config: dict
    name: str
    
    def setup(self):
        """
        C = number of site classes
        
        
        Flax Module Parameters
        -----------------------
        tkf_lam_mu_logits: ArrayLike (2,)
            first value is logit for lambda, second is for offset
        
        r_extend_logits: ArrayLike (C,)
            logits for TKF fragment extension probability, r
        
        """
        ### unpack config
        self.num_tkf_site_classes = self.config['num_tkf_site_classes']
        
        # initializing lamda, offset
        init_lam_offset_logits = self.config.get( 'init_lambda_offset_logits', 
                                                  [-2, -5] )
        init_lam_offset_logits = jnp.array(init_lam_offset_logits, dtype=float)
        self.lam_min_val, self.lam_max_val = self.config.get( 'lambda_range', 
                                                               [1e-4, 2] )
        self.offs_min_val, self.offs_max_val = self.config.get( 'offset_range', 
                                                                [1e-4, 0.333] )
        
        # initializing r extension prob
        init_r_extend_logits = self.config.get( 'init_r_extend_logits',
                                               [-x/10 for x in 
                                                range(1, self.num_tkf_site_classes+1)] )
        init_r_extend_logits = jnp.array(init_r_extend_logits, dtype=float)
        self.r_extend_min_val, self.r_extend_max_val = self.config.get( 'r_range', 
                                                                [1e-4, 0.999] )
        
        # were options at one point, but I'm fixing the values now
        self.sigmoid_temp = 1
        
        
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
        """
        Arguments
        ----------
        t_array : ArrayLike, (T,)
            branch lengths, times for marginalizing over
        
        class_probs : ArrayLike, (C,)
            support for classes i.e. P(ending at class c)
        
        sow_intermediates : bool
            switch for tensorboard logging
          
        Returns
        -------
        out_dict : dict
            out_dict["joint"]: (T,C,C,4,4)
                score transitions in joint probability calculation
                
            out_dict["marginal"]: (2,2)
                score transitions in marginal probability calculation
            
            out_dict["conditional"]: (T,C,C,4,4)
                score transitions in conditional probability calculation
        
        use_approx : Tuple( ArrayLike, ArrayLike ), ( (T,), (T,) )
            where tkf approximation formulas were used
            
        """
        # lam, mu are of size () (i.e. just floats)
        out = self.logits_to_indel_rates(lam_mu_logits = self.tkf_lam_mu_logits,
                                         lam_min_val = self.lam_min_val,
                                         lam_max_val = self.lam_max_val,
                                         offs_min_val = self.offs_min_val,
                                         offs_max_val = self.offs_max_val)
        lam, mu, offset = out
        del out
        
        # r_extend is of size (C,)
        r_extend = bound_sigmoid(x = self.r_extend_logits,
                                   min_val = self.r_extend_min_val,
                                   max_val = self.r_extend_max_val)
        
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
        out_dict, approx_flags_dict = stable_tkf(mu = mu, 
                                                 offset = offset,
                                                 t_array = t_array)
        
        # add to these dictionaries before filling out matrix
        out_dict['log_offset'] = jnp.log(offset)
        out_dict['log_one_minus_offset'] = jnp.log1p(-offset)
        approx_flags_dict['t_array'] = t_array
        
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
        
        matrix_dict = self.return_all_matrices(offset=offset,
                                               class_probs=class_probs,
                                               r_ext_prob = r_extend,
                                               joint_matrix=joint_matrix)
        return matrix_dict, approx_flags_dict
        
    
    def fill_joint_tkf92(self,
                        out_dict,
                        r_extend,
                        class_probs):
        """
        C = number of site classes
        S = number of regular transitions, 4 here: M, I, D, START/END
        
        Arguments
        ----------
        out_dict : dict
            contains values for calculating matrix terms: lambda, mu, 
            alpha, beta, gamma, 1 - alpha, 1 - beta, 1 - gamma
            (all in log space)
        
        r_extend : ArrayLike, (C,)
            fragment extension probabilities
        
        class_probs : ArrayLike, (C,)
            support for the classes i.e. P(end at class c)
          
        Returns
        -------
        out : ArrayLike, (T, C_from, C_to, S_from=4, S_to=4)
            joint loglike of transitions
        
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
                            offset,
                            class_probs,
                            r_ext_prob,
                            joint_matrix):
        """
        T = times
        C = number of site classes
        
        
        Arguments
        ---------
        offset : ArrayLike, ()
            1 - (lam/mu)
        
        r_extend : ArrayLike, (C,)
            fragment extension probabilities
        
        class_probs : ArrayLike, (C,)
            support for the classes i.e. P(end at class c)
          
        joint_matrix : ArrayLike, (T, 4, 4)
        
        
        Returns
        -------
        (returned_dictionary)["joint"]: (T, C, C, 4, 4)
        (returned_dictionary)["marginal"]: ( C, C, 2, 2)
        (returned_dictionary)["conditional"]: (T, C, C, 4, 4)
        
        """
        # output is: (C, C, 4, 4)
        marginal_matrix = MargTKF92TransitionLogprobs(offset=offset,
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
    like TKF91TransitionLogprobs, but load values from a file
    
    NOTE: mu is provided directly, no need for offset
    
    B = batch size; number of samples
    C = number of site classes
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
        
    Initialize with
    ----------------
    config : dict (but nothing used here)
        config["tkf_err"] : float
            error term for tkf approximation
            DEFAULT: 1e-4
            
        config["tkf_params_file"] : str
            loads a dictionary with two values:
                
                (output object)['lam_mu'] : ArrayLike, (2,)
                    initial values for logits that determine lambda, offset
                
                (output object)['r_extend'] : ArrayLike, (C,)
                    TKF fragment extension probabilities
                    
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
        
    logits_to_indel_rates
        converts lambda/offset logits to lambda/mu values
    
    tkf_params
        from lambda and mu, calculate TKF alpha, beta, gamma
        
    
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
        self.tkf_err = self.config.get('tkf_err', 1e-4)
        in_file = self.config['filenames']['tkf_params_file']

        # set this manually
        self.approx_floor = -1e-4
        
        with open(in_file,'rb') as f:
            in_dict = pickle.load(f)
            self.tkf_lam_mu = in_dict['lam_mu']
            self.r_extend = in_dict['r_extend']
    
    def __call__(self,
                 t_array,
                 class_probs,
                 sow_intermediates: bool):
        """
        Arguments
        ----------
        t_array : ArrayLike, (T,)
            branch lengths, times for marginalizing over
        
        class_probs : ArrayLike, (C,)
            support for classes i.e. P(ending at class c)
        
        sow_intermediates : bool
            switch for tensorboard logging
          
        Returns
        -------
        out_dict : dict
            out_dict["joint"]: (T,C,C,4,4)
                score transitions in joint probability calculation
                
            out_dict["marginal"]: (2,2)
                score transitions in marginal probability calculation
            
            out_dict["conditional"]: (T,C,C,4,4)
                score transitions in conditional probability calculation
        
        use_approx : Tuple( ArrayLike, ArrayLike ), ( (T,), (T,) )
            where tkf approximation formulas were used
            
        """
        lam = self.tkf_lam_mu[0]
        mu = self.tkf_lam_mu[1]
        offset = 1 - (lam/mu)
        r_extend = self.r_extend
        num_site_classes = r_extend.shape[0]
        use_approx = False
        
        # get alpha, beta, gamma
        out_dict, _ = stable_tkf(mu = mu, 
                                                 offset = offset,
                                                 t_array = t_array)
        
        # add to these dictionaries before filling out matrix
        out_dict['log_offset'] = jnp.log(offset)
        out_dict['log_one_minus_offset'] = jnp.log1p(-offset)
        
        # (T, C_from, C_to, S_from=4, S_to=4)
        joint_matrix =  self.fill_joint_tkf92(out_dict, 
                                              r_extend,
                                              class_probs)
        
        matrix_dict = self.return_all_matrices(offset=offset,
                                               class_probs=class_probs,
                                               r_ext_prob=r_extend,
                                               joint_matrix=joint_matrix)
        return matrix_dict, None
