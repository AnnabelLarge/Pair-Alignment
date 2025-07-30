#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:42:28 2024

@author: annabel


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
        
        out_vec = safe_log( jnp.concatenate( [p_emit, 1-p_emit] ) ) #(2,)
        
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
        config["init_mu_offset_logits"] : Tuple, (2,)
            initial values for logits that determine lambda, offset
            DEFAULT: -2, -5
        
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
        
    logits_to_indel_rates
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
        tkf_mu_offset_logits: ArrayLike (2,)
            first value is logit for mu, second is for offset
        
        """
        ### unpack config
        # initializing lamda, offset
        init_mu_offset_logits = self.config.get( 'init_mu_offset_logits',
                                                [-2, -5] )
        init_mu_offset_logits = jnp.array(init_mu_offset_logits, dtype=float)
        self.mu_min_val, self.mu_max_val = self.config.get( 'mu_range', 
                                                             [1e-4, 2] )
        self.offs_min_val, self.offs_max_val = self.config.get( 'offset_range', 
                                                                [1e-4, 0.333] )
        
        # which tkf function
        tkf_function_name = self.config.get('tkf_function', 'switch_tkf')
        
        # were options at one point, but I'm fixing the values now
        self.sigmoid_temp = 1
        
        
        ### initialize logits for mu, offset
        # with default values:
        # init mu: 0.11929100006818771
        # init offset: 0.0023280500900000334
        self.tkf_mu_offset_logits = self.param('TKF91 lambda, mu',
                                            lambda rng, shape, dtype: init_mu_offset_logits,
                                            init_mu_offset_logits.shape,
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
        out = self.logits_to_indel_rates(mu_offset_logits = self.tkf_mu_offset_logits,
                                         mu_min_val = self.mu_min_val,
                                         mu_max_val = self.mu_max_val,
                                         offs_min_val = self.offs_min_val,
                                         offs_max_val = self.offs_max_val)
        mu, offset = out
        lam = mu * (1-offset)
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
            
            self.sow_histograms_scalars(mat= lam, 
                                        label=f'{self.name}/lam', 
                                        which='scalars')
            
            self.sow_histograms_scalars(mat= mu, 
                                        label=f'{self.name}/mu', 
                                        which='scalars')
        
        joint_matrix =  self.fill_joint_tkf91(out_dict)
        
        matrix_dict = self.return_all_matrices(offset=offset,
                                               joint_matrix=joint_matrix)
        matrix_dict['log_corr'] = 0
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
                           max_val = mu_max_val,
                           temperature = self.sigmoid_temp)
        
        # mu
        offset = bound_sigmoid(x = mu_offset_logits[1],
                               min_val = offs_min_val,
                               max_val = offs_max_val,
                               temperature = self.sigmoid_temp)

        return (mu, offset)
    
    
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
        marginal_matrix = get_tkf91_single_seq_marginal_transition_logprobs(offset=offset)
        
        # output is same as joint: (T, 4, 4)
        cond_matrix = get_cond_transition_logprobs(marg_matrix=marginal_matrix, 
                                             joint_matrix=joint_matrix)
        
        return {'joint': joint_matrix,
                'marginal': marginal_matrix,
                'conditional': cond_matrix}
    


class TKF91TransitionLogprobsFromFile(TKF91TransitionLogprobs):
    """
    like TKF91TransitionLogprobs, but load values from a file
    
    NOTE: lambda and mu are provided directly, no need for offset
    
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        > a quantized array of times per sample (T = T', where T' <= T)
        
        
    Initialize with
    ----------------
    config : dict (but nothing used here)
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
        tkf_function_name = self.config.get('tkf_function', 'switch_tkf')
        
        # read file
        if in_file.endswith('.pkl'):
            with open(in_file,'rb') as f:
                self.param_dict = pickle.load(f)
                
        elif in_file.endswith('.txt') or in_file.endswith('.tsv'):
            param_dict = {}
            with open(in_file,'r') as f:
                for line in f:
                    if not line.startswith('#'):
                        param_name, value = line.strip().split('\t')
                        param_dict[param_name] = jnp.array( float(value) )
            self.param_dict = param_dict
        
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
        lam = self.param_dict['lambda']
        mu = self.param_dict['mu']
        offset = 1 - (lam /mu)
        
        # get alpha, beta, gamma
        out_dict, _ = self.tkf_function(mu = mu, 
                                        offset = offset,
                                        t_array = t_array)
        out_dict['log_offset'] = jnp.log(offset)
        out_dict['log_one_minus_offset'] = jnp.log1p(-offset)
        
        joint_matrix =  self.fill_joint_tkf91(out_dict)
        
        matrix_dict = self.return_all_matrices(offset=offset,
                                               joint_matrix=joint_matrix)
        matrix_dict['log_corr'] = 0
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
        config["init_mu_offset_logits"] : Tuple, (2,)
            initial values for logits that determine mu, offset
            DEFAULT: -2, -5
        
        config["mu_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines mu
            DEFAULT: -1e-4, 2
        
        config["offset_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines offset 
            (which determines mu)
            DEFAULT: -1e-4, 0.333
            
        config["init_r_extend_logits"] : Tuple, (C,)
            initial values for logits that determine mu, offset
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
        converts mu/offset logits to mu/offset values
    """
    config: dict
    name: str
    
    def setup(self):
        """
        C = number of site classes
        
        
        Flax Module Parameters
        -----------------------
        tkf_mu_offset_logits: ArrayLike (2,)
            first value is logit for mu, second is for offset
        
        r_extend_logits: ArrayLike (C,)
            logits for TKF fragment extension probability, r
        
        """
        ### unpack config
        self.num_tkf_fragment_classes = self.config['num_tkf_fragment_classes']
        
        # initializing lamda, offset
        init_mu_offset_logits = self.config.get( 'init_mu_offset_logits', 
                                                  [-2, -5] )
        init_mu_offset_logits = jnp.array(init_mu_offset_logits, dtype=float)
        self.mu_min_val, self.mu_max_val = self.config.get( 'mu_range', 
                                                            [1e-4, 2] )
        self.offs_min_val, self.offs_max_val = self.config.get( 'offset_range', 
                                                                [1e-4, 0.333] )
        
        # initializing r extension prob
        init_r_extend_logits = self.config.get( 'init_r_extend_logits',
                                               [-x/10 for x in 
                                                range(1, self.num_tkf_fragment_classes+1)] )
        init_r_extend_logits = jnp.array(init_r_extend_logits, dtype=float)
        self.r_extend_min_val, self.r_extend_max_val = self.config.get( 'r_range', 
                                                                [1e-4, 0.999] )
        
        # which tkf function
        tkf_function_name = self.config.get('tkf_function', 'switch_tkf')
        
        # were options at one point, but I'm fixing the values now
        self.sigmoid_temp = 1
        
        
        ### initialize logits for mu, offset
        # with default values:
        # init mu: 0.11929100006818771
        # init offset: 0.0023280500900000334
        self.tkf_mu_offset_logits = self.param('TKF92 lambda, mu',
                                            lambda rng, shape, dtype: init_mu_offset_logits,
                                            init_mu_offset_logits.shape,
                                            jnp.float32)
        
        # up to num_tkf_fragment_classes different r extension probabilities
        # with default first 10 values: 
        #   0.40004998, 0.38006914, 0.3601878, 0.34050342, 0.32110974
        #   0.30209476, 0.2835395, 0.26551658, 0.24808942, 0.2313115
        self.r_extend_logits = self.param('TKF92 r extension prob',
                                          lambda rng, shape, dtype: init_r_extend_logits,
                                          init_r_extend_logits.shape,
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
                 log_class_probs,
                 sow_intermediates: bool):
        """
        Arguments
        ----------
        t_array : ArrayLike, (T,)
            branch lengths, times for marginalizing over
        
        log_class_probs : ArrayLike, (C,)
            support for classes i.e. logP(ending at class c)
        
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
        class_probs = jnp.exp(log_class_probs)
        
        # lam, mu are of size () (i.e. just floats)
        out = self.logits_to_indel_rates(mu_offset_logits = self.tkf_mu_offset_logits,
                                         mu_min_val = self.mu_min_val,
                                         mu_max_val = self.mu_max_val,
                                         offs_min_val = self.offs_min_val,
                                         offs_max_val = self.offs_max_val)
        mu, offset = out
        lam = mu * (1-offset)
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
        
        
        # correction factors for S->I transition
        matrix_dict['log_corr'] = jnp.log(lam/mu) - jnp.log( r_extend + (1-r_extend)*(lam/mu) )
        
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
          
        joint_matrix : ArrayLike, (T, C, C, 4, 4)
        
        
        Returns
        -------
        (returned_dictionary)["joint"]: (T, C, C, 4, 4)
        (returned_dictionary)["marginal"]: ( C, C, 2, 2)
        (returned_dictionary)["conditional"]: (T, C, C, 4, 4)
        
        """
        # output is: (C, C, 4, 4)
        marginal_matrix = get_tkf92_single_seq_marginal_transition_logprobs(offset=offset,
                                                      class_probs=class_probs,
                                                      r_ext_prob=r_ext_prob)
        
        # output is: (T, C, C, 4, 4)
        cond_matrix = get_cond_transition_logprobs(marg_matrix=marginal_matrix, 
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
        config["tkf_params_file"] : str
            contains values for lambda, mu, r-extension
                    
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
        in_file = self.config['filenames']['tkf_params_file']
        tkf_function_name = self.config.get('tkf_function', 'switch_tkf')

        # read file
        if in_file.endswith('.pkl'):
            with open(in_file,'rb') as f:
                self.param_dict = pickle.load(f)
        
        elif in_file.endswith('.txt') or in_file.endswith('.tsv'):
            param_dict = {}
            r_extend = []
            with open(in_file,'r') as f:
                for line in f:
                    if not line.startswith('#'):
                        param_name, value = line.strip().split('\t')
                        value = float(value)
                        
                        if param_name.startswith('r_extend'):
                            r_extend.append(value)
                        else:
                            param_dict[param_name] = jnp.array(value)
            param_dict['r_extend'] = jnp.array(r_extend)
            self.param_dict = param_dict
        
        err = f'KEYS SEEN: {self.param_dict.keys()}'
        assert 'lambda' in self.param_dict.keys(), err
        assert 'mu' in self.param_dict.keys(), err
        assert 'r_extend' in self.param_dict.keys(), err
        
        # pick tkf function
        if tkf_function_name == 'regular_tkf':
            self.tkf_function = regular_tkf
        elif tkf_function_name == 'approx_tkf':
            self.tkf_function = approx_tkf
        elif tkf_function_name == 'switch_tkf':
            self.tkf_function = switch_tkf
                    
        
    def __call__(self,
                 t_array,
                 log_class_probs,
                 sow_intermediates: bool):
        """
        Arguments
        ----------
        t_array : ArrayLike, (T,)
            branch lengths, times for marginalizing over
        
        log_class_probs : ArrayLike, (C,)
            support for classes i.e. logP(ending at class c)
        
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
        lam = self.param_dict['lambda']
        mu = self.param_dict['mu']
        offset = 1 - (lam /mu)
        r_extend = self.param_dict['r_extend']
        num_site_classes = r_extend.shape[0]
        class_probs = jnp.exp(log_class_probs)
        
        # get alpha, beta, gamma
        out_dict, _ = self.tkf_function(mu = mu, 
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
        
        # correction factors for S->I transition
        matrix_dict['log_corr'] = jnp.log(lam/mu) - jnp.log( r_extend + (1-r_extend)*(lam/mu) )
        
        return matrix_dict, None
