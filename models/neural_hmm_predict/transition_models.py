#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 13:43:50 2025

@author: annabel

tkf91 models, pred_config entries:
-----------------------------------
'GlobalTKF91',
- (OPTIONAL) pred_config['init_mu_offset_logits'] 
- (OPTIONAL) pred_config['mu_range']
- (OPTIONAL) pred_config['offset_range']
- (OPTIONAL) pred_config['tkf_function']

 'LocalTKF91',
- (OPTIONAL) pred_config['mu_range']
- (OPTIONAL) pred_config['offset_range']
- (OPTIONAL) pred_config['tkf_function']

'GlobalTKF91FromFile'
- pred_config['filenames']['tkf_params_file']
- (OPTIONAL) pred_config['tkf_function']


tkf92, pred_config entries:
----------------------------
 'TKF92GlobalRateGlobalFragSize',
- (OPTIONAL) pred_config['init_mu_offset_logits'] 
- (OPTIONAL) pred_config['mu_range']
- (OPTIONAL) pred_config['offset_range']
- (OPTIONAL) pred_config['init_r_extend_logits']
- (OPTIONAL) pred_config['r_range']
- (OPTIONAL) pred_config['tkf_function']

 'TKF92GlobalRateLocalFragSize',
- (OPTIONAL) pred_config['init_mu_offset_logits'] 
- (OPTIONAL) pred_config['mu_range']
- (OPTIONAL) pred_config['offset_range']
- (OPTIONAL) pred_config['r_range']
- (OPTIONAL) pred_config['tkf_function']

 'TKF92LocalRateLocalFragSize',
- (OPTIONAL) pred_config['mu_range']
- (OPTIONAL) pred_config['offset_range']
- (OPTIONAL) pred_config['r_range']
- (OPTIONAL) pred_config['tkf_function']

'GlobalTKF92FromFile'
- pred_config['filenames']['tkf_params_file']
- (OPTIONAL) pred_config['tkf_function']

"""
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm 

from models.BaseClasses import (neuralTKFModuleBase, 
                                ModuleBase)
from models.neural_hmm_predict.model_functions import (bound_sigmoid,
                                                       safe_log,
                                                       logsumexp_with_arr_lst,
                                                       log_one_minus_x,
                                                       switch_tkf,
                                                       regular_tkf,
                                                       approx_tkf,
                                                       logprob_tkf91,
                                                       logprob_tkf92)


# add another tensorboard recording utililty function
class transitionModuleBase(neuralTKFModuleBase):
    def maybe_record_indel_interms_to_tboard(self,
                                             mu,
                                             offset,
                                             r_extend,
                                             tkf_params_dict,
                                             sow_intermediates):
        if sow_intermediates:
            lam = mu * (1-offset)
            lam = jnp.squeeze(lam)
            mu_to_rec = jnp.squeeze(mu)
            
            self.sow_histograms_scalars(mat= lam, 
                                        label=f'{self.name}/lam', 
                                        which='scalars')
            
            self.sow_histograms_scalars(mat= mu_to_rec, 
                                        label=f'{self.name}/mu', 
                                        which='scalars')
            
            
            del lam, mu_to_rec
        
            if r_extend is not None:
                r_to_rec = jnp.squeeze(r_extend)
                self.sow_histograms_scalars(mat= r_to_rec, 
                                            label=f'{self.name}/r_extend', 
                                            which='scalars')
                del r_to_rec
            
            self.sow_histograms_scalars(mat= jnp.exp(tkf_params_dict['log_alpha']), 
                                        label=f'{self.name}/tkf_alpha', 
                                        which='scalars')
            
            self.sow_histograms_scalars(mat= jnp.exp(tkf_params_dict['log_beta']), 
                                        label=f'{self.name}/tkf_beta', 
                                        which='scalars')
            
            self.sow_histograms_scalars(mat= jnp.exp(tkf_params_dict['log_gamma']), 
                                        label=f'{self.name}/tkf_gamma', 
                                        which='scalars')


###############################################################################
### GLOBAL indel rates   ######################################################
###############################################################################
class GlobalTKF91(transitionModuleBase):
    """
    CONDITIONAL LOG-PROBABILITY!!! calculating logP(align_i|align_{i-1},Anc,Desc_{...i-1},t)
    
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        
    Initialize with
    ----------------
    config : dict 
        config["init_mu_offset_logits"] : Tuple, (1, 1, 2)
            initial values for logits that determine mu, offset
            DEFAULT: -2, -5
        
        config["mu_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines mu
            DEFAULT: -1e-4, 2
        
        config["offset_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines offset 
            (which determines lambda)
            DEFAULT: -1e-4, 0.333
            
    name : str
        class name, for flax
    
    
    Methods here
    ------------
    setup
    __call__
    get_r_ext_prob
        placeholder; returns None
    """
    config: dict
    name: str
    
    def setup(self):
        """
        Flax Module Parameters
        -----------------------
        tkf_mu_offset_logits: ArrayLike (1,1,2)
            first value is logit for mu, second is for offset
        """
        ### Mu, Offset
        # initial values
        init_mu_offset_logits = self.config.get( 'init_mu_offset_logits', 
                                                  [-2, -5] ) #(2,)
        init_mu_offset_logits = jnp.array(init_mu_offset_logits, dtype=float) #(2,)
        init_mu_offset_logits = init_mu_offset_logits[None,None,:] #(1,1,2)
        
        # setting limits for bound sigmoid function
        self.mu_min_val, self.mu_max_val = self.config.get( 'mu_range', 
                                                            [1e-4, 2] )
        self.offs_min_val, self.offs_max_val = self.config.get( 'offset_range', 
                                                                [1e-4, 0.333] )
        
        # create the parameters in flax
        # with default values:
        # init mu: 0.11929100006818771
        # init offset: 0.0023280500900000334
        self.tkf_mu_offset_logits = self.param('mu, offset',
                                            lambda rng, shape, dtype: init_mu_offset_logits,
                                            init_mu_offset_logits.shape,
                                            jnp.float32) #(1,1,2)
        
        
        ### decide tkf function
        tkf_function_name = self.config.get('tkf_function', 'switch_tkf')
        tkf_fn_registry = {'regular_tkf': regular_tkf,
                           'approx_tkf': approx_tkf,
                           'switch_tkf': switch_tkf}
        self.tkf_function = tkf_fn_registry[tkf_function_name]

        ### declare the logprob function
        self.cond_logprob_fn = logprob_tkf91
        
        
    def __call__(self,
                 t_array,
                 unique_time_per_sample: bool,
                 sow_intermediates: bool,
                 *args,
                 **kwargs):
        """
        T: number of times
        B: batch size
        L_align: length of alignment
        
        
        Arguments
        ----------
        t_array : ArrayLike, (T,)
            branch lengths, times for marginalizing over
        
        unique_time_per_sample : Bool
            whether there's one time per sample, or a grid of times you'll 
            marginalize over
            
        sow_intermediates : bool
            switch for tensorboard logging
          
        Returns
        -------
        cond_logprob : ArrayLike
            > if unique time per sample: (B, 1, 4, 4) 
            > if not unique time per sample: (T, 1, 1, 4, 4) 
            log-probability matrix for transitions
        
        use_approx : Tuple( ArrayLike, ArrayLike ), ( (T,), (T,) ) or ( (B,), (B,) )
            where tkf approximation formulas were used
            
        """
        ### logits to values
        # get mu and offset
        mu = self.apply_bound_sigmoid_activation( logits = self.tkf_mu_offset_logits[...,0],
                                                  min_val = self.mu_min_val,
                                                  max_val = self.mu_max_val,
                                                  param_name = 'tkf mu',
                                                  sow_intermediates = False ) #(1,1)
        
        offset = self.apply_bound_sigmoid_activation( logits = self.tkf_mu_offset_logits[...,1],
                                                  min_val = self.offs_min_val,
                                                  max_val = self.offs_max_val,
                                                  param_name = 'tkf offset',
                                                  sow_intermediates = False ) #(1,1)
        
        # get r_extend
        r_extend = self.get_r_ext_prob() #placeholder
        
        
        ### get tkf alpha, beta, gamma
        # contents of out_dict ( all ArrayLike[float32], (T,B,L_align) or (B,L_align) ):
        #   out_dict['log_alpha']
        #   out_dict['log_one_minus_alpha']
        #   out_dict['log_beta']
        #   out_dict['log_one_minus_beta']
        #   out_dict['log_gamma']
        #   out_dict['log_one_minus_gamma']
        #
        # contents of approx_flags_dict ( all ArrayLike[float32], (1,) ):
        #   out_dict['log_one_minus_alpha']
        #   out_dict['log_beta']
        #   out_dict['log_one_minus_gamma']
        #   out_dict['log_gamma']
        tkf_params_dict, approx_flags_dict = self.tkf_function(mu = mu, 
                                                               offset = offset,
                                                               t_array = t_array,
                                                               unique_time_per_sample = unique_time_per_sample)
        
        # record values to tensorboard
        self.maybe_record_indel_interms_to_tboard(mu = mu,
                                                  offset = offset,
                                                  r_extend = r_extend,
                                                  tkf_params_dict = tkf_params_dict,
                                                  sow_intermediates = sow_intermediates)
        
        # cond_logprob is either:
        # (T, 1, 1, 4, 4), or
        # (B, 1, 4, 4)
        cond_logprob =  self.cond_logprob_fn( tkf_params_dict = tkf_params_dict,
                                              r_extend = r_extend,
                                              offset = offset,
                                              unique_time_per_sample = unique_time_per_sample ) 
        
        intermed_params_dict = {'lambda': mu * (1-offset),
                                'mu': mu,
                                'r_extend': r_extend}
        
        return cond_logprob, approx_flags_dict, intermed_params_dict
        
    
    def get_r_ext_prob(self):
        """
        placeholder; r would actually be zero, but this plays poorly with 
          log-transformed functions
        """
        return None
    
    
class TKF92GlobalRateGlobalFragSize(GlobalTKF91):
    """
    CONDITIONAL LOG-PROBABILITY!!! calculating logP(align_i|align_{i-1},Anc,Desc_{...i-1},t)
    
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        
    Initialize with
    ----------------
    config : dict 
        config["init_mu_offset_logits"] : Tuple, (1, 1, 2)
            initial values for logits that determine mu, offset
            DEFAULT: -2, -5
        
        config["mu_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines mu
            DEFAULT: -1e-4, 2
        
        config["offset_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines offset 
            (which determines lambda)
            DEFAULT: -1e-4, 0.333
            
        config["init_r_extend_logits"] : Tuple, (1,1)
            initial values for logits that determine mu, offset
            DEFAULT: -1/10
        
        config["r_range"]
            range for bound sigmoid activation that determines TKF r
            DEFAULT: -1e-4, 0.999
            
    name : str
        class name, for flax
    
    
    Methods here
    -------------
    get_r_ext_prob
        gets r extension probability for tkf92 instead
    
    Inherited from GlobalTKF91
    ----------------------------
    setup
    __call__
    """
    config: dict
    name: str
    
    def setup(self):
        """
        Flax Module Parameters
        -----------------------
        tkf_mu_offset_logits: ArrayLike (1,1,2)
            first value is logit for mu, second is for offset
        
        r_extend_logits: ArrayLike (1,1)
            logits for TKF fragment extension probability, r
        
        """
        super().setup()
        
        ### extra stuff for TKF92: R extension probability 
        # overwrite cond_logprob_fn
        self.cond_logprob_fn = logprob_tkf92
        
        # initializing r extension prob
        init_r_extend_logits = self.config.get( 'init_r_extend_logits', [-1/10] ) #(1,)
        init_r_extend_logits = jnp.array(init_r_extend_logits, dtype=float) #(1,)
        init_r_extend_logits = init_r_extend_logits[None,:] #(1,1)
        
        # setting limits for bound sigmoid function
        self.r_extend_min_val, self.r_extend_max_val = self.config.get( 'r_range', 
                                                                [1e-4, 0.999] )
        
        # create the parameters in flax
        # with default values: 0.40004998
        self.r_extend_logits = self.param('r extension prob',
                                          lambda rng, shape, dtype: init_r_extend_logits,
                                          init_r_extend_logits.shape,
                                          jnp.float32) #(1,1)
    
    def get_r_ext_prob(self):
        """
        return r extension probability for tkf92
        """
        return self.apply_bound_sigmoid_activation( logits = self.r_extend_logits,
                                                    min_val = self.r_extend_min_val,
                                                    max_val = self.r_extend_max_val,
                                                    param_name = 'tkf92 r',
                                                    sow_intermediates = False )


class TKF92GlobalRateLocalFragSize(GlobalTKF91):
    """
    CONDITIONAL LOG-PROBABILITY!!! calculating logP(align_i|align_{i-1},Anc,Desc_{...i-1},t)
    
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        
    Initialize with
    ----------------
    config : dict 
        config["init_mu_offset_logits"] : Tuple, (1, 1, 2)
            initial values for logits that determine mu, offset
            DEFAULT: -2, -5
        
        config["mu_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines mu
            DEFAULT: -1e-4, 2
        
        config["offset_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines offset 
            (which determines lambda)
            DEFAULT: -1e-4, 0.333
            
        config["r_range"]
            range for bound sigmoid activation that determines TKF r
            DEFAULT: -1e-4, 0.999
            
    name : str
        class name, for flax
    
    
    Methods here
    -------------
    __call__
    
    Inherited from GlobalTKF91
    ----------------------------
    setup
    get_r_ext_prob (not used)
    """
    config: dict
    name: str
    
    def setup(self):
        """
        Flax Module Parameters
        -----------------------
        tkf_mu_offset_logits: ArrayLike (1,1,2)
            first value is logit for mu, second is for offset
        
        projection to r extend:
            kernel: [fill in later]
            bias: [fill in later]
        
        """
        super().setup()
        
        ### extra stuff for TKF92: R extension probability 
        # overwrite cond_logprob_fn
        self.cond_logprob_fn = logprob_tkf92
        
        # setting limits for bound sigmoid function
        self.r_extend_min_val, self.r_extend_max_val = self.config.get( 'r_range', 
                                                                [1e-4, 0.999] )
        
        # linear projection to R
        name = f'{self.name}/Project to R extension prob'
        self.final_project_to_r = nn.Dense(features = 1,
                                           use_bias = True,
                                           name = name)
    
    def __call__(self,
                 datamat,
                 t_array,
                 unique_time_per_sample: bool,
                 sow_intermediates: bool):
        """
        T: number of times
        B: batch size
        H: input hidden dim
        L_align: length of alignment
        
        
        Arguments
        ----------
        datamat : ArrayLike, (B, L_align, H)
        
        t_array : ArrayLike, (T,)
            branch lengths, times for marginalizing over
        
        unique_time_per_sample : Bool
            whether there's one time per sample, or a grid of times you'll 
            marginalize over
            
        sow_intermediates : bool
            switch for tensorboard logging
          
        Returns
        -------
        cond_logprob : ArrayLike
            > if unique time per sample: (B, L_align, 4, 4) 
            > if not unique time per sample: (T, B, L_align, 4, 4) 
            log-probability matrix for transitions
        
        use_approx : Tuple( ArrayLike, ArrayLike ), ( (T,), (T,) ) or ( (B,), (B,) )
            where tkf approximation formulas were used
            
        """
        ### logits to values
        # r: (B, L_align, H) -> (B, L_align, 1) -> (B, L_align)
        r_logits = self.final_project_to_r(datamat)[...,0]
        r_extend = self.apply_bound_sigmoid_activation( logits = r_logits,
                                                        min_val = self.r_extend_min_val,
                                                        max_val = self.r_extend_max_val,
                                                        param_name = 'tkf92 r',
                                                        sow_intermediates = False )
        
        # get mu and offset
        mu = self.apply_bound_sigmoid_activation( logits = self.tkf_mu_offset_logits[...,0],
                                                  min_val = self.mu_min_val,
                                                  max_val = self.mu_max_val,
                                                  param_name = 'tkf mu',
                                                  sow_intermediates = False ) #(1,1)
        
        offset = self.apply_bound_sigmoid_activation( logits = self.tkf_mu_offset_logits[...,1],
                                                  min_val = self.offs_min_val,
                                                  max_val = self.offs_max_val,
                                                  param_name = 'tkf offset',
                                                  sow_intermediates = False ) #(1,1)
        
        
        ### get tkf alpha, beta, gamma
        # contents of out_dict ( all ArrayLike[float32], (T,B,L_align) or (B,L_align) ):
        #   out_dict['log_alpha']
        #   out_dict['log_one_minus_alpha']
        #   out_dict['log_beta']
        #   out_dict['log_one_minus_beta']
        #   out_dict['log_gamma']
        #   out_dict['log_one_minus_gamma']
        #
        # contents of approx_flags_dict ( all ArrayLike[float32], (1,) ):
        #   out_dict['log_one_minus_alpha']
        #   out_dict['log_beta']
        #   out_dict['log_one_minus_gamma']
        #   out_dict['log_gamma']
        tkf_params_dict, approx_flags_dict = self.tkf_function(mu = mu, 
                                                               offset = offset,
                                                               t_array = t_array,
                                                               unique_time_per_sample = unique_time_per_sample)
        
        # record values to tensorboard
        self.maybe_record_indel_interms_to_tboard(mu = mu,
                                                  offset = offset,
                                                  r_extend = r_extend,
                                                  tkf_params_dict = tkf_params_dict,
                                                  sow_intermediates = sow_intermediates)
        
        # cond_logprob is either:
        # (T, B, L_align, 4, 4), or
        # (B, L_align, 4, 4)
        cond_logprob =  self.cond_logprob_fn( tkf_params_dict = tkf_params_dict,
                                              r_extend = r_extend,
                                              offset = offset,
                                              unique_time_per_sample = unique_time_per_sample ) 
        
        intermed_params_dict = {'lambda': mu * (1-offset),
                                'mu': mu,
                                'r_extend': r_extend}
        
        return cond_logprob, approx_flags_dict, intermed_params_dict


###############################################################################
### GLOBAL indel rates, parameters read from a file   #########################
###############################################################################
class GlobalTKF91FromFile(neuralTKFModuleBase):
    """
    same as GlobalTKF91, but load params from file
    
    CONDITIONAL LOG-PROBABILITY!!! calculating logP(align_i|align_{i-1},Anc,Desc_{...i-1},t)
    
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        
    Initialize with
    ----------------
    config : dict tkf_params_file
        config["filenames"]["tkf_params_file"] : dict
            
    name : str
        class name, for flax
    
    
    Methods here
    ------------
    setup
    __call__
    get_r_ext_prob
        placeholder; returns None
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
        
        
        ### read file
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
        
        
        ### declare the logprob function
        self.cond_logprob_fn = logprob_tkf91
    
    
    def __call__(self,
                 t_array,
                 unique_time_per_sample: bool,
                 *args,
                 **kwargs):
        """
        T: number of times
        B: batch size
        L_align: length of alignment
        
        
        Arguments
        ----------
        t_array : ArrayLike, (T,)
            branch lengths, times for marginalizing over
        
        unique_time_per_sample : Bool
            whether there's one time per sample, or a grid of times you'll 
            marginalize over
            
        Returns
        -------
        cond_logprob : ArrayLike
            > if unique time per sample: (B, 1, 4, 4) 
            > if not unique time per sample: (T, 1, 1, 4, 4) 
            log-probability matrix for transitions
        
        use_approx : Tuple( ArrayLike, ArrayLike ), ( (T,), (T,) ) or ( (B,), (B,) )
            where tkf approximation formulas were used
            
        """
        # get mu and offset
        lam = self.param_dict['lambda']
        mu = self.param_dict['mu']
        offset = 1 - (lam /mu)
        
        # get r_extend
        r_extend = self.get_r_ext_prob() #placeholder
        
        ### get tkf alpha, beta, gamma
        # contents of out_dict ( all ArrayLike[float32], (T,B,L_align) or (B,L_align) ):
        #   out_dict['log_alpha']
        #   out_dict['log_one_minus_alpha']
        #   out_dict['log_beta']
        #   out_dict['log_one_minus_beta']
        #   out_dict['log_gamma']
        #   out_dict['log_one_minus_gamma']
        #
        # contents of approx_flags_dict ( all ArrayLike[float32], (1,) ):
        #   out_dict['log_one_minus_alpha']
        #   out_dict['log_beta']
        #   out_dict['log_one_minus_gamma']
        #   out_dict['log_gamma']
        tkf_params_dict, _ = self.tkf_function(mu = mu, 
                                               offset = offset,
                                               t_array = t_array,
                                               unique_time_per_sample = unique_time_per_sample)
        # cond_logprob is either:
        # (T, 1, 1, 4, 4), or
        # (B, 1, 4, 4)
        cond_logprob =  self.cond_logprob_fn( tkf_params_dict = tkf_params_dict,
                                              r_extend = r_extend,
                                              offset = offset,
                                              unique_time_per_sample = unique_time_per_sample ) 
        
        return cond_logprob, None, None
    
    def get_r_ext_prob(self):
        return None


class GlobalTKF92FromFile(GlobalTKF91FromFile):
    """
    same as TKF92GlobalRateGlobalFragSize, but load params from file
    
    CONDITIONAL LOG-PROBABILITY!!! calculating logP(align_i|align_{i-1},Anc,Desc_{...i-1},t)
    
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        
    Initialize with
    ----------------
    config : dict tkf_params_file
        config["filenames"]["tkf_params_file"] : dict
            
    name : str
        class name, for flax
    
    
    Methods here
    ------------
    setup
    __call__
    get_r_ext_prob
        placeholder; returns None
    """
    config: dict
    name: str
    
    def setup(self):
        """
        Flax Module Parameters
        -----------------------
        None
        """
        super().setup()
        
        # make sure r_extend is in the parameter dictionary
        err = f'KEYS SEEN: {self.param_dict.keys()}'
        assert 'r_extend' in self.param_dict.keys(), err
        
        # overwrite the logprob function
        self.cond_logprob_fn = logprob_tkf92
    
    def get_r_ext_prob(self):
        return self.param_dict['r_extend']
        

###############################################################################
### LOCAL indel rates   #######################################################
###############################################################################
class LocalTKF91(transitionModuleBase):
    """
    (idk if I'll use this, but it's useful to have this before defining
       the next TKF92 function: local rates and local fragment sites)
    
    CONDITIONAL LOG-PROBABILITY!!! calculating logP(align_i|align_{i-1},Anc,Desc_{...i-1},t)
    
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        
    Initialize with
    ----------------
    config : dict 
        config["mu_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines mu
            DEFAULT: -1e-4, 2
        
        config["offset_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines offset 
            (which determines lambda)
            DEFAULT: -1e-4, 0.333
            
    name : str
        class name, for flax
    
    
    Methods here
    ------------
    setup
    __call__
    get_r_ext_prob
        placeholder; returns None
    """
    config: dict
    name: str
    
    def setup(self):
        """
        Flax Module Parameters
        -----------------------
        tkf_mu_offset: 
            kernel: [fill in later]
            bias: [fill in later]
        """
        name = f'{self.name}/Project to mu, offset'
        self.mu_offset_final_project = nn.Dense(features = 2,
                                                use_bias = True,
                                                name = name)
        self.mu_min_val, self.mu_max_val = self.config.get( 'mu_range', 
                                                            [1e-4, 2] )
        self.offs_min_val, self.offs_max_val = self.config.get( 'offset_range', 
                                                                [1e-4, 0.333] )
        
        # decide tkf function
        tkf_function_name = self.config.get('tkf_function', 'switch_tkf')
        tkf_fn_registry = {'regular_tkf': regular_tkf,
                           'approx_tkf': approx_tkf,
                           'switch_tkf': switch_tkf}
        self.tkf_function = tkf_fn_registry[tkf_function_name]
        
        # declare the logprob function
        self.cond_logprob_fn = logprob_tkf91
        
        
    def __call__(self,
                 datamat,
                 t_array,
                 unique_time_per_sample: bool,
                 sow_intermediates: bool):
        """
        T: number of times
        B: batch size
        H: input hidden dim
        L_align: length of alignment
        
        
        Arguments
        ----------
        datamat : ArrayLike, (B, L_align, H)
        
        t_array : ArrayLike, (T,)
            branch lengths, times for marginalizing over
        
        unique_time_per_sample : Bool
            whether there's one time per sample, or a grid of times you'll 
            marginalize over
            
        sow_intermediates : bool
            switch for tensorboard logging
          
        Returns
        -------
        cond_logprob : ArrayLike
            > if unique time per sample: (B, L_align, 4, 4) 
            > if not unique time per sample: (T, B, L_align, 4, 4) 
            log-probability matrix for transitions
        
        use_approx : Tuple( ArrayLike, ArrayLike ), ( (T,), (T,) ) or ( (B,), (B,) )
            where tkf approximation formulas were used
            
        """
        ### logits to values
        # get mu and offset; (B, L_align, H) -> (B, L_align, 2)
        tkf_mu_offset_logits = self.mu_offset_final_project(datamat)
        
        mu = self.apply_bound_sigmoid_activation( logits = tkf_mu_offset_logits[...,0],
                                                  min_val = self.mu_min_val,
                                                  max_val = self.mu_max_val,
                                                  param_name = 'tkf mu',
                                                  sow_intermediates = False ) #(B,L_align)
        
        offset = self.apply_bound_sigmoid_activation( logits = tkf_mu_offset_logits[...,1],
                                                      min_val = self.offs_min_val,
                                                      max_val = self.offs_max_val,
                                                      param_name = 'tkf offset',
                                                      sow_intermediates = False ) #(B,L_align)
        
        r_extend = self.get_r_ext_prob(datamat)
        
        
        ### get tkf alpha, beta, gamma
        # contents of out_dict ( all ArrayLike[float32], (T,B,L_align) or (B,L_align) ):
        #   out_dict['log_alpha']
        #   out_dict['log_one_minus_alpha']
        #   out_dict['log_beta']
        #   out_dict['log_one_minus_beta']
        #   out_dict['log_gamma']
        #   out_dict['log_one_minus_gamma']
        #
        # contents of approx_flags_dict ( all ArrayLike[float32], (1,) ):
        #   out_dict['log_one_minus_alpha']
        #   out_dict['log_beta']
        #   out_dict['log_one_minus_gamma']
        #   out_dict['log_gamma']
        tkf_params_dict, approx_flags_dict = self.tkf_function(mu = mu, 
                                                               offset = offset,
                                                               t_array = t_array,
                                                               unique_time_per_sample = unique_time_per_sample)
        
        # record values to tensorboard
        self.maybe_record_indel_interms_to_tboard(mu = mu,
                                                  offset = offset,
                                                  r_extend = r_extend,
                                                  tkf_params_dict = tkf_params_dict,
                                                  sow_intermediates = sow_intermediates)
        
        # cond_logprob is either:
        # (T, B, L_align, 4, 4), or
        # (B, L_align, 4, 4)
        cond_logprob =  self.cond_logprob_fn( tkf_params_dict = tkf_params_dict,
                                              r_extend = r_extend,
                                              offset = offset,
                                              unique_time_per_sample = unique_time_per_sample ) 
        
        intermed_params_dict = {'lambda': mu * (1-offset),
                                'mu': mu,
                                'r_extend': r_extend}
        
        return cond_logprob, approx_flags_dict, intermed_params_dict
        
    
    def get_r_ext_prob(self, 
                       *args, 
                       **kwargs):
        """
        placeholder; r would actually be zero, but this plays poorly with 
          log-transformed functions
        """
        return None


class TKF92LocalRateLocalFragSize(LocalTKF91):
    """
    CONDITIONAL LOG-PROBABILITY!!! calculating logP(align_i|align_{i-1},Anc,Desc_{...i-1},t)
    
    B = batch size; number of samples
    T = number of branch lengths; this could be: 
        > an array of times for all samples (T; marginalize over these later)
        > an array of time per sample (T=B)
        
    Initialize with
    ----------------
    config : dict 
        config["init_mu_offset_logits"] : Tuple, (1, 1, 2)
            initial values for logits that determine mu, offset
            DEFAULT: -2, -5
        
        config["mu_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines mu
            DEFAULT: -1e-4, 2
        
        config["offset_range"] : Tuple, (2,)
            range for bound sigmoid activation that determines offset 
            (which determines lambda)
            DEFAULT: -1e-4, 0.333
            
        config["r_range"]
            range for bound sigmoid activation that determines TKF r
            DEFAULT: -1e-4, 0.999
            
    name : str
        class name, for flax
    
    
    Methods here
    -------------
    setup
    __call__
    
    Inherited from LocalTKF91
    ----------------------------
    setup (run before this setup)
    get_r_ext_prob (not used)
    """
    config: dict
    name: str
    
    def setup(self):
        """
        Flax Module Parameters
        -----------------------
        tkf_mu_offset: 
            kernel: [fill in later]
            bias: [fill in later]
            
        
        projection to r extend:
            kernel: [fill in later]
            bias: [fill in later]
        
        """
        super().setup()
        
        ### extra stuff for TKF92: R extension probability 
        # overwrite cond_logprob_fn
        self.cond_logprob_fn = logprob_tkf92
        
        # setting limits for bound sigmoid function
        self.r_extend_min_val, self.r_extend_max_val = self.config.get( 'r_range', 
                                                                [1e-4, 0.999] )
        
        # linear projection to R
        name = f'{self.name}/Project to R extension prob'
        self.final_project_to_r = nn.Dense(features = 1,
                                           use_bias = True,
                                           name = name)
        
    def get_r_ext_prob(self,
                       datamat):
        # r: (B, L_align, H) -> (B, L_align, 1) -> (B, L_align)
        r_logits = self.final_project_to_r(datamat)[...,0]
        r_extend = self.apply_bound_sigmoid_activation( logits = r_logits,
                                                        min_val = self.r_extend_min_val,
                                                        max_val = self.r_extend_max_val,
                                                        param_name = 'tkf92 r',
                                                        sow_intermediates = False )
        return r_extend
        