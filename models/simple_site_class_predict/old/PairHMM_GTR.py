#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 04:33:00 2025

@author: annabel


functions:
===========
'_lse_over_equl_logprobs_per_class',
'_lse_over_match_logprobs_per_class',
'_score_alignment',


flax modules:
==============
'IndpSites',
'IndpSitesLoadAll',
'IndpSitesHKY85',
'IndpSitesHKY85LoadAll',
"""
import numpy as np
import pickle

# jumping jax and leaping flax
from flax import linen as nn
import jax
from jax._src.typing import Array, ArrayLike
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax.scipy.special import logsumexp

from models.BaseClasses import ModuleBase
from models.simple_site_class_predict.emission_models import (EqulDistLogprobsFromCounts,
                                                              EqulDistLogprobsPerClass,
                                                              EqulDistLogprobsFromFile,
                                                              GTRRateMat,
                                                              GTRRateMatFromFile,
                                                              SiteClassLogprobs,
                                                              SiteClassLogprobsFromFile,
                                                              HKY85RateMat,
                                                              HKY85RateMatFromFile,
                                                              get_cond_logprob_emit_at_match_per_class,
                                                              get_joint_logprob_emit_at_match_per_class,
                                                              lse_over_match_logprobs_per_class,
                                                              lse_over_equl_logprobs_per_class)
from utils.pairhmm_helpers import (bound_sigmoid,
                                   safe_log,
                                   log_one_minus_x)



###############################################################################
### Helper functions   ########################################################
###############################################################################
def _score_geom_length( lens: ArrayLike,
                        logprob_emit: ArrayLike ):
    """
    score a geometric length
    
    B = batch; number of alignments
    N = sequence length
    
    
    Arguments
    ----------
    lens : ArrayLike, (B,)
    logprob_emit : ArrayLike, (1,)
        
    Returns
    -------
    length_score : ArrayLike, (B,)
    
    """
    # 1 - P(emit)
    # log( 1 - P(emit) )
    log_one_minus_prob_emit = log_one_minus_x(logprob_emit) #(1,)
    
    # P(length) = P(emit)**(N) * ( 1-P(emit) )
    # logP(length) = N*P(emit) + log( 1-P(emit) )
    length_score = lens * logprob_emit + log_one_minus_prob_emit #(B,)
    
    return length_score


def _score_alignment( subCounts: ArrayLike,
                      logprob_emit_at_match: ArrayLike,
                      logprob_emit: ArrayLike ):
    """
    score an alignment from summary counts
    
    B = batch; number of alignments
    A = alphabet size
    T = branch length; time
    
    
    Arguments
    ----------
    subCounts : ArrayLike, (B,A,A)
    logprob_emit_at_match : ArrayLike, (T,A,A)
        
    Returns
    -------
    logprob_perSamp_perTime : ArrayLike, (T,B)
    
    """
    ### emissions at match sites
    # subCounts is (B,A,A)
    # logprob_emit_at_match is (T,A,A)
    # match_emit_score is (T,B)
    match_emit_score = jnp.einsum('tij,bij->tb',
                                  logprob_emit_at_match, 
                                  subCounts)
    
    
    ### length is geometrically distributed
    align_lens = subCounts.sum(axis=(-1,-2))
    length_score = _score_geom_length( lens = align_lens,
                                       logprob_emit = logprob_emit ) #(B,)
    
    
    ### sum
    logprob_perSamp_perTime = (match_emit_score +
                               length_score[None,:]) #(T,B)
    
    return logprob_perSamp_perTime


def _score_sequence( seqCounts: ArrayLike,
                     log_equl_dist: ArrayLike,
                     logprob_emit: ArrayLike ):
    """
    score a single sequence from summary counts
    
    B = batch; number of alignments
    A = alphabet size
    
    
    Arguments
    ----------
    seqCounts : ArrayLike, (B,A)
    log_equl_dist : ArrayLike, (A)
        
    Returns
    -------
    logprob_perSamp_perTime : ArrayLike, (B,)
    
    """
    ### emissions at insert sites
    # seqCounts is (B,A)
    # log_equl_dist is (A,)
    # seq_emit_score is (B)
    seq_emit_score = jnp.einsum('i,bi->b',
                                log_equl_dist, 
                                seqCounts)
    
    
    ### length is geometrically distributed
    seq_lens = seqCounts.sum(axis=(-1)) #(B,)
    length_score = _score_geom_length( lens = seq_lens,
                                       logprob_emit = logprob_emit ) #(B,)
    
    
    ### sum
    logprob_perSamp_perTime = seq_emit_score + length_score #(B,)
    
    return logprob_perSamp_perTime
    
    
    
    
###############################################################################
### PairHMM with GTR substitution model, some TKF indel model   ###############
###############################################################################
class SubsOnlyIndpSites(ModuleBase):
    """
    General time reversible model, with no indels or gap positions 
        (they are ignored)
    
    Initialize with
    ----------------
    config : dict
        config['num_emit_site_classes'] :  int
            number of emission site classes
        
        config['exponential_dist_param'] : int, optional
            There is an exponential prior over time; this provides the
            parameter for this during marginalization over times
        
    name : str
        class name, for flax
    
    Main methods here
    -----------------
    setup
    
    __call__
        unpack batch and calculate logP(anc, desc, align)
    
    calculate_all_loglikes
        calculate logP(anc, desc, align), logP(anc), logP(desc), and
        logP(desc, align | anc)
    
    write_params
        write parameters to files
    
    return_bound_sigmoid_limits
        after initializing model, get the limits for bound_sigmoid activations
    
    
    Other methods
    --------------
    _init_rate_matrix_module
        decide what function to use for rate matrix
    
    _get_scoring_matrices
        get all matrices needed to score sequences
    
    _marginalize_over_times
        handles time marginalization
    
    _joint_logprob_align
        calculate logP(anc, desc, align)
    
    
    Methods inherited from models.model_utils.BaseClasses.ModuleBase
    -----------------------------------------------------------------
    sow_histograms_scalars
        for tensorboard logging
    """
    config: dict
    name: str
    
    def setup(self):
        num_emit_site_classes = self.config['num_emit_site_classes']
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        
        ### how to score single sequences
        if num_emit_site_classes == 1:
            self.equl_dist_module = EqulDistLogprobsFromCounts(config = self.config,
                                                       name = f'get equilibrium')
        elif num_emit_site_classes > 1:
            # activation function is softmax
            self.equl_dist_module = EqulDistLogprobsPerClass(config = self.config,
                                                     name = f'get equilibrium')
        
        
        ### rate matrix to score emissions from match sites
        # activation function is scaled+shifted sigmoid 
        out = self._init_rate_matrix_module(self.config)
        self.rate_matrix_module, self.subst_model_type = out
        del out
        
        
        ### length is geometrically distributed with parameter p = P(emit)
        # activation function is standard sigmoid
        # init with P(emit) = sigmoid(3.0) ~= 0.95257413
        self.p_emit_logit = self.param('geom length p',
                                       jnp.full((1,), 3.0, dtype=jnp.float32)
                                       )
        
        ### probabilities for rate classes themselves
        # activation function is either scaled+shifted sigmoid or softplus
        self.site_class_probability_module = SiteClassLogprobs(config = self.config,
                                                  name = f'get site class probabilities')
        
    
    def __call__(self,
                 batch,
                 t_array,
                 sow_intermediates: bool):
        """
        Use this during active model training
        
        returns:
            - loss: average across the batch, based on length-normalized
                    joint log-likelihood
                    
            - aux_dict: has the following keys and values
              1.) 'joint_neg_logP': sum down the length
              2.) 'joint_neg_logP_length_normed': sum down the length,  
                  normalized by alignment length
        """
        # get scoring matrices for joint probability
        out = self._get_scoring_matrices(t_array=t_array,
                                        sow_intermediates=sow_intermediates)
        log_equl_dist = out['log_equl_dist']
        joint_logprob_emit_at_match = out['joint_logprob_emit_at_match']
        del out
        
        # for geometric length
        logprob_emit = nn.log_sigmoid(self.p_emit_logit) #(1,)
        
        if sow_intermediates:
            lab = f'{self.name}/geometric length P(emit)'
            self.sow_histograms_scalars(mat = logprob_emit,
                                        label=lab,
                                        which='scalars')
            del lab
            
        # calculate loglikelihoods
        aux_dict = self._joint_logprob_align( batch=batch,
                                              t_array=t_array,
                                              joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                              logprob_emit=logprob_emit,
                                              sow_intermediates=sow_intermediates )
        
        loss = jnp.mean( aux_dict['joint_neg_logP_length_normed'] )
        aux_dict['used_tkf_beta_approx'] = ( jnp.array([False]), jnp.array([False]) )
        
        return loss, aux_dict
    
    
    def calculate_all_loglikes(self,
                               batch,
                               t_array):
        """
        Use this during final eval
        
        returns all four loglikelihoods in a dictionary:
        
        1.) 'joint_neg_logP': P(anc, desc, align)
        2.) 'joint_neg_logP_length_normed': P(anc, desc, align), normalized 
            by desired length (set by self.norm_by)
        3.) 'anc_neg_logP': P(anc)
        4.) 'anc_neg_logP_length_normed': P(anc), normalized by ancestor 
            length
        5.) 'desc_neg_logP': P(desc)
        6.) 'desc_neg_logP_length_normed': P(desc), normalized by descendant 
            length
        7.) 'cond_neg_logP': P(desc, align | anc)
        8.) 'cond_neg_logP_length_normed': P(desc, align | anc), normalized 
            by desired length (set by self.norm_by)
        """
        
        ####################################
        ### prep matrices, unpack values   #
        ####################################
        # unpack batch: (B, ...)
        subCounts = batch[0] #(B, 20, 20)
        anc_emitCounts = subCounts.sum(axis=-1) #(B, 20)
        desc_emitCounts = subCounts.sum(axis=-2) #(B, 20)
        seq_lenths = subCounts.sum(axis=(-2, -1)) #(B,)
        
        # get scoring matrices for joint and marginal probabilities 
        #   (conditional comes from dividing joint by marginal)
        out = self._get_scoring_matrices(t_array=t_array,
                                        sow_intermediates=False)
                                        
        log_equl_dist = out['log_equl_dist'] #(A,)
        joint_logprob_emit_at_match = out['joint_logprob_emit_at_match'] #(T,A,A)
        del out

        # for geometric length
        logprob_emit = nn.log_sigmoid(self.p_emit_logit) #(1,)

        
        #########################
        ### joint probability   #
        #########################
        # out has two items:
        # joint_neg_logP: (T,B)
        # joint_neg_logP_length_normed: (T,B)
        out = self._joint_logprob_align( batch=batch,
                                         t_array=t_array,
                                         joint_logprob_emit_at_match=joint_logprob_emit_at_match,
                                         logprob_emit=logprob_emit,
                                         sow_intermediates=False ) #(T,B)
        
        
        ##############################
        ### marginal probabilities   #
        ##############################
        # score ancestor
        anc_neg_logP = -_score_sequence( seqCounts = anc_emitCounts,
                                         log_equl_dist = log_equl_dist,
                                         logprob_emit = logprob_emit ) #(T,B)
        anc_neg_logP_length_normed = anc_neg_logP / seq_lenths[None,:] #(T,B)
        
        out['anc_neg_logP'] = anc_neg_logP #(T,B)
        out['anc_neg_logP_length_normed'] = anc_neg_logP_length_normed #(T,B)
        
        # score descendant
        desc_neg_logP = -_score_sequence( seqCounts = desc_emitCounts,
                                          log_equl_dist = log_equl_dist,
                                          logprob_emit = logprob_emit ) #(T,B)
        desc_neg_logP_length_normed = desc_neg_logP / seq_lenths[None,:] #(T,B)
        
        out['desc_neg_logP'] = desc_neg_logP #(T,B)
        out['desc_neg_logP_length_normed'] = desc_neg_logP_length_normed #(T,B)
        
        
        #####################################################
        ### calculate conditional from joint and marginal   #
        #####################################################
        cond_neg_logP = -( -out['joint_neg_logP'] - -anc_neg_logP ) #(T,B)
        cond_neg_logP_length_normed = cond_neg_logP / seq_lenths[None,:] #(T,B)
        
        out['cond_neg_logP'] = cond_neg_logP
        out['cond_neg_logP_length_normed'] = cond_neg_logP_length_normed
        
        return out
    
    
    def write_params(self,
                     t_array,
                     out_folder: str):
        with open(f'{out_folder}/activations_used.tsv','w') as g:
            act = self.rate_matrix_module.rate_mult_activation
            g.write(f'activation for rate multipliers: {act}\n')
            g.write(f'activation for exchangeabiliites: bound_sigmoid\n')
        
        
        ##########################
        ### the final matrices   #
        ##########################  
        out = self._get_scoring_matrices(t_array=t_array,
                                        sow_intermediates=False)
        
        scaled_rate_mat_per_class = out['scaled_rate_mat_per_class']
        for c in range(scaled_rate_mat_per_class.shape[0]):
            mat_to_save = scaled_rate_mat_per_class[c,...]
            
            with open(f'{out_folder}/class-{c}_rate_matrix_times_rho.npy', 'wb') as g:
                np.save(g, mat_to_save)
            
            np.savetxt( f'{out_folder}/ASCII_class-{c}_rate_matrix_times_rho.tsv', 
                        np.array(mat_to_save), 
                        fmt = '%.4f',
                        delimiter= '\t' )
            
            del mat_to_save, g
            

        # matrix that you apply expm() to
        to_expm = np.squeeze( out['to_expm'] )
        
        with open(f'{out_folder}/to_expm.npy', 'wb') as g:
            np.save(g, to_expm)
        
        if len(to_expm.shape) <= 2:
            np.savetxt( f'{out_folder}/ASCII_to_expm.tsv', 
                        to_expm, 
                        fmt = '%.4f',
                        delimiter= '\t' )
        
        del to_expm, g
        
        # other emission matrices; exponentiate them first
        for key in ['log_equl_dist', 
                    'joint_logprob_emit_at_match']:
            mat = np.exp(out[key])
            new_key = key.replace('logprob','prob')
            
            with open(f'{out_folder}/{new_key}.npy', 'wb') as g:
                np.save(g, mat)
            
            mat = np.squeeze(mat)
            if len(mat.shape) <= 2:
                np.savetxt( f'{out_folder}/ASCII_{new_key}.tsv', 
                            np.array(mat), 
                            fmt = '%.4f',
                            delimiter= '\t' )
            
            del key, mat, g
        
        
        ###############
        ### extract   #
        ###############
        ### site class probs
        if 'class_logits' in dir(self.site_class_probability_module):
            class_probs = nn.softmax(self.site_class_probability_module.class_logits)
            with open(f'{out_folder}/PARAMS_class_probs.txt','w') as g:
                [g.write(f'{elem.item()}\n') for elem in class_probs]
        
        
        ### emissions
        # exchangeabilities
        if 'exchangeabilities_logits_vec' in dir(self.rate_matrix_module):
            exch_logits = self.rate_matrix_module.exchangeabilities_logits_vec
            exchangeabilities = self.rate_matrix_module.exchange_activation( exch_logits )
            
            if self.subst_model_type == 'GTR':
                np.savetxt( f'{out_folder}/PARAMS_exchangeabilities.tsv', 
                            np.array(exchangeabilities), 
                            fmt = '%.4f',
                            delimiter= '\t' )
                
                with open(f'{out_folder}/PARAMS_exchangeabilities.npy','wb') as g:
                    jnp.save(g, exchangeabilities)
            
            elif self.subst_model_type == 'HKY85':
                with open(f'{out_folder}/PARAMS_HKY85RateMat_model.txt','w') as g:
                    g.write(f'transition rate, ti: {exchangeabilities[1]}')
                    g.write(f'transition rate, tv: {exchangeabilities[0]}')
                
        # emissions: rate multipliers
        if 'rate_mult_logits' in dir(self.rate_matrix_module):
            rate_mult_logits = self.rate_matrix_module.rate_mult_logits
            rate_mult = self.rate_matrix_module.rate_multiplier_activation( rate_mult_logits )

            with open(f'{out_folder}/PARAMS_rate_multipliers.txt','w') as g:
                [g.write(f'{elem.item()}\n') for elem in rate_mult]
        
        # emissions: equilibrium distribution
        if 'logits' in dir(self.equl_dist_module):
            equl_logits = self.equl_dist_module.logits
            equl_dist = nn.softmax( equl_logits, axis=1 )
            
            np.savetxt( f'{out_folder}/PARAMS_equilibriums.tsv', 
                        np.array(equl_dist), 
                        fmt = '%.4f',
                        delimiter= '\t' )
            
            with open(f'{out_folder}/PARAMS-ARR_equilibriums.npy','wb') as g:
                jnp.save(g, equl_dist)
                
                
        ### sequence length
        geom_p_emit = nn.sigmoid(self.p_emit_logit) #(1,)
        with open(f'{out_folder}/PARAMS_geom_seq_len.txt','w') as g:
            g.write(f'P(emit): {geom_p_emit}\n')
            g.write(f'1-P(emit): {1 - geom_p_emit}\n')
        
    
    def return_bound_sigmoid_limits(self):
        ### rate_matrix_module
        # exchangeabilities
        exchange_min_val = self.rate_matrix_module.exchange_min_val
        exchange_max_val = self.rate_matrix_module.exchange_max_val
        
        #rate multiplier
        if self.rate_mult_activation == 'bound_sigmoid':
            rate_mult_min_val = self.rate_matrix_module.rate_mult_min_val
            rate_mult_max_val = self.rate_matrix_module.rate_mult_max_val
        
        params_range = { "exchange_min_val": exchange_min_val,
                         "exchange_max_val": exchange_max_val,
                         "rate_mult_min_val": rate_mult_min_val,
                         "rate_mult_max_val": rate_mult_max_val
                         }
        
        return params_range
    
    def _init_rate_matrix_module(self, config):
        mod = GTRRateMat( config = self.config,
                          name = f'get rate matrix' )
        return mod, 'GTR'
        
    def _get_scoring_matrices(self,
                             t_array,
                             sow_intermediates: bool):
        # Probability of each site class; is one, if no site clases
        log_class_probs = self.site_class_probability_module(sow_intermediates = sow_intermediates) #(C,)
        
        
        ######################################################
        ### build log-transformed equilibrium distribution   #
        ### use this to score single sequences               #
        ######################################################
        log_equl_dist_per_class = self.equl_dist_module(sow_intermediates = sow_intermediates) # (C, A)
        log_equl_dist = lse_over_equl_logprobs_per_class(log_class_probs = log_class_probs,
                                                         log_equl_dist_per_class = log_equl_dist_per_class) #(A,)
        
        
        ####################################################
        ### build substitution log-probability matrix      #
        ### use this to score paired emissions             #
        ####################################################
        # rho * Q
        scaled_rate_mat_per_class = self.rate_matrix_module(logprob_equl = log_equl_dist_per_class,
                                                            sow_intermediates = sow_intermediates) #(C, A, A)
        
        # conditional probability
        # cond_logprob_emit_at_match_per_class is (T, C, A, A)
        # to_expm is (T, C, A, A)
        out = get_cond_logprob_emit_at_match_per_class(t_array = t_array,
                                                        scaled_rate_mat_per_class = scaled_rate_mat_per_class)
        cond_logprob_emit_at_match_per_class, to_expm = out 
        del out
        
        # joint probability
        joint_logprob_emit_at_match_per_class = get_joint_logprob_emit_at_match_per_class( cond_logprob_emit_at_match_per_class = cond_logprob_emit_at_match_per_class,
                                                                        log_equl_dist_per_class = log_equl_dist_per_class) #(T, C, A, A)
        joint_logprob_emit_at_match = lse_over_match_logprobs_per_class(log_class_probs = log_class_probs,
                                               joint_logprob_emit_at_match_per_class = joint_logprob_emit_at_match_per_class) #(T, A, A)
        
        out_dict = {'log_equl_dist': log_equl_dist, #(A,)
                    'joint_logprob_emit_at_match': joint_logprob_emit_at_match, #(T,A,A)
                    'rate_mat_times_rho': scaled_rate_mat_per_class, #(C,A,A)
                    'to_expm': to_expm, #(T,C,A,A)
                    'cond_logprob_emit_at_match': cond_logprob_emit_at_match_per_class} #(T,C,A,A)
        return out_dict
    
    def _marginalize_over_times(self,
                               logprob_perSamp_perTime,
                               exponential_dist_param,
                               t_array,
                               sow_intermediates: bool):
        ### constants to add (multiply by)
        # logP(t_k) = exponential distribution
        logP_time = ( jnp.log(exponential_dist_param) - 
                      (exponential_dist_param * t_array) )
        log_t_grid = jnp.log( t_array[1:] - t_array[:-1] )
        
        # kind of a hack, but repeat the last time array value
        log_t_grid = jnp.concatenate( [log_t_grid, log_t_grid[-1][None] ], axis=0)
        
        
        ### add in log space, multiply in probability space; logsumexp
        logP_perSamp_perTime_withConst = ( logprob_perSamp_perTime +
                                           logP_time[:,None] +
                                           log_t_grid[:,None] )
        
        if sow_intermediates:
            lab = f'{self.name}/time_marginalization/before logsumexp'
            self.sow_histograms_scalars(mat= logP_perSamp_perTime_withConst, 
                                        label=lab, 
                                        which='scalars')
            del lab
        
        
        logP_perSamp_raw = logsumexp(logP_perSamp_perTime_withConst, axis=0)
        
        if sow_intermediates:
            lab = f'{self.name}/time_marginalization/after logsumexp'
            self.sow_histograms_scalars(mat= logP_perSamp_raw, 
                                        label=lab, 
                                        which='scalars')
            del lab
        
        return logP_perSamp_raw
    
    def _joint_logprob_align( self,
                             batch,
                             t_array,
                             joint_logprob_emit_at_match,
                             logprob_emit,
                             sow_intermediates: bool):
        # unpack batch: (B, ...)
        subCounts = batch[0] #(B, 20, 20)
        
        # score alignments
        joint_logprob_perSamp_perTime = _score_alignment( subCounts = subCounts,
                                                          logprob_emit_at_match = joint_logprob_emit_at_match,
                                                          logprob_emit = logprob_emit ) #(T, B)
        
        # marginalize over times, if required
        if t_array.shape[0] > 1:
            joint_neg_logP = -self._marginalize_over_times(logprob_perSamp_perTime = joint_logprob_perSamp_perTime,
                                        exponential_dist_param = self.exponential_dist_param,
                                        t_array = t_array,
                                        sow_intermediates = sow_intermediates) #(B,)
        else:
            joint_neg_logP = -joint_logprob_perSamp_perTime[0,:] #(B,)
        
        # normalize by length of alignment
        joint_neg_logP_length_normed = joint_neg_logP / subCounts.sum(axis=(-2, -1))[None,:]
        
        return {'joint_neg_logP': joint_neg_logP,
                'joint_neg_logP_length_normed': joint_neg_logP_length_normed}
    
    
class SubsOnlyIndpSitesLoadAll(SubsOnlyIndpSites):
    """
    same as SubsOnlyIndpSites, but load values (i.e. no free parameters)
    
    
    Initialize with
    ----------------
    config : dict
        config['num_emit_site_classes'] :  int
            number of emission site classes
            
        config['exponential_dist_param'] : int, optional
            There is an exponential prior over time; this provides the
            parameter for this during marginalization over times
        
        config['filenames']['geom_length_params_file'] : str
            Name of the file to load geometric length parameter from ()
        
    name : str
        class name, for flax
    
    Methods
    --------
    setup
    
    write_params
        write parameters to files
    
    _init_rate_matrix_module
        decide what function to use for rate matrix
    
    inherited from IndpSites
    ----------------------------
    __call__
        unpack batch and calculate logP(anc, desc, align)
    
    calculate_all_loglikes
        calculate logP(anc, desc, align), logP(anc), logP(desc), and
        logP(desc, align | anc)
    
    return_bound_sigmoid_limits
        after initializing model, get the limits for bound_sigmoid activations
    
    _get_scoring_matrices
        get all matrices needed to score sequences
    
    _marginalize_over_times
        handles time marginalization
    
    _joint_logprob_align
        calculate logP(anc, desc, align)
        
    """
    config: dict
    name: str
    
    def setup(self):
        num_emit_site_classes = self.config['num_emit_site_classes']
        self.exponential_dist_param = self.config.get('exponential_dist_param', 1)
        file_with_transit_probs = self.config['filenames']['geom_length_params_file']
        
        ### how to score emissions from indel sites
        self.equl_dist_module = EqulDistLogprobsFromFile(config = self.config,
                                                   name = f'get equilibrium')
        
        
        ### rate matrix to score emissions from match sites
        out = self._init_rate_matrix_module(self.config)
        self.rate_matrix_module, self.subst_model_type = out
        del out
        
        
        ### length is geometrically distributed with parameter p = P(emit)
        # if saved as a numpy matrix
        if file_with_transit_probs.endswith('.npy'):
            with open(file_with_transit_probs,'rb') as f:
                p_emit = jnp.load(f) #(1,)
        
        # if saved as a flat text file
        elif ( file_with_transit_probs.endswith('.txt') ) or ( file_with_transit_probs.endswith('.tsv') ):
            with open(file_with_transit_probs,'r') as f:
                p_emit = jnp.array( [ float(f.readline().strip().split()[-1]) ] ) #(1,)
        
        # convert to logit
        self.p_emit_logit = -jnp.log( (1 / p_emit) - 1 )
        
        
        ### probability of site classes
        self.site_class_probability_module = SiteClassLogprobsFromFile(config = self.config,
                                                 name = f'get site class probabilities')
        
        
    def write_params(self,
                     t_array,
                     out_folder: str):
        out = self._get_scoring_matrices(t_array=t_array,
                                        sow_intermediates=False)
        
        
        scaled_rate_mat_per_class = out['scaled_rate_mat_per_class']
        for c in range(scaled_rate_mat_per_class.shape[0]):
            mat_to_save = scaled_rate_mat_per_class[c,...]
            
            with open(f'{out_folder}/class-{c}_rate_matrix_times_rho.npy', 'wb') as g:
                np.save(g, mat_to_save)
            
            np.savetxt( f'{out_folder}/ASCII_class-{c}_rate_matrix_times_rho.tsv', 
                        np.array(mat_to_save), 
                        fmt = '%.4f',
                        delimiter= '\t' )
            
            del mat_to_save, g
            

        # matrix that you apply expm() to
        to_expm = np.squeeze( out['to_expm'] )
        
        with open(f'{out_folder}/to_expm.npy', 'wb') as g:
            np.save(g, to_expm)
        
        if len(to_expm.shape) <= 2:
            np.savetxt( f'{out_folder}/ASCII_to_expm.tsv', 
                        to_expm, 
                        fmt = '%.4f',
                        delimiter= '\t' )
        
        del to_expm, g
        
        # other emission matrices; exponentiate them first
        for key in ['log_equl_dist', 
                    'joint_logprob_emit_at_match']:
            mat = np.exp(out[key])
            new_key = key.replace('logprob','prob')
            
            with open(f'{out_folder}/{new_key}.npy', 'wb') as g:
                np.save(g, mat)
            
            mat = np.squeeze(mat)
            if len(mat.shape) <= 2:
                np.savetxt( f'{out_folder}/ASCII_{new_key}.tsv', 
                            np.array(mat), 
                            fmt = '%.4f',
                            delimiter= '\t' )
            
            del key, mat, g
            
            
    def _init_rate_matrix_module(self, config):
        mod = GTRRateMatFromFile( config = self.config,
                                  name = f'get rate matrix' )
        return mod, 'GTR'


###############################################################################
### PairHMM with HKY85 substitution model (no indel scores)  ##################
###############################################################################
class SubsOnlyIndpSitesHKY85(SubsOnlyIndpSites):
    """
    Same as SubsOnlyIndpSites, but uses HKY85

    
    Initialize with
    ----------------
    config : dict
        config['num_emit_site_classes'] :  int
            number of emission site classes
        
        config['indel_model_type'] : {TKF91, TKF92}
            which indel model
            
        config['gap_tok'] :  int
            token that represents gaps; usually 43
        
        config['norm_loss_by'] :  {desc_len, align_len}, optional
            what length to normalize loglikelihood by
            Default is 'desc_len'
        
        config['exponential_dist_param'] : int, optional
            There is an exponential prior over time; this provides the
            parameter for this during marginalization over times
        
    name : str
        class name, for flax
    
    Methods
    -------
    _joint_logprob_align
        calculate logP(anc, desc, align)
        
    Inherited from IndpSites
    ---------------------------
    setup
    
    __call__
        unpack batch and calculate logP(anc, desc, align)
    
    calculate_all_loglikes
        calculate logP(anc, desc, align), logP(anc), logP(desc), and
        logP(desc, align | anc)
    
    write_params
        write parameters to files
    
    return_bound_sigmoid_limits
        after initializing model, get the limits for bound_sigmoid activations
        
    _init_rate_matrix_module
        decide what function to use for rate matrix
    
    _get_scoring_matrices
        get all matrices needed to score sequences
    
    _marginalize_over_times
        handles time marginalization
    
    Methods inherited from models.model_utils.BaseClasses.ModuleBase
    -----------------------------------------------------------------
    sow_histograms_scalars
        for tensorboard logging
    """
    config: dict
    name: str

    def _init_rate_matrix_module(self, config):
        mod = HKY85RateMat( config = self.config,
                            name = f'get rate matrix' )
        return mod, 'HKY85'


class SubsOnlyIndpSitesHKY85LoadAll(SubsOnlyIndpSitesLoadAll):
    """
    same as SubsOnlyIndpSitesLoadAll, but use HKY85
    
    
    Initialize with
    ----------------
    config : dict
        config['num_emit_site_classes'] :  int
            number of emission site classes
        
        config['indel_model_type'] : {TKF91, TKF92}
            which indel model
            
        config['gap_tok'] :  int
            token that represents gaps; usually 43
        
        config['norm_loss_by'] :  {desc_len, align_len}, optional
            what length to normalize loglikelihood by
            Default is 'desc_len'
        
        config['exponential_dist_param'] : int, optional
            There is an exponential prior over time; this provides the
            parameter for this during marginalization over times
        
    name : str
        class name, for flax
    
    Methods
    --------
    _init_rate_matrix_module
        decide what function to use for rate matrix
    
    inherited from IndpSitesLoadAll
    ----------------------------------
    setup
    
    write_params
        write parameters to files
    
    inherited from IndpSites
    ----------------------------
    __call__
        unpack batch and calculate logP(anc, desc, align)
    
    calculate_all_loglikes
        calculate logP(anc, desc, align), logP(anc), logP(desc), and
        logP(desc, align | anc)
    
    return_bound_sigmoid_limits
        after initializing model, get the limits for bound_sigmoid activations
    
    _get_scoring_matrices
        get all matrices needed to score sequences
    
    _marginalize_over_times
        handles time marginalization
    
    _joint_logprob_align
        calculate logP(anc, desc, align)
    
    """
    config: dict
    name: str

    def _init_rate_matrix_module(self, config):
        mod = HKY85RateMatFromFile( config = self.config,
                                    name = f'get rate matrix' )
        return mod, 'HKY85'
    
