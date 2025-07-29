#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:33:01 2025

@author: annabel

Load parameters and evaluate likelihoods for markovian
  site class model

"""
# general python
import os
import shutil
from tqdm import tqdm
from time import process_time
from time import time as wall_clock_time
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None # this is annoying
import pickle
from functools import partial
import platform
import argparse
import json

# jax/flax stuff
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.linalg import expm
import optax

# pytorch imports
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# custom function/classes imports (in order of appearance)
from train_eval_fns.build_optimizer import build_optimizer
from utils.write_config import write_config
from utils.edit_argparse import (enforce_valid_defaults,
                                 fill_with_default_values,
                                 share_top_level_args)
from utils.train_eval_utils import setup_training_dir
from utils.train_eval_utils import (determine_seqlen_bin, 
                                           determine_alignlen_bin)
from utils.tensorboard_recording_utils import (write_times,
                                               write_optional_outputs_during_training)
from utils.train_eval_utils import write_timing_file

# specific to training this model
from dloaders.init_full_len_dset import init_full_len_dset



###############
### HELPERS   #
###############
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
    b_for_logsumexp = jnp.array([1.0, -1.0])
    out = logsumexp(a = a_for_logsumexp,
                    b = b_for_logsumexp,
                    axis = -1)
    
    return out

def log_dot_bigger(log_vec, log_mat):
    return logsumexp(log_vec[:, :, None, :] + log_mat, axis=1)




#####################
### MAIN FUNCTION   #
#####################
def train_pairhmm_markovian_sites_pure_jax(args, dataloader_dict: dict):
    ###########################################################################
    ### 0: CHECK CONFIG; IMPORT APPROPRIATE MODULES   #########################
    ###########################################################################
    err = (f"{args.pred_model_type} is not pairhmm_markovian_sites; "+
           f"using the wrong training script")
    assert args.pred_model_type == 'pairhmm_markovian_sites', err
    del err
    
    
    ### edit the argparse object in-place
    fill_with_default_values(args)
    enforce_valid_defaults(args)
    share_top_level_args(args)
    
    if not args.update_grads:
        print('DEBUG MODE: DISABLING GRAD UPDATES')
    
    
    ###########################################################################
    ### 1: SETUP   ############################################################
    ###########################################################################
    ### initial setup of misc things
    # setup the working directory (if not done yet) and this run's sub-directory
    setup_training_dir(args)
    
    # initial random key, to carry through execution
    rngkey = jax.random.key(args.rng_seednum)
    
    # create a new logfile
    with open(args.logfile_name,'w') as g:
        if not args.update_grads:
            g.write('DEBUG MODE: DISABLING GRAD UPDATES\n\n')
            
        g.write(f'PairHMM TKF92 with markovian site classes over emissions\n')
        g.write( (f'  - Number of site classes: '+
                  f'{args.pred_config["num_emit_site_classes"]}\n' )
                )
        g.write(f'  - Normalizing losses by: {args.norm_loss_by}\n')
    
    
    ### save updated config, provide filename for saving model parameters
    finalpred_save_model_filename = args.model_ckpts_dir + '/'+ f'FINAL_PRED.pkl'
    write_config(args = args, out_dir = args.model_ckpts_dir)
    
    
    ### extract data from dataloader_dict
    training_dset = dataloader_dict['training_dset']
    training_dl = dataloader_dict['training_dl']
    test_dset = dataloader_dict['test_dset']
    test_dl = dataloader_dict['test_dl']
    args.pred_config['training_dset_aa_counts'] = training_dset.aa_counts
    
    
    ###########################################################################
    ### 2: INITIALIZE MODEL PARTS, OPTIMIZER  #################################
    ###########################################################################
    print('2: model init')
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'2: model init\n')
        
        
    optimizer = optax.adamw(learning_rate = args.optimizer_config["peak_value"])    
    t_array = training_dset.return_time_array()
    
    # dims
    C = args.pred_config["num_tkf_site_classes"]
    A = args.emission_alphabet_size
    num_exchange = 190 #for proteins, hard code
    T = t_array.shape
    
    
    #######################
    ### initial guesses   #
    #######################
    ### one for all components
    init_lam_logit = 0.1
    init_offset_logit = 0.1
    
    
    ### unique to each component
    out = jax.random.split( rngkey, 3 )
    rngkey, r_extend_key, exch_vec_key = out
    init_exch_vec_logits = jax.random.normal( key=exch_vec_key,
                                              shape=(num_exchange,) )
    init_r_extend_logits = jax.random.normal( key=r_extend_key,
                                              shape=(C,) )
    
    
    ### initial parameter dictionary
    params = {'lam_logits': init_lam_logit,
              'offset_logits': init_offset_logit,
              'r_extend_logits': init_r_extend_logits,
              'exch_vec_logits': init_exch_vec_logits
              }
    hparams = {}
    opt_state = optimizer.init(params)
    
    
    ### only fit these if C > 1:
    if C > 1:
        out = jax.random.split( rngkey, 4 )
        rngkey, equl_dist_key, rate_mult_key, class_probs_key = out
        del out
        
        init_equl_logits = jax.random.normal( key=equl_dist_key,
                                              shape = (C,A) )
        init_rate_mult_logits = jax.random.norm( key=rate_mult_key,
                                                 shape = (C,) )        
        init_class_probs_logits = jax.random.normal( key=class_probs_key,
                                                     shape=(C,) )
        params['equl_logits'] = init_equl_logits
        params['rate_mult_logits'] = init_rate_mult_logits
        params['class_probs_logits'] = init_class_probs_logits
        
    
    else:
        equl_dist = training_dset.aa_counts / training_dset.aa_counts.sum()
        hparams['equl_dist'] = equl_dist[None,:]
    
    
    
    ###########################################################################
    ### 3: DEFINE TRAINING FUNCTION   #########################################
    ###########################################################################
    print(f'3: define training function')
    
    def train_one_batch(batch,
                        params,
                        hparams,
                        max_align_len):
        seqs = batch[1][:, :max_align_len, :]
        B = seqs.shape[0]
        L = seqs.shape[1]
        desc_len = (seqs[:,:,0] != 0).sum(axis=1) - 1
        
        def apply_model(p):
            #####################################
            ### PROBABILITY OF INSERT/DELETES   #
            #####################################
            if C > 1:
                log_equl_dist = jax.nn.log_softmax(p['equl_logits'],
                                               axis=1)
                equl_dist = jnp.exp(log_equl_dist)
                
            elif C == 1:
                log_equl_dist = safe_log( hparams['equl_dist'] )
                equl_dist = hparams['equl_dist']
            
            A = equl_dist.shape[1]
            T = t_array.shape
            
            #########################
            ### BUILD RATE MATRIX   #
            #########################
            exch_raw_vector = bounded_sigmoid(p['exch_vec_logits'],
                                              min_val = 1e-4,
                                              max_val = 10)
            
            # fill matrix: (190,) -> (20, 20)
            upper_tri_exchang = jnp.zeros( (A, A) )
            idxes = jnp.triu_indices(A, k=1)  
            upper_tri_exchang = upper_tri_exchang.at[idxes].set(exch_raw_vector)
            square_exch_mat = (upper_tri_exchang + upper_tri_exchang.T)
            
            # make instantaneous rate matrix: chi * pi = Q
            rate_mat_without_diags = jnp.einsum('ij, cj -> cij', 
                                                square_exch_mat, 
                                                equl_dist)
            
            row_sums = rate_mat_without_diags.sum(axis=2) 
            ones_diag = jnp.eye( A, dtype=bool )[None,:,:]
            ones_diag = jnp.broadcast_to( ones_diag, (C,
                                                      ones_diag.shape[1],
                                                      ones_diag.shape[2]) )
            diags_to_add = -jnp.einsum('ci,cij->cij', row_sums, ones_diag)
            subst_rate_mat = rate_mat_without_diags + diags_to_add
            
            # normalize it
            diag = jnp.einsum("cii->ci", subst_rate_mat) 
            norm_factor = -jnp.sum(diag * equl_dist, axis=1)[:,None,None]
            subst_rate_mat = subst_rate_mat / norm_factor
            
            
            # multiply by rate multipliers: pi(x_c) * matexp( rho * Q * t)
            if C > 1:
                rate_mult = bounded_sigmoid( p['rate_mult_logits'],
                                             min_val = 0.01,
                                             max_val = 10 )
                to_expm = subst_rate_mat * rate_mult[:, None, None]
                
            else:
                rate_mult = jnp.array([1])
                to_exp = subst_rate_mat
            
            
            ####################################
            ### PROBABILITY OF SUBSTITUTIONS   #
            ####################################
            cond_prob = expm(subst_rate_mat * t_array[...,None,None,None])
            cond_logprob = safe_log(cond_prob)
            joint_logprob_subs = cond_logprob + log_equl_dist[None,...,None]
            
            
            
            ################################
            ### BUILD JOINT TKF92 MATRIX   #
            ################################
            ### transform params
            lam = bounded_sigmoid(p['lam_logits'],
                                  min_val = 1e-4,
                                  max_val = 3)
            
            offset = bounded_sigmoid(p['offset_logits'],
                                     min_val = 1e-4,
                                     max_val = 0.8)
            
            mu = lam / (1-offset)
            
            r_ext_prob = bounded_sigmoid( p['r_extend_logits'],
                                          min_val = 1e-4,
                                          max_val = 0.8 )
            
            
            ### alpha, beta, gamma (in log space)
            use_approx = False
            
            final_shape = (T, C)
            r_extend = r_ext_prob
            mu_per_t = mu * t_array
            lam_per_t = lam * t_array
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
                return ( safe_log(1 - args.pred_config["tkf_err"]) + 
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
    
            ### build matrix in log space
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
            log_tkf92_transmat = jnp.stack([ jnp.stack([  log_a,   log_b, log_c_h, log_mi_e], axis=-1),
                               jnp.stack([  log_f,   log_g, log_c_h, log_mi_e], axis=-1),
                               jnp.stack([  log_p,   log_q,   log_r,  log_d_e], axis=-1),
                               jnp.stack([log_s_m[None,:], log_s_i[None,:], log_s_d[None,:],  log_s_e[None,:]], axis=-1)
                              ], axis=-2)
            
            
            #####################################################
            ### jax.lax.scan version of the forward algorithm   #
            #####################################################
            if C > 1:
                log_class_probs = jax.nn.log_softmax( p['class_probs_logits'] )
            
            elif C == 1:
                log_class_probs = jnp.array([0])
            
            ### init with <s> -> any
            # these are all (B,)
            prev_state = seqs[:,0,2]
            anc_toks = seqs[:,1,0]
            desc_toks = seqs[:,1,1]
            curr_state = seqs[:,1,2]
            
            curr_state = jnp.where(curr_state != 5, curr_state, 4)
            prev_state = jnp.where(prev_state != 5, prev_state, 4)
            
            
            ### score emissions
            e = jnp.zeros( (T, C, B) )
            e = e + jnp.where( curr_state == 1,
                                 joint_logprob_subs[:,:,anc_toks-3, desc_toks-3],
                                 0 )
            e = e + jnp.where( curr_state == 2,
                                 log_equl_dist[:,desc_toks-3],
                                 0 )
            e = e + jnp.where( curr_state == 3,
                                 log_equl_dist[:,anc_toks-3],
                                 0 )
            
            ### score log_tkf92_transmations
            # (B,)
            prev_state_expanded = prev_state[None, None, :, None] 
            
            # (T, C, B, col)
            selected_rows = jnp.take_along_axis(arr = log_tkf92_transmat, 
                                                indices = prev_state_expanded-1, 
                                                axis=2)
            
            curr_state_expanded = curr_state[None, None, :, None]
            selected_trans = jnp.take_along_axis( arr = selected_rows,
                                                  indices = curr_state_expanded-1,
                                                  axis = 3)
            
            # T, C, B
            tr = selected_trans[...,0] + log_class_probs[None, :, None]
            
            init_carry = {'alpha': (tr + e),
                          'state': curr_state}
    
                         
            def scan_fn(carry_dict, index):
                prev_alpha = carry_dict['alpha']
                prev_state = carry_dict['state']
                
                prev_state = seqs[:,index-1,2]
                anc_toks = seqs[:,index,0]
                desc_toks = seqs[:,index,1]
                curr_state = seqs[:,index,2]
                
                curr_state = jnp.where(curr_state != 5, curr_state, 4)
                prev_state = jnp.where(prev_state != 5, prev_state, 4)
                
                ### emissions
                e = jnp.zeros( (T, C, B,) )
                e = e + jnp.where( curr_state == 1,
                                     joint_logprob_subs[:,:,anc_toks-3, desc_toks-3],
                                     0 )
                e = e + jnp.where( curr_state == 2,
                                     log_equl_dist[:,desc_toks-3],
                                     0 )
                e = e + jnp.where( curr_state == 3,
                                     log_equl_dist[:,anc_toks-3],
                                     0 )
                
                ### log_tkf92_transmation probabilities
                # (B,)
                prev_state_expanded = prev_state[None, None, :, None] 
            
                # (T, C, B, col)
                selected_rows = jnp.take_along_axis(arr = log_tkf92_transmat, 
                                                    indices = prev_state_expanded-1, 
                                                    axis=2)
            
                curr_state_expanded = curr_state[None, None, :, None]
            
                # T, C, B
                tr = jnp.take_along_axis( arr = selected_rows,
                                          indices = curr_state_expanded-1,
                                          axis = 3)
                tr = tr[...,0]
            
                def main_body(in_carry):
                    # T, C, C, B
                    tr_per_class = tr[:, :, None, :] + log_class_probs[None, None, :, None]
                    
                    # output is T, C, B
                    return e + log_dot_bigger(log_vec = in_carry, log_mat = tr_per_class)
                
                def end(in_carry):
                    # output is T, C, B
                    return tr + in_carry
                
                ### alpha update, in log space ONLY if curr_state is not pad
                new_alpha = jnp.where(curr_state != 0,
                                      jnp.where( curr_state != 4,
                                                 main_body(prev_alpha),
                                                 end(prev_alpha) ),
                                      prev_alpha )
                                         
                new_carry_dict = {'alpha': new_alpha,
                                  'state': curr_state}
                
                return (new_carry_dict, None)
    
            idx_arr = jnp.array( [i for i in range(2, L)] )
            scan_out,_ = jax.lax.scan( f = scan_fn,
                                       init = init_carry,
                                       xs = idx_arr,
                                       length = idx_arr.shape[0] )
            all_fw_scores = logsumexp(scan_out['alpha'], axis=1)[0,:]  / desc_len
            loss = -jnp.mean(all_fw_scores )
            
            aux_dict = {'all_scores': all_fw_scores,
                        'class_probs': jnp.exp(log_class_probs),
                        'lam': lam,
                        'mu': mu,
                        'r_extend': r_ext_prob,
                        'exch_mat': square_exch_mat,
                        'rate_mult': rate_mult}
            
            return loss, aux_dict
        
        
        ################################
        ### OUTSIDE OF APPLY_FN HERE   #
        ################################
        val_grad_fn = jax.value_and_grad(apply_model, has_aux = True)
        (loss, aux_dict), grads = val_grad_fn(params)
        
        return (loss, grads, aux_dict)
    
    
    ###########################################################################
    ### 4: DEFINE EVAL FUNCTION   #############################################
    ###########################################################################
    print(f'4: define eval function')
    
    def eval_one_batch(batch,
                       params,
                       hparams,
                       max_align_len):
        seqs = batch[1][:, :max_align_len, :]
        B = seqs.shape[0]
        L = seqs.shape[1]
        T = t_array.shape
        desc_len = (seqs[:,:,0] != 0).sum(axis=1) - 1
        
        #####################################
        ### PROBABILITY OF INSERT/DELETES   #
        #####################################
        if C > 1:
            log_equl_dist = jax.nn.log_softmax(params['exch_vec_logits'],
                                           axis=1)
            equl_dist = jnp.exp(log_equl_dist)
            
        elif C == 1:
            log_equl_dist = safe_log( hparams['equl_dist'] )
            equl_dist = hparams['equl_dist']
        
        A = equl_dist.shape[1]
        
        
        #########################
        ### BUILD RATE MATRIX   #
        #########################
        exch_raw_vector = bounded_sigmoid(params['exch_vec_logits'],
                                          min_val = 1e-4,
                                          max_val = 10)
        
        # fill matrix: (190,) -> (20, 20)
        upper_tri_exchang = jnp.zeros( (A, A) )
        idxes = jnp.triu_indices(A, k=1)  
        upper_tri_exchang = upper_tri_exchang.at[idxes].set(exch_raw_vector)
        square_exch_mat = (upper_tri_exchang + upper_tri_exchang.T)
        
        # make instantaneous rate matrix: chi * pi = Q
        rate_mat_without_diags = jnp.einsum('ij, cj -> cij', 
                                            square_exch_mat, 
                                            equl_dist)
        
        row_sums = rate_mat_without_diags.sum(axis=2) 
        ones_diag = jnp.eye( A, dtype=bool )[None,:,:]
        ones_diag = jnp.broadcast_to( ones_diag, (C,
                                                  ones_diag.shape[1],
                                                  ones_diag.shape[2]) )
        diags_to_add = -jnp.einsum('ci,cij->cij', row_sums, ones_diag)
        subst_rate_mat = rate_mat_without_diags + diags_to_add
        
        # normalize it
        diag = jnp.einsum("cii->ci", subst_rate_mat) 
        norm_factor = -jnp.sum(diag * equl_dist, axis=1)[:,None,None]
        subst_rate_mat = subst_rate_mat / norm_factor
        
        
        # multiply by rate multipliers: pi(x_c) * matexp( rho * Q * t)
        if C > 1:
            rate_mult = bounded_sigmoid( params['rate_mult_logits'],
                                         min_val = 0.01,
                                         max_val = 10 )
            to_expm = subst_rate_mat * rate_mult[:, None, None]
            
        else:
            rate_mult = jnp.array([1])
            to_exp = subst_rate_mat
        
        
        ####################################
        ### PROBABILITY OF SUBSTITUTIONS   #
        ####################################
        cond_prob = expm(subst_rate_mat * t_array[...,None,None,None])
        cond_logprob = safe_log(cond_prob)
        joint_logprob_subs = cond_logprob + log_equl_dist[None,...,None]
        
        
        
        ################################
        ### BUILD JOINT TKF92 MATRIX   #
        ################################
        ### transform params
        lam = bounded_sigmoid(params['lam_logits'],
                              min_val = 1e-4,
                              max_val = 3)
        
        offset = bounded_sigmoid(params['offset_logits'],
                                 min_val = 1e-4,
                                 max_val = 0.8)
        
        mu = lam / (1-offset)
        
        r_ext_prob = bounded_sigmoid( params['r_extend_logits'],
                                      min_val = 1e-4,
                                      max_val = 0.8 )
        
        
        ### alpha, beta, gamma (in log space)
        use_approx = False
        
        final_shape = (T, C)
        r_extend = r_ext_prob
        mu_per_t = mu * t_array
        lam_per_t = lam * t_array
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
            return ( safe_log(1 - args.pred_config['tkf_err']) + 
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

        ### build matrix in log space
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
        log_tkf92_transmat = jnp.stack([ jnp.stack([  log_a,   log_b, log_c_h, log_mi_e], axis=-1),
                           jnp.stack([  log_f,   log_g, log_c_h, log_mi_e], axis=-1),
                           jnp.stack([  log_p,   log_q,   log_r,  log_d_e], axis=-1),
                           jnp.stack([log_s_m[None,:], log_s_i[None,:], log_s_d[None,:],  log_s_e[None,:]], axis=-1)
                          ], axis=-2)
        
        
        #####################################################
        ### jax.lax.scan version of the forward algorithm   #
        #####################################################
        if C > 1:
            log_class_probs = jax.nn.log_softmax( params['class_probs_logits'] )
        
        elif C == 1:
            log_class_probs = jnp.array([0])
        
        ### init with <s> -> any
        # these are all (B,)
        prev_state = seqs[:,0,2]
        anc_toks = seqs[:,1,0]
        desc_toks = seqs[:,1,1]
        curr_state = seqs[:,1,2]
        
        curr_state = jnp.where(curr_state != 5, curr_state, 4)
        prev_state = jnp.where(prev_state != 5, prev_state, 4)
        
        
        ### score emissions
        e = jnp.zeros( (T, C, B) )
        e = e + jnp.where( curr_state == 1,
                             joint_logprob_subs[:,:,anc_toks-3, desc_toks-3],
                             0 )
        e = e + jnp.where( curr_state == 2,
                             log_equl_dist[:,desc_toks-3],
                             0 )
        e = e + jnp.where( curr_state == 3,
                             log_equl_dist[:,anc_toks-3],
                             0 )
        
        ### score log_tkf92_transmations
        # (B,)
        prev_state_expanded = prev_state[None, None, :, None] 
        
        # (T, C, B, col)
        selected_rows = jnp.take_along_axis(arr = log_tkf92_transmat, 
                                            indices = prev_state_expanded-1, 
                                            axis=2)
        
        curr_state_expanded = curr_state[None, None, :, None]
        selected_trans = jnp.take_along_axis( arr = selected_rows,
                                              indices = curr_state_expanded-1,
                                              axis = 3)
        
        # T, C, B
        tr = selected_trans[...,0] + log_class_probs[None, :, None]
        
        init_carry = {'alpha': (tr + e),
                      'state': curr_state}

                     
        def scan_fn(carry_dict, index):
            prev_alpha = carry_dict['alpha']
            prev_state = carry_dict['state']
            
            prev_state = seqs[:,index-1,2]
            anc_toks = seqs[:,index,0]
            desc_toks = seqs[:,index,1]
            curr_state = seqs[:,index,2]
            
            curr_state = jnp.where(curr_state != 5, curr_state, 4)
            prev_state = jnp.where(prev_state != 5, prev_state, 4)
            
            ### emissions
            e = jnp.zeros( (T, C, B,) )
            e = e + jnp.where( curr_state == 1,
                                 joint_logprob_subs[:,:,anc_toks-3, desc_toks-3],
                                 0 )
            e = e + jnp.where( curr_state == 2,
                                 log_equl_dist[:,desc_toks-3],
                                 0 )
            e = e + jnp.where( curr_state == 3,
                                 log_equl_dist[:,anc_toks-3],
                                 0 )
            
            ### log_tkf92_transmation probabilities
            # (B,)
            prev_state_expanded = prev_state[None, None, :, None] 
        
            # (T, C, B, col)
            selected_rows = jnp.take_along_axis(arr = log_tkf92_transmat, 
                                                indices = prev_state_expanded-1, 
                                                axis=2)
        
            curr_state_expanded = curr_state[None, None, :, None]
        
            # T, C, B
            tr = jnp.take_along_axis( arr = selected_rows,
                                      indices = curr_state_expanded-1,
                                      axis = 3)
            tr = tr[...,0]
        
            def main_body(in_carry):
                # T, C, C, B
                tr_per_class = tr[:, :, None, :] + log_class_probs[None, None, :, None]
                
                # output is T, C, B
                return e + log_dot_bigger(log_vec = in_carry, log_mat = tr_per_class)
            
            def end(in_carry):
                # output is T, C, B
                return tr + in_carry
            
            ### alpha update, in log space ONLY if curr_state is not pad
            new_alpha = jnp.where(curr_state != 0,
                                  jnp.where( curr_state != 4,
                                             main_body(prev_alpha),
                                             end(prev_alpha) ),
                                  prev_alpha )
                                     
            new_carry_dict = {'alpha': new_alpha,
                              'state': curr_state}
            
            return (new_carry_dict, None)

        idx_arr = jnp.array( [i for i in range(2, L)] )
        scan_out,_ = jax.lax.scan( f = scan_fn,
                                   init = init_carry,
                                   xs = idx_arr,
                                   length = idx_arr.shape[0] )
        all_fw_scores = logsumexp(scan_out['alpha'], axis=1)[0,:]  / desc_len
        loss = -jnp.mean(all_fw_scores )
        
        aux_dict = {'all_scores': all_fw_scores,
                    'class_probs': jnp.exp(log_class_probs),
                    'lam': lam,
                    'mu': mu,
                    'r_extend': r_ext_prob,
                    'exch_mat': square_exch_mat,
                    'rate_mult': rate_mult}
        
        return loss, aux_dict
    
    
    ###########################################################################
    ### 5: JIT COMPILE FUNCTIONS AND TRAIN   ##################################
    ###########################################################################
    # fn to handle jit-compiling according to alignment length
    parted_determine_alignlen_bin = partial(determine_alignlen_bin,  
                                            chunk_length = args.chunk_length,
                                            seq_padding_idx = args.seq_padding_idx)
    jitted_determine_alignlen_bin = jax.jit(parted_determine_alignlen_bin)
    del parted_determine_alignlen_bin
    
    # train and eval
    train_fn_jitted = jax.jit(train_one_batch, 
                              static_argnames = ['max_align_len'])
    
    eval_fn_jitted = jax.jit(eval_one_batch,
                             static_argnames = ['max_align_len'])
    
    
    ############################
    ### simple training loop   #
    ############################
    print(f'5: main training loop')
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'5: main training loop\n')
    
    # when to save/what to save
    best_epoch = -1
    best_test_loss = 999999
    best_params = params
    
    # quit training if test loss increases for X epochs in a row
    prev_test_loss = 999999
    early_stopping_counter = 0
    
    # rng key for train
    rngkey, training_rngkey = jax.random.split(rngkey, num=2)
    
    # record time spent at each phase (use numpy array to store)
    all_train_set_times = np.zeros( (args.num_epochs,2) )
    all_eval_set_times = np.zeros( (args.num_epochs,2) )
    all_epoch_times = np.zeros( (args.num_epochs,2) )
    
    for epoch_idx in tqdm( range(args.num_epochs) ):
        epoch_real_start = wall_clock_time()
        epoch_cpu_start = process_time()
        
        ave_epoch_train_loss = 0
        ave_epoch_train_perpl = 0

#__4___8: epoch level (two tabs)          
        ##############################################
        ### 3.1: train and update model parameters   #
        ##############################################
        # start timer
        train_real_start = wall_clock_time()
        train_cpu_start = process_time()
        
        for batch_idx, batch in enumerate(training_dl):
            batch_epoch_idx = epoch_idx * len(training_dl) + batch_idx

#__4___8__12: batch level (three tabs)          
            # unpack briefly to get max len and number of samples in the 
            #   batch; place in some bin (this controls how many jit 
            #   compilations you do)
            batch_max_alignlen = jitted_determine_alignlen_bin(batch = batch).item()
            
            ### run function to train on one batch of samples
            out = train_fn_jitted(batch=batch, 
                                  params=params,
                                  hparams=hparams,
                                  max_align_len=batch_max_alignlen)
            
            loss, grads, aux_dict = out
            del out
            
            
            ### update params
            updates, opt_state = optimizer.update(grads, opt_state, params)  
            params = optax.apply_updates(params, updates)
        
            
            ### add to recorded metrics for this epoch
            weight = args.batch_size / len(training_dset)
            ave_epoch_train_loss += loss * weight
            ave_epoch_train_perpl += np.exp(-loss) * weight
            del weight, loss


#__4___8: epoch level (two tabs)
        ### manage timing
        # stop timer
        train_real_end = wall_clock_time()
        train_cpu_end = process_time()

        # record the CPU+system and wall-clock (real) time
        all_train_set_times[epoch_idx, 0] = train_real_end - train_real_start
        all_train_set_times[epoch_idx, 1] = train_cpu_end - train_cpu_start
        
        del train_cpu_start, train_cpu_end
        del train_real_start, train_real_end
        
        
        ##############################################################
        ### 3.3: also check current performance on held-out test set #
        ##############################################################
        # Note: it's possible to output intermediates for these points too;
        # but right now, that's not collected
        ave_epoch_test_loss = 0
        ave_epoch_test_perpl = 0
        
        # start timer
        eval_real_start = wall_clock_time()
        eval_cpu_start = process_time()
        
        for batch_idx, batch in enumerate(test_dl):
            # unpack briefly to get max len and number of samples in the 
            #   batch; place in some bin (this controls how many jit 
            #   compilations you do)
            batch_max_alignlen = jitted_determine_alignlen_bin(batch = batch).item()
                
            loss, _ = eval_fn_jitted(batch=batch, 
                                          params=params,
                                          hparams=hparams,
                                          max_align_len=batch_max_alignlen)
            
#__4___8__12: batch level (three tabs)
            ### add to total loss for this epoch; weight by number of
            ###   samples/valid tokens in this batch
            weight = args.batch_size / len(test_dset)
            ave_epoch_test_loss += loss * weight
            ave_epoch_test_perpl += jnp.exp(-loss) * weight
            del weight, loss
        
            
#__4___8: epoch level (two tabs) 
        ### manage timing
        # stop timer
        eval_real_end = wall_clock_time()
        eval_cpu_end = process_time()
        
        # also record for later
        all_eval_set_times[epoch_idx, 0] = eval_real_end - eval_real_start
        all_eval_set_times[epoch_idx, 1] = eval_cpu_end - eval_cpu_start
        
        del eval_cpu_start, eval_cpu_end
        del eval_real_start, eval_real_end
        
        
#__4___8: epoch level (two tabs) 
        ##########################################################
        ### 3.5: if this is the best epoch TEST loss,            #
        ###      save the model params and args for later eval   #
        ##########################################################
        if ave_epoch_test_loss < best_test_loss:
            with open(args.logfile_name,'a') as g:
                g.write((f'New best test loss at epoch {epoch_idx}: ') +
                        (f'{ave_epoch_test_loss}\n'))
            
            # update "best" recordings
            best_test_loss = ave_epoch_test_loss
            best_params = params
            best_epoch = epoch_idx
            
            # save param dict and aux dict
            with open(f'{args.model_ckpts_dir}/PARAMS.pkl','wb') as g:
                pickle.dump(best_params, g)
                
            with open(f'{args.model_ckpts_dir}/HYPERPARAMS.pkl','wb') as g:
                pickle.dump(hparams, g)
            
            with open(f'{args.out_arrs_dir}/TRAIN-SET_AUX-DICT.pkl','wb') as g:
                pickle.dump(aux_dict, g)
            
            
            
#__4___8: epoch level (two tabs) 
        ###########################
        ### 3.6: EARLY STOPPING   #
        ###########################
        ### condition 1: if test loss stagnates or starts to go up, compared
        ###              to previous epoch's test loss
        cond1 = jnp.allclose (prev_test_loss, 
                              jnp.minimum (prev_test_loss, ave_epoch_test_loss), 
                              atol=args.early_stop_cond1_atol)

        ### condition 2: if test loss is substatially worse than best test loss
        cond2 = (ave_epoch_test_loss - best_test_loss) > args.early_stop_cond2_gap

        if cond1 or cond2:
            early_stopping_counter += 1
        else:
            early_stopping_counter = 0
        
        if early_stopping_counter == args.patience:
            # record in the raw ascii logfile
            with open(args.logfile_name,'a') as g:
                g.write(f'\n\nEARLY STOPPING AT {epoch_idx}:\n')
            
            # record time spent at this epoch
            epoch_real_end = wall_clock_time()
            epoch_cpu_end = process_time()
            all_epoch_times[epoch_idx, 0] = epoch_real_end - epoch_real_start
            all_epoch_times[epoch_idx, 1] = epoch_cpu_end - epoch_cpu_start
            
            del epoch_cpu_start, epoch_cpu_end
            del epoch_real_start, epoch_real_end
            
            # rage quit
            break
        

#__4___8: epoch level (two tabs) 
        ### before next epoch, do this stuff
        # remember this epoch's loss for next iteration
        prev_test_loss = ave_epoch_test_loss
        
        # record time spent at this epoch
        epoch_real_end = wall_clock_time()
        epoch_cpu_end = process_time()
        all_epoch_times[epoch_idx, 0] = epoch_real_end - epoch_real_start
        all_epoch_times[epoch_idx, 1] = epoch_cpu_end - epoch_cpu_start
        
        
        del epoch_cpu_start, epoch_cpu_end, epoch_real_start, epoch_real_end
        
        
    ###########################################################################
    ### loop through dataloaders and  #########################################
    ### score with best params        #########################################
    ###########################################################################
    del params
    
    with open(args.logfile_name,'a') as g:
        g.write(f'SCORING ALL TRAIN SEQS\n\n')
    
    ### train set
    final_ave_train_loss = 0
    final_ave_train_perpl = 0
    for batch_idx, batch in enumerate(training_dl):
        batch_max_alignlen = jitted_determine_alignlen_bin(batch = batch).item()
            
        train_loss, train_aux_dict  = eval_fn_jitted(batch=batch, 
                                                     params=best_params,
                                                     hparams=hparams,
                                                     max_align_len=batch_max_alignlen)
        
        with open(f'{args.out_arrs_dir}/FINAL-EVAL-TRAIN-SET_AUX-DICT.pkl','wb') as g:
            pickle.dump(train_aux_dict, g)
        
        final_loglikes = training_dset.retrieve_sample_names(batch[-1])
        final_loglikes['logP/normlength'] = train_aux_dict['all_scores']
        final_loglikes['perplexity'] = jnp.exp( -train_aux_dict['all_scores'] )
        final_loglikes['dataloader_idx'] = batch[-1]
        
        wf = ( args.batch_size / len(training_dset) )
        final_ave_train_loss += final_loglikes['logP/normlength'].mean() * wf
        final_ave_train_perpl  += final_loglikes['perplexity'].mean() * wf
        
        if args.save_per_sample_losses:
            final_loglikes.to_csv((f'{logfile_dir}/train-set_pt{batch_idx}_'+
                                  'FINAL-LOGLIKES.tsv'), sep='\t')
            
        del batch_max_alignlen
    
    
    ### repeat for eval set
    final_ave_test_loss = 0
    final_ave_test_perpl = 0
    for batch_idx, batch in enumerate(test_dl):
        batch_max_alignlen = jitted_determine_alignlen_bin(batch = batch).item()
            
        test_loss, test_aux_dict  = eval_fn_jitted(batch=batch, 
                                                     params=best_params,
                                                     hparams=hparams,
                                                     max_align_len=batch_max_alignlen)
        
        with open(f'{args.out_arrs_dir}/FINAL-EVAL-TEST-SET_AUX-DICT.pkl','wb') as g:
            pickle.dump(test_aux_dict, g)
        
        final_loglikes = test_dset.retrieve_sample_names(batch[-1])
        final_loglikes['logP/normlength'] = test_aux_dict['all_scores']
        final_loglikes['perplexity'] = jnp.exp( -test_aux_dict['all_scores'] )
        final_loglikes['dataloader_idx'] = batch[-1]
        
        wf = ( args.batch_size / len(test_dset) )
        final_ave_test_loss += final_loglikes['logP/normlength'].mean() * wf
        final_ave_test_perpl  += final_loglikes['perplexity'].mean() * wf
        
        if args.save_per_sample_losses:
            final_loglikes.to_csv((f'{logfile_dir}/test-set_pt{batch_idx}_'+
                                  'FINAL-LOGLIKES.tsv'), sep='\t')
            
        del batch_max_alignlen
    
    
    ###########################################
    ### update the logfile with final losses  #
    ###########################################
    to_write = {'RUN': args.training_wkdir,
                f'train_ave_{args.loss_type}_loss_seqlen_normed': final_ave_train_loss,
                'train_perplexity': final_ave_test_perpl,
                'train_ece': jnp.exp(-final_ave_train_loss) ,
                f'test_ave_{args.loss_type}_loss_seqlen_normed': final_ave_test_loss,
                'test_perplexity': final_ave_test_perpl,
                'test_ece': jnp.exp(-final_ave_test_loss)
                }
    
    with open(f'{args.logfile_dir}/AVE-LOSSES.tsv','w') as g:
        for k, v in to_write.items():
            g.write(f'{k}\t{v}\n')
    
    
    ##########################################
    ### write best params to text file too   #
    ##########################################
    with open(f'{args.out_arrs_dir}/FINAL-PARAMS.tsv','w') as g:
        g.write(f'insert rate, lambda: {test_aux_dict["lam"]}\n')
        g.write(f'delete rate, mu: {test_aux_dict["mu"]}\n')
        g.write(f'extension prob, r: {test_aux_dict["r_extend"]}\n')
        g.write(f'class_probs: {test_aux_dict["class_probs"]}\n')
        g.write(f'rate multipliers: {test_aux_dict["rate_mult"]}\n')
    
    with open(f'{args.out_arrs_dir}/FINAL-EXCH.npy','wb') as g:
        jnp.save(g, test_aux_dict["exch_mat"])
        
    