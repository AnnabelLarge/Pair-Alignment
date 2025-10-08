#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 13:23:18 2025

@author: annabel
"""
import pickle
import numpy as np

def read_mat(file):
    with open(file,'rb') as f:
        return np.load(f)

def read_pkl(file):
    with open(file,'rb') as f:
        return pickle.load(f)

folder = 'NAN-ERR_10TKF92-F81-domain-mix_4Rates_seed8100'


###################
### read params   #
###################
### mixture proportions
# P(c_dom)
domain_class_probs = read_mat(f'{folder}/out_arrs/PARAMS-MAT_test-set_pt0_domain_class_probs.npy') #(C_dom)

# P(c_frag | c_dom)
fragment_class_probs = read_mat(f'{folder}/out_arrs/PARAMS-MAT_test-set_pt0_frag_class_probs.npy') #(C_dom, C_frag)

# P(c_site | c_frag, c_dom)
site_class_probs = read_mat(f'{folder}/out_arrs/PARAMS-MAT_test-set_pt0_site_class_probs.npy') #(C_dom*C_frag, C_sites)

# P(rate_k | c_site, c_frag, c_dom)
rate_mult_probs = read_mat(f'{folder}/out_arrs/PARAMS-MAT_test-set_pt0_rate_mult_probs.npy') #(C_dom*C_frag, C_sites, K_rates)


### joint probabilities of mixtures
# P(c_frag, c_dom) = P(c_frag | c_dom) * P(c_dom)
frag_dom_jointprob = fragment_class_probs * domain_class_probs[:, None] #(C_dom, C_frag)

# P(c_site, c_frag, c_dom) = P(c_sites | c_frag, c_dom) * P(c_frag, c_dom)
frag_dom_jointprob = frag_dom_jointprob.flatten() #(C_dom * C_frag)
site_frag_dom_jointprob = site_class_probs * frag_dom_jointprob[:, None] #(C_dom*C_frag, C_sites)

# P(rate_k, c_site, c_frag, c_dom) = # P(rate_k | c_site, c_frag, c_dom) * P(c_site, c_frag, c_dom)
rate_site_frag_dom_jointprob = rate_mult_probs * site_frag_dom_jointprob[:,:,None] #(C_dom*C_frag, C_sites, K_rates)



### rate multipliers
rate_multipliers = read_mat(f'{folder}/out_arrs/PARAMS-MAT_test-set_pt0_rate_multipliers.npy') #(C_dom*C_frag, C_sites, K_rates)


### equilibrium distributions (already marginalized over site classes and rate multipliers)
equl_dist = read_mat(f'{folder}/out_arrs/PARAMS-MAT_test-set_pt0_equilibriums-per-site-class.npy') #(C_dom*C_frag, C_sites, A)


### fragment-level tkf92 parameters
fragment_indel_params = read_pkl(f'{folder}/out_arrs/PARAMS-DICT_test-set_pt0_fragment_tkf92_indel_params.pkl')
fragment_insert_rate = fragment_indel_params['lambda'] #(C_dom,)
fragment_delete_rate = fragment_indel_params['mu'] #(C_dom,)
fragment_ext_probs = np.array( fragment_indel_params['r_extend'] ) #(C_dom, C_frag)
fragment_mean_lens = 1 / (1 - fragment_ext_probs) #(C_dom, C_frag)
del fragment_indel_params


### domain-level tkf91 parameters
domain_indel_params = read_pkl(f'{folder}/out_arrs/PARAMS-DICT_test-set_pt0_top_level_tkf91_indel_params.pkl')
domain_insert_rate = domain_indel_params['lambda'] #float
domain_delete_rate = domain_indel_params['mu'] #float
del domain_indel_params



#################################
### expected parameter values   #
#################################
### expected rate multiplier (should be one)
# \sum P(k_rate, c_sites, c_frag, c_dom) P( \rho | k_rate, c_sites, c_frag, c_dom )
ave_rate_mult = (rate_multipliers * rate_site_frag_dom_jointprob).sum()
assert np.isclose(exp_rate_mult, 1.0)


### expected equilibrium distribution
# \sum P(c_sites, c_frag, c_dom) P( \pi | c_sites, c_frag, c_dom )
ave_equl_dist = (equl_dist * site_frag_dom_jointprob[:,:,None]).sum(axis=(0,1)) #(A,)


### expected fragment-level TKF92 params
# \sum P(c_dom) P( \lambda | c_dom )
ave_frag_insert = (fragment_insert_rate * domain_class_probs).sum()

# \sum P(c_dom) P( \mu | c_dom )
ave_frag_delete = (fragment_delete_rate * domain_class_probs).sum()

# \sum P(c_frag, c_dom) P( r | c_frag, c_dom )
ave_frag_ext_prob = (fragment_ext_probs.flatten() * frag_dom_jointprob).sum()
ave_mean_frag_len = (fragment_mean_lens.flatten() * frag_dom_jointprob).sum()

