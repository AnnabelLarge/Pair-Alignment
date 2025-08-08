#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 19:35:50 2025

@author: annabel_large
"""
import numpy as np

def load_mat(file):
    with open(file,'rb') as f:
        return np.load(f)


# # transit: pass
# true = load_mat(f'RESULTS_eval_frag-mix/out_arrs/PARAMS-MAT_test-set_pt0_joint_prob_transit_matrix.npy')[:,0,0,...]
# loaded = load_mat(f'RESULTS_train_indp-sites/out_arrs/PARAMS-MAT_test-set_pt0_joint_prob_transit_matrix.npy')
# assert np.allclose(true, loaded), 'transit differs'


# # equl dist: pass
# true = load_mat(f'RESULTS_eval_frag-mix/out_arrs/PARAMS-MAT_test-set_pt0_prob_emit_at_indel.npy')
# loaded = load_mat(f'RESULTS_train_indp-sites/out_arrs/PARAMS-MAT_test-set_pt0_prob_emit_at_indel.npy')
# assert np.allclose(true, loaded), 'equl differs'


# # rate mults: pass
# true = load_mat(f'RESULTS_eval_frag-mix/out_arrs/PARAMS-MAT_test-set_pt0_rate_multipliers.npy')
# loaded = load_mat(f'RESULTS_train_indp-sites/out_arrs/PARAMS-MAT_test-set_pt0_rate_multipliers.npy')
# assert np.allclose(true, loaded), 'rate multipliers differs'

# # site class probs: PASS
# true = load_mat(f'RESULTS_eval_frag-mix/out_arrs/PARAMS-MAT_test-set_pt0_site_class_probs.npy')
# loaded = load_mat(f'RESULTS_train_indp-sites/out_arrs/PARAMS-MAT_test-set_pt0_site_class_probs.npy')
# assert np.allclose(true, loaded), 'PROBS of site classes differs'

# rate mult probs: FAIL
true = load_mat(f'RESULTS_eval_frag-mix/out_arrs/PARAMS-MAT_test-set_pt0_rate_mult_probs.npy')[0,...]
loaded = load_mat(f'RESULTS_train_indp-sites/out_arrs/PARAMS-MAT_test-set_pt0_rate_mult_probs.npy')
assert np.allclose(true, loaded), 'PROBS of rate multipliers differs'





# # joint match: FAIL
# true = load_mat(f'RESULTS_eval_frag-mix/out_arrs/PARAMS-MAT_test-set_pt0_joint_prob_emit_at_match.npy')[:,0,...]
# loaded = load_mat(f'RESULTS_train_indp-sites/out_arrs/PARAMS-MAT_test-set_pt0_joint_prob_emit_at_match.npy')
# assert np.allclose(true, loaded), 'subs differs'

