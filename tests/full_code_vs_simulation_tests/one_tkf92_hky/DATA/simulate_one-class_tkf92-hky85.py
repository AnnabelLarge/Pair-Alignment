#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 05:03:36 2025

@author: annabel

simulate sequences with multiclass TKF92, assuming independent site classes
"""
import jax
from jax import numpy as jnp

import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
from itertools import product

from funcs.utils import safe_log
from funcs.emission_models import (equl_from_counts, 
                                   hky85_rate_mat,
                                   joint_logprob_subs)
from funcs.transition_models import joint_TKF92_with_class_probs
from funcs.fake_inputs_params import (generate_fake_counts,
                                      generate_fake_alignments)


n_samples = 2000


C = 1
rngnum = 24
rngkey = jax.random.key(rngnum)
param_file_prefixes = f'basic-hky85-tkf92_simul'
dset_name = f'{n_samples*2}SAMPS-{param_file_prefixes}'

# later, randomly validate 10% of the data
validate_size = int( (n_samples * 2) * 0.1 )
select_key, rngkey = jax.random.split(rngkey, 2)
validate_samps = jax.random.choice( key=select_key, 
                                    a= jnp.array(range(n_samples)), 
                                    shape=(validate_size,), 
                                    replace=False)
validate_samps = np.array(validate_samps).tolist()


###############################################################################
### define transition and emission distributions   ############################
###############################################################################
### params associated with transitions
true_lam = jnp.array([0.995/10])
true_mu = jnp.array([1.0/10])
true_r_ext = jnp.array([0.2])
true_class_probs = jnp.array([1.])


### params associated with emissions 
true_ti = jnp.array([1.5])
true_tv = jnp.array([1.1])
true_rate_mult = jnp.array([1.])

# hyperparams
t_array = jnp.array([1.0])

# dims
T = t_array.shape[0]
C = true_class_probs.shape[0]

# marginals (at insert and delete sites)
true_indel_emit_prob = jnp.array([25,  25, 20,  20])
true_indel_emit_prob = true_indel_emit_prob/true_indel_emit_prob.sum()
true_indel_emit_prob = true_indel_emit_prob[None,:]


### make sure everything is the expected size
### DON'T CHANGE THIS SECTION
assert true_lam.shape == (1,)
assert true_mu.shape == (1,)
assert true_r_ext.shape == (1,)
assert true_ti.shape == (1,)
assert true_tv.shape == (1,)

assert true_class_probs.shape == (C,)
assert true_rate_mult.shape == (C,)

assert true_indel_emit_prob.shape == (C, 4)


####################
### get matrices   #
####################
# joint P(anc,desc) (at matches)
true_rate_mat = hky85_rate_mat( ti=true_ti,
                                tv=true_tv,
                                equl=true_indel_emit_prob,
                                norm=True )

pred_rate_mat = hky85_rate_mat( ti=jnp.array([1.7168959379196167]),
                                tv=jnp.array([1.2683467864990234]),
                                equl=true_indel_emit_prob,
                                norm=True )


out = joint_logprob_subs( rate_matrix = (true_rate_mat * true_rate_mult[:, None, None]),
                          equl_prob = true_indel_emit_prob,
                          equl_logprob = safe_log( true_indel_emit_prob ),
                          time = t_array )
true_match_emit_prob = out[0]
del out

out = joint_logprob_subs( rate_matrix = (pred_rate_mat * true_rate_mult[:, None, None]),
                          equl_prob = true_indel_emit_prob,
                          equl_logprob = safe_log( true_indel_emit_prob ),
                          time = t_array )
pred_match_emit_prob = out[0]
del out







#%%
### transitions
out = joint_TKF92_with_class_probs( lam = true_lam,
                                    mu = true_mu,
                                    r_ext_prob = true_r_ext,
                                    class_probs = jnp.array([1]),
                                    use_approx = False,
                                    t_array = t_array,
                                    tkf_err = 1e-4 )
true_transit_logprob, tkf_params = out
del out

true_transit_prob = jnp.exp(true_transit_logprob)


### make sure everything is the expected size
### DON'T CHANGE THIS SECTION
assert true_match_emit_prob.shape == (T, C, 4, 4)
assert true_transit_prob.shape == (T, 1, 1, 4, 4)

true_transit_prob = true_transit_prob[:,0,0,...]



###############################################################################
### simulate alignments   #####################################################
###############################################################################
def pad_seqs(arrays):
    max_length = max(arr.shape[0] for arr in arrays)
    padded_arrays = [jnp.pad(arr, (0, max_length - arr.shape[0])) for arr in arrays]
    return jnp.stack(padded_arrays)

def generate_new_keys(L, rngkey):
    curr_key = jax.random.fold_in(rngkey,  L)
    out = jax.random.split(curr_key, 3)
    state_key = out[0]
    class_key = out[1]
    emit_key = out[2]
    
    return(state_key,
           class_key,
           emit_key)

def emit_at_match(emit_at_match_probs, 
                  c,
                  rngkey):
    nucleotides = [3, 4, 5, 6]
    pairs = list(product(nucleotides, repeat=2))
    
    row_idx, col_idx = zip(*pairs)
    row_idx = jnp.array(row_idx)-3
    col_idx = jnp.array(col_idx)-3
    mat_to_index = emit_at_match_probs[0,:,:]
    flattened_probs = mat_to_index[row_idx, col_idx]
    
    align_col = jax.random.choice( key=rngkey, 
                                   a=jnp.array(pairs), 
                                   replace=True, 
                                   p=flattened_probs )
    
    return jnp.array( [align_col[0].item(), align_col[1].item(), 1] )

def emit_at_indel( emit_at_indel_probs,
                   c,
                   is_ins,
                   rngkey,
                   gap_tok = 7 ):
    indel = jax.random.choice( key=rngkey, 
                               a=jnp.array([3, 4, 5, 6]), 
                               replace=True, 
                               p=emit_at_indel_probs[c,:] )
    
    if is_ins:
        return jnp.array( [gap_tok, indel.item(), 2] )
    
    elif not is_ins:
        return jnp.array( [indel.item(), gap_tok, 3] )

def emit( curr_state,
          prob_arrays,
          class_key,
          emit_key ):
    # unpack
    emit_at_match_probs = prob_arrays[0]
    emit_at_indel_probs = prob_arrays[1]
    del prob_arrays
    
    site_class_at_emit = 0
    
    # what to emit at first state
    # match
    if curr_state == 0:
        emit_key = jax.random.fold_in(emit_key, curr_state)
        align_col = emit_at_match( emit_at_match_probs=emit_at_match_probs, 
                                   c=site_class_at_emit,
                                   rngkey=emit_key )
    
    # insert
    elif curr_state == 1:
        emit_key = jax.random.fold_in(emit_key, curr_state)
        align_col = emit_at_indel( emit_at_indel_probs=emit_at_indel_probs,
                                   c=site_class_at_emit,
                                   is_ins=True,
                                   rngkey=emit_key,
                                   gap_tok=43 )
    
    # delete
    elif curr_state == 2:
        emit_key = jax.random.fold_in(emit_key, curr_state)
        align_col = emit_at_indel( emit_at_indel_probs=emit_at_indel_probs,
                                   c=site_class_at_emit,
                                   is_ins=False,
                                   rngkey=emit_key,
                                   gap_tok=43 )
    
    return align_col, site_class_at_emit

def generate_alignment( class_probs,
                        emit_at_match_probs,
                        emit_at_indel_probs,
                        transit_probs,
                        max_len,
                        step,
                        rng_key ):
    C = class_probs.shape[0]
    alignment = [ jnp.array([1,1,4]) ]
    anc = [ jnp.array(1) ]
    desc = [ jnp.array(1) ]
    
    counts_dict = {}
    counts_dict['match_counts'] = np.zeros( (4,4),dtype='int32' )
    counts_dict['ins_counts'] = np.zeros( (4,),dtype='int32' )
    counts_dict['del_counts'] = np.zeros( (4,),dtype='int32' )
    counts_dict['transit_counts'] = np.zeros( (4, 4),dtype='int32' )
    counts_dict['all_emit_counts'] = np.zeros( (4,),dtype='int32' )
    
    
    ### <s> -> first state
    # random keys
    L = len(alignment)
    int_for_fold_in = 1000000 * step + L
    out = generate_new_keys(int_for_fold_in, rngkey)
    state_key, class_key, emit_key = out
    del out
    
    # which state to transition to; for tkf91, this is the same regardless 
    #   of whether curr_state is start or emit
    #   remap to: M = 0, I = 1, D = 2, E = 3
    next_state_options = jnp.array( [ 0, 1, 2, 3 ] )
    transition_probs = transit_probs[-1, :]
    curr_state = jax.random.choice( key=state_key, 
                                    a=next_state_options, 
                                    replace=True, 
                                    p=transition_probs )
    
    if curr_state != 3:
        # emit
        prob_arrays = (emit_at_match_probs,
                       emit_at_indel_probs)
        align_col, site_class_at_emit = emit( curr_state=curr_state,
                                             prob_arrays=prob_arrays,
                                             class_key=class_key,
                                             emit_key=emit_key )
        anc_tok_generated = align_col[0]
        desc_tok_generated = align_col[1]

        # updates at every alignment column        
        counts_dict['transit_counts'][-1, curr_state]+=1
        alignment.append(align_col)
        
        # updates specific to match
        if curr_state == 0:
            anc.append(anc_tok_generated)
            desc.append(desc_tok_generated)
            counts_dict['match_counts'][anc_tok_generated-3, desc_tok_generated-3] += 1
            counts_dict['all_emit_counts'][anc_tok_generated-3] += 1
            counts_dict['all_emit_counts'][desc_tok_generated-3] += 1
        
        # updates specific to ins
        elif curr_state == 1:
            desc.append(desc_tok_generated)
            counts_dict['ins_counts'][desc_tok_generated-3]+=1
            counts_dict['all_emit_counts'][desc_tok_generated-3] += 1
        
        # updates specific to del
        elif curr_state == 2:
            anc.append(anc_tok_generated)
            counts_dict['del_counts'][anc_tok_generated-3]+=1
            counts_dict['all_emit_counts'][anc_tok_generated-3] += 1
    
    elif curr_state == 3:
        print('Start -> End')
        return None
    
    
    ### keep generating until reaching end (or a max length)
    prev_state = curr_state
    while prev_state != 3:
        # random keys
        L = len(alignment)
        int_for_fold_in = - (1000000 * step + L)
        out = generate_new_keys(int_for_fold_in, rngkey)
        state_key, class_key, emit_key = out
        del out
    
        # remap to: M = 0, I = 1, D = 2, E = 3
        next_state_options = jnp.array( [ 0, 1, 2, 3 ] )
        transition_probs = transit_probs[prev_state, :]
        curr_state = jax.random.choice( key=state_key, 
                                        a=next_state_options, 
                                        replace=True, 
                                        p=transition_probs )
    
        if curr_state != 3:
            prob_arrays = (emit_at_match_probs,
                           emit_at_indel_probs)
            align_col, _  = emit( curr_state=curr_state,
                                                  prob_arrays=prob_arrays,
                                                  class_key=class_key,
                                                  emit_key=emit_key )
            anc_tok_generated = align_col[0]
            desc_tok_generated = align_col[1]
            
            # updates at every alignment column        
            counts_dict['transit_counts'][prev_state, curr_state]+=1
            alignment.append(align_col)
            
            # updates specific to match
            if curr_state == 0:
                anc.append(anc_tok_generated)
                desc.append(desc_tok_generated)
                counts_dict['match_counts'][anc_tok_generated-3, desc_tok_generated-3] += 1
                counts_dict['all_emit_counts'][anc_tok_generated-3] += 1
                counts_dict['all_emit_counts'][desc_tok_generated-3] += 1
            
            # updates specific to ins
            elif curr_state == 1:
                desc.append(desc_tok_generated)
                counts_dict['ins_counts'][desc_tok_generated-3]+=1
                counts_dict['all_emit_counts'][desc_tok_generated-3] += 1
            
            # updates specific to del
            elif curr_state == 2:
                anc.append(anc_tok_generated)
                counts_dict['del_counts'][anc_tok_generated-3]+=1
                counts_dict['all_emit_counts'][anc_tok_generated-3] += 1
        
        # end alignment
        elif curr_state == 3:
            # make sure anc and desc both have some length
            if (len(anc) == 1) or (len(desc)==1):
                print('Empty ancestor or descendant!')
                return None
            
            counts_dict['transit_counts'][prev_state, curr_state]+=1
            anc.append( jnp.array(2) )
            desc.append( jnp.array(2) )
            alignment.append( jnp.array([2, 2, 5]) )
            
            out = {'alignment': jnp.stack(alignment, axis=0).astype('uint8'),
                   'ancestor': jnp.stack(anc, axis=0).astype('uint8'),
                   'descendant': jnp.stack(desc, axis=0).astype('uint8'),
                   'counts_dict': counts_dict}
            return out
        
        if len(alignment) > max_len:
            print('MAX REACHED')
            return None
        
        prev_state = curr_state
    
    
    
    
all_aligned_mats = []
all_anc = []
all_desc = []
all_counts_sets = []
all_hidden_class_sequences = []
max_align_len = 0
max_single_seq_len = 0
step = 0
t = 0
while len(all_aligned_mats) < n_samples:    
    if len(all_aligned_mats) % 500 == 0:
        print( f'{len(all_aligned_mats)}/{n_samples} samples' )
    
    sample_rng_key = jax.random.fold_in(rngkey, step)
    out = generate_alignment( class_probs = true_class_probs,
                              emit_at_indel_probs = true_indel_emit_prob,
                              emit_at_match_probs = true_match_emit_prob[t,...],
                              transit_probs = true_transit_prob[t,...],
                              rng_key = sample_rng_key,
                              step = step,
                              max_len = 1000 )
    
    if out is not None:
        alignment = out['alignment']
        ancestor = out['ancestor']
        descendant = out['descendant']
        counts_dict = out['counts_dict']
        
        align_len = alignment.shape[0]
        anc_len = ancestor.shape[0]
        desc_len = descendant.shape[0]
        
        if max([anc_len, desc_len]) > max_single_seq_len:
            max_single_seq_len = max([anc_len, desc_len])
        
        if align_len > max_align_len:
            max_align_len = align_len
        
        all_aligned_mats.append( alignment )
        all_anc.append(ancestor)
        all_desc.append(descendant)
        all_counts_sets.append(counts_dict)
        del alignment, ancestor, descendant, counts_dict, 
        del align_len, anc_len, desc_len, out, sample_rng_key
    
    step += 1

del step, t


###############################################################################
### prepare outputs for Pair Align   ##########################################
###############################################################################

###############################
### concatenate all samples   #
###############################
def pad_and_stack(arr_lst, pad_to):
    out = []
    for arr in arr_lst:
        padding_len = pad_to - arr.shape[0]
        
        
        if len(arr.shape)==1:
            padded_arr = jnp.pad( arr, 
                                  (0, padding_len)
                                  )
        
        elif len(arr.shape)==2:
            padded_arr = jnp.pad( arr, 
                                  ( (0, padding_len), (0,0) ) 
                                  )
        
        out.append(padded_arr)
        
    return jnp.stack(out, axis=0)
    

### stack the generated alignments, true hidden site classes
all_aligned_mats = pad_and_stack( all_aligned_mats, 
                           max_align_len )

all_anc = pad_and_stack( all_anc, 
                         max_single_seq_len )

all_desc = pad_and_stack( all_desc, 
                          max_single_seq_len )

all_match_counts = []
all_ins_counts = []
all_del_counts = []
all_transit_counts = []
all_emit_counts = np.zeros( (4,),dtype='int32' )

for d in all_counts_sets:
    all_match_counts.append( d['match_counts'] )
    all_ins_counts.append( d['ins_counts'] )
    all_del_counts.append( d['del_counts'] )
    all_transit_counts.append( d['transit_counts'] )
    all_emit_counts += d['all_emit_counts']
    
all_match_counts = jnp.stack( all_match_counts, axis=0 )
all_ins_counts = jnp.stack( all_ins_counts, axis=0 )
all_del_counts = jnp.stack( all_del_counts, axis=0 )
all_transit_counts = jnp.stack( all_transit_counts, axis=0 )
del all_counts_sets


########################
### validate outputs   #
########################
# validate the total emission counts
for i in [3,4,5,6]:
    pred_val = all_emit_counts[i-3]
    true_val = (all_aligned_mats[:,:,[0,1]] == i).sum()
    assert pred_val == true_val
    
# loop through all samples to validate
for b in validate_samps:
    ### 1.) validate full length inputs
    check_align = all_aligned_mats[b,...]
    pred_anc = all_anc[b,...]
    pred_desc = all_desc[b,...]

    # only one bos, and it's at the beginning
    assert check_align[0,0] == 1
    assert check_align[0,1] == 1

    check_bos = (check_align == 1).sum(axis=0)    
    assert check_bos[0] == 1 
    assert check_bos[1] == 1 
    del check_bos
    
    assert check_align[0,2] == 4
    assert (check_align[...,2] == 4).sum(axis=0) == 1
    
    # only one eos, and it's at the end
    align_len = (check_align[...,2]!=0).sum(axis=0)
    anc_len = ( (pred_anc!=0) & (pred_anc!=43) ).sum(axis=0)
    desc_len = ( (pred_desc!=0) & (pred_anc!=43) ).sum(axis=0)
    
    check_eos = (check_align == 2).sum(axis=0)    
    assert check_eos[0] == 1
    assert check_eos[1] == 1 
    del check_eos
    assert (check_align[...,2] == 5).sum(axis=0) == 1
    
    assert check_align[align_len-1, 0] == 2
    assert check_align[align_len-1, 1] == 2
    assert check_align[align_len-1, 2] == 5
    assert pred_anc[anc_len-1] == 2
    assert pred_desc[desc_len-1] == 2
    
    # make sure the ungapped ancestor and descendant given in the alignment
    #   matches the ungapped ancestor and descendant returned by the simulation
    #   function
    pred_anc = pred_anc[(pred_anc!=0)]
    pred_desc = pred_desc[(pred_desc!=0)]
    
    true_anc = []
    true_desc = []
    for l in range(check_align.shape[0]):
        a = check_align[l,0]
        d = check_align[l,1]
        p = check_align[l,2]
        
        # pad
        if p == 0:
            break
        
        # start, match, end
        elif p in [4, 1, 5]:
            true_anc.append(a)
            true_desc.append(d)
        
        # ins
        elif p == 2:
            true_desc.append(d)
        
        # del
        elif p == 3:
            true_anc.append(a)
        
        else:
            raise ValueError(f'unforseen alignment token: {p}')

    assert jnp.allclose( jnp.stack(true_anc), pred_anc )  
    assert jnp.allclose( jnp.stack(true_desc), pred_desc )  
    del true_anc, true_desc, a, d, p, l
    
    
    ### 2.)validate counts sets
    check_match_counts = all_match_counts[b,...]
    check_ins_counts = all_ins_counts[b,...]
    check_del_counts = all_del_counts[b,...]
    check_transit_counts = all_transit_counts[b,...]
    
    # check alignment length by total emissions
    checksum = ( check_match_counts.sum() +
                 check_ins_counts.sum() +
                 check_del_counts.sum() )
    assert np.allclose( align_len-2, checksum )
    del checksum
    
    # check alignment length by transition counts
    checksum = check_transit_counts.sum( axis=(-2,-1) )
    assert np.allclose( align_len, checksum+1 )
    del checksum
    
    # check ancestor length by total emissions
    checksum = ( check_match_counts.sum() +
                 check_del_counts.sum() )
    assert np.allclose( anc_len-2, checksum )
    
    # check ancestor length by corresponding transition counts
    checksum = check_transit_counts[:,[0,2]].sum()
    assert np.allclose( anc_len-2, checksum )
    del checksum
    
    # check descendant length by total emissions
    checksum = ( check_match_counts.sum() +
                 check_ins_counts.sum() )
    assert np.allclose( desc_len-2, checksum )
    del checksum
    
    # check descendant length by corresponding transition counts
    checksum = check_transit_counts[:,[0,1]].sum()
    assert np.allclose( desc_len-2, checksum )
    del checksum
    del check_match_counts, check_ins_counts, check_del_counts
    del align_len, anc_len, desc_len, b, pred_anc, pred_desc
    del check_align, check_transit_counts
    

####################################################################
### once validated, stack ungapped seqs, reverse pairs, and save   #
####################################################################
ungapped_seqs = jnp.stack( [all_anc, all_desc], axis=-1 )
del all_anc, all_desc

### create reverse pair set, append
ungapped_seqs = jnp.concatenate( [ungapped_seqs, ungapped_seqs[...,[1,0]] ] )
all_match_counts = jnp.concatenate( [all_match_counts,
                                     jnp.transpose(all_match_counts, (0,2,1))],
                                   axis=0)
new_all_ins_counts = jnp.concatenate( [all_ins_counts, all_del_counts], axis=0 )
new_all_del_counts = jnp.concatenate( [all_del_counts, all_ins_counts], axis=0 )

all_ins_counts = new_all_ins_counts
all_del_counts = new_all_del_counts
del new_all_ins_counts, new_all_del_counts


# for aligned inputs, need to transform ins to del and vice versa
second_mat_seqs = all_aligned_mats[...,[1,0]]

state_for_second_mat = all_aligned_mats[...,2]
ins_pos = (state_for_second_mat == 2)
del_pos = (state_for_second_mat == 3)

state_for_second_mat = jnp.where( ins_pos,
                                  3,
                                  state_for_second_mat )

state_for_second_mat = jnp.where( del_pos,
                                  2,
                                  state_for_second_mat )

to_append = jnp.concatenate( [second_mat_seqs, state_for_second_mat[...,None]], axis=-1 )
all_aligned_mats = jnp.concatenate( [all_aligned_mats, to_append], axis=0 )

del second_mat_seqs, state_for_second_mat, ins_pos, del_pos, to_append


# rearrange transition counts matrix
rv_m_m = all_transit_counts[:, 0, 0]
rv_m_i = all_transit_counts[:, 0, 2]
rv_m_d = all_transit_counts[:, 0, 1]
rv_m_e = all_transit_counts[:, 0, 3]
row1 = jnp.stack([rv_m_m,
                  rv_m_i,
                  rv_m_d,
                  rv_m_e],
                 axis=-1)
del rv_m_m, rv_m_i, rv_m_d, rv_m_e

rv_i_m = all_transit_counts[:, 2, 0]
rv_i_i = all_transit_counts[:, 2, 2]
rv_i_d = all_transit_counts[:, 2, 1]
rv_i_e = all_transit_counts[:, 2, 3]
row2 = jnp.stack([rv_i_m,
                  rv_i_i,
                  rv_i_d,
                  rv_i_e],
                 axis=-1)
del rv_i_m, rv_i_i, rv_i_d, rv_i_e

rv_d_m = all_transit_counts[:, 1, 0]
rv_d_i = all_transit_counts[:, 1, 2]
rv_d_d = all_transit_counts[:, 1, 1]
rv_d_e = all_transit_counts[:, 1, 3]
row3 = jnp.stack([rv_d_m,
                  rv_d_i,
                  rv_d_d,
                  rv_d_e],
                 axis=-1)
del rv_d_m, rv_d_i, rv_d_d, rv_d_e

rv_e_m = all_transit_counts[:, 4, 0]
rv_e_i = all_transit_counts[:, 4, 2]
rv_e_d = all_transit_counts[:, 4, 1]
rv_e_e = all_transit_counts[:, 4, 3]
row4 = jnp.stack([rv_e_m,
                  rv_e_i,
                  rv_e_d,
                  rv_e_e],
                 axis=-1)
del rv_e_m, rv_e_i, rv_e_d, rv_e_e

rv_transit_counts = jnp.stack( [row1, row2, row3, row4], axis=-2 )
del row1, row2, row3, row4

all_transit_counts = jnp.concatenate( [all_transit_counts,
                                       rv_transit_counts],
                                     axis=0)
del rv_transit_counts


### assign names to each ancestor, descendant, and alignment pair
anc_names = [f'FakeAnc{i}' for i in range(n_samples)]
desc_names = [f'FakeDesc{i}' for i in range(n_samples)]
fw_pair_names = [f'FW_pair{i}' for i in range(n_samples)]
rv_pair_names = [f'RV_pair{i}' for i in range(n_samples)]

# subtract 2 for senitnel tokens
align_lens_wo_sents = (all_aligned_mats[...,2] != 0).sum(axis=1) - 2
anc_lens_wo_sents = ( (all_aligned_mats[...,0] != 0) & (all_aligned_mats[...,0] != 43) ).sum(axis=1) - 2
desc_lens_wo_sents = ( (all_aligned_mats[...,1] != 0) & (all_aligned_mats[...,1] != 43) ).sum(axis=1) - 2

# count main transition types
num_matches = all_match_counts.sum(axis=(-2, -1)).astype(int)
num_ins = all_ins_counts.sum(axis=-1).astype(int)
num_del = all_del_counts.sum(axis=-1).astype(int)

align_lens_wo_sents = list(align_lens_wo_sents)
anc_lens_wo_sents = list(anc_lens_wo_sents)
desc_lens_wo_sents = list(desc_lens_wo_sents)
num_matches = list(num_matches)
num_ins = list(num_ins)
num_del = list(num_del)


### compile metadata (both forward and reverse)
metadata = {'pairID': fw_pair_names + rv_pair_names,
           'ancestor': anc_names + desc_names,
           'descendant': desc_names + anc_names,
           'pfam': 'SIMULPF00001', 
           'anc_seq_len': anc_lens_wo_sents, 
           'desc_seq_len': desc_lens_wo_sents,
           'alignment_len': align_lens_wo_sents,
           'num_matches': num_matches,
           'num_ins': num_ins,
           'num_del': num_del 
           }
metadata = pd.DataFrame(metadata)


# validate the lengths in the metadata file
assert (( metadata['num_matches'] +
         metadata['num_ins'] +
         metadata['num_del'] ) == metadata['alignment_len']).all()

assert (( metadata['num_matches'] +
         metadata['num_del'] ) == metadata['anc_seq_len']).all()

assert (( metadata['num_matches'] +
         metadata['num_ins'] ) == metadata['desc_seq_len']).all()


# validate transition lengths again... just because
# alignment needs offset of one, but anc and desc lengths are fine as-is
checksum = all_transit_counts.sum( axis=(-2,-1) )
assert jnp.allclose( checksum-1, jnp.stack(align_lens_wo_sents) )
del checksum

checksum = all_transit_counts[...,[0,2]].sum( axis=(-2,-1) )
assert jnp.allclose( checksum, jnp.stack(anc_lens_wo_sents) )
del checksum

checksum = all_transit_counts[...,[0,1]].sum( axis=(-2,-1) )
assert jnp.allclose( checksum, jnp.stack(desc_lens_wo_sents) )
del checksum

# expand to 5x5
all_transit_counts_np = np.array(all_transit_counts)
matrix_with_zero_row = np.insert(all_transit_counts_np, 4, 0, axis=1)
all_transit_counts_to_save = np.insert(matrix_with_zero_row, 3, 0, axis=2)


### write everything
# numpy arrays
arrs_to_save = [ all_aligned_mats,
                 ungapped_seqs,
                 all_match_counts,
                 all_ins_counts,
                 all_del_counts,
                 all_transit_counts_to_save,
                 all_emit_counts
                 ]

suffixes = [ 'aligned_mats',
             'seqs_unaligned',
             'subCounts',
             'insCounts',
             'delCounts',
             'transCounts_five_by_five',
             'NuclCounts'
             ]




###############
### save arrs #
###############
with open(f'{param_file_prefixes}_TO_LOAD_ti_tv.npy','wb') as g:
    jnp.save(g, jnp.squeeze( jnp.stack([true_ti, true_tv]) ) )
    del true_ti, true_tv, g

with open(f'{param_file_prefixes}_TO_LOAD_tkf92_dict.pkl','wb') as g:
    out_dict = {'lam_mu': jnp.squeeze( jnp.array([true_lam, true_mu]) ),
                'r_extend': true_r_ext}
    pickle.dump(out_dict, g)
    del out_dict, true_lam, true_mu, true_r_ext, g
    
with open(f'{param_file_prefixes}_TO_LOAD_toy_align_times.txt','w') as g:
    [g.write(f'{t}'+'\n') for t in t_array]
    del t_array, g

with open(f'{param_file_prefixes}_TO_LOAD_rate_multiplier.npy','wb') as g:
    jnp.save(g, true_rate_mult )
    del true_rate_mult, g

with open(f'{param_file_prefixes}_TO_LOAD_equl.npy','wb') as g:
    jnp.save(g, true_indel_emit_prob )
    del true_indel_emit_prob, g

with open(f'{param_file_prefixes}_TO_LOAD_class_probs.npy','wb') as g:
    jnp.save(g, true_class_probs )
    del true_class_probs, g


for i in range( len(arrs_to_save) ):
    with open(f'{dset_name}_{suffixes[i]}.npy', 'wb') as g:
        jnp.save( g, arrs_to_save[i] )

metadata.to_csv(f'{dset_name}_metadata.tsv', sep='\t')


