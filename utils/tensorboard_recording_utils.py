#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 22:25:03 2025

@author: annabel
"""
from collections.abc import MutableMapping
import jax.numpy as jnp
import numpy as np
import pandas as pd


########################
### smaller functions  #
########################
def flatten_convert(dictionary, 
                    parent_key=None, 
                    separator = '/'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_convert(value, 
                                          new_key, 
                                          separator=separator).items()
                          )
        else:
            items.append((new_key, np.array(value)))
    
    return dict(items)


def calc_stats(mat, 
               name):
    # don't include last forward-slash
    name = name[:-1] if name[-1] == '/' else name
    
    # stats
    perc_zeros = (mat==0).sum() / mat.size
    mean_without_zeros = mat.sum() / (mat!=0).sum()
    max_vals = mat.max()
    min_vals = mat.min()
    mean_vals = mat.mean()
    variance = mat.var()
    
    # output dict (with forward-slash added back in)
    to_write = {f'{name}/MAX': mat.max().item(),
                f'{name}/MIN': mat.min().item(),
                f'{name}/MEAN': mat.mean().item(),
                f'{name}/VAR': mat.var().item(),
                f'{name}/MEAN-WITHOUT-ZEROS': mean_without_zeros.item(),
                f'{name}/PERC-ZEROS': perc_zeros.item()}
    
    return to_write

def update_stats_dict(old_dict, new_dict):
    # if no values have been written yet, initialize with new_dict
    if old_dict == dict():
        return new_dict
    
    # otherwise, add to values
    for key in new_dict.keys():
        old_val = old_dict[key]
        new_val = new_dict[key]
        
        # maxes: update with new max
        if key.endswith('MAX'):
            new_max = np.max( [old_val, new_val] )
            old_dict[key] = new_max
        
        # mins: update with new min
        elif key.endswith('MIN'):
            new_min = np.min( [old_val, new_val] )
            old_dict[key] = new_min
        
        # means: take new average (treat as same sizes for now, but a more 
        #        precise updated average would take into account the number 
        #        of elements in previous matrix and new matrix)
        elif key.endswith('MEAN'):
            old_dict[key] = (old_val + new_val)/2
        
        # variance: kind of sloppy, but use average of variances for now
        #           (again, more precise measurement would take unequal sample
        #           sizes into account...)
        elif key.endswith('VAR'):
            old_dict[key] = (old_val + new_val)/2
        
        # means without zeros: take new average, different sample sizes 
        #                      disclaimer bla bla bla
        elif key.endswith('MEAN-WITHOUT-ZEROS'):
            old_dict[key] = (old_val + new_val)/2
        
        # percent zeros: take new average, different sample sizes 
        #                      disclaimer bla bla bla
        elif key.endswith('PERC-ZEROS'):
            old_dict[key] = (old_val + new_val)/2
    
    return old_dict
    
def format_tag(top_layer_name, layer_name):
    # reformat the tags to be: (tag1 | tag2 | tag3)/lowest_tag
    raw_tag = top_layer_name + layer_name
    raw_tag = raw_tag.split('/')
    prefix = ' | '.join(raw_tag[0:-1])
    suffix = raw_tag[-1]
    tag = f'{prefix}/{suffix}'
    return tag


#####################################
### tensorboard writing functions   #
#####################################
def write_times(cpu_start, 
                cpu_end, 
                real_start, 
                real_end, 
                tag, 
                step, 
                writer_obj):
    writer_obj.add_scalar(tag =f'Code Timing | {tag}/CPU+sys time', 
                      scalar_value = cpu_end - cpu_start, 
                      global_step = step)
    
    writer_obj.add_scalar(tag =f'Code Timing | {tag}/Real time', 
                      scalar_value = real_end - real_start, 
                      global_step = step)


def write_scalars_from_dict(flat_dict, 
                            top_layer_name, 
                            writer_obj, 
                            global_step):
    for layer_name, layer_values in flat_dict.items():
        tag = format_tag(top_layer_name, layer_name)
        
        writer_obj.add_scalar(tag=tag,
                          scalar_value =layer_values,
                          global_step=global_step)
        
        
def write_histograms_from_dict(flat_dict, 
                               top_layer_name, 
                               writer_obj, 
                               global_step):
    for layer_name, layer_values in flat_dict.items():
        tag = format_tag(top_layer_name, layer_name)
        
        # writer_obj.add_histogram(tag=tag,
        #                      values=layer_values,
        #                      global_step=global_step)

def write_stats_to_tabular(flat_dict,
                           writer_obj):
    ### reorganize into dictionaries grouped by tag
    to_table = {}

    for key, val in flat_dict.items():
        key_parts = key.split('/')
        top_level_tag = '/'.join(key_parts[:-1]).replace('/',' | ')
        bottom_tag = key_parts[-1]
        
        if top_level_tag not in to_table.keys():
            to_table[top_level_tag] = [{'stat':bottom_tag,'value':val}]
        
        elif top_level_tag in to_table.keys():
            to_table[top_level_tag].append( {'stat':bottom_tag,'value':val} )
            
            
    ### convert each table to a markdown table output and write to text
    for top_level_tag, dict_to_table in to_table.items():
        markdown_table = pd.DataFrame(dict_to_table).to_markdown()
        writer_obj.add_text(tag = top_level_tag,
                            text_string = markdown_table,
                            global_step = 0)
    


#################################################
### functions used in training and final eval   #
#################################################
def weight_summary_stats(all_trainstates,
                         tag_prefix):
    """
    keys have naming convention:
        tag_prefix/WEIGHTS/layer_name/statistic_name
    
    when written, will look like
        tag_prefix | WEIGHTS | layer_name/statistic_name
        
    layer_name should also contain ancestor/descendant/outproj distinction, 
        since the flax modules include this in the name 
    
    tag_prefix denotes when during training loop (like "in train loop" or 
        "in final eval")
    
    out_dict is a flat dictionary (i.e. NOT nested)
    """
    out_dict = {}
    
    for tstate in all_trainstates:
        param_dict = flatten_convert( tstate.params.get('params', dict()) )
        
        for layer_name, param_mat in param_dict.items():
            
            ## for some reason, the first two labels keep getting duplicated; 
            ## just manually remove these
            #layer_name = '/'.join( layer_name.split('/')[2:] )
            
            layer_for_tag = f'{tag_prefix}/WEIGHTS/'+layer_name
            to_add = calc_stats(mat = param_mat, 
                                name = layer_for_tag)
            out_dict = {**out_dict, **to_add}
    
    return out_dict
            

###########################################
### functions used during training loop   #
###########################################
def grads_summary_stats(gradient_dictionaries,
                        tag_prefix):
    """
    keys have naming convention:
        tag_prefix/GRADIENTS/layer_name/statistic_name
    
    when written, will look like
        tag_prefix | GRADIENTS | layer_name/statistic_name
        
    layer_name should also contain ancestor/descendant/outproj distinction, 
        since the flax modules include this in the name 
    
    tag_prefix denotes when during training loop (like "in train loop" or 
        "in final eval")
    
    out_dict is a flat dictionary (i.e. NOT nested)
    """
    out_dict = {}
    for which_module_grad in ['enc_gradient',
                              'dec_gradient',
                              'finalpred_gradient']:
        grad_dict = gradient_dictionaries[which_module_grad]
        grad_dict = grad_dict.get('params', dict() )
        grad_dict = flatten_convert( grad_dict )
        
        for layer_name, grad_mat in grad_dict.items():
            ## for some reason, the first two labels keep getting duplicated; 
            ## just manually remove these
            #layer_name = '/'.join( layer_name.split('/')[2:] )
            
            layer_for_tag = f'{tag_prefix}/GRADIENTS/'+layer_name
                
            to_add = calc_stats(mat = grad_mat, 
                                name = layer_for_tag)
            out_dict = {**out_dict, **to_add}
    
    return out_dict


def write_adam_optimizer_summary_stats(all_trainstates,
                                       writer_obj,
                                       global_step):
    """
    keys have naming convention:
        ADAM OPTIMIZER (varname)/layer_name/statistic_name
    
    when written, will look like
        ADAM OPTIMIZER (varname) | layer_name | statistic_name
        
    layer_name should also contain ancestor/descendant/outproj distinction, 
        since the flax modules include this in the name 
    
    writes to tensorboard without returning any values
    """
    out_dict = {}
    
    for tstate in all_trainstates:
        ### mu
        mu = tstate.opt_state.inner_opt_state[0].mu.get( 'params', dict() )
        mu = flatten_convert( mu )
        
        for layer_name, param_mat in mu.items():
            ## for some reason, the first two labels keep getting duplicated; 
            ## just manually remove these
            #layer_name = '/'.join( layer_name.split('/')[2:] )
        
            layer_for_tag = f'ADAM OPTIMIZER (mu)/'+layer_name
            to_add = calc_stats(mat = param_mat, 
                                name = layer_for_tag)
            out_dict = {**out_dict, **to_add}
        
        ### nu
        nu = tstate.opt_state.inner_opt_state[0].nu.get( 'params', dict() )
        nu = flatten_convert( nu )

        for layer_name, param_mat in nu.items():
            ## for some reason, the first two labels keep getting duplicated; 
            ## just manually remove these
            #layer_name = '/'.join( layer_name.split('/')[2:] )
            
            layer_for_tag = f'ADAM OPTIMIZER (nu)/'+layer_name
            to_add = calc_stats(mat = param_mat, 
                                name = layer_for_tag)
            out_dict = {**out_dict, **to_add}
        
    write_scalars_from_dict(flat_dict=out_dict, 
                            top_layer_name='',
                            writer_obj=writer_obj, 
                            global_step=global_step)


def write_optimizer_updates(all_updates,
                            writer_obj,
                            global_step):
    """
    keys have naming convention:
        PARAM UPDATE/layer_name/statistic_name
    
    when written, will look like
        PARAM UPDATE | layer_name | statistic_name
        
    layer_name should also contain ancestor/descendant/outproj distinction, 
        since the flax modules include this in the name 
    
    writes to tensorboard without returning any values
    """
    out_dict = {}
    
    for update_obj in all_updates:
        update_dict = update_obj.get( 'params', dict() )
        update_dict = flatten_convert( update_dict )
        
        for layer_name, param_mat in update_dict.items():
            ## for some reason, the first two labels keep getting duplicated; 
            ## just manually remove these
            #layer_name = '/'.join( layer_name.split('/')[2:] )
        
            layer_for_tag = f'PARAM UPDATE/'+layer_name
            to_add = calc_stats(mat = param_mat, 
                                name = layer_for_tag)
            out_dict = {**out_dict, **to_add}
        
    write_scalars_from_dict(flat_dict=out_dict, 
                            top_layer_name='',
                            writer_obj=writer_obj, 
                            global_step=global_step)


def write_optional_outputs_during_training(writer_obj, 
                                           all_trainstates,
                                           global_step, 
                                           dict_of_values, 
                                           interms_for_tboard, 
                                           write_histograms_flag):
    """
    in the training loop, could record the following (under certain flags)
    
    scalars:
        - sowed intermediates statistics
        - weights (calculate statistics first!)
        - gradients (calculate statistics first!)
        - optimizer states (mu, nu, updates)  (calculate statistics first!)
        
    histograms (periodically):
        - weights
        - gradients
    """
    ### intermediates sowed by the models
    if interms_for_tboard['encoder_sow_outputs']:
        flat_dict = flatten_convert( dict_of_values['anc_layer_metrics']['scalars'] )
        write_scalars_from_dict(flat_dict=flat_dict,  
                                top_layer_name=f'IN TRAIN LOOP/ANC_INTERMS/',
                                writer_obj=writer_obj,
                                global_step=global_step)
        del flat_dict
    
    if interms_for_tboard['decoder_sow_outputs']:
        flat_dict = flatten_convert( dict_of_values['desc_layer_metrics']['scalars'] )
        write_scalars_from_dict(flat_dict=flat_dict,
                                top_layer_name=f'IN TRAIN LOOP/DESC_INTERMS/',
                                writer_obj=writer_obj,                        
                                global_step=global_step)
        del flat_dict
    
    if interms_for_tboard['finalpred_sow_outputs']:
        flat_dict = flatten_convert( dict_of_values['pred_layer_metrics']['scalars'] )
        write_scalars_from_dict(flat_dict=flat_dict, 
                                top_layer_name=f'IN TRAIN LOOP/FINALPRED_INTERMS/',
                                writer_obj=writer_obj, 
                                global_step=global_step)
        del flat_dict
    
    
    ### weights; already flattened with top_layer_name
    if interms_for_tboard['weights']:
        flat_dict = weight_summary_stats(all_trainstates = all_trainstates,
                                         tag_prefix = 'IN TRAIN LOOP')
        write_scalars_from_dict(flat_dict=flat_dict, 
                                top_layer_name='',
                                writer_obj=writer_obj, 
                                global_step=global_step)
        del flat_dict
        
        # also possibly output histogram
        if write_histograms_flag:
            for tstate in all_trainstates:
                param_dict = flatten_convert( tstate.params.get('params', dict()) )
                write_histograms_from_dict(flat_dict=param_dict, 
                                           top_layer_name='IN TRAIN LOOP/WEIGHTS/',
                                           writer_obj=writer_obj, 
                                           global_step=global_step)
            del param_dict
        
            
    
    ### gradients; also already flattened with top_layer_name
    if interms_for_tboard['gradients']:
        gradient_dictionaries = {key: val for key, val in dict_of_values.items() if key in 
                                 ['enc_gradient', 'dec_gradient','finalpred_gradient']}        
        flat_dict = grads_summary_stats(gradient_dictionaries = gradient_dictionaries,
                                tag_prefix = 'IN TRAIN LOOP') 
        write_scalars_from_dict(flat_dict=flat_dict, 
                                top_layer_name='',
                                writer_obj=writer_obj, 
                                global_step=global_step)
        del flat_dict
        
        # also possibly output histogram
        if write_histograms_flag:
            for key, grad_dict in gradient_dictionaries.items():
                grad_dict = flatten_convert( grad_dict.get('params',dict()) )
                write_histograms_from_dict(flat_dict=grad_dict, 
                                           top_layer_name='IN TRAIN LOOP/GRADIENTS/',
                                           writer_obj=writer_obj, 
                                           global_step=global_step)
            del grad_dict
    
    
    ### optimizer updates; functions defined above
    if interms_for_tboard['optimizer']:
        # mu, nu
        write_adam_optimizer_summary_stats(all_trainstates = all_trainstates,
                                           writer_obj=writer_obj,
                                           global_step=global_step)
        
        # updates
        all_updates = [dict_of_values.get(item, dict()) for item in
                       ['encoder_updates', 'decoder_updates', 'finalpred_updates']
                       ]
        write_optimizer_updates(all_updates = all_updates,
                                writer_obj=writer_obj,
                                global_step=global_step)
        
        

###########################################
### functions used in final eval wrapper  #
###########################################
def calc_stats_during_final_eval(all_trainstates,
                                 dict_of_values, 
                                 interms_for_tboard,
                                 top_level_tag):
    """
    calculate stats for the following:
        - embeddings
        - final logits
        - final logprobs
    
    # todo: could add stats for attention weights...
    
    (stats already calculated for sowed intermediates)
    
    """
    out_dict = dict()
    
    flags_keynames = [('ancestor_embeddings', 'final_ancestor_embeddings'),
                      ('descendant_embeddings', 'final_descendant_embeddings'),
                      ('final_logprobs', 'final_logprobs')]
    
    arrays_dict = dict()
    for flagname, keyname in flags_keynames:
        if (interms_for_tboard[flagname] and
            keyname in dict_of_values.keys()):
            to_add = calc_stats(mat = dict_of_values[keyname], 
                                name = f'FINAL-EVAL/{top_level_tag}/{keyname}')
            arrays_dict = {**arrays_dict, **to_add}
            del to_add
    out_dict = {**out_dict, **arrays_dict}
    
    # check final outputs from forward pass separately
    if interms_for_tboard['forward_pass_outputs']:
        for keyname in dict_of_values.keys():
            if keyname.startswith('FPO_'):
                to_add = calc_stats(mat = dict_of_values[keyname],
                                    name = f'FINAL-EVAL/{top_level_tag}/{keyname.replace("FPO_","")}')
    
    # stats for sowed intermediates are already calculated, so just flatten
    #   and add
    sowed_dict = dict()
    if interms_for_tboard['encoder_sow_outputs']:
        flat_dict = flatten_convert( dict_of_values['anc_layer_metrics']['scalars'],
                                    parent_key = f'FINAL-EVAL/{top_level_tag}/ANC_INTERMS')
        sowed_dict = {**sowed_dict, **flat_dict}
        del flat_dict
    
    if interms_for_tboard['decoder_sow_outputs']:
        flat_dict = flatten_convert( dict_of_values['desc_layer_metrics']['scalars'],
                                    parent_key = f'FINAL-EVAL/{top_level_tag}/DESC_INTERMS' )
        sowed_dict = {**sowed_dict, **flat_dict}
        del flat_dict
    
    if interms_for_tboard['finalpred_sow_outputs']:
        flat_dict = flatten_convert( dict_of_values['pred_layer_metrics']['scalars'],
                                    parent_key = f'FINAL-EVAL/{top_level_tag}/FINALPRED_INTERMS' )
        sowed_dict = {**sowed_dict, **flat_dict}
        del flat_dict
    out_dict = {**out_dict, **sowed_dict}
    del sowed_dict
    
    return out_dict
    
    